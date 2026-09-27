"""WDM 梳测试：相位噪声 + 定时偏差下的 BER 验证。

测试内容:
1. WDM 合波/分波基本功能
2. 多通道在相位噪声和定时偏差下的 BER 表现
"""

import numpy as np

from physics_nlo_eval.channel.wdm import ChannelExtractor, ChannelSpec, WDMDemux, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.dsp.carrier import PhaseNoiseCompensation
from physics_nlo_eval.dsp.time_domain import TimingRecovery
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.rrc import RRCFilter

# ============================================================================
# 辅助函数
# ============================================================================


def make_wdm_plan(n_channels: int, spacing_hz: float) -> WDMPlan:
    """构建等间隔、以参考频率为中心的灵活栅格规划。

    等价于旧 ``WDMMux(channel_spacing_hz=...)`` 的通道布局：
    通道频偏对称分布在参考频率两侧。
    """
    center = (n_channels - 1) / 2
    return WDMPlan(
        reference_freq_hz=193.1e12,
        channels=[
            ChannelSpec(channel_id=i, freq_offset_hz=(i - center) * spacing_hz) for i in range(n_channels)
        ],
    )


def add_phase_noise(symbols: np.ndarray, linewidth_hz: float, symbol_rate: float) -> np.ndarray:
    """给符号添加维纳过程相位噪声."""
    n_samples = symbols.shape[-1]
    dt = 1 / symbol_rate
    phase_increment_std = np.sqrt(2 * np.pi * linewidth_hz * dt)
    phase_noise = np.cumsum(np.random.randn(n_samples) * phase_increment_std)
    return symbols * np.exp(1j * phase_noise)


def apply_fractional_delay(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    """对信号施加分数样点延迟（通过 Farrow 插值）."""
    n_pol, n = sig.shape
    out = np.zeros_like(sig, dtype=np.complex128)

    for p in range(n_pol):
        for i in range(n):
            fi = i - delay_samples
            base = int(np.floor(fi))
            mu = fi - base

            if base - 1 < 0 or base + 2 >= n:
                out[p, i] = sig[p, i]
                continue

            xm1 = sig[p, base - 1]
            x0 = sig[p, base]
            x1 = sig[p, base + 1]
            x2 = sig[p, base + 2]

            c0 = x0
            c1 = -(1 / 3) * xm1 - 0.5 * x0 + x1 - (1 / 6) * x2
            c2 = 0.5 * xm1 - x0 + 0.5 * x1
            c3 = -(1 / 6) * xm1 + 0.5 * x0 - 0.5 * x1 + (1 / 6) * x2

            out[p, i] = ((c3 * mu + c2) * mu + c1) * mu + c0

    return out


def add_awgn(symbols: np.ndarray, eb_n0_db: float, bits_per_symbol: int) -> np.ndarray:
    """给符号添加 AWGN 噪声."""
    eb_n0 = 10 ** (eb_n0_db / 10)
    es = np.mean(np.abs(symbols) ** 2)
    eb = es / bits_per_symbol
    n0 = eb / eb_n0
    sigma = np.sqrt(n0 / 2)
    noise = sigma * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape))
    return symbols + noise


# ============================================================================
# 测试 1: WDM 合波/分波基本功能
# ============================================================================


def _verify_channel_ber(rx_sig, all_bits, channel_idx, rx_rrc, demapper, desc="通道"):
    """验证单通道 BER 的辅助函数 (含 RRC 匹配滤波和延迟搜索)."""
    # 匹配滤波
    rx_sig_filtered = rx_rrc.forward(rx_sig)

    # 下采样到 1 SPS
    data_1sps = rx_sig_filtered.data[:, ::4]
    rx_sig_1sps = Signal(data_1sps, rx_sig_filtered.ctx)
    rx_sig_1sps.ctx.sps = 1

    rx_bits = demapper.forward(rx_sig_1sps)

    # 延迟搜索 (RRC 群时延对齐)
    best_ber = 1.0
    for delay in range(0, 40):
        test_bits = rx_bits[delay : delay + len(all_bits[channel_idx])]
        if len(test_bits) < len(all_bits[channel_idx]):
            break
        errs = np.sum(all_bits[channel_idx] != test_bits)
        ber = errs / len(all_bits[channel_idx])
        if ber < best_ber:
            best_ber = ber

    print(f"  {desc} {channel_idx}: BER = {best_ber:.2e}")
    return best_ber


def test_wdm_mux_demux_basic():
    """Test basic WDM mux/demux functionality with RRC pulse shaping."""
    print("\n" + "=" * 60)
    print("测试 1: WDM 合波/分波基本功能 (RRC 脉冲成形)")
    print("=" * 60)

    np.random.seed(42)

    modulation = "SP-QPSK"
    num_symbols = 4000
    symbol_rate = 50e9
    n_channels = 3
    channel_spacing_hz = 75e9

    mapper = Mapper(modulation=modulation)
    tx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=False)
    rx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=True)
    demapper = Demapper(modulation=modulation)

    channels = []
    all_bits = []

    for _i in range(n_channels):
        tx_bits = np.random.randint(0, 2, size=num_symbols * 2, dtype=bool)
        all_bits.append(tx_bits)

        sig_base = mapper.forward(tx_bits)
        sig_base.ctx.sample_rate = symbol_rate
        sig_base.ctx.sym_rate = symbol_rate
        sig_rrc = tx_rrc.forward(sig_base)
        channels.append(sig_rrc)

    wdm_mux = WDMMux(wdm_plan=make_wdm_plan(n_channels, channel_spacing_hz), target_sample_rate_hz=400e9)
    wdm_signal = wdm_mux.combine(channels)

    wdm_demux = WDMDemux()
    demuxed_signal = wdm_demux.forward(wdm_signal)

    all_passed = True
    for i in range(n_channels):
        extractor = ChannelExtractor(channel_idx=i)
        rx_sig = extractor.forward(demuxed_signal)

        best_ber = _verify_channel_ber(rx_sig, all_bits, i, rx_rrc, demapper)
        status = "[PASS]" if best_ber < 1e-4 else "[FAIL]"
        print(f"  通道 {i}: BER = {best_ber:.2e} {status}")
        if best_ber > 1e-4:
            all_passed = False

    assert all_passed, "test_wdm_mux_demux_basic: 存在通道 BER 超过阈值 1e-4，见上方各通道输出"


def test_wdm_with_phase_noise_bps():
    """测试 WDM 在相位噪声下的 BER (BPS 补偿)."""
    print("\n" + "=" * 60)
    print("测试 2: WDM 相位噪声补偿 (BPS 方法)")
    print("=" * 60)

    np.random.seed(42)

    modulation = "SP-QPSK"
    num_symbols = 4000
    symbol_rate = 50e9
    n_channels = 3
    channel_spacing_hz = 75e9
    linewidth_hz = 100e3  # 100 kHz 载波线宽 (更合理的值)

    mapper = Mapper(modulation=modulation)
    tx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=False)
    rx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=True)
    demapper = Demapper(modulation=modulation)

    # BPS 相位噪声补偿
    pn_comp = PhaseNoiseCompensation(method="BPS", n_test_phases=64, block_size=128)

    channels = []
    all_bits = []

    for _i in range(n_channels):
        tx_bits = np.random.randint(0, 2, size=num_symbols * 2, dtype=bool)
        all_bits.append(tx_bits)

        sig_base = mapper.forward(tx_bits)
        sig_base.ctx.sample_rate = symbol_rate
        sig_base.ctx.sym_rate = symbol_rate

        sig_rrc = tx_rrc.forward(sig_base)

        # 添加相位噪声
        sig_noisy = Signal(data=add_phase_noise(sig_rrc.data, linewidth_hz, symbol_rate * 4), ctx=sig_rrc.ctx)
        channels.append(sig_noisy)

    wdm_mux = WDMMux(wdm_plan=make_wdm_plan(n_channels, channel_spacing_hz), target_sample_rate_hz=400e9)
    wdm_signal = wdm_mux.combine(channels)

    wdm_demux = WDMDemux()
    demuxed_signal = wdm_demux.forward(wdm_signal)

    all_passed = True
    for i in range(n_channels):
        extractor = ChannelExtractor(channel_idx=i)
        rx_sig = extractor.forward(demuxed_signal)

        # 匹配滤波
        rx_sig_filtered = rx_rrc.forward(rx_sig)

        # 相位噪声补偿 (BPS)
        rx_sig_comp = pn_comp.forward(rx_sig_filtered)

        # 下采样到 1 SPS
        data_1sps = rx_sig_comp.data[:, ::4]
        rx_sig_1sps = Signal(data_1sps, rx_sig_comp.ctx)
        rx_sig_1sps.ctx.sps = 1

        rx_bits = demapper.forward(rx_sig_1sps)

        # 延迟搜索
        best_ber = 1.0
        for delay in range(0, 40):
            test_bits = rx_bits[delay : delay + len(all_bits[i])]
            if len(test_bits) < len(all_bits[i]):
                break
            errs = np.sum(all_bits[i] != test_bits)
            ber = errs / len(all_bits[i])
            if ber < best_ber:
                best_ber = ber

        status = "[PASS]" if best_ber < 1e-3 else "[FAIL]"
        print(f"  通道 {i}: BER = {best_ber:.2e} {status}")
        if best_ber > 1e-3:
            all_passed = False

    assert all_passed, "test_wdm_with_phase_noise_bps: 存在通道 BER 超过阈值 1e-3，见上方各通道输出"


def test_wdm_with_timing_offset():
    """测试 WDM 在定时偏差下的 BER (Oerder-Meyer 定时恢复)."""
    print("\n" + "=" * 60)
    print("测试 3: WDM 定时偏差补偿 (Oerder-Meyer 方法)")
    print("=" * 60)

    np.random.seed(42)

    modulation = "SP-QPSK"
    num_symbols = 4000
    symbol_rate = 50e9
    n_channels = 3
    channel_spacing_hz = 75e9
    delay_samples = 1.5  # 分数样点定时偏差

    mapper = Mapper(modulation=modulation)
    tx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=False)
    rx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=True)
    demapper = Demapper(modulation=modulation)

    # Oerder-Meyer 定时恢复
    timing_recovery = TimingRecovery(sps_in=4, block_size=1000)

    channels = []
    all_bits = []

    for _i in range(n_channels):
        tx_bits = np.random.randint(0, 2, size=num_symbols * 2, dtype=bool)
        all_bits.append(tx_bits)

        sig_base = mapper.forward(tx_bits)
        sig_base.ctx.sample_rate = symbol_rate
        sig_base.ctx.sym_rate = symbol_rate

        sig_rrc = tx_rrc.forward(sig_base)

        # 添加分数样点定时偏差
        sig_delayed = Signal(data=apply_fractional_delay(sig_rrc.data, delay_samples), ctx=sig_rrc.ctx)
        channels.append(sig_delayed)

    wdm_mux = WDMMux(wdm_plan=make_wdm_plan(n_channels, channel_spacing_hz), target_sample_rate_hz=400e9)
    wdm_signal = wdm_mux.combine(channels)

    wdm_demux = WDMDemux()
    demuxed_signal = wdm_demux.forward(wdm_signal)

    all_passed = True
    for i in range(n_channels):
        extractor = ChannelExtractor(channel_idx=i)
        rx_sig = extractor.forward(demuxed_signal)

        # 匹配滤波
        rx_sig_filtered = rx_rrc.forward(rx_sig)

        # 定时恢复 (输出 1 SPS)
        rx_sig_synced = timing_recovery.forward(rx_sig_filtered)

        rx_bits = demapper.forward(rx_sig_synced)

        # 延迟搜索
        best_ber = 1.0
        for delay in range(0, 40):
            test_bits = rx_bits[delay : delay + len(all_bits[i])]
            if len(test_bits) < len(all_bits[i]):
                break
            errs = np.sum(all_bits[i] != test_bits)
            ber = errs / len(all_bits[i])
            if ber < best_ber:
                best_ber = ber

        status = "[PASS]" if best_ber < 1e-3 else "[FAIL]"
        print(f"  通道 {i}: BER = {best_ber:.2e} {status}")
        if best_ber > 1e-3:
            all_passed = False

    assert all_passed, "test_wdm_with_timing_offset: 存在通道 BER 超过阈值 1e-3，见上方各通道输出"


def test_wdm_with_pilot_assisted_vv():
    """测试 WDM 在相位噪声下的 BER (V&V 导频辅助方法).

    V&V 必须工作在 1 SPS，且需要导频信息来消除 M 次方的 ±π/2 模糊度。
    在 WDM 环境下，使用 FIR 滤波 + 直接 ::4 下采样替代 TimingRecovery，
    避免插值引入的相位畸变影响 V&V 的块平均跟踪。
    """
    print("\n" + "=" * 60)
    print("测试 4: WDM 导频辅助 V&V 相位噪声补偿")
    print("=" * 60)

    np.random.seed(42)

    modulation = "SP-QPSK"
    num_symbols = 20000
    symbol_rate = 50e9
    n_channels = 3
    channel_spacing_hz = 75e9
    linewidth_hz = 1e6  # 1 MHz 载波线宽

    # 导频配置: 每 32 符号插入 4 个导频
    pilot_symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    pilot_info = {"period": 32, "symbols": pilot_symbols}

    # Mapper 和 Demapper 都要配置导频信息
    mapper = Mapper(modulation=modulation, power_norm=True, pilot_info=pilot_info)
    demapper = Demapper(modulation=modulation, power_norm=True)

    tx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=False)
    rx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=True)

    # V&V 相位噪声补偿 (在 1 SPS 下使用)
    pn_comp = PhaseNoiseCompensation(method="V&V", block_size=128)

    channels = []
    all_bits = []

    for _i in range(n_channels):
        bits_per_symbol = mapper.bits_per_symbol
        tx_bits = np.random.randint(0, 2, size=num_symbols * bits_per_symbol, dtype=bool)
        all_bits.append(tx_bits)

        # Mapper.forward 会自动插入导频
        sig_base = mapper.forward(tx_bits)
        sig_base.ctx.sample_rate = symbol_rate
        sig_base.ctx.sym_rate = symbol_rate
        # 导频信息已在 sig_base.ctx.meta['pilot_info'] 中

        sig_rrc = tx_rrc.forward(sig_base)

        # 添加相位噪声
        sig_noisy = Signal(data=add_phase_noise(sig_rrc.data, linewidth_hz, symbol_rate * 4), ctx=sig_rrc.ctx)
        channels.append(sig_noisy)

    wdm_mux = WDMMux(wdm_plan=make_wdm_plan(n_channels, channel_spacing_hz), target_sample_rate_hz=400e9)
    wdm_signal = wdm_mux.combine(channels)

    # 使用 FIR 滤波的 WDMDemux
    wdm_demux = WDMDemux(filter_type="fir")
    demuxed_signal = wdm_demux.forward(wdm_signal)

    all_passed = True
    for i in range(n_channels):
        extractor = ChannelExtractor(channel_idx=i)
        rx_sig = extractor.forward(demuxed_signal)

        # 匹配 RRC 滤波
        rx_sig_filtered = rx_rrc.forward(rx_sig)

        # 直接 ::4 下采样到 1 SPS (不经过 TimingRecovery)
        data_1sps = rx_sig_filtered.data[:, ::4]
        rx_sig_1sps = Signal(data_1sps, rx_sig_filtered.ctx)
        rx_sig_1sps.ctx.sps = 1

        # V&V 相位噪声补偿 (在 1 SPS 下应用)
        rx_sig_comp = pn_comp.forward(rx_sig_1sps)

        rx_bits = demapper.forward(rx_sig_comp)

        # 延迟搜索
        best_ber = 1.0
        for delay in range(0, 40):
            test_bits = rx_bits[delay : delay + len(all_bits[i])]
            if len(test_bits) < len(all_bits[i]):
                break
            errs = np.sum(all_bits[i] != test_bits)
            ber = errs / len(all_bits[i])
            if ber < best_ber:
                best_ber = ber

        status = "[PASS]" if best_ber < 1e-3 else "[FAIL]"
        print(f"  通道 {i}: BER = {best_ber:.2e} {status}")
        if best_ber > 1e-3:
            all_passed = False

    assert all_passed, "test_wdm_with_pilot_assisted_vv: 存在通道 BER 超过阈值 1e-3，见上方各通道输出"


def test_wdm_high_order_qam():
    """测试 WDM 高阶 QAM (DP-16QAM) 性能."""
    print("\n" + "=" * 60)
    print("测试 5: WDM 高阶 QAM (DP-16QAM)")
    print("=" * 60)

    np.random.seed(42)

    modulation = "DP-16QAM"
    num_symbols = 4000
    symbol_rate = 50e9
    n_channels = 3
    channel_spacing_hz = 75e9

    mapper = Mapper(modulation=modulation)
    tx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=False)
    rx_rrc = RRCFilter(sps=4, roll_off=0.1, is_matched_filter=True)
    demapper = Demapper(modulation=modulation)

    channels = []
    all_bits = []

    for _i in range(n_channels):
        tx_bits = np.random.randint(0, 2, size=num_symbols * 4, dtype=bool)  # 16QAM: 4 bits/symbol
        all_bits.append(tx_bits)

        sig_base = mapper.forward(tx_bits)
        sig_base.ctx.sample_rate = symbol_rate
        sig_base.ctx.sym_rate = symbol_rate

        sig_rrc = tx_rrc.forward(sig_base)
        channels.append(sig_rrc)

    wdm_mux = WDMMux(wdm_plan=make_wdm_plan(n_channels, channel_spacing_hz), target_sample_rate_hz=400e9)
    wdm_signal = wdm_mux.combine(channels)

    wdm_demux = WDMDemux()
    demuxed_signal = wdm_demux.forward(wdm_signal)

    all_passed = True
    for i in range(n_channels):
        extractor = ChannelExtractor(channel_idx=i)
        rx_sig = extractor.forward(demuxed_signal)

        # 匹配滤波
        rx_sig_filtered = rx_rrc.forward(rx_sig)

        # 下采样到 1 SPS
        data_1sps = rx_sig_filtered.data[:, ::4]
        rx_sig_1sps = Signal(data_1sps, rx_sig_filtered.ctx)
        rx_sig_1sps.ctx.sps = 1

        rx_bits = demapper.forward(rx_sig_1sps)

        # 延迟搜索
        best_ber = 1.0
        for delay in range(0, 40):
            test_bits = rx_bits[delay : delay + len(all_bits[i])]
            if len(test_bits) < len(all_bits[i]):
                break
            errs = np.sum(all_bits[i] != test_bits)
            ber = errs / len(all_bits[i])
            if ber < best_ber:
                best_ber = ber

        status = "[PASS]" if best_ber < 1e-3 else "[FAIL]"
        print(f"  通道 {i}: BER = {best_ber:.2e} {status}")
        if best_ber > 1e-3:
            all_passed = False

    assert all_passed, "test_wdm_high_order_qam: 存在通道 BER 超过阈值 1e-3，见上方各通道输出"


def run_all_tests():
    """运行所有 WDM 测试."""
    print("\n" + "=" * 60)
    print("WDM 梳测试套件")
    print("  相位噪声 + 定时偏差下的 BER 验证")
    print("=" * 60)

    try:
        test_wdm_mux_demux_basic()
        test_wdm_with_phase_noise_bps()
        test_wdm_with_timing_offset()
        test_wdm_with_pilot_assisted_vv()
        test_wdm_high_order_qam()

        print("\n" + "=" * 60)
        print("所有 WDM 测试完成!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试出错: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
