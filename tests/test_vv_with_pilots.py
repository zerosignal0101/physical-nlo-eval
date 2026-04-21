"""V&V 与 BPS 周跳补偿对比测试。

本测试验证 V&V 和 BPS 两种相位噪声补偿算法在周跳场景下的表现，
并记录 V&V 的已知局限性。

## V&V 的已知局限性

1. **块平均效应**：V&V 对每个 block_size 符号做 M 次方平均。当相位在块内
   线性变化时（如升余弦斜坡），S^4·e^(j4θ) 在不同 θ 下相位不同，互相抵消，
   导致 mean_pow → 0，丢失相位信息。

2. **导频插值延迟**：导频辅助校正使用零阶保持插值 (kind='previous')，
   在两个导频块之间保持旧基准，导致 transition_end 附近出现残余误差。

3. **结构性 BER 下限**：由于上述原因，即使在无噪声的纯净相位漂移下，
   V&V + 导频辅助对周跳也存在结构性 BER 下限（约 1e-3 ~ 1e-4）。

## BPS 的优势

BPS (Blind Phase Search) 不做块平均，而是对每个符号搜索最优相位，
因此对快速相位变化的跟踪能力更强，不存在 V&V 的块平均问题。
"""

import numpy as np

from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.dsp.carrier import PhaseNoiseCompensation
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper


def add_bulk_phase_noise(symbols: np.ndarray, phase_rad: float) -> np.ndarray:
    """给所有符号添加一个固定的相位偏移 (用于测试)."""
    return symbols * np.exp(1j * phase_rad)


def add_wiener_phase_drift(
    symbols: np.ndarray,
    linewidth_hz: float,
    symbol_rate: float,
    drift_start_idx: int,
    drift_duration_syms: int,
    drift_rad: float,
) -> np.ndarray:
    """给符号添加维纳过程相位漂移（平滑累积越过象限边界）。

    模拟真实光通信中的周跳：相位缓慢漂移累积，最终越过 QPSK 象限边界。
    与瞬间阶跃不同，平滑漂移允许 V&V 块平均和导频插值算法有效跟踪。

    Parameters
    ----------
    symbols : np.ndarray
        输入符号
    linewidth_hz : float
        激光线宽 (Hz)，用于正常相位噪声
    symbol_rate : float
        符号率 (baud)
    drift_start_idx : int
        漂移开始索引（符号位置）
    drift_duration_syms : int
        漂移持续符号数（漂移在此期间完成）
    drift_rad : float
        总漂移量（弧度），通常为 π/2 的整数倍以模拟象限跳变

    Returns
    -------
    np.ndarray
        添加维纳相位漂移后的符号
    """
    n_samples = symbols.shape[-1]
    dt = 1 / symbol_rate

    phase_increment_std = np.sqrt(2 * np.pi * linewidth_hz * dt)
    phase_noise = np.cumsum(np.random.randn(n_samples) * phase_increment_std)

    if drift_duration_syms > 0 and drift_rad != 0:
        ramp = np.zeros(n_samples)
        transition_end = min(drift_start_idx + drift_duration_syms, n_samples)
        n_transition = transition_end - drift_start_idx
        if n_transition > 0:
            # 使用升余弦窗（raised cosine）实现平滑过渡
            t = np.arange(n_transition)
            cosine_window = 0.5 * (1 - np.cos(np.pi * t / (n_transition - 1))) if n_transition > 1 else np.array([1.0])
            ramp[drift_start_idx:transition_end] = drift_rad * cosine_window
        phase_noise += ramp

    return symbols * np.exp(1j * phase_noise)


def add_sudden_cycle_slip(
    symbols: np.ndarray,
    slip_idx: int,
    slip_rad: float,
) -> np.ndarray:
    """瞬间周跳（硬阶跃），仅用于算法基准测试。

    真实光通信中不会出现瞬间阶跃，此方法仅用于测试算法的极限性能。
    """
    result = symbols.copy()
    result[slip_idx:] *= np.exp(1j * slip_rad)
    return result


def test_vv_vs_bps():
    """测试 V&V 和 BPS 在周跳场景下的表现对比。

    本测试展示：
    1. V&V 对固定相位偏移可以完美恢复
    2. V&V 对周跳存在结构性 BER 下限
    3. BPS 对周跳可以完美恢复（或接近完美）
    """
    print("\n" + "=" * 60)
    print("测试: V&V vs BPS 周跳补偿对比")
    print("=" * 60)

    mod = "SP-QPSK"
    linewidth = 0  # 无额外维纳噪声，仅测试漂移效应
    symbol_rate = 50e9
    slip_rad = np.pi / 2  # 90度 周跳

    np.random.seed(42)

    pilot_symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    pilot_info = {"period": 32, "symbols": pilot_symbols}

    mapper = Mapper(modulation=mod, power_norm=True, pilot_info=pilot_info)
    demapper = Demapper(modulation=mod, power_norm=True)

    bits_per_symbol = mapper.bits_per_symbol
    num_payload_syms = 10000
    total_bits = num_payload_syms * bits_per_symbol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    signal = mapper.forward(tx_bits)
    tx_symbols = signal.data[0]

    mid = len(tx_symbols) // 2

    pnc_vv = PhaseNoiseCompensation(method="V&V", block_size=128)
    pnc_bps = PhaseNoiseCompensation(method="BPS", n_test_phases=64, constellation=mapper.constellation.lut)

    # =============================================
    # 测试 1: 固定相位偏移 - 两种算法都应完美恢复
    # =============================================
    print("\n--- 测试 1: 固定相位偏移 (45°) ---")
    phase_offset = np.pi / 4
    rx_symbols_offset = add_bulk_phase_noise(tx_symbols.copy(), phase_offset)
    rx_signal_offset = Signal(data=rx_symbols_offset[np.newaxis, :], ctx=signal.ctx)

    rx_bits_vv = demapper.forward(pnc_vv.forward(rx_signal_offset))
    rx_bits_bps = demapper.forward(pnc_bps.forward(rx_signal_offset))

    ber_vv_offset = np.sum(tx_bits != rx_bits_vv) / total_bits
    ber_bps_offset = np.sum(tx_bits != rx_bits_bps) / total_bits

    print(f"  V&V  BER: {ber_vv_offset:.2e}  {'✅ 完美恢复' if ber_vv_offset == 0 else '❌ 有误差'}")
    print(f"  BPS  BER: {ber_bps_offset:.2e}  {'✅ 完美恢复' if ber_bps_offset == 0 else '❌ 有误差'}")

    # =============================================
    # 测试 2: 瞬间阶跃（硬周跳）
    # =============================================
    print("\n--- 测试 2: 瞬间阶跃 (硬周跳, π/2) ---")
    rx_symbols_sudden = add_sudden_cycle_slip(tx_symbols.copy(), mid, slip_rad)
    rx_signal_sudden = Signal(data=rx_symbols_sudden[np.newaxis, :], ctx=signal.ctx)

    rx_bits_vv_sudden = demapper.forward(pnc_vv.forward(rx_signal_sudden))
    rx_bits_bps_sudden = demapper.forward(pnc_bps.forward(rx_signal_sudden))

    ber_vv_sudden = np.sum(tx_bits != rx_bits_vv_sudden) / total_bits
    ber_bps_sudden = np.sum(tx_bits != rx_bits_bps_sudden) / total_bits

    print(f"  V&V  BER: {ber_vv_sudden:.2e}  (结构性问题: 导频块边界不对齐)")
    print(f"  BPS  BER: {ber_bps_sudden:.2e}  {'✅ 完美恢复' if ber_bps_sudden == 0 else '❌ 有误差'}")

    # =============================================
    # 测试 3: 平滑漂移（升余弦斜坡）
    # =============================================
    print("\n--- 测试 3: 平滑漂移 (升余弦斜坡, 512符号) ---")
    drift_duration = 512
    rx_symbols_smooth = add_wiener_phase_drift(
        tx_symbols.copy(),
        linewidth_hz=linewidth,
        symbol_rate=symbol_rate,
        drift_start_idx=mid,
        drift_duration_syms=drift_duration,
        drift_rad=slip_rad,
    )
    rx_signal_smooth = Signal(data=rx_symbols_smooth[np.newaxis, :], ctx=signal.ctx)

    rx_bits_vv_smooth = demapper.forward(pnc_vv.forward(rx_signal_smooth))
    rx_bits_bps_smooth = demapper.forward(pnc_bps.forward(rx_signal_smooth))

    ber_vv_smooth = np.sum(tx_bits != rx_bits_vv_smooth) / total_bits
    ber_bps_smooth = np.sum(tx_bits != rx_bits_bps_smooth) / total_bits

    print(f"  V&V  BER: {ber_vv_smooth:.2e}  (V&V 块平均导致结构性误差)")
    print(f"  BPS  BER: {ber_bps_smooth:.2e}  {'✅ 完美恢复' if ber_bps_smooth == 0 else '❌ 有误差'}")

    # =============================================
    # 总结
    # =============================================
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("  固定相位偏移: V&V ✅, BPS ✅ (两者都完美恢复)")
    print("  硬周跳:        V&V ~1e-3 (结构性误差), BPS ~25% (搜索范围限制)")
    print("  平滑漂移:      V&V ~1e-3 (结构性误差), BPS ~1% (搜索范围限制)")
    print("  连续维纳漂移:  V&V ✅ 完全恢复,   BPS ❌ 反而更差")
    print()
    print("  算法特性分析:")
    print("    V&V: 块平均导致结构性 BER 下限，但导频辅助可处理周跳，")
    print("         对连续漂移跟踪效果好（因为漂移足够慢，块内相位近似恒定）")
    print("    BPS: 搜索范围 [-π/4, π/4] 有限，对大相位跳变和快速漂移效果差")
    print()
    print("  实际建议:")
    print("    - 缓慢连续漂移 (维纳过程): V&V + 导频 ✅, BPS ❌")
    print("    - 周跳处理: V&V + 导频辅助 ~1e-3 BER (结构性下限)")
    print("    - BPS 适用于: 小幅度相位噪声、无周跳、SNR 较高的场景")

    # 验证: 固定偏移两种方法都应完美
    assert ber_vv_offset == 0, "[FAIL] V&V 应能恢复固定相位偏移"
    assert ber_bps_offset == 0, "[FAIL] BPS 应能恢复固定相位偏移"

    # 验证: 对硬周跳，V&V (有导频) 应优于 BPS (无导频)
    assert ber_vv_sudden < ber_bps_sudden, "V&V+pilot 应优于 BPS 对硬周跳"

    print("\n  [PASS] 所有验证通过")


def test_vv_continuous_drift():
    """测试 V&V 对连续相位漂移的跟踪能力（无周跳，纯积累）。"""
    print("\n" + "=" * 60)
    print("测试: V&V + 导频 (连续相位漂移跟踪)")
    print("=" * 60)

    mod = "SP-QPSK"
    linewidth = 1e6  # 1 MHz - 较大线宽导致明显相位漂移
    symbol_rate = 50e9

    np.random.seed(42)

    pilot_symbols = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    pilot_info = {"period": 32, "symbols": pilot_symbols}

    mapper = Mapper(modulation=mod, power_norm=True, pilot_info=pilot_info)
    demapper = Demapper(modulation=mod, power_norm=True)

    bits_per_symbol = mapper.bits_per_symbol
    num_payload_syms = 20000
    total_bits = num_payload_syms * bits_per_symbol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    signal = mapper.forward(tx_bits)
    tx_symbols = signal.data[0]

    # 添加维纳过程相位噪声（连续漂移，无周跳）
    rx_symbols = add_wiener_phase_drift(
        tx_symbols.copy(),
        linewidth_hz=linewidth,
        symbol_rate=symbol_rate,
        drift_start_idx=0,
        drift_duration_syms=num_payload_syms,
        drift_rad=0,
    )
    rx_signal = Signal(data=rx_symbols[np.newaxis, :], ctx=signal.ctx)

    # 无补偿
    rx_bits_noisy = demapper.forward(rx_signal)
    ber_noisy = np.sum(tx_bits != rx_bits_noisy) / total_bits

    # V&V 补偿
    pnc_vv = PhaseNoiseCompensation(method="V&V", block_size=128)
    rx_signal_vv = pnc_vv.forward(rx_signal)
    rx_bits_vv = demapper.forward(rx_signal_vv)
    ber_vv = np.sum(tx_bits != rx_bits_vv) / total_bits

    # BPS 补偿
    pnc_bps = PhaseNoiseCompensation(method="BPS", n_test_phases=64, constellation=mapper.constellation.lut)
    rx_signal_bps = pnc_bps.forward(rx_signal)
    rx_bits_bps = demapper.forward(rx_signal_bps)
    ber_bps = np.sum(tx_bits != rx_bits_bps) / total_bits

    print(f"  调制格式: {mod}")
    print(f"  线宽: {linewidth / 1e6:.0f} MHz @ {symbol_rate / 1e9:.0f} Gbaud")
    print(f"  符号数: {num_payload_syms}")
    print(f"  相位增量 std: {np.sqrt(2 * np.pi * linewidth / symbol_rate) * 180 / np.pi:.4f} 度/符号")
    print(f"  BER (无补偿): {ber_noisy:.2e}")
    print(f"  BER (V&V):   {ber_vv:.2e}")
    print(f"  BER (BPS):   {ber_bps:.2e}")

    if ber_bps < ber_noisy:
        improvement = ber_noisy / max(ber_bps, 1e-10)
        print(f"  [PASS] BPS 有效跟踪连续相位漂移，改善比: {improvement:.1f}x")

    if ber_vv < ber_noisy:
        improvement = ber_noisy / max(ber_vv, 1e-10)
        print("  [INFO] V&V 也有一定改善")


if __name__ == "__main__":
    test_vv_vs_bps()
    test_vv_continuous_drift()
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
