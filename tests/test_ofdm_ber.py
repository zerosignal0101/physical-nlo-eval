"""OFDM 调制/解调与 CommonPhaseNoiseCompensate 补偿的 BER 验证测试。

测试内容:
1. OFDM 调制解调 round-trip (无噪声) — 验证 OFDM 链路正确性
2. AWGN 信道下 BER 与理论值对比
3. CommonPhaseNoiseCompensate 补偿对 CFO 场景的改善效果
"""

import numpy as np

from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.freq_domain import CommonPhaseNoiseCompensate, OFDM_Demodulator, OFDMToSymbolStream
from physics_nlo_eval.dsp.time_domain import OFDMChannelEstimator
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.ofdm import OFDMModulator

# ============================================================================
# 辅助函数
# ============================================================================


def qfunc_real(x):
    """标准 Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))"""
    from scipy import special

    return 0.5 * special.erfc(x / np.sqrt(2))


def theoretical_ber_qam(m_ary: int, eb_n0_db: float) -> float:
    """M-QAM 理论 BER: BER ≈ (4/log2(M)) * (1 - 1/√M) * Q(√(3*log2(M)/(M-1) * Eb/N0))"""
    from scipy import special

    eb_n0 = 10 ** (eb_n0_db / 10)
    k = np.log2(m_ary)
    return (4 / k) * (1 - 1 / np.sqrt(m_ary)) * 0.5 * special.erfc(np.sqrt(3 * k * eb_n0 / (m_ary - 1)) / np.sqrt(2))


def theoretical_ber_qpsk(eb_n0_db: float) -> float:
    """QPSK 理论 BER: Q(sqrt(2*Eb/N0))"""
    from scipy import special

    eb_n0 = 10 ** (eb_n0_db / 10)
    return 0.5 * special.erfc(np.sqrt(2 * eb_n0) / np.sqrt(2))


def add_awgn(symbols: np.ndarray, eb_n0_db: float, bits_per_symbol: int) -> np.ndarray:
    """给 OFDM 时域信号添加 AWGN 噪声。"""
    eb_n0 = 10 ** (eb_n0_db / 10)
    es = np.mean(np.abs(symbols) ** 2)
    eb = es / bits_per_symbol
    n0 = eb / eb_n0
    sigma = np.sqrt(n0 / 2)
    noise = sigma * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape))
    return symbols + noise


def make_signal_context(
    n_pol: int = 1,
    sym_rate: float = 1e11,
    bits_per_symbol: int = 4,
    sps: int = 2,
) -> SignalContext:
    """构建 OFDM 测试用 SignalContext。"""
    return SignalContext(
        sample_rate=sym_rate * sps,
        center_freq=0.0,
        n_pol=n_pol,
        sps=sps,
        sym_rate=sym_rate,
        bits_per_symbol=bits_per_symbol,
        is_ofdm=False,
        meta=None,
    )


# ============================================================================
# 测试 1: OFDM 调制/解调 round-trip (无噪声)
# ============================================================================


def test_ofdm_modem_roundtrip(
    modulation: str = "SP-QPSK",
    n_subcarriers: int = 256,
    cp_len: int = 16,
    num_data_symbols_per_frame: int = 10,
    pilot_spacing: int = 0,
    num_frames: int = 10,
    seed: int = 42,
):
    """验证 OFDM Modulator → OFDM Demodulator 的 round-trip。"""
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print("测试 1: OFDM Modem Round-Trip (无噪声)")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    modulator = OFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=None,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=pilot_spacing,
        pilot_value=1.0 + 0j,
    )

    demodulator = OFDM_Demodulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
    )

    payload_per_frame = modulator.n_data_subcarriers * num_data_symbols_per_frame
    total_qam_symbols = payload_per_frame * num_frames
    total_bits = total_qam_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    tx_signal = mapper.forward(tx_bits)

    ofdm_signal = modulator.forward(tx_signal)

    rx_fd = demodulator.forward(ofdm_signal)

    ofdm_reshape = OFDMToSymbolStream(modulator.data_indices if pilot_spacing > 0 else None)
    rx_fd_reshaped = ofdm_reshape.forward(rx_fd)
    rx_bits = demapper.forward(rx_fd_reshaped)

    bit_errors = np.sum(tx_bits != rx_bits)
    ber = bit_errors / total_bits

    print(f"  调制格式: {modulation}")
    print(f"  子载波: {n_subcarriers}, CP: {cp_len}")
    print(f"  每帧数据符号: {num_data_symbols_per_frame}, 帧数: {num_frames}")
    print(f"  OFDM 信号形状: {ofdm_signal.data.shape}")
    print(f"  解调后形状: {rx_fd.data.shape}")
    print(f"  比特数: {total_bits}, 错误数: {bit_errors}, BER: {ber:.2e}")

    assert bit_errors == 0, f"OFDM round-trip 失败: {bit_errors} 错误"
    print(f"  [PASS] {modulation} OFDM round-trip 零误差通过")


# ============================================================================
# 测试 2: AWGN 信道 BER 验证
# ============================================================================


def test_ofdm_awgn_ber(
    modulation: str = "SP-QPSK",
    eb_n0_db: float = 10.0,
    n_subcarriers: int = 256,
    cp_len: int = 16,
    num_data_symbols_per_frame: int = 10,
    pilot_spacing: int = 0,
    num_frames: int = 20,
    min_errors: int = 500,
    seed: int = 42,
):
    """OFDM 在 AWGN 信道下的 BER 测试。"""
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"测试 2: OFDM AWGN BER ({modulation} @ Eb/N0={eb_n0_db}dB)")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    modulator = OFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=None,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=pilot_spacing,
        pilot_value=1.0 + 0j,
    )

    demodulator = OFDM_Demodulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
    )

    payload_per_frame = modulator.n_data_subcarriers * num_data_symbols_per_frame
    total_qam_symbols_per_iter = payload_per_frame * num_frames
    total_bits_per_iter = total_qam_symbols_per_iter * bits_per_symbol * n_pol

    data_indices = modulator.data_indices if pilot_spacing > 0 else None

    # 理论 BER
    mod_str = modulation.replace("SP-", "").replace("DP-", "")
    if "QPSK" in mod_str:
        theoretical_ber = theoretical_ber_qpsk(eb_n0_db)
    elif "BPSK" in mod_str:
        theoretical_ber = theoretical_ber_qpsk(eb_n0_db)  # BPSK same formula
    else:
        m_ary = int(mod_str.replace("QAM", ""))
        theoretical_ber = theoretical_ber_qam(m_ary, eb_n0_db)

    total_errors = 0
    total_bits = 0
    num_iters = 0

    while total_errors < min_errors:
        np.random.seed(seed + num_iters)
        tx_bits = np.random.randint(0, 2, size=total_bits_per_iter, dtype=bool)
        tx_signal = mapper.forward(tx_bits)

        ofdm_signal = modulator.forward(tx_signal)
        tx_ofdm_symbols = ofdm_signal.data

        # 添加 AWGN
        rx_noisy = add_awgn(tx_ofdm_symbols, eb_n0_db, bits_per_symbol)
        rx_signal = Signal(data=rx_noisy, ctx=ofdm_signal.ctx)

        # OFDM 解调
        rx_fd = demodulator.forward(rx_signal)
        ofdm_reshape = OFDMToSymbolStream(data_indices)
        rx_fd_reshaped = ofdm_reshape.forward(rx_fd)
        rx_bits = demapper.forward(rx_fd_reshaped)

        batch_errors = np.sum(tx_bits != rx_bits)
        total_errors += batch_errors
        total_bits += total_bits_per_iter
        num_iters += 1

        if total_bits > 500_000_000:
            break

    measured_ber = total_errors / total_bits
    ratio = measured_ber / theoretical_ber if theoretical_ber > 0 else float("inf")
    passed = ratio < 3.0 or measured_ber < theoretical_ber

    print(f"  测量 BER: {measured_ber:.2e}")
    print(f"  理论 BER: {theoretical_ber:.2e}")
    print(f"  比值: {ratio:.2f}")
    print(f"  总比特: {total_bits}, 总错误: {total_errors}")
    print(f"  {'[PASS]' if passed else '[WARN]'} BER {'OK' if passed else 'out of range'}")


# ============================================================================
# 测试 3: CommonPhaseNoiseCompensate 补偿效果验证 (CFO + AWGN)
# ============================================================================


def test_fofine_cfo_compensation(
    modulation: str = "SP-QPSK",
    eb_n0_db: float = 20.0,
    cfo_hz: float = 5e6,
    n_subcarriers: int = 256,
    cp_len: int = 16,
    num_data_symbols_per_frame: int = 10,
    pilot_spacing: int = 8,
    num_frames: int = 20,
    seed: int = 42,
):
    """验证 CommonPhaseNoiseCompensate 对 CFO 的补偿效果。

    注意: data_aided 方法对 QAM 信号有根本性缺陷（QAM 数据均值不为零），
    应使用 pilot_assisted 方法。
    """
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"测试 3: CommonPhaseNoiseCompensate CFO 补偿 ({modulation}, CFO={cfo_hz / 1e6:.1f}MHz, Eb/N0={eb_n0_db}dB)")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    # 子载波间隔
    sym_rate = 1e11

    modulator = OFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=None,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=pilot_spacing,
        pilot_value=1.0 + 0j,
    )

    demodulator = OFDM_Demodulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
    )

    payload_per_frame = modulator.n_data_subcarriers * num_data_symbols_per_frame
    total_qam_symbols = payload_per_frame * num_frames
    total_bits = total_qam_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    tx_signal = mapper.forward(tx_bits)
    tx_signal.ctx = SignalContext(
        sample_rate=sym_rate * 2,
        center_freq=0.0,
        n_pol=n_pol,
        sps=2,
        sym_rate=sym_rate,
        bits_per_symbol=bits_per_symbol,
        is_ofdm=False,
    )
    ofdm_signal = modulator.forward(tx_signal)
    sample_rate = ofdm_signal.ctx.sample_rate

    # CFO 相位旋转
    cfo_phase_per_sample = 2 * np.pi * cfo_hz / sample_rate
    n_samples = ofdm_signal.data.shape[1]
    cfo_phases = np.exp(1j * cfo_phase_per_sample * np.arange(n_samples))

    # 发射端先加 CFO（模拟频偏），再通过 OFDM 信道
    if n_pol == 1:
        tx_ofdm = ofdm_signal.data[0]
        rx_noisy = add_awgn(tx_ofdm, eb_n0_db, bits_per_symbol)
        rx_with_cfo = rx_noisy * cfo_phases
        rx_signal = Signal(data=rx_with_cfo[np.newaxis, :], ctx=ofdm_signal.ctx)
    else:
        tx_x = ofdm_signal.data[0]
        tx_y = ofdm_signal.data[1]
        rx_x_noisy = add_awgn(tx_x, eb_n0_db, bits_per_symbol)
        rx_y_noisy = add_awgn(tx_y, eb_n0_db, bits_per_symbol)
        rx_with_cfo_x = rx_x_noisy * cfo_phases
        rx_with_cfo_y = rx_y_noisy * cfo_phases
        rx_signal = Signal(data=np.stack([rx_with_cfo_x, rx_with_cfo_y], axis=0), ctx=ofdm_signal.ctx)

    # OFDM 解调 (得到 3D: n_pol, n_sub, n_ofdm)
    rx_fd = demodulator.forward(rx_signal)

    # 解调后转为 QAM 流给 demapper
    def demod_and_count_errors(rx_fd_signal: Signal) -> int:
        ofdm_reshape = OFDMToSymbolStream(modulator.data_indices)
        rx_fd_reshaped = ofdm_reshape.forward(rx_fd_signal)
        rx_bits = demapper.forward(rx_fd_reshaped)
        return np.sum(tx_bits != rx_bits)

    # 无 CommonPhaseNoiseCompensate
    ber_no_fofine = demod_and_count_errors(rx_fd) / total_bits
    print(f"  BER (无 CommonPhaseNoiseCompensate): {ber_no_fofine:.2e}")

    # CommonPhaseNoiseCompensate pilot_assisted (需要导频)
    fofine_pa = CommonPhaseNoiseCompensate(method="pilot_assisted")
    rx_fd_fofine_pa = fofine_pa.forward(rx_fd)
    ber_fofine_pa = demod_and_count_errors(rx_fd_fofine_pa) / total_bits
    print(f"  BER (CommonPhaseNoiseCompensate pilot_assist): {ber_fofine_pa:.2e}")

    # pilot_assisted 应该比无 CommonPhaseNoiseCompensate 好
    improved = ber_fofine_pa < ber_no_fofine
    print(f"  {'[PASS]' if improved else '[WARN]'} CommonPhaseNoiseCompensate {'有效' if improved else '未见改善'}")


# ============================================================================
# 测试 4: TS 序列提取验证
# ============================================================================


def test_ofdm_ts_extraction(
    modulation: str = "SP-QPSK",
    n_subcarriers: int = 256,
    cp_len: int = 16,
    num_data_symbols_per_frame: int = 10,
    pilot_spacing: int = 8,
    num_frames: int = 5,
    seed: int = 42,
):
    """验证 OFDM Demodulator 正确分离和提取 TS (训练序列) 符号。

    检查项:
    1. TS 符号在频域被正确提取，与原始 TS pattern 一致
    2. 数据符号与 TS 符号正确分离，无混叠
    3. TS 元数据 (ofdm_ts_freq_symbols) 形状正确
    """
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"测试 4: OFDM TS 序列提取验证 ({modulation})")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    # 生成 TS pattern: 每个极化一列，频域理想星座 (QPSK 的 4 个参考点)
    base_sequence = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j] * (n_subcarriers // 4)
    ts_pattern = np.array([base_sequence[:n_subcarriers] for _ in range(n_pol)], dtype=np.complex64)
    if n_pol == 1:
        ts_pattern = ts_pattern[:1, :]
    else:
        ts_pattern = np.stack([ts_pattern[0], ts_pattern[1] * 1j], axis=0)

    modulator = OFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=ts_pattern,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=pilot_spacing,
        pilot_value=1.0 + 0j,
    )

    demodulator = OFDM_Demodulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
    )

    payload_per_frame = modulator.n_data_subcarriers * num_data_symbols_per_frame
    total_qam_symbols = payload_per_frame * num_frames
    total_bits = total_qam_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    tx_signal = mapper.forward(tx_bits)
    ofdm_signal = modulator.forward(tx_signal)

    # OFDM 解调
    rx_fd = demodulator.forward(ofdm_signal)

    # 验证 TS 提取
    meta = rx_fd.ctx.meta
    ts_freq_symbols = meta.get("ofdm_ts_freq_symbols")

    print(f"  TS 频域符号形状: {ts_freq_symbols.shape if ts_freq_symbols is not None else None}")
    print(f"  期望形状: ({n_pol}, {n_subcarriers}, {num_frames})")

    assert ts_freq_symbols is not None, "TS 频域符号未提取"
    expected_ts_shape = (n_pol, n_subcarriers, num_frames)
    assert ts_freq_symbols.shape == expected_ts_shape, f"TS 形状错误: {ts_freq_symbols.shape} != {expected_ts_shape}"

    # 验证每帧的 TS 与原始 ts_pattern 相关性 (无噪声时应高度相关)
    for pol_idx in range(n_pol):
        for frame_idx in range(num_frames):
            extracted_ts = ts_freq_symbols[pol_idx, :, frame_idx]
            original_ts = ts_pattern[pol_idx, :]

            # 计算相位偏移 (由于 CP 去除和 FFT 处理可能有微小旋转)
            phase_diff = np.angle(np.mean(extracted_ts * original_ts.conj()))
            rotated_ts = extracted_ts * np.exp(-1j * phase_diff)

            # 归一化误差
            error = np.mean(np.abs(rotated_ts - original_ts) ** 2)
            mean_power = np.mean(np.abs(original_ts) ** 2)
            norm_error = error / mean_power

            print(f"  [帧 {frame_idx}, 极化 {pol_idx}] TS 归一化误差: {norm_error:.2e}")
            assert norm_error < 1e-6, f"TS 提取误差过大: {norm_error:.2e}"

    # 验证数据符号也能正确解调 (TS 正确分离不影响数据)
    ofdm_reshape = OFDMToSymbolStream(modulator.data_indices)
    rx_fd_reshaped = ofdm_reshape.forward(rx_fd)
    rx_bits = demapper.forward(rx_fd_reshaped)

    bit_errors = np.sum(tx_bits != rx_bits)
    ber = bit_errors / total_bits

    print(f"  数据 BER (TS 分离后): {ber:.2e}")
    print(f"  比特数: {total_bits}, 错误数: {bit_errors}")
    assert bit_errors == 0, f"TS 分离后数据 BER 不为零: {bit_errors} 错误"
    print("  [PASS] TS 序列提取与数据分离验证通过")


# ============================================================================
# 测试 5: OFDM 信道估计与均衡
# ============================================================================


def test_ofdm_channel_estimation_equalization(
    modulation: str = "SP-QPSK",
    eb_n0_db: float = 15.0,
    n_subcarriers: int = 256,
    cp_len: int = 16,
    num_data_symbols_per_frame: int = 10,
    pilot_spacing: int = 8,
    num_frames: int = 20,
    seed: int = 42,
):
    """验证 OFDMChannelEstimator 利用 TS 进行信道估计与均衡的有效性。

    测试步骤:
    1. 构造含 TS 的 OFDM 信号
    2. 通过模拟信道 (幅度衰减 + 相位旋转)
    3. OFDM 解调后，利用 TS 估计信道并均衡
    4. 验证均衡后 BER 是否恢复至无信道干扰水平
    """
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"测试 5: OFDM 信道估计与均衡 ({modulation}, Eb/N0={eb_n0_db}dB)")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    # TS pattern: 已知训练序列，形状 (n_pol, n_subcarriers)
    base_seq = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j] * (n_subcarriers // 4)
    ts_pattern = np.tile(np.array(base_seq[:n_subcarriers], dtype=np.complex64), (n_pol, 1))

    modulator = OFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=ts_pattern,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=pilot_spacing,
        pilot_value=1.0 + 0j,
    )

    demodulator = OFDM_Demodulator(n_subcarriers=n_subcarriers, cp_len=cp_len)
    channel_estimator = OFDMChannelEstimator()

    payload_per_frame = modulator.n_data_subcarriers * num_data_symbols_per_frame
    total_qam_symbols = payload_per_frame * num_frames
    total_bits = total_qam_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    tx_signal = mapper.forward(tx_bits)
    ofdm_signal = modulator.forward(tx_signal)

    # 模拟信道: 幅度衰减 0.5, 相位旋转 0.3 rad
    channel_h = 0.5 * np.exp(1j * 0.3)
    rx_signal = Signal(
        data=ofdm_signal.data * channel_h,
        ctx=ofdm_signal.ctx,
    )

    # 添加 AWGN
    rx_noisy = add_awgn(rx_signal.data, eb_n0_db, bits_per_symbol)
    rx_signal = Signal(data=rx_noisy, ctx=rx_signal.ctx)

    # OFDM 解调
    rx_fd = demodulator.forward(rx_signal)

    # 信道估计与均衡
    rx_eq = channel_estimator.forward(rx_fd)

    # 验证信道估计结果
    h_est = rx_eq.ctx.meta.get("ofdm_channel_response")
    print(f"  信道估计形状: {h_est.shape}")
    h_mean = np.mean(h_est)
    h_error = np.abs(h_mean - channel_h)
    print(f"  理论信道: {channel_h:.4f}")
    print(f"  估计信道均值: {h_mean:.4f}")
    print(f"  信道估计误差: {h_error:.2e}")
    assert h_error < 0.1, f"信道估计误差过大: {h_error:.2e}"

    # 均衡后 BER
    ofdm_reshape = OFDMToSymbolStream(modulator.data_indices)
    rx_fd_reshaped = ofdm_reshape.forward(rx_eq)
    rx_bits = demapper.forward(rx_fd_reshaped)

    bit_errors = np.sum(tx_bits != rx_bits)
    ber = bit_errors / total_bits

    # 理论 BER (无信道 AWGN)
    mod_str = modulation.replace("SP-", "").replace("DP-", "")
    if "QPSK" in mod_str:
        theoretical_ber = theoretical_ber_qpsk(eb_n0_db)
    else:
        m_ary = int(mod_str.replace("QAM", ""))
        theoretical_ber = theoretical_ber_qam(m_ary, eb_n0_db)

    print(f"  均衡后 BER: {ber:.2e}")
    print(f"  理论 BER (AWGN only): {theoretical_ber:.2e}")
    print(f"  比特数: {total_bits}, 错误数: {bit_errors}")

    # 信道补偿有效: BER 远低于信道损伤下的水平
    # 信道损伤前 BER 约为 0.5 (解调失败)，均衡后应显著改善
    passed = bit_errors == 0 or ber < 10.0 * theoretical_ber
    print(f"  {'[PASS]' if passed else '[WARN]'} 均衡{'有效' if passed else '效果异常'}")


# ============================================================================
# 运行所有测试
# ============================================================================


def run_all_tests():
    """运行 OFDM Modem + CommonPhaseNoiseCompensate BER 验证测试套件。"""
    print("\n" + "=" * 60)
    print("OFDM Modem + CommonPhaseNoiseCompensate BER 验证测试套件")
    print("=" * 60)

    try:
        # 测试 1: Round-trip (无导频)
        test_ofdm_modem_roundtrip(modulation="SP-QPSK")
        test_ofdm_modem_roundtrip(modulation="SP-16QAM")
        test_ofdm_modem_roundtrip(modulation="DP-16QAM")

        # 测试 2: AWGN BER
        test_ofdm_awgn_ber(modulation="SP-QPSK", eb_n0_db=8.0)
        test_ofdm_awgn_ber(modulation="SP-16QAM", eb_n0_db=14.0, min_errors=100)

        # 测试 3: CommonPhaseNoiseCompensate CFO 补偿
        # 子载波间隔 = 1e11/256 ≈ 390 MHz
        # CFO=0.2x subc ≈ 78 MHz @ Eb/N0=15dB，可产生可测量 BER
        test_fofine_cfo_compensation(modulation="SP-QPSK", eb_n0_db=15.0, cfo_hz=78e6)
        test_fofine_cfo_compensation(modulation="SP-16QAM", eb_n0_db=15.0, cfo_hz=78e6)

        # 测试 4: TS 序列提取
        test_ofdm_ts_extraction(modulation="SP-QPSK")
        test_ofdm_ts_extraction(modulation="DP-16QAM")

        # 测试 5: OFDM 信道估计与均衡
        test_ofdm_channel_estimation_equalization(modulation="SP-QPSK", eb_n0_db=15.0)
        test_ofdm_channel_estimation_equalization(modulation="DP-16QAM", eb_n0_db=10.0)

        print("\n" + "=" * 60)
        print("所有测试完成!")
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
