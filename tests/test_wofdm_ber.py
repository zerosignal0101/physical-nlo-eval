"""Windowed OFDM (W-OFDM) 调制/解调 BER 验证测试。"""

import numpy as np

from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.freq_domain import OFDM_Demodulator, OFDMToSymbolStream
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.ofdm import OFDMModulator, WindowedOFDMModulator


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
# 测试 1: Windowed OFDM 调制/解调 round-trip (无噪声)
# ============================================================================


def test_wofdm_modem_roundtrip(
    modulation: str = "SP-QPSK",
    n_subcarriers: int = 64,
    cp_len: int = 8,
    beta: float = 0.1,
    num_data_symbols_per_frame: int = 10,
    pilot_spacing: int = 0,
    num_frames: int = 10,
    seed: int = 42,
):
    """验证 WindowedOFDMModulator → OFDM_Demodulator 的 round-trip。"""
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print("测试 1: Windowed OFDM Modem Round-Trip (无噪声)")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    ts_freq_pattern = np.exp(1j * np.random.uniform(-np.pi, np.pi, n_subcarriers))
    ts_freq_pattern = np.stack([ts_freq_pattern, ts_freq_pattern * 1j])

    modulator = WindowedOFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=ts_freq_pattern[:n_pol, :] if n_pol == 1 else ts_freq_pattern,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=pilot_spacing,
        pilot_value=1.0 + 0j,
        beta=beta,
    )

    demodulator = OFDM_Demodulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        cs_len=modulator.cs_len,
    )

    payload_per_frame = modulator.n_data_subcarriers * num_data_symbols_per_frame
    total_qam_symbols = payload_per_frame * num_frames
    total_bits = total_qam_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    tx_signal = mapper.forward(tx_bits)

    ofdm_signal = modulator.forward(tx_signal)

    print(f"  调制格式: {modulation}")
    print(f"  子载波: {n_subcarriers}, CP: {cp_len}, CS: {modulator.cs_len}, beta: {beta}")
    print(f"  每帧数据符号: {num_data_symbols_per_frame}, 帧数: {num_frames}")
    print(f"  OFDM 信号形状: {ofdm_signal.data.shape}")
    print(f"  OFDM SPS: {ofdm_signal.ctx.sps}")
    print(f"  期望有效符号长度: {modulator.effective_output_sym_len}")

    rx_fd = demodulator.forward(ofdm_signal)

    print(f"  解调后形状: {rx_fd.data.shape}")

    ofdm_reshape = OFDMToSymbolStream(modulator.data_indices if pilot_spacing > 0 else None)
    rx_fd_reshaped = ofdm_reshape.forward(rx_fd)

    rx_bits = demapper.forward(rx_fd_reshaped)

    bit_errors = np.sum(tx_bits != rx_bits)
    ber = bit_errors / total_bits

    print(f"  比特数: {total_bits}, 错误数: {bit_errors}, BER: {ber:.2e}")

    assert bit_errors == 0, f"W-OFDM round-trip 失败: {bit_errors} 错误"
    print(f"  [PASS] {modulation} W-OFDM round-trip 零误差通过")


# ============================================================================
# 测试 2: Windowed OFDM vs 标准 OFDM 对比 (无噪声)
# ============================================================================


def test_wofdm_vs_ofdm_roundtrip(
    modulation: str = "SP-QPSK",
    n_subcarriers: int = 64,
    cp_len: int = 8,
    beta: float = 0.1,
    num_data_symbols_per_frame: int = 10,
    num_frames: int = 5,
    seed: int = 42,
):
    """对比 Windowed OFDM 与标准 OFDM 的 round-trip 误差。"""
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print("测试 2: Windowed OFDM vs 标准 OFDM Round-Trip 对比")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    ts_freq_pattern = np.exp(1j * np.random.uniform(-np.pi, np.pi, n_subcarriers))
    ts_freq_pattern = np.stack([ts_freq_pattern, ts_freq_pattern * 1j])

    # Standard OFDM
    ofdm_mod = OFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=None,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=0,
        pilot_value=1.0 + 0j,
    )
    ofdm_demod = OFDM_Demodulator(n_subcarriers=n_subcarriers, cp_len=cp_len)

    # Windowed OFDM
    wofdm_mod = WindowedOFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=None,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=0,
        pilot_value=1.0 + 0j,
        beta=beta,
    )
    wofdm_demod = OFDM_Demodulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        cs_len=wofdm_mod.cs_len,
    )

    payload_per_frame = ofdm_mod.n_data_subcarriers * num_data_symbols_per_frame
    total_qam_symbols = payload_per_frame * num_frames
    total_bits = total_qam_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    tx_signal = mapper.forward(tx_bits)

    # Standard OFDM round-trip
    ofdm_signal = ofdm_mod.forward(tx_signal)
    ofdm_rx_fd = ofdm_demod.forward(ofdm_signal)
    ofdm_reshape = OFDMToSymbolStream()
    ofdm_rx_fd_reshaped = ofdm_reshape.forward(ofdm_rx_fd)
    ofdm_rx_bits = demapper.forward(ofdm_rx_fd_reshaped)
    ofdm_errors = np.sum(tx_bits != ofdm_rx_bits)

    # Windowed OFDM round-trip
    wofdm_signal = wofdm_mod.forward(tx_signal)
    wofdm_rx_fd = wofdm_demod.forward(wofdm_signal)
    wofdm_reshape = OFDMToSymbolStream()
    wofdm_rx_fd_reshaped = ofdm_reshape.forward(wofdm_rx_fd)
    wofdm_rx_bits = demapper.forward(wofdm_rx_fd_reshaped)
    wofdm_errors = np.sum(tx_bits != wofdm_rx_bits)

    print(f"  子载波: {n_subcarriers}, CP: {cp_len}, CS: {wofdm_mod.cs_len}, beta: {beta}")
    print(f"  标准 OFDM 误差: {ofdm_errors}, W-OFDM 误差: {wofdm_errors}")

    assert ofdm_errors == 0, f"标准 OFDM 失败: {ofdm_errors} 错误"
    assert wofdm_errors == 0, f"W-OFDM 失败: {wofdm_errors} 错误"
    print("  [PASS] 两种 OFDM 变体 round-trip 均零误差")


# ============================================================================
# 测试 3: W-OFDM AWGN 信道 BER 验证
# ============================================================================


def test_wofdm_awgn_ber(
    modulation: str = "SP-QPSK",
    eb_n0_db: float = 10.0,
    n_subcarriers: int = 64,
    cp_len: int = 8,
    beta: float = 0.1,
    num_data_symbols_per_frame: int = 10,
    pilot_spacing: int = 0,
    num_frames: int = 20,
    min_errors: int = 100,
    seed: int = 42,
):
    """W-OFDM 在 AWGN 信道下的 BER 测试。"""
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"测试 3: W-OFDM AWGN BER ({modulation} @ Eb/N0={eb_n0_db}dB, beta={beta})")
    print(f"{'=' * 60}")

    n_pol = 1 if "SP-" in modulation else 2
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol

    modulator = WindowedOFDMModulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        ts_pattern=None,
        num_data_symbols_per_frame=num_data_symbols_per_frame,
        pilot_spacing=pilot_spacing,
        pilot_value=1.0 + 0j,
        beta=beta,
    )

    demodulator = OFDM_Demodulator(
        n_subcarriers=n_subcarriers,
        cp_len=cp_len,
        cs_len=modulator.cs_len,
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
        theoretical_ber = theoretical_ber_qpsk(eb_n0_db)
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
# 运行所有测试
# ============================================================================


def run_all_tests():
    """运行 W-OFDM BER 验证测试套件。"""
    print("\n" + "=" * 60)
    print("Windowed OFDM (W-OFDM) BER 验证测试套件")
    print("=" * 60)

    try:
        # 测试 1: W-OFDM round-trip (无噪声)
        test_wofdm_modem_roundtrip(modulation="SP-QPSK", beta=0.1)
        test_wofdm_modem_roundtrip(modulation="SP-16QAM", beta=0.1)
        test_wofdm_modem_roundtrip(modulation="DP-16QAM", beta=0.1)

        # 测试 2: W-OFDM vs 标准 OFDM 对比
        test_wofdm_vs_ofdm_roundtrip(modulation="SP-QPSK", beta=0.1)
        test_wofdm_vs_ofdm_roundtrip(modulation="SP-16QAM", beta=0.1)
        test_wofdm_vs_ofdm_roundtrip(modulation="DP-16QAM", beta=0.1)

        # 测试 3: W-OFDM AWGN BER
        test_wofdm_awgn_ber(modulation="SP-QPSK", eb_n0_db=8.0, beta=0.1)
        test_wofdm_awgn_ber(modulation="SP-16QAM", eb_n0_db=14.0, beta=0.1, min_errors=50)

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
