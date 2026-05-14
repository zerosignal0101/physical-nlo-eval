"""Multi-span Fiber + IdealAmplifier + Raman + 完整接收 DSP 测试。

使用 dt ≤ 7.6 fs 时域网格满足 Raman 带宽要求。
升采样使用 Resampler, CD 补偿使用 ChromaticDispersionComp,
解调使用 Demapper + EVM/BER。
"""

import numpy as np
from pynlo.utility.chi3 import raman_from_dat

from physics_nlo_eval.channel.fiber import FiberChannel, IdealAmplifier
from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp, Resampler
from physics_nlo_eval.metrics.ber_evm import BERCalculator, Demapper
from physics_nlo_eval.tx.mapper import Constellation, Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter

# ============================================================================
# 参数
# ============================================================================
MODULATION = "DP-QPSK"
SYMBOL_RATE_HZ = 50e9
RRC_SPS = 4
RRC_ROLL_OFF = 0.1
LAUNCH_POWER_DBM_PER_POL = 0.0
FIBER_LEN_KM = 80.0
ALPHA_DB_PER_KM = 0.2
BETA2_PS2_PER_KM = -20.0
BETA3_PS3_PER_KM = 0.1e-3
N2_SI = 2.6e-20
A_EFF_UM2 = 80.0
AMP_NF_DB = 5.5
COARSE_STEP_M = 500.0
NUM_SYMBOLS = 256
RMAN_SPS = 4000  # Raman 网格 sps: sr = 4000 × 50e9 = 200 THz, dt = 5.0 fs < 7.6 fs
SEED = 42
RAMAN_FILE = "/home/zsig/Documents/Codes/PyNLO/pynlo/utility/RamanGain.dat"


def make_signal():
    """生成 DP-QPSK 信号 (sps=4, sr=200 GHz)."""
    np.random.seed(SEED)
    mapper = Mapper(modulation=MODULATION, power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    tx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF)
    iq_mod = IQModulator(launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL)

    total_bits = NUM_SYMBOLS * mapper.bits_per_symbol * mapper.n_pol
    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)

    sig = mapper.forward(tx_bits)
    sig_rrc = tx_rrc.forward(sig)
    sig_optical = iq_mod.forward(sig_rrc)
    return sig_optical, tx_bits, mapper.constellation


def create_raman_response(n_points: int, dt_s: float) -> np.ndarray:
    """生成 Raman 响应函数."""
    _, nonlinear_v = raman_from_dat(
        n_points, dt_s, filepath=RAMAN_FILE,
        raman_frac=0.18, analytic=True,
    )
    return nonlinear_v


def measure_evm_ber(signal: Signal, tx_bits: np.ndarray,
                    constellation: Constellation,
                    total_length_km: float = 0.0,
                    total_beta2: float = 0.0):
    """完整接收处理 + EVM/BER 测量.

    total_length_km / total_beta2: 多 span 累积的色散参数，
    若为 0 则尝试从 signal.ctx.meta 读取单 span 信息。
    """
    # CD 补偿
    if total_length_km > 0:
        avg_beta2 = total_beta2 / total_length_km if total_beta2 != 0 else None
        cdc = ChromaticDispersionComp(
            fiber_length_km=total_length_km,
            beta2_ps2_per_km=avg_beta2,
        )
    else:
        cdc = ChromaticDispersionComp()
    signal = cdc.forward(signal)

    # 匹配滤波
    rx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=True)
    rx_matched = rx_rrc.forward(signal)

    # EVM
    ber_calc = BERCalculator(ignore_headers=30)
    evm = ber_calc.calculate_evm(rx_matched, constellation, tx_bits)

    # BER: 降采样 + 解映射
    symbols_1sps = Signal(
        data=rx_matched.data[:, ::RRC_SPS],
        ctx=SignalContext(
            sample_rate=rx_matched.ctx.sample_rate / RRC_SPS,
            center_freq=rx_matched.ctx.center_freq,
            n_pol=rx_matched.ctx.n_pol, sps=1,
            sym_rate=rx_matched.ctx.sym_rate,
            bits_per_symbol=rx_matched.ctx.bits_per_symbol,
            meta=rx_matched.ctx.meta,
        ),
    )
    demapper = Demapper(modulation=MODULATION, power_norm=True)
    rx_bits = demapper.forward(symbols_1sps)
    ber = ber_calc.calculate_ber(tx_bits, rx_bits)
    return evm, ber


RESAMPLER_DOWN = Resampler(target_sps=RRC_SPS)


# ============================================================================
# 测试 1: Resampler 升/降采样保真度
# ============================================================================


def test_resampler_fidelity():
    """验证 Resampler 升/降采样 round-trip 对信号无损."""
    print("=" * 70)
    print("Test 1: Resampler Fidelity (sps=4 → 4000 → 4)")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()
    n_samps_orig = sig_optical.num_samples
    print(f"  Original: sr={sig_optical.ctx.sample_rate/1e9:.2f} GHz, "
          f"sps={sig_optical.ctx.sps}, n={n_samps_orig}")
    print(f"  dt={1/sig_optical.ctx.sample_rate*1e15:.2f} fs")

    # 升采样
    resampler_up = Resampler(target_sps=RMAN_SPS)
    sig_up = resampler_up.forward(sig_optical)
    actual_dt_fs = 1.0 / sig_up.ctx.sample_rate * 1e15
    print(f"  Upsampled: sr={sig_up.ctx.sample_rate/1e12:.2f} THz, "
          f"sps={sig_up.ctx.sps}, n={sig_up.num_samples}, "
          f"dt={actual_dt_fs:.2f} fs (≤7.6 fs: {'YES' if actual_dt_fs <= 7.6 else 'NO'})")

    assert actual_dt_fs <= 7.6, f"dt={actual_dt_fs:.2f} fs exceeds 7.6 fs threshold"

    # 降采样
    resampler_down = Resampler(target_sps=RRC_SPS)
    sig_down = resampler_down.forward(sig_up)
    print(f"  Downsampled: sr={sig_down.ctx.sample_rate/1e9:.2f} GHz, "
          f"sps={sig_down.ctx.sps}, n={sig_down.num_samples}")

    assert sig_down.num_samples == n_samps_orig, \
        f"Sample count mismatch: {sig_down.num_samples} vs {n_samps_orig}"

    # EVM
    evm, _ = measure_evm_ber(sig_down, tx_bits, constellation)
    print(f"  EVM (round-trip): {evm:.4f}%")
    assert evm < 1.0, f"Resampler introduces {evm:.4f}% EVM"
    print(f"  [PASS] Resampler fidelity OK")

    return True


# ============================================================================
# 测试 2: 单 Span Raman + 接收 DSP
# ============================================================================


def test_single_span_raman():
    """单 span 光纤 (Raman+CD+Kerr) + 放大器 + 接收补偿."""
    print("\n" + "=" * 70)
    print("Test 2: Single Span with Raman + CD compensation")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()
    p_launch_total = np.mean(np.abs(sig_optical.data) ** 2) * sig_optical.data.shape[0]

    # 升采样至 Raman 网格
    resampler_up = Resampler(target_sps=RMAN_SPS)
    sig_up = resampler_up.forward(sig_optical)
    n_points = sig_up.num_samples
    dt_s = 1.0 / sig_up.ctx.sample_rate

    raman_r3 = create_raman_response(n_points, dt_s)
    print(f"  Raman grid: n={n_points:.1e}, dt={dt_s*1e15:.2f} fs")

    # 光纤 (含 Raman + CD + Kerr)
    fiber = FiberChannel(
        length_m=FIBER_LEN_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        coarse_step_size_m=COARSE_STEP_M,
        n2_si=N2_SI,
        a_eff_um2=A_EFF_UM2,
        raman_r3=raman_r3,
        compute_ctx_name="cupy",
    )
    rx_up = fiber.forward(sig_up)
    p_after_fiber = np.mean(np.abs(rx_up.data) ** 2) * rx_up.data.shape[0]
    print(f"  After fiber: {10*np.log10(p_after_fiber*1e3):.2f} dBm")

    # IdealAmplifier
    amp = IdealAmplifier(nf_db=AMP_NF_DB)
    rx_amp = amp.forward(rx_up)

    # 降采样回 sps=4
    rx_down = RESAMPLER_DOWN.forward(rx_amp)

    # EVM/BER (含 CD 补偿)
    evm, ber = measure_evm_ber(rx_down, tx_bits, constellation,
                                total_length_km=FIBER_LEN_KM,
                                total_beta2=BETA2_PS2_PER_KM * FIBER_LEN_KM)
    snr = -20 * np.log10(evm / 100) if evm > 0 else 99.0

    print(f"\n  [Results] EVM={evm:.4f}%, SNR={snr:.2f} dB, BER={ber:.2e}")

    # 基本有效性: 不应彻底损坏
    assert evm < 40, f"EVM={evm:.2f}% unreasonably high after CD compensation"
    print(f"  [PASS] Single-span Raman with CD compensation OK")

    return True


# ============================================================================
# 测试 3: 多 Span Raman (对象复用)
# ============================================================================


def test_multi_span_raman():
    """多 span Raman 传播 + 对象复用 + 完整接收链."""
    print("\n" + "=" * 70)
    print("Test 3: Multi-Span Raman (Object Reuse)")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()
    n_pol = sig_optical.data.shape[0]
    total_len_km = 0

    # 升采样
    resampler_up = Resampler(target_sps=RMAN_SPS)
    sig_up = resampler_up.forward(sig_optical)
    n_points = sig_up.num_samples
    dt_s = 1.0 / sig_up.ctx.sample_rate
    raman_r3 = create_raman_response(n_points, dt_s)

    print(f"  Grid: n={n_points:.1e}, dt={dt_s*1e15:.2f} fs")

    # 可复用对象
    fiber = FiberChannel(
        length_m=FIBER_LEN_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        coarse_step_size_m=COARSE_STEP_M,
        n2_si=N2_SI,
        a_eff_um2=A_EFF_UM2,
        raman_r3=raman_r3,
        compute_ctx_name="cupy",
    )
    amp = IdealAmplifier(nf_db=AMP_NF_DB)

    signal = sig_up
    num_spans = 3
    total_len_km = 0
    total_beta2 = 0

    print(f"\n  {'Span':>5s} {'P_amp(dBm)':>11s} {'EVM(%)':>9s} "
          f"{'SNR(dB)':>9s} {'BER':>11s}")
    print(f"  {'-'*5} {'-'*11} {'-'*9} {'-'*9} {'-'*11}")

    for span_idx in range(1, num_spans + 1):
        signal = fiber.forward(signal)
        signal = amp.forward(signal)
        p_after = np.mean(np.abs(signal.data) ** 2) * n_pol
        total_len_km += FIBER_LEN_KM
        total_beta2 += BETA2_PS2_PER_KM * FIBER_LEN_KM

        # 降采样 + 接收
        rx_down = RESAMPLER_DOWN.forward(signal)
        evm, ber = measure_evm_ber(rx_down, tx_bits, constellation,
                                    total_length_km=total_len_km,
                                    total_beta2=total_beta2)
        snr = -20 * np.log10(evm / 100) if evm > 0 else 99.0

        print(f"  {span_idx:5d} {10*np.log10(p_after*1e3):11.2f} {evm:9.4f} "
              f"{snr:9.2f} {ber:11.2e}")

    # 最终
    rx_down = RESAMPLER_DOWN.forward(signal)
    final_evm, final_ber = measure_evm_ber(rx_down, tx_bits, constellation,
                                            total_length_km=total_len_km,
                                            total_beta2=total_beta2)
    final_snr = -20 * np.log10(final_evm / 100) if final_evm > 0 else 99.0
    print(f"\n  Final ({num_spans} spans): EVM={final_evm:.4f}%, "
          f"SNR={final_snr:.2f} dB, BER={final_ber:.2e}")

    assert final_evm < 40, f"Multi-span EVM={final_evm:.2f}% too high"
    print(f"  [PASS] Multi-span Raman propagation OK")

    return True


# 复用降采样对象 (test_multi_span_raman 需要)
resampler_down = Resampler(target_sps=RRC_SPS)


# ============================================================================
# 测试 4: Raman ON vs OFF 对比
# ============================================================================


def test_raman_on_vs_off():
    """对比 Raman 开启和关闭时的信号特征."""
    print("\n" + "=" * 70)
    print("Test 4: Raman ON vs OFF")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()

    sig_up = Resampler(target_sps=RMAN_SPS).forward(sig_optical)
    n_points = sig_up.num_samples
    dt_s = 1.0 / sig_up.ctx.sample_rate
    raman_r3 = create_raman_response(n_points, dt_s)

    results = {}
    for label, use_raman in [("Raman ON", True), ("Raman OFF", False)]:
        fiber = FiberChannel(
            length_m=FIBER_LEN_KM * 1e3,
            alpha_db_per_km=ALPHA_DB_PER_KM,
            beta2_ps2_per_km=BETA2_PS2_PER_KM,
            beta3_ps3_per_km=BETA3_PS3_PER_KM,
            coarse_step_size_m=COARSE_STEP_M,
            n2_si=N2_SI,
            a_eff_um2=A_EFF_UM2,
            raman_r3=raman_r3 if use_raman else None,
            compute_ctx_name="cupy",
        )
        rx = fiber.forward(sig_up)
        rx_amp = IdealAmplifier(nf_db=AMP_NF_DB).forward(rx)
        rx_down = RESAMPLER_DOWN.forward(rx_amp)

        evm, ber = measure_evm_ber(rx_down, tx_bits, constellation,
                                    total_length_km=FIBER_LEN_KM,
                                    total_beta2=BETA2_PS2_PER_KM * FIBER_LEN_KM)
        snr = -20 * np.log10(evm / 100) if evm > 0 else 99.0
        results[label] = {"evm": evm, "ber": ber, "snr": snr}

        print(f"  {label}: EVM={evm:.4f}%, SNR={snr:.2f} dB, BER={ber:.2e}")

    for label, r in results.items():
        assert 0.1 < r["evm"] < 40, f"{label} EVM={r['evm']:.2f}% out of range"

    print(f"  [PASS] Raman comparison valid")
    return True


# ============================================================================
# 运行
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(f"Multi-Span Raman + IdealAmplifier + RxDSP Suite")
    print(f"dt={1/50e9/RMAN_SPS*1e15:.2f} fs (≤7.6 fs), {NUM_SYMBOLS} symbols")
    print("=" * 70)

    results = {}
    for name, fn in [
        ("fidelity", test_resampler_fidelity),
        ("single_span", test_single_span_raman),
        ("multi_span", test_multi_span_raman),
        ("raman_on_off", test_raman_on_vs_off),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name}: {'PASSED' if ok else 'FAILED'}")
    if all(results.values()):
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
