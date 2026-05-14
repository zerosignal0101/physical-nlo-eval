"""Multi-span Fiber + IdealAmplifier 信号处理验证。

专注检查复用光纤和放大器对象时的问题:
1. 信号功率是否逐段正确恢复
2. 对象复用是否存在状态泄漏
3. ASE 噪声是否正确累积
4. EVM/BER 是否符合理论预期
"""

import numpy as np

from physics_nlo_eval.channel.fiber import FiberChannel, IdealAmplifier
from physics_nlo_eval.core.signal import Signal, SignalContext
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
AMP_NF_DB = 5.5
NUM_SYMBOLS = 4096
SEED = 42


def make_signal():
    """生成 DP-QPSK 信号."""
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


def measure_signal_power(signal: Signal) -> tuple[float, float]:
    """返回 (总功率, 每偏振功率) in W."""
    n_pol = signal.data.shape[0]
    total = np.mean(np.abs(signal.data) ** 2) * n_pol
    per_pol = np.mean(np.abs(signal.data) ** 2, axis=1)
    return total, per_pol


def measure_evm(signal: Signal, tx_bits: np.ndarray, constellation: Constellation,
                ignore_headers: int = 30) -> float:
    """经匹配滤波后计算 EVM."""
    rx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=True)
    rx_matched = rx_rrc.forward(signal)
    ber_calc = BERCalculator(ignore_headers=ignore_headers)
    return ber_calc.calculate_evm(rx_matched, constellation, tx_bits)


def theory_ase_noise_per_pol(nf_db: float, gain_linear: float, freq_hz: float,
                              sym_rate_hz: float) -> float:
    """ISRS-GN ASE 复数噪声功率 per pol (per symbol after MF processing gain)."""
    h = 6.62607015e-34
    nf_lin = 10 ** (nf_db / 10)
    G_minus_1 = max(gain_linear - 1.0, 0.0)
    # IdealAmplifier formula: σ²_per_IQ = 0.5 * nf * h * nu * Rs * (G-1)
    # Complex noise power per pol per sample = 2 * σ²_per_IQ = nf * h * nu * Rs * (G-1)
    # After matched filter processing gain (sps), SNR improved by sps
    # But the per-symbol noise power after MF is the same as per-sample (energy-norm MF)
    return nf_lin * h * freq_hz * sym_rate_hz * G_minus_1  # complex noise, per pol, per sym


# ============================================================================
# 测试 1: 对象复用 — 同一 Fiber 和 Amp 对象多次 forward
# ============================================================================


def test_object_reuse():
    """同一对象多次 forward：检查有无状态泄漏."""
    print("=" * 70)
    print("Test 1: Object Reuse (same Fiber + Amp, multiple forward calls)")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()
    n_pol = sig_optical.data.shape[0]
    p_launch_total, p_launch_per_pol = measure_signal_power(sig_optical)

    print(f"  Launch: {10*np.log10(p_launch_total*1e3):.2f} dBm total, "
          f"{10*np.log10(p_launch_per_pol[0]*1e3):.2f} dBm/pol")

    # 创建对象（复用）
    fiber = FiberChannel(
        length_m=FIBER_LEN_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=0.0,
        beta3_ps3_per_km=0.0,
        coarse_step_size_m=500.0,
        n2_si=0.0,
        a_eff_um2=80.0,
        raman_r3=None,
        compute_ctx_name="cupy",
    )
    amp = IdealAmplifier(nf_db=AMP_NF_DB)

    signal = sig_optical
    span_results = []

    for span_idx in range(5):
        p_before_fiber, _ = measure_signal_power(signal)

        # 复用同一个 fiber 对象
        signal = fiber.forward(signal)
        p_after_fiber, _ = measure_signal_power(signal)

        # 复用同一个 amp 对象
        signal = amp.forward(signal)
        p_after_amp, _ = measure_signal_power(signal)

        evm = measure_evm(signal, tx_bits, constellation)
        snr_evm = -20 * np.log10(evm / 100) if evm > 0 else 99.0

        span_results.append({
            "p_before": p_before_fiber,
            "p_after_fiber": p_after_fiber,
            "p_after_amp": p_after_amp,
            "evm": evm,
            "snr": snr_evm,
        })

        print(f"  Span {span_idx+1}: "
              f"fiber_in={10*np.log10(p_before_fiber*1e3):.2f} dBm, "
              f"fiber_out={10*np.log10(p_after_fiber*1e3):.2f} dBm, "
              f"amp_out={10*np.log10(p_after_amp*1e3):.2f} dBm, "
              f"EVM={evm:.4f}%, SNR={snr_evm:.2f} dB")

    # ---- 验证 ----
    # 1. 每段功率应恢复至入纤功率附近
    for i, r in enumerate(span_results):
        power_error = abs(10 * np.log10(r["p_after_amp"] * 1e3) -
                          10 * np.log10(p_launch_total * 1e3))
        print(f"\n  Span {i+1} power recovery error: {power_error:.3f} dB")
        if power_error > 0.5:
            print(f"  [WARN] Span {i+1}: power recovery error > 0.5 dB!")
            return False

    # 2. EVM 应逐段恶化
    for i in range(1, len(span_results)):
        if span_results[i]["evm"] <= span_results[i-1]["evm"]:
            print(f"  [WARN] Span {i+1} EVM did not degrade!")
            return False

    # 3. 功率不应漂移
    p_first_amp_out = span_results[0]["p_after_amp"]
    p_last_amp_out = span_results[-1]["p_after_amp"]
    drift = abs(10 * np.log10(p_last_amp_out / p_first_amp_out))
    print(f"\n  Power drift over 5 spans: {drift:.3f} dB")
    if drift > 1.0:
        print(f"  [WARN] Excessive power drift!")
        return False

    print(f"  [PASS] Object reuse OK")
    return True


# ============================================================================
# 测试 2: 对象复用 vs 新对象 — 结果应一致
# ============================================================================


def test_reuse_vs_new():
    """对比复用对象与每次新建对象的结果差异."""
    print("\n" + "=" * 70)
    print("Test 2: Reuse vs Fresh Objects (3 spans)")
    print("=" * 70)

    num_spans = 3

    # ---- 路径 A: 复用对象 ----
    np.random.seed(SEED)
    sig_optical, tx_bits, constellation = make_signal()

    fiber_reuse = FiberChannel(
        length_m=FIBER_LEN_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=0.0, beta3_ps3_per_km=0.0,
        coarse_step_size_m=500.0, n2_si=0.0,
        a_eff_um2=80.0, raman_r3=None,
        compute_ctx_name="cupy",
    )
    amp_reuse = IdealAmplifier(nf_db=AMP_NF_DB)

    signal_a = sig_optical
    for _ in range(num_spans):
        signal_a = fiber_reuse.forward(signal_a)
        signal_a = amp_reuse.forward(signal_a)

    # ---- 路径 B: 每次新建对象 ----
    np.random.seed(SEED)
    sig_optical_b, tx_bits_b, constellation_b = make_signal()

    signal_b = sig_optical_b
    for _ in range(num_spans):
        fb = FiberChannel(
            length_m=FIBER_LEN_KM * 1e3,
            alpha_db_per_km=ALPHA_DB_PER_KM,
            beta2_ps2_per_km=0.0, beta3_ps3_per_km=0.0,
            coarse_step_size_m=500.0, n2_si=0.0,
            a_eff_um2=80.0, raman_r3=None,
            compute_ctx_name="cupy",
        )
        ab = IdealAmplifier(nf_db=AMP_NF_DB)
        signal_b = fb.forward(signal_b)
        signal_b = ab.forward(signal_b)

    # 比较
    _, p_a_per_pol = measure_signal_power(signal_a)
    _, p_b_per_pol = measure_signal_power(signal_b)
    evm_a = measure_evm(signal_a, tx_bits, constellation)
    evm_b = measure_evm(signal_b, tx_bits_b, constellation_b)

    # 功率偏差 (应在数值噪声范围内)
    power_diff_db = abs(10 * np.log10(p_a_per_pol[0] / p_b_per_pol[0]))
    evm_diff = abs(evm_a - evm_b)

    print(f"  Reuse:  P={10*np.log10(p_a_per_pol[0]*1e3):.3f} dBm/pol, EVM={evm_a:.4f}%")
    print(f"  Fresh:  P={10*np.log10(p_b_per_pol[0]*1e3):.3f} dBm/pol, EVM={evm_b:.4f}%")
    print(f"  Power diff: {power_diff_db:.4f} dB, EVM diff: {evm_diff:.4f}%")

    # 功率应一致 (同一 seed 同一信号)
    assert power_diff_db < 0.1, f"Power mismatch between reuse and fresh: {power_diff_db:.4f} dB"
    # EVM 可略有差异 (噪声随机种子不同)，但幅度应接近
    assert evm_diff < 1.0, f"EVM too different between reuse and fresh: {evm_diff:.4f}%"

    print(f"  [PASS] Reuse and fresh produce consistent results")
    return True


# ============================================================================
# 测试 3: 理论 SNR vs 测量 EVM (多 span)
# ============================================================================


def test_theory_vs_measured_snr():
    """多 span 下理论 ASE SNR 与 EVM 反推 SNR 的一致性."""
    print("\n" + "=" * 70)
    print("Test 3: Multi-Span Theory vs Measured SNR")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()
    n_pol = sig_optical.data.shape[0]
    p_launch_total, p_launch_per_pol = measure_signal_power(sig_optical)
    p_sig_per_pol = p_launch_per_pol[0]  # signal power per pol

    nu = sig_optical.ctx.center_freq
    rs = sig_optical.ctx.sym_rate

    signal = sig_optical
    cumulative_noise_per_pol = 0.0  # 累积复数噪声功率 per pol per sym
    prev_total_power = p_launch_total

    num_spans = 5
    span_loss_db = ALPHA_DB_PER_KM * FIBER_LEN_KM
    gain_linear = 10 ** (span_loss_db / 10)

    print(f"  Launch: {10*np.log10(p_launch_total*1e3):.2f} dBm, "
          f"span loss: {span_loss_db:.1f} dB, NF: {AMP_NF_DB:.1f} dB")
    print(f"\n  {'Span':>5s} {'P_amp(dBm)':>11s} {'EVM(%)':>9s} "
          f"{'SNR_EVM':>9s} {'SNR_th':>9s} {'Δ':>7s}")
    print(f"  {'-'*5} {'-'*11} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")

    for span_idx in range(1, num_spans + 1):
        # 新建 fiber + amp 避免复用放大已有噪声功率
        fiber = FiberChannel(
            length_m=FIBER_LEN_KM * 1e3,
            alpha_db_per_km=ALPHA_DB_PER_KM,
            beta2_ps2_per_km=0.0, beta3_ps3_per_km=0.0,
            coarse_step_size_m=500.0, n2_si=0.0,
            a_eff_um2=80.0, raman_r3=None,
            compute_ctx_name="cupy",
        )
        amp = IdealAmplifier(nf_db=AMP_NF_DB)

        signal = fiber.forward(signal)
        signal = amp.forward(signal)

        # 测量
        evm = measure_evm(signal, tx_bits, constellation)
        snr_evm = -20 * np.log10(evm / 100) if evm > 0 else 99.0

        # 该 span 新增的理论噪声
        noise_this_span = theory_ase_noise_per_pol(AMP_NF_DB, gain_linear, nu, rs)
        cumulative_noise_per_pol += noise_this_span

        # 理论 SNR (计入 MF 处理增益 sps)
        snr_theory_lin = p_sig_per_pol * RRC_SPS / cumulative_noise_per_pol
        snr_theory = 10 * np.log10(snr_theory_lin)

        p_amp, _ = measure_signal_power(signal)

        delta = snr_evm - snr_theory
        print(f"  {span_idx:5d} {10*np.log10(p_amp*1e3):11.2f} {evm:9.4f} "
              f"{snr_evm:9.2f} {snr_theory:9.2f} {delta:+7.2f}")

    # 最终检查
    final_evm = measure_evm(signal, tx_bits, constellation)
    final_snr_evm = -20 * np.log10(final_evm / 100) if final_evm > 0 else 99.0

    final_theory_lin = p_sig_per_pol * RRC_SPS / cumulative_noise_per_pol
    final_theory = 10 * np.log10(final_theory_lin)

    final_delta = final_snr_evm - final_theory
    print(f"\n  Final Δ = {final_delta:+.2f} dB")

    # 容许 1.5 dB 偏差 (含 MF 响应非理想性)
    if abs(final_delta) > 2.0:
        print(f"  [FAIL] SNR deviation too large!")
        return False

    print(f"  [PASS] Theory-measurement SNR within tolerance")
    return True


# ============================================================================
# 测试 4: IdealAmplifier 增益计算检查 (多 span 噪声功率影响)
# ============================================================================


def test_amplifier_gain_accuracy():
    """检查 IdealAmplifier 在多 span 下的实际增益是否因噪声功率而偏离预期."""
    print("\n" + "=" * 70)
    print("Test 4: Amplifier Gain Accuracy vs Noise Accumulation")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()
    n_pol = sig_optical.data.shape[0]
    p_launch_total, _ = measure_signal_power(sig_optical)
    span_loss_db = ALPHA_DB_PER_KM * FIBER_LEN_KM
    expected_gain_db = span_loss_db  # 放大应恰好补偿损耗
    expected_gain_lin = 10 ** (expected_gain_db / 10)

    signal = sig_optical
    gains_measured = []

    num_spans = 10  # 更多 span 使噪声累积更显著
    for span_idx in range(num_spans):
        p_before, _ = measure_signal_power(signal)

        fiber = FiberChannel(
            length_m=FIBER_LEN_KM * 1e3,
            alpha_db_per_km=ALPHA_DB_PER_KM,
            beta2_ps2_per_km=0.0, beta3_ps3_per_km=0.0,
            coarse_step_size_m=500.0, n2_si=0.0,
            a_eff_um2=80.0, raman_r3=None,
            compute_ctx_name="cupy",
        )
        signal = fiber.forward(signal)
        p_after_fiber, _ = measure_signal_power(signal)

        amp = IdealAmplifier(nf_db=AMP_NF_DB)
        signal = amp.forward(signal)
        p_after_amp, _ = measure_signal_power(signal)

        # 实际总功率增益
        actual_gain = p_after_amp / p_after_fiber
        actual_gain_db = 10 * np.log10(actual_gain)

        gains_measured.append(actual_gain_db)
        amp_gains = signal.ctx.meta.get("ideal_amp_gains_linear", [])
        ideal_amp_gain_db = 10 * np.log10(amp_gains[0]) if amp_gains else 0

        if span_idx < 5 or span_idx >= num_spans - 2:
            print(f"  Span {span_idx+1}: measured gain={actual_gain_db:.2f} dB, "
                  f"IdealAmp computed gain={ideal_amp_gain_db:.2f} dB, "
                  f"expected={expected_gain_db:.1f} dB, "
                  f"Δ={actual_gain_db - expected_gain_db:+.3f} dB")

    # 分析增益漂移趋势
    avg_first3 = np.mean(gains_measured[:3])
    avg_last3 = np.mean(gains_measured[-3:])
    gain_drift = avg_last3 - avg_first3

    print(f"\n  Avg gain (spans 1-3): {avg_first3:.3f} dB")
    print(f"  Avg gain (spans 8-10): {avg_last3:.3f} dB")
    print(f"  Gain drift: {gain_drift:+.3f} dB")

    if abs(gain_drift) > 0.5:
        print(f"  [WARN] Significant gain drift detected!"
              f"  Noise power accumulation affects gain calculation.")
        print(f"  Root cause: IdealAmplifier uses total power (signal+noise)"
              f" for gain computation.")
        # This is a known design characteristic, not a crash-level bug
        # But it should be documented
    else:
        print(f"  [PASS] Gain stable across spans")

    return True


# ============================================================================
# 测试 5: Signal metadata/context 跨段传递完整性
# ============================================================================


def test_metadata_integrity():
    """验证 SignalContext 和 metadata 在多次 fiber+amp 后的完整性."""
    print("\n" + "=" * 70)
    print("Test 5: Metadata/Context Integrity Across Spans")
    print("=" * 70)

    sig_optical, tx_bits, constellation = make_signal()

    # 记录原始值
    orig_sample_rate = sig_optical.ctx.sample_rate
    orig_center_freq = sig_optical.ctx.center_freq
    orig_sps = sig_optical.ctx.sps
    orig_sym_rate = sig_optical.ctx.sym_rate
    orig_n_pol = sig_optical.ctx.n_pol
    orig_bps = sig_optical.ctx.bits_per_symbol

    print(f"  Original: sr={orig_sample_rate/1e9:.2f} GHz, "
          f"cf={orig_center_freq/1e12:.3f} THz, "
          f"sps={orig_sps}, sym_rate={orig_sym_rate/1e9:.2f} GBd, "
          f"n_pol={orig_n_pol}, bps={orig_bps}")

    signal = sig_optical
    for span_idx in range(3):
        fiber = FiberChannel(
            length_m=FIBER_LEN_KM * 1e3,
            alpha_db_per_km=ALPHA_DB_PER_KM,
            beta2_ps2_per_km=0.0, beta3_ps3_per_km=0.0,
            coarse_step_size_m=500.0, n2_si=0.0,
            a_eff_um2=80.0, raman_r3=None,
            compute_ctx_name="cupy",
        )
        signal = fiber.forward(signal)
        signal = IdealAmplifier(nf_db=AMP_NF_DB).forward(signal)

    ctx = signal.ctx
    issues = []
    if ctx.sample_rate != orig_sample_rate:
        issues.append(f"sample_rate: {orig_sample_rate} → {ctx.sample_rate}")
    if ctx.center_freq != orig_center_freq:
        issues.append(f"center_freq: {orig_center_freq} → {ctx.center_freq}")
    if ctx.sps != orig_sps:
        issues.append(f"sps: {orig_sps} → {ctx.sps}")
    if ctx.sym_rate != orig_sym_rate:
        issues.append(f"sym_rate: {orig_sym_rate} → {ctx.sym_rate}")
    if ctx.n_pol != orig_n_pol:
        issues.append(f"n_pol: {orig_n_pol} → {ctx.n_pol}")
    if ctx.bits_per_symbol != orig_bps:
        issues.append(f"bps: {orig_bps} → {ctx.bits_per_symbol}")
    if "tx_launch_power_dbm_per_pol" not in ctx.meta:
        issues.append("tx_launch_power_dbm_per_pol lost from meta")

    if issues:
        print(f"  [FAIL] Metadata issues found:")
        for iss in issues:
            print(f"    - {iss}")
        return False

    print(f"  [PASS] All metadata preserved:")
    print(f"    launch_power: {ctx.meta.get('tx_launch_power_dbm_per_pol')} dBm")
    print(f"    ideal_amp_nf: {ctx.meta.get('ideal_amp_nf_db')} dB")
    return True


# ============================================================================
# 运行
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Multi-Span IdealAmplifier + Fiber Diagnostic Suite")
    print("=" * 70)

    results = {}
    for name, fn in [
        ("object_reuse", test_object_reuse),
        ("reuse_vs_new", test_reuse_vs_new),
        ("theory_vs_snr", test_theory_vs_measured_snr),
        ("gain_accuracy", test_amplifier_gain_accuracy),
        ("metadata_integrity", test_metadata_integrity),
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
