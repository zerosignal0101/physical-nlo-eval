"""
WDM + 光纤 (ISRS/NLO) + IdealAmplifier 测试

测试目标：
1. WDM 系统通过启用 ISRS (拉曼散射) 的光纤，验证 Raman 倾斜效应
2. 使用 IdealAmplifier 恢复入纤功率并添加 ASE 噪声
3. 验证功率恢复的正确性

"""

import numpy as np
from pynlo.utility.chi3 import raman_from_dat

from physics_nlo_eval.channel.fiber import FiberChannel, IdealAmplifier
from physics_nlo_eval.channel.wdm import ChannelSpec, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.metrics.ber_evm import BERCalculator
from physics_nlo_eval.tx.mapper import Constellation, Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter

# ============================================================================
# 参数配置 - 优化用于快速验证
# ============================================================================

# WDM 规划 - 5 通道，100 GHz 间隔
WDM_PLAN = WDMPlan(
    reference_freq_hz=193.1e12,
    channels=[
        ChannelSpec(channel_id=0, freq_offset_hz=-200e9),
        ChannelSpec(channel_id=1, freq_offset_hz=-100e9),
        ChannelSpec(channel_id=2, freq_offset_hz=0e9),
        ChannelSpec(channel_id=3, freq_offset_hz=100e9),
        ChannelSpec(channel_id=4, freq_offset_hz=200e9),
    ],
)

# 调制参数
MODULATION = "DP-QPSK"
SYMBOL_RATE_HZ = 50e9
RRC_SPS = 4
RRC_ROLL_OFF = 0.1

# 入纤功率 (每个偏振)
LAUNCH_POWER_DBM_PER_POL = 6.0  # 6 dBm/pol

# 光纤参数
FIBER_LENGTH_KM = 80.0  # 80 km 单跨段
ALPHA_DB_PER_KM = 0.2
BETA2_PS2_PER_KM = -20.0
BETA3_PS3_PER_KM = 0.1e-3
N2_SI = 2.6e-20  # 非线性折射率 (silica)
A_EFF_UM2 = 80.0
COARSE_STEP_SIZE_M = 500.0  # 500 m 步长

# 仿真参数
NUM_SYMBOLS = 512  # 减少符号数以加快计算
SEED = 42


# ============================================================================
# 辅助函数
# ============================================================================


def create_raman_response(n_points: int, dt_s: float) -> np.ndarray:
    """为给定网格创建拉曼响应函数。"""
    _, nonlinear_v = raman_from_dat(
        n_points,
        dt_s,
        filepath="/home/zsig/Documents/Codes/PyNLO/pynlo/utility/RamanGain.dat",
        raman_frac=0.18,
        analytic=True,
    )
    return nonlinear_v


def calculate_grid_dt(n_points: int, bandwidth_hz: float) -> float:
    """计算 PyNLO Grid 的时间步长 dt。"""
    dv = bandwidth_hz / (n_points - 1)
    dt = 1.0 / (n_points * dv)
    return dt


def measure_channel_power_fft(signal: Signal, wdm_plan: WDMPlan, channel_idx: int) -> float:
    """通过 FFT 测量指定通道的功率。

    用于从宽带 WDM 信号中提取各通道功率。
    """
    n_points = signal.num_samples
    data = signal.data
    if data.ndim == 3:
        # 堆叠 WDM: (n_channels, n_pol, N_samples) -> (n_pol, N_samples)
        data = data[channel_idx]

    # 合并两个偏振
    combined = data[0] + data[1]

    # FFT 获取频谱 - 从 sample_rate 计算 dt
    dt = 1.0 / signal.ctx.sample_rate
    spec = np.fft.fftshift(np.fft.fft(combined))
    freq = np.fft.fftshift(np.fft.fftfreq(n_points, d=dt))

    # 找到通道中心频率
    ch = wdm_plan.channels[channel_idx]
    center_freq = wdm_plan.reference_freq_hz + ch.freq_offset_hz
    idx = np.argmin(np.abs(freq - center_freq))

    # 在通道中心附近求和功率
    window = 500
    start = max(0, idx - window)
    end = min(len(spec), idx + window)
    power = np.mean(np.abs(spec[start:end]) ** 2)
    return power


def test_raman_tilt():
    """测试 Raman 倾斜效应：对比有/无拉曼的通道功率差异。

    Returns
    -------
    bool
        测试是否通过 (Raman tilt > 0.5 dB 视为有效)
    """
    np.random.seed(SEED)

    print(f"\n{'=' * 70}")
    print("Test: Raman Tilt Verification")
    print(f"{'=' * 70}")

    # ---- TX 组件 ----
    mapper = Mapper(modulation=MODULATION, power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    tx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=False)
    iq_mod = IQModulator(launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL)

    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = NUM_SYMBOLS * bits_per_symbol * n_pol

    # ---- 生成通道 ----
    channels = []
    for ch_spec in WDM_PLAN.channels:
        tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
        sig = mapper.forward(tx_bits)
        sig_rrc = tx_rrc.forward(sig)
        sig_optical = iq_mod.forward(sig_rrc)
        channels.append(sig_optical)

    # ---- WDM 合波 (不指定采样率，使用自动采样率) ----
    wdm_mux = WDMMux(wdm_plan=WDM_PLAN)
    wdm_signal = wdm_mux.combine(channels)
    print(f"  WDM: {len(WDM_PLAN.channels)} ch, sr={wdm_signal.ctx.sample_rate / 1e9:.2f} GHz, "
          f"shape={wdm_signal.data.shape}, n_samples={wdm_signal.num_samples}")

    # ---- 创建拉曼响应 ----
    n_points = wdm_signal.num_samples
    bandwidth = wdm_signal.ctx.sample_rate
    dt = calculate_grid_dt(n_points, bandwidth)
    print(f"  Grid: n={n_points}, dt={dt * 1e15:.3f} fs, sr={bandwidth / 1e12:.2f} THz")

    raman_r3 = create_raman_response(n_points, dt)
    print(f"  Raman r3: shape={raman_r3.shape}, max={np.max(np.abs(raman_r3)):.4f}")

    # ---- 光纤传播：80 km ----
    fiber_with_raman = FiberChannel(
        length_m=FIBER_LENGTH_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        coarse_step_size_m=COARSE_STEP_SIZE_M,
        n2_si=N2_SI,
        a_eff_um2=A_EFF_UM2,
        slope_a_eff_um2_per_nm=0.05,
        ref_lambda_nm=1550.0,
        d_pmd_ps_per_sqrt_km=0.0,
        raman_r3=raman_r3,
        compute_ctx_name="cupy",
    )

    fiber_no_raman = FiberChannel(
        length_m=FIBER_LENGTH_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        coarse_step_size_m=COARSE_STEP_SIZE_M,
        n2_si=N2_SI,
        a_eff_um2=A_EFF_UM2,
        slope_a_eff_um2_per_nm=0.05,
        ref_lambda_nm=1550.0,
        d_pmd_ps_per_sqrt_km=0.0,
        raman_r3=None,
        compute_ctx_name="cupy",
    )

    print(f"\n  [Fiber Propagation: {FIBER_LENGTH_KM} km]")
    print("  Computing with Raman...")
    rx_with_raman = fiber_with_raman.forward(wdm_signal)

    print("  Computing without Raman...")
    rx_no_raman = fiber_no_raman.forward(wdm_signal)

    # ---- 测量通道功率 (通过 FFT) ----
    print("\n  [Channel Powers After 80km Fiber]")

    ch_powers_raman = []
    ch_powers_no_raman = []

    for i, ch in enumerate(WDM_PLAN.channels):
        p_raman = measure_channel_power_fft(rx_with_raman, WDM_PLAN, i)
        p_no_raman = measure_channel_power_fft(rx_no_raman, WDM_PLAN, i)
        ch_powers_raman.append(p_raman)
        ch_powers_no_raman.append(p_no_raman)

        db_raman = 10 * np.log10(p_raman * 1e3) if p_raman > 0 else -np.inf
        db_no_raman = 10 * np.log10(p_no_raman * 1e3) if p_no_raman > 0 else -np.inf
        diff = db_raman - db_no_raman
        print(f"    Ch{i} ({ch.freq_offset_hz / 1e9:.0f} GHz): "
              f"Raman={db_raman:.2f} dBm, NoRaman={db_no_raman:.2f} dBm, diff={diff:+.2f} dB")

    # ---- Raman 倾斜计算 ----
    tilt_raman = 10 * np.log10(ch_powers_raman[-1] / ch_powers_raman[0])
    tilt_no_raman = 10 * np.log10(ch_powers_no_raman[-1] / ch_powers_no_raman[0])

    print(f"\n  [Raman Tilt Results]")
    print(f"    With Raman: {tilt_raman:+.2f} dB (Ch4-Ch0, should be negative ~-1 to -3 dB)")
    print(f"    No Raman: {tilt_no_raman:+.2f} dB (should be ~0 dB)")
    print(f"    Net tilt due to ISRS: {tilt_raman - tilt_no_raman:+.2f} dB")

    # 验证：Raman 倾斜应该明显 (负值，至少 > 0.5 dB)
    tilt_threshold = 0.5
    raman_tilt_valid = abs(tilt_raman - tilt_no_raman) > tilt_threshold

    status = "[PASS]" if raman_tilt_valid else "[FAIL]"
    print(f"\n  Raman tilt test: {status} (net tilt = {tilt_raman - tilt_no_raman:+.2f} dB, threshold = {tilt_threshold} dB)")

    return raman_tilt_valid


def test_ideal_amp_ase():
    """测试 IdealAmplifier 的 ASE 噪声和功率恢复。

    使用单通道信号，无 Raman，验证放大器能否正确恢复功率并添加 ASE 噪声。
    不使用堆叠 WDM，直接使用单个通道信号。

    Returns
    -------
    bool
        测试是否通过。
    """
    np.random.seed(SEED)

    print(f"\n{'=' * 70}")
    print("Test: IdealAmplifier ASE Noise and Power Recovery")
    print(f"{'=' * 70}")

    # ---- TX 组件 ----
    mapper = Mapper(modulation=MODULATION, power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    tx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=False)
    iq_mod = IQModulator(launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL)

    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = NUM_SYMBOLS * bits_per_symbol * n_pol

    # ---- 单通道信号 ----
    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    sig = mapper.forward(tx_bits)
    sig_rrc = tx_rrc.forward(sig)
    sig_optical = iq_mod.forward(sig_rrc)

    orig_power_w = np.mean(np.abs(sig_optical.data) ** 2) * n_pol
    orig_power_dbm = 10 * np.log10(orig_power_w * 1e3)
    print(f"  Original signal power: {orig_power_dbm:.2f} dBm")

    # ---- 通过光纤衰减 ----
    fiber = FiberChannel(
        length_m=FIBER_LENGTH_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        coarse_step_size_m=COARSE_STEP_SIZE_M,
        n2_si=0.0,  # 关闭非线性，仅色散和损耗
        a_eff_um2=A_EFF_UM2,
        raman_r3=None,  # 关闭 Raman
        compute_ctx_name="cupy",
    )

    print(f"  Fiber: {FIBER_LENGTH_KM} km, loss = {ALPHA_DB_PER_KM * FIBER_LENGTH_KM:.1f} dB")
    rx_signal = fiber.forward(sig_optical)

    rx_power_w = np.mean(np.abs(rx_signal.data) ** 2) * rx_signal.data.shape[0]
    rx_power_dbm = 10 * np.log10(rx_power_w * 1e3)
    print(f"  After fiber: {rx_power_dbm:.2f} dBm")

    # ---- IdealAmplifier 放大并添加 ASE ----
    amp = IdealAmplifier(nf_db=5.5)
    rx_amplified = amp.forward(rx_signal)

    recovered_power_w = np.mean(np.abs(rx_amplified.data) ** 2) * rx_amplified.data.shape[0]
    recovered_power_dbm = 10 * np.log10(recovered_power_w * 1e3)
    power_error_db = recovered_power_dbm - orig_power_dbm

    print(f"\n  [IdealAmplifier Results]")
    print(f"    Recovered power: {recovered_power_dbm:.2f} dBm")
    print(f"    Original power: {orig_power_dbm:.2f} dBm")
    print(f"    Power error: {power_error_db:+.2f} dB")

    # ---- 验证 ----
    power_tolerance_db = 0.5
    power_recovery_passed = abs(power_error_db) < power_tolerance_db

    # 检查 ASE 噪声是否被添加 (噪声方差 > 0)
    # 比较放大后信号和原始信号的统计特性
    signal_variance = np.var(np.abs(sig_optical.data))
    amplified_variance = np.var(np.abs(rx_amplified.data))
    noise_added = amplified_variance > signal_variance * 1.01  # 允许一点数值误差

    status_power = "[PASS]" if power_recovery_passed else "[FAIL]"
    status_ase = "[PASS]" if noise_added else "[FAIL]"

    print(f"\n  [Verification]")
    print(f"    Power recovery: {status_power} (error = {power_error_db:+.2f} dB, tol = ±{power_tolerance_db} dB)")
    print(f"    ASE noise added: {status_ase} (variance increased by {10*np.log10(amplified_variance/signal_variance):+.2f} dB)")

    return power_recovery_passed and noise_added


def verify_ase_snr_consistency():
    """验证 IdealAmplifier 的 ASE 噪声水平 — 双路径交叉验证。

    采用两个独立的方法检测噪声水平:
    A. 直接测量法: 两次放大器运行，差分提取纯噪声，与理论对比。
    B. EVM 转换法: 完整接收链 (匹配滤波 + EVM → SNR)，与理论对比。

    使用较高衰减 (32 dB 光纤) 使 SNR 落在 10-20 dB 范围，
    该范围内 SNR 偏差对 QoT 评估才具有实际意义。

    ASE 理论公式 (IdealAmplifier 实现):
        σ²_ASE = 0.5 * NF * h * ν * R_s * (G-1)   [per I/Q, per pol]
        复数噪声功率 per pol = 2 * σ²_ASE
        信号功率 per pol = P_target / 2
        SNR_lin = P_sig_per_pol / (2 * σ²_ASE)

    Returns
    -------
    bool
        噪声一致性验证是否通过。
    """
    print(f"\n{'=' * 70}")
    print("Verification: ASE Noise SNR Consistency (Dual-Path)")
    print(f"{'=' * 70}")

    # ---- 参数 (使用更长的光纤使 SNR 低于 20 dB) ----
    fiber_len_km = 160.0   # 160 km → 32 dB loss → SNR ≈ 20 dB
    nf_db = 7.0            # 宽松的 NF 便于更容易观察噪声
    seed_a = 42
    seed_b = 54321

    # ---- TX 链 ----
    np.random.seed(SEED)
    mapper = Mapper(modulation=MODULATION, power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    tx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=False)
    iq_mod = IQModulator(launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL)

    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = NUM_SYMBOLS * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    sig = mapper.forward(tx_bits)
    sig_rrc = tx_rrc.forward(sig)
    sig_optical = iq_mod.forward(sig_rrc)

    p_target_w = np.mean(np.abs(sig_optical.data) ** 2) * n_pol
    print(f"  Launch power: {10 * np.log10(p_target_w * 1e3):.2f} dBm "
          f"({p_target_w * 1e3:.2f} mW total)")

    # ---- 光纤 (纯衰减) ----
    fiber = FiberChannel(
        length_m=fiber_len_km * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=0.0,
        beta3_ps3_per_km=0.0,
        coarse_step_size_m=COARSE_STEP_SIZE_M,
        n2_si=0.0,
        a_eff_um2=A_EFF_UM2,
        raman_r3=None,
        compute_ctx_name="cupy",
    )

    rx_signal = fiber.forward(sig_optical)
    rx_power_w = np.mean(np.abs(rx_signal.data) ** 2) * rx_signal.data.shape[0]
    expected_loss_db = ALPHA_DB_PER_KM * fiber_len_km
    actual_loss_db = 10 * np.log10(p_target_w * 1e3) - 10 * np.log10(rx_power_w * 1e3)
    print(f"  After {fiber_len_km} km: {10 * np.log10(rx_power_w * 1e3):.2f} dBm "
          f"(loss {actual_loss_db:.1f} dB, expected {expected_loss_db:.1f} dB)")

    # ---- 物理常数和参数 ----
    h = 6.62607015e-34
    nu = sig_optical.ctx.center_freq
    rs_hz = sig_optical.ctx.sym_rate
    nf_linear = 10 ** (nf_db / 10)
    gain_linear = p_target_w / rx_power_w
    G_minus_1 = max(gain_linear - 1.0, 0.0)

    # 理论 ASE 噪声
    ase_var_per_iq = 0.5 * nf_linear * h * nu * rs_hz * G_minus_1
    theory_noise_power_per_pol = 2.0 * ase_var_per_iq  # complex noise power
    p_sig_per_pol = p_target_w / 2.0
    snr_theory_lin = p_sig_per_pol / theory_noise_power_per_pol
    snr_theory_db = 10.0 * np.log10(snr_theory_lin)

    print(f"\n  [Theoretical ASE]")
    print(f"    NF = {nf_db:.1f} dB,  Gain = {10*np.log10(gain_linear):.1f} dB")
    print(f"    σ²_ASE (per I/Q) = {ase_var_per_iq:.4e} W")
    print(f"    Theory SNR = {snr_theory_db:.2f} dB")

    # ========================================================================
    # 方法 A: 直接噪声测量 (两次运行差分)
    # ========================================================================
    print(f"\n  [Method A: Direct Noise Measurement (Dual-Run Difference)]")

    amp_a = IdealAmplifier(nf_db=nf_db)
    amp_b = IdealAmplifier(nf_db=nf_db)

    np.random.seed(seed_a)
    out_a = amp_a.forward(rx_signal)

    np.random.seed(seed_b)
    out_b = amp_b.forward(rx_signal)

    # diff = (s+n1) - (s+n2) = n1 - n2, Var(diff) = 2*Var(n)
    noise_diff = out_a.data - out_b.data
    # 测量复数噪声总功率 per pol
    measured_complex_noise_power_per_pol = (
        0.5 * np.mean(np.abs(noise_diff) ** 2, axis=1)
    )

    noise_meas_db = 10 * np.log10(measured_complex_noise_power_per_pol * 1e3)
    noise_theory_db = 10 * np.log10(theory_noise_power_per_pol * 1e3)

    print(f"    Measured noise power per pol: {measured_complex_noise_power_per_pol[0]:.4e} W")
    print(f"    Theory noise power per pol:   {theory_noise_power_per_pol:.4e} W")
    print(f"    Ratio (meas/theory): {measured_complex_noise_power_per_pol[0] / theory_noise_power_per_pol:.4f} "
          f"= {10*np.log10(measured_complex_noise_power_per_pol[0] / theory_noise_power_per_pol):+.2f} dB")

    # 检查每个偏振的噪声一致性
    method_a_ok = True
    for pol in range(2):
        ratio = measured_complex_noise_power_per_pol[pol] / theory_noise_power_per_pol
        if abs(10 * np.log10(ratio)) > 0.5:
            method_a_ok = False
            print(f"    Pol{pol}: ratio={ratio:.3f} ({10*np.log10(ratio):+.2f} dB) — OUT OF TOLERANCE")

    # ========================================================================
    # 方法 B: 完整接收链 + EVM → SNR
    # ========================================================================
    print(f"\n  [Method B: EVM-Based SNR (Full RX Chain)]")

    ber_calc = BERCalculator(ignore_headers=30)
    constellation_obj = Constellation(MODULATION, power_norm=True)

    # 对两次运行都进行 EVM 计算，取平均
    evm_list = []
    for out in [out_a, out_b]:
        rx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=True)
        rx_matched = rx_rrc.forward(out)
        evm = ber_calc.calculate_evm(rx_matched, constellation_obj, tx_bits)
        snr_evm = -20.0 * np.log10(evm / 100.0)
        evm_list.append(snr_evm)
        print(f"    EVM = {evm:.3f}%,  SNR = {snr_evm:.2f} dB")

    # EVM 端理论 SNR 需计入匹配滤波器处理增益 (sps 倍)
    # 匹配滤波器在符号采样点将噪声带宽压缩为 R_s，等效噪声功率降至 1/sps
    snr_evm_theory_db = snr_theory_db + 10.0 * np.log10(RRC_SPS)

    snr_evm_avg = np.mean(evm_list)
    delta_evm = snr_evm_avg - snr_evm_theory_db

    print(f"\n  [Cross-Validation]")
    print(f"    Raw ASE SNR (pre-MF)       = {snr_theory_db:.2f} dB")
    print(f"    MF gain (sps={RRC_SPS})           = +{10 * np.log10(RRC_SPS):.2f} dB")
    print(f"    Theory SNR (post-MF)       = {snr_evm_theory_db:.2f} dB")
    print(f"    SNR from EVM (avg)         = {snr_evm_avg:.2f} dB")
    print(f"    Δ (EVM - Theory)            = {delta_evm:+.2f} dB")

    # ---- 判定 ----
    tolerance_dir = 0.5   # 直接测量容许 0.5 dB
    tolerance_evm = 2.0   # EVM 端容许 2 dB (含滤波器/降采样影响)
    method_b_ok = abs(delta_evm) < tolerance_evm

    all_ok = method_a_ok and method_b_ok

    status = "[PASS]" if all_ok else "[FAIL]"
    details = []
    if not method_a_ok:
        details.append("Direct noise mismatch")
    if not method_b_ok:
        details.append(f"EVM SNR mismatch (Δ={delta_evm:+.2f} dB)")

    print(f"\n  Overall: {status} ({'; '.join(details) if details else 'OK'})")

    return all_ok


def run_all_tests():
    """运行所有测试场景。"""
    print("\n" + "=" * 70)
    print("WDM + Fiber (ISRS/NLO) + IdealAmplifier Test Suite")
    print("=" * 70)

    results = {}

    # # 测试 1: Raman 倾斜效应
    # print("\n" + "-" * 70)
    # print("Test 1: Raman Tilt Effect")
    # print("-" * 70)
    # try:
    #     results["raman_tilt"] = test_raman_tilt()
    # except Exception as e:
    #     print(f"  [ERROR] {e}")
    #     import traceback
    #     traceback.print_exc()
    #     results["raman_tilt"] = False

    # 测试 2: IdealAmplifier ASE 和功率恢复
    print("\n" + "-" * 70)
    print("Test 2: IdealAmplifier ASE + Power Recovery")
    print("-" * 70)
    try:
        results["ideal_amp"] = test_ideal_amp_ase()
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        results["ideal_amp"] = False

    # 验证: ASE SNR 一致性 (EVM vs. 理论)
    print("\n" + "-" * 70)
    print("Verification: ASE SNR Consistency (EVM vs Theory)")
    print("-" * 70)
    try:
        results["ase_snr_consistency"] = verify_ase_snr_consistency()
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        results["ase_snr_consistency"] = False

    # 总结
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
