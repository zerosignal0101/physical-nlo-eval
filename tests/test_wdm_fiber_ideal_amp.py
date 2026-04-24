"""
WDM + 光纤 (ISRS/NLO) + IdealAmplifier 测试

测试目标：
1. WDM 系统通过启用 ISRS (拉曼散射) 和 NLO (非线性) 的光纤
2. 使用 IdealAmplifier 恢复入纤功率
3. 验证功率恢复是否正确

"""

import numpy as np
from pynlo.utility.chi3 import raman_from_dat

from physics_nlo_eval.channel.fiber import IdealAmplifier
from physics_nlo_eval.channel.wdm import ChannelSpec, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter

# ============================================================================
# 参数配置 (用于演示 SRS 功率倾斜效应)
# ============================================================================

# WDM 规划 - 扩展到 5 通道，更宽间隔以突出 SRS 效应
WDM_PLAN = WDMPlan(
    reference_freq_hz=193.1e12,
    channels=[
        ChannelSpec(channel_id=0, freq_offset_hz=-200e9),  # 间隔 200 GHz
        ChannelSpec(channel_id=1, freq_offset_hz=-100e9),
        ChannelSpec(channel_id=2, freq_offset_hz=0e9),
        ChannelSpec(channel_id=3, freq_offset_hz=100e9),
        ChannelSpec(channel_id=4, freq_offset_hz=200e9),
    ],
)
TARGET_SAMPLE_RATE_HZ = 600e9

# 调制参数
MODULATION = "DP-QPSK"
SYMBOL_RATE_HZ = 50e9
RRC_SPS = 4
RRC_ROLL_OFF = 0.1

# 入纤功率 (每个偏振) - 提高功率以增强 SRS 效应
LAUNCH_POWER_DBM_PER_POL = 6.0  # 6 dBm/pol = 4 mW/pol, 总共 ~12 dBm for 5 ch

# 光纤参数 - 使用更长链路和多跨段以累积 SRS 效应
FIBER_LENGTH_KM = 400.0  # 400 km (4 x 100 km spans)
ALPHA_DB_PER_KM = 0.2
BETA2_PS2_PER_KM = -20.0
BETA3_PS3_PER_KM = 0.1e-3
N2_SI = 2.6e-20  # 非线性折射率 (silica)
A_EFF_UM2 = 80.0
NUM_SPANS = 4  # 4 跨段，每跨段 100 km
SPAN_LENGTH_KM = FIBER_LENGTH_KM / NUM_SPANS  # 每跨段长度
COARSE_STEP_SIZE_M = 1000.0  # 初始猜测值，PyNLO 内部会自动调整

# 仿真参数
NUM_SYMBOLS = 4000
SEED = 42


# ============================================================================
# 辅助函数
# ============================================================================


def create_raman_response(n_points: int, dt_s: float, wdm_signal) -> np.ndarray:
    """为给定网格创建拉曼响应函数。

    Parameters
    ----------
    n_points : int
        网格点数 (应与信号长度匹配)。
    dt_s : float
        时间步长 (秒)。
    wdm_signal : Signal
        WDM 信号，用于获取网格信息。

    Returns
    -------
    np.ndarray
        拉曼响应函数 (rfft 域)，形状为 (n_points//2 + 1,)。
    """
    # 使用 raman_from_dat 从数据文件创建拉曼响应 (与 PyNLO 测试脚本一致)
    _, nonlinear_v = raman_from_dat(
        n_points,
        dt_s,
        filepath="D:/C-Codes/Optical-Networks/PyNLO/pynlo/utility/RamanGain.dat",
        raman_frac=0.18,
        analytic=True,
    )

    return nonlinear_v


def calculate_grid_dt(n_points: int, bandwidth_hz: float) -> float:
    """计算 PyNLO Grid 的时间步长 dt。

    PyNLO Grid.create uses:
        dv = bandwidth / (n - 1)
        dt = 1.0 / (n * dv)
    """
    dv = bandwidth_hz / (n_points - 1)
    dt = 1.0 / (n_points * dv)
    return dt


def measure_channel_power(signal: Signal, channel_idx: int | None = None) -> float:
    """测量通道功率。

    Parameters
    ----------
    signal : Signal
        输入信号。
    channel_idx : int | None
        通道索引。对于堆叠 WDM 信号需要指定。

    Returns
    -------
    float
        总功率 (W)，两个偏振合计。
    """
    if signal.ctx.meta.get("is_stacked_wdm_signal") and channel_idx is not None:
        # 堆叠 WDM: (n_channels, n_pol, N_samples)
        ch_data = signal.data[channel_idx]
        n_pol = ch_data.shape[0]
        return np.mean(np.abs(ch_data) ** 2) * n_pol
    else:
        # 普通信号: (n_pol, N_samples)
        n_pol = signal.data.shape[0]
        return np.mean(np.abs(signal.data) ** 2) * n_pol


def test_wdm_fiber_ideal_amp(
    modulation: str = "DP-QPSK",
    fiber_length_km: float = 80.0,
    nlo_enable: bool = True,
    raman_enable: bool = True,
    seed: int = 42,
    multi_span: bool = False,
):
    """WDM + 光纤 (ISRS/NLO) + IdealAmplifier 功率恢复测试。

    Parameters
    ----------
    modulation : str
        调制格式。
    fiber_length_km : float
        光纤长度 (km)。
    nlo_enable : bool
        是否启用 NLO (非线性效应)。
    raman_enable : bool
        是否启用拉曼散射 (ISRS)。
    seed : int
        随机种子。
    multi_span : bool
        是否使用多跨段 (每跨段后有 IdealAmplifier 恢复功率)。

    Returns
    -------
    bool
        测试是否通过。
    """
    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print("WDM Fiber + IdealAmplifier Test")
    print(f"{'=' * 70}")
    print(f"  Modulation: {modulation}")
    print(f"  Fiber: {fiber_length_km} km, NLO={nlo_enable}, Raman={raman_enable}, MultiSpan={multi_span}")
    print(f"  Launch Power: {LAUNCH_POWER_DBM_PER_POL} dBm/pol")

    # ---- TX 组件 ----
    mapper = Mapper(modulation=modulation, power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    tx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=False)
    iq_mod = IQModulator(launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL)

    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = NUM_SYMBOLS * bits_per_symbol * n_pol

    # ---- 生成通道 ----
    channels = []
    all_bits = []
    for ch_spec in WDM_PLAN.channels:
        tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
        all_bits.append(tx_bits)

        sig = mapper.forward(tx_bits)
        sig_rrc = tx_rrc.forward(sig)
        sig_optical = iq_mod.forward(sig_rrc)

        avg_pwr_w = np.mean(np.abs(sig_optical.data) ** 2) * n_pol
        print(
            f"  Ch{ch_spec.channel_id} ({ch_spec.freq_offset_hz / 1e9:.0f} GHz): "
            f"launch power = {10 * np.log10(avg_pwr_w * 1e3):.2f} dBm"
        )

        channels.append(sig_optical)

    # ---- WDM 合波 ----
    wdm_mux = WDMMux(wdm_plan=WDM_PLAN, target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ)
    wdm_signal = wdm_mux.combine(channels)
    print(
        f"  WDM: {len(WDM_PLAN.channels)} ch, sr={wdm_signal.ctx.sample_rate / 1e9:.0f} GHz, "
        f"shape={wdm_signal.data.shape}"
    )

    # 计算入纤总功率
    launch_power_total_w = 0.0
    for _, ch in enumerate(channels):
        ch_power = np.mean(np.abs(ch.data) ** 2) * ch.ctx.n_pol
        launch_power_total_w += ch_power
    print(f"  Total launch power: {10 * np.log10(launch_power_total_w * 1e3):.2f} dBm")

    # ---- 创建拉曼响应 (如果启用) ----
    raman_r3 = None
    if raman_enable:
        n_points = wdm_signal.num_samples
        bandwidth = wdm_signal.ctx.sample_rate
        dt = calculate_grid_dt(n_points, bandwidth)
        print(f"  Raman: n={n_points}, dt={dt * 1e12:.2f} ps")
        raman_r3 = create_raman_response(n_points, dt, wdm_signal)
        print(f"  Raman r3: shape={raman_r3.shape}, max={np.max(np.abs(raman_r3)):.4f}")

        # 验证拉曼响应可以正常工作
        from physics_nlo_eval.channel.fiber import FiberChannel

        # 创建测试光纤检查 r3 是否被正确传递
        test_fiber = FiberChannel(
            length_m=1000,
            alpha_db_per_km=0.2,
            beta2_ps2_per_km=-20.0,
            beta3_ps3_per_km=0.1e-3,
            coarse_step_size_m=1000.0,
            n2_si=2.6e-20,
            a_eff_um2=80.0,
            raman_r3=raman_r3,
            compute_ctx_name="cupy",
        )
        print(
            f"  FiberChannel raman_r3: shape={test_fiber.raman_r3.shape if test_fiber.raman_r3 is not None else None}"
        )

    # ---- 多跨段传播 ----
    span_length_m = SPAN_LENGTH_KM * 1e3 if multi_span else fiber_length_km * 1e3
    actual_num_spans = NUM_SPANS if multi_span else 1

    print(f"\n  [Fiber Propagation: {actual_num_spans} span(s), {span_length_m / 1e3:.0f} km each]")
    current_signal = wdm_signal

    amp = IdealAmplifier(nf_db=5.5)

    for span_idx in range(actual_num_spans):
        # 光纤
        fiber = FiberChannel(
            length_m=span_length_m,
            alpha_db_per_km=ALPHA_DB_PER_KM,
            beta2_ps2_per_km=BETA2_PS2_PER_KM,
            beta3_ps3_per_km=BETA3_PS3_PER_KM,
            coarse_step_size_m=COARSE_STEP_SIZE_M,  # 细步长利于 SRS
            n2_si=N2_SI if nlo_enable else 0.0,
            a_eff_um2=A_EFF_UM2,
            slope_a_eff_um2_per_nm=0.05,
            ref_lambda_nm=1550.0,
            d_pmd_ps_per_sqrt_km=0.0,
            raman_r3=raman_r3,
            compute_ctx_name="cupy",
        )

        rx_optical = fiber.forward(current_signal)

        # 打印每跨段后的功率
        powers_db = []
        for i, _ in enumerate(WDM_PLAN.channels):
            rx_power_w = measure_channel_power(rx_optical, i)
            rx_power_dbm = 10 * np.log10(rx_power_w * 1e3) if rx_power_w > 0 else -np.inf
            powers_db.append(rx_power_dbm)

        print(
            f"  Span {span_idx + 1}: "
            + " ".join([f"Ch{i}={powers_db[i]:.1f}" for i in range(len(WDM_PLAN.channels))])
            + f" (tilt={powers_db[-1] - powers_db[0]:.1f} dB)"
        )

        # 每跨段后用 IdealAmplifier 恢复功率
        if span_idx < actual_num_spans - 1:  # 除了最后一跨段
            rx_amplified = amp.forward(rx_optical)
            current_signal = rx_amplified

    # 最后一跨段后的放大
    rx_optical = current_signal if multi_span else rx_optical
    rx_amplified = amp.forward(rx_optical)

    # ---- 打印恢复结果 ----
    print("\n  [IdealAmplifier - Power Recovery After Last Span]")
    total_recovered_power_w = 0.0
    recovered_powers_db = []
    for i, _ in enumerate(WDM_PLAN.channels):
        recovered_power_w = measure_channel_power(rx_amplified, i)
        total_recovered_power_w += recovered_power_w
        recovered_power_dbm = 10 * np.log10(recovered_power_w * 1e3) if recovered_power_w > 0 else -np.inf

        # 计算原始入纤功率
        orig_power_w = np.mean(np.abs(channels[i].data) ** 2) * channels[i].ctx.n_pol
        orig_power_dbm = 10 * np.log10(orig_power_w * 1e3)
        power_error_db = recovered_power_dbm - orig_power_dbm
        recovered_powers_db.append(recovered_power_dbm)

        print(
            f"  Ch{i}: recovered = {recovered_power_dbm:.2f} dBm, "
            f"original = {orig_power_dbm:.2f} dBm, error = {power_error_db:.2f} dB"
        )

    total_recovered_dbm = 10 * np.log10(total_recovered_power_w * 1e3)
    total_original_dbm = 10 * np.log10(launch_power_total_w * 1e3)
    total_error_db = total_recovered_dbm - total_original_dbm
    print(
        f"  Total: recovered = {total_recovered_dbm:.2f} dBm, original = {total_original_dbm:.2f} dBm, "
        f"error = {total_error_db:.2f} dB"
    )

    # ---- 验证 ----
    print("\n  [Verification]")
    # 功率误差容限: ±0.5 dB
    power_tolerance_db = 0.5
    all_passed = True

    for i, _ in enumerate(WDM_PLAN.channels):
        orig_power_w = np.mean(np.abs(channels[i].data) ** 2) * channels[i].ctx.n_pol
        recovered_power_w = measure_channel_power(rx_amplified, i)
        power_error_db = 10 * np.log10(recovered_power_w / orig_power_w) if orig_power_w > 0 else np.inf

        passed = abs(power_error_db) < power_tolerance_db
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  Ch{i}: power error = {power_error_db:.3f} dB (tol = ±{power_tolerance_db} dB) {status}")

    return all_passed


def run_all_tests():
    """运行所有测试场景。"""
    print("\n" + "=" * 70)
    print("WDM + Fiber (ISRS/NLO) + IdealAmplifier Test Suite")
    print("=" * 70)

    all_success = True

    # # 测试 1: 单跨段 + Raman (ISRS) - 展示功率倾斜
    # print("\n" + "-" * 70)
    # print("Test 1: Single Span + Raman (ISRS) - Power tilt demonstration")
    # print("-" * 70)
    # try:
    #     success = test_wdm_fiber_ideal_amp(
    #         modulation="DP-QPSK",
    #         fiber_length_km=100.0,
    #         nlo_enable=True,
    #         raman_enable=True,
    #         seed=42,
    #         multi_span=False,
    #     )
    #     if not success:
    #         all_success = False
    # except Exception as e:
    #     print(f"  [ERROR] {e}")
    #     import traceback
    #     traceback.print_exc()
    #     all_success = False

    # 测试 2: 多跨段 + Raman (累积 ISRS 效应)
    print("\n" + "-" * 70)
    print("Test 2: Multi-Span + Raman (Cumulative ISRS)")
    print("-" * 70)
    try:
        success = test_wdm_fiber_ideal_amp(
            modulation="DP-QPSK",
            fiber_length_km=FIBER_LENGTH_KM,
            nlo_enable=True,
            raman_enable=True,
            seed=42,
            multi_span=True,
        )
        if not success:
            all_success = False
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback

        traceback.print_exc()
        all_success = False

    # # 测试 3: 多跨段 + Linear (无 SRS) - 对比
    # print("\n" + "-" * 70)
    # print("Test 3: Multi-Span + Linear (No SRS) - Baseline comparison")
    # print("-" * 70)
    # try:
    #     success = test_wdm_fiber_ideal_amp(
    #         modulation="DP-QPSK",
    #         fiber_length_km=FIBER_LENGTH_KM,
    #         nlo_enable=False,
    #         raman_enable=False,
    #         seed=42,
    #         multi_span=True,
    #     )
    #     if not success:
    #         all_success = False
    # except Exception as e:
    #     print(f"  [ERROR] {e}")
    #     import traceback
    #     traceback.print_exc()
    #     all_success = False

    print("\n" + "=" * 70)
    if all_success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 70)

    return all_success


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
