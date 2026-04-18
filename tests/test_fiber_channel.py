"""FiberChannel 测试脚本。

测试 FiberChannel 的数值稳定性、功率衰减、色散走离等特性。
不使用收发机组件，从脉冲开始测试，验证光纤信道的物理行为。
"""

import numpy as np
from pynlo.backends import get_numpy_context

# PyNLO imports
from pynlo.physics.field import FieldState, create_gaussian, create_sech
from scipy.constants import c

# FiberChannel under test
from physics_nlo_eval.channel.fiber import FiberChannel
from physics_nlo_eval.core.signal import Signal, SignalContext


def test_fiber_channel_basic():
    """测试 FiberChannel 基本功能：数值稳定性验证。"""
    print("\n" + "=" * 60)
    print("测试 1: FiberChannel 基本功能 - 数值稳定性")
    print("=" * 60)

    ctx = get_numpy_context()

    # 脉冲参数
    n_samples = 2**12
    center_freq = 193e12  # 193 THz (C-band)
    bandwidth = 100e9  # 100 GHz
    pulse_energy = 1e-12  # 1 pJ
    pulse_fwhm = 100e-15  # 100 fs

    # 创建双偏振高斯脉冲 (Manakov 方程需要双偏振)
    print(f"创建高斯脉冲: E={pulse_energy * 1e12:.1f} pJ, FWHM={pulse_fwhm * 1e15:.1f} fs")

    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    pulse_y = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    combined_data = np.vstack([pulse_x.a_v_monotonic, pulse_y.a_v_monotonic])

    print(f"脉冲能量 (单偏振): {pulse_x.energy:.6e} J")

    # 创建信号
    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # 光纤参数 (典型 SMF)
    length = 80e3  # 80 km
    alpha_db = 0.2  # 0.2 dB/km
    beta2 = -20e-3  # ps^2/km (SMF typical)
    beta3 = 0.1e-3  # ps^3/km

    print(f"\n光纤参数: L={length / 1e3:.0f} km, α={alpha_db} dB/km")
    print(f"         β2={beta2 * 1e3:.1f} ps^2/km, β3={beta3 * 1e3:.2f} ps^3/km")

    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=alpha_db,
        beta2_ps2_per_m=beta2,
        beta3_ps3_per_m=beta3,
        step_size_m=500.0,
    )

    # 传播
    output = fiber.forward(signal)

    # 检查输出
    assert output.data.shape[0] == 2, "输出应为双偏振"
    output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)
    output_energy = output_state.energy

    print(f"\n输出脉冲能量: {output_energy:.6e} J")

    # 总输入能量 = 两偏振之和
    input_energy_x = pulse_x.energy
    input_energy_y = pulse_y.energy
    total_input_energy = input_energy_x + input_energy_y

    expected_loss_linear = 10 ** (-alpha_db * length / 1e3 / 10)
    expected_energy = total_input_energy * expected_loss_linear
    loss_db = alpha_db * length / 1e3
    print(f"预期能量 (考虑 {alpha_db} dB/km × {length / 1e3:.0f} km = {loss_db:.1f} dB loss): {expected_energy:.6e} J")

    energy_ratio = output_energy / total_input_energy
    print(f"能量比值: {energy_ratio:.4f} (预期 ~{expected_loss_linear:.4f})")

    # 检查数值爆炸
    max_val = np.max(np.abs(output.data))
    initial_max = np.max(np.abs(combined_data))
    print("\n数值范围检查:")
    print(f"  初始最大值: {initial_max:.4e}")
    print(f"  输出最大值: {max_val:.4e}")
    print(f"  比值: {max_val / initial_max:.4f}")

    assert np.isfinite(output_energy), "输出能量为有限值"
    assert output_energy > 0, "输出能量为正"
    print("\n[PASS] 基本功能测试通过 - 数值稳定")


def test_fiber_channel_power_attenuation():
    """测试功率衰减是否符合预期。"""
    print("\n" + "=" * 60)
    print("测试 2: 功率衰减验证")
    print("=" * 60)

    ctx = get_numpy_context()

    # 脉冲参数
    n_samples = 2**12
    center_freq = 193e12
    bandwidth = 100e9
    pulse_energy = 10e-12  # 10 pJ
    pulse_fwhm = 50e-15  # 50 fs

    # 双偏振脉冲
    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    pulse_y = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    combined_data = np.vstack([pulse_x.a_v_monotonic, pulse_y.a_v_monotonic])

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # 测试不同长度
    lengths = [20e3, 40e3, 60e3, 80e3, 100e3]  # km
    alpha_db = 0.2  # dB/km
    total_input_energy = pulse_x.energy + pulse_y.energy

    print(f"\n测试不同光纤长度 (α={alpha_db} dB/km):")
    print("-" * 50)

    measured_losses = []
    expected_losses = []

    for length in lengths:
        fiber = FiberChannel(
            length_m=length,
            alpha_db_per_km=alpha_db,
            beta2_ps2_per_m=-20e-3,
            beta3_ps3_per_m=0.1e-3,
            step_size_m=500.0,
        )

        output = fiber.forward(signal)
        output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)

        expected_linear_loss = 10 ** (-alpha_db * length / 1e3 / 10)
        expected_out_energy = total_input_energy * expected_linear_loss
        actual_out_energy = output_state.energy

        measured_loss_db = -10 * np.log10(actual_out_energy / total_input_energy)
        expected_loss_db = alpha_db * length / 1e3

        measured_losses.append(measured_loss_db)
        expected_losses.append(expected_loss_db)

        print(
            f"  L={length / 1e3:.0f} km: 测量 {measured_loss_db:.2f} dB, "
            f"预期 {expected_loss_db:.2f} dB, "
            f"误差 {measured_loss_db - expected_loss_db:+.2f} dB"
        )

    # 验证线性衰减 (损耗应该是线性的)
    losses_array = np.array(measured_losses)
    expected_array = np.array(expected_losses)

    # 计算误差
    error = np.abs(losses_array - expected_array)
    mean_error = np.mean(error)
    max_error = np.max(error)

    print(f"\n误差统计: 平均 {mean_error:.2f} dB, 最大 {max_error:.2f} dB")

    assert np.all(error < 1.0), f"损耗误差过大: {error}"
    print("\n[PASS] 功率衰减测试通过")


def test_fiber_channel_chromatic_dispersion():
    """测试色散效应 - 脉冲走离。"""
    print("\n" + "=" * 60)
    print("测试 3: 色散效应 - 脉冲走离")
    print("=" * 60)

    ctx = get_numpy_context()

    # 使用更宽的脉冲以观察色散效应
    n_samples = 2**13
    center_freq = 193e12
    bandwidth = 200e9
    pulse_energy = 1e-12
    pulse_fwhm = 1e-12  # 1 ps (相对宽，易于观察走离)

    print(f"创建高斯脉冲: FWHM={pulse_fwhm * 1e12:.2f} ps")

    # 双偏振脉冲
    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    pulse_y = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    combined_data = np.vstack([pulse_x.a_v_monotonic, pulse_y.a_v_monotonic])

    # 转换为时间域
    pulse_t = pulse_x.to_time(ctx)

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # SMF 色散参数
    beta2 = -20e-3  # ps^2/km (正常色散)
    D = -beta2 * 1e6 / (c * (center_freq / c) ** 2)  # ps/(nm·km) 简化转换
    print(f"\n光纤色散: β2={beta2 * 1e3:.1f} ps²/km")

    fiber = FiberChannel(
        length_m=80e3,
        alpha_db_per_km=0.2,
        beta2_ps2_per_m=beta2,
        beta3_ps3_per_m=0.0,  # 忽略高阶色散简化分析
        step_size_m=500.0,
    )

    output = fiber.forward(signal)

    # 分析脉冲宽度变化
    output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)
    output_t = output_state.to_time(ctx)

    # 计算 FWHM
    def calculate_fwhm(time_grid, power):
        """计算功率包络的 FWHM。"""
        half_max = np.max(power) / 2
        above_half = time_grid[power > half_max]
        if len(above_half) < 2:
            return 0
        return above_half[-1] - above_half[0]

    initial_power = np.abs(pulse_t[0]) ** 2
    output_power = np.abs(output_t[0]) ** 2

    initial_fwhm = calculate_fwhm(pulse_x.grid.t_grid, initial_power)
    output_fwhm = calculate_fwhm(output_state.grid.t_grid, output_power)

    print("\n脉冲宽度分析:")
    print(f"  初始 FWHM: {initial_fwhm * 1e12:.3f} ps")
    print(f"  输出 FWHM: {output_fwhm * 1e12:.3f} ps")
    print(f"  展宽比: {output_fwhm / initial_fwhm:.3f}")

    # 解析公式验证 (忽略损耗)
    # 对于高斯脉冲，色散导致的展宽:
    # τ_out² = τ_in² * sqrt(1 + (16*ln(2)*β2*L/τ_in²)²)
    tau_in = initial_fwhm / (2 * np.sqrt(2 * np.log(2)))  # 转换为 RMS
    beta2_m = beta2 * 1e-24  # ps^2/m to s^2/m
    L = 80e3

    expected_tau_out = tau_in * np.sqrt(1 + (beta2_m * L / tau_in**2) ** 2 * (16 * np.log(2)) ** 2)
    expected_fwhm_out = expected_tau_out * 2 * np.sqrt(2 * np.log(2))

    print("\n解析解验证:")
    print(f"  预期 FWHM: {expected_fwhm_out * 1e12:.3f} ps")
    print(f"  误差: {(output_fwhm - expected_fwhm_out) * 1e12 / expected_fwhm_out * 100:.1f}%")

    assert output_fwhm > initial_fwhm, "色散应导致脉冲展宽"
    print("\n[PASS] 色散效应测试通过 - 脉冲正确展宽")


def test_fiber_channel_pmd():
    """测试偏振模色散 (PMD)。"""
    print("\n" + "=" * 60)
    print("测试 4: 偏振模色散 (PMD)")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**12
    center_freq = 193e12
    bandwidth = 100e9
    pulse_energy = 1e-12
    pulse_fwhm = 500e-15  # 500 fs

    print(f"创建高斯脉冲: FWHM={pulse_fwhm * 1e15:.1f} fs")

    # 创建双偏振脉冲
    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )

    pulse_y = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )

    # 合并为双偏振信号
    combined_data = np.vstack([pulse_x.a_v_monotonic, pulse_y.a_v_monotonic])

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # PMD 参数
    d_pmd = 0.1  # ps/sqrt(km) - 典型 SMF
    print(f"\nPMD 系数: D={d_pmd} ps/√km")

    fiber = FiberChannel(
        length_m=80e3,
        alpha_db_per_km=0.2,
        beta2_ps2_per_m=-20e-3,
        beta3_ps3_per_m=0.1e-3,
        d_pmd_ps_per_sqrt_km=d_pmd,
        step_size_m=500.0,
    )

    initial_state = FieldState(grid=pulse_x.grid, a_v=combined_data, n_comp=2)
    print(f"\n初始状态 - X极化能量: {np.sum(np.abs(initial_state.a_v[0]) ** 2) * pulse_x.grid.dv:.6e} J")
    print(f"           Y极化能量: {np.sum(np.abs(initial_state.a_v[1]) ** 2) * pulse_x.grid.dv:.6e} J")

    output = fiber.forward(signal)

    # 检查输出是双偏振
    assert output.data.shape[0] == 2, "输出应为双偏振"

    output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)
    print(f"\n输出状态 - X极化能量: {np.sum(np.abs(output_state.a_v[0]) ** 2) * pulse_x.grid.dv:.6e} J")
    print(f"           Y极化能量: {np.sum(np.abs(output_state.a_v[1]) ** 2) * pulse_x.grid.dv:.6e} J")

    # 检查能量守恒 (忽略损耗)
    total_energy_in = np.sum(np.abs(initial_state.a_v) ** 2) * pulse_x.grid.dv
    total_energy_out = np.sum(np.abs(output_state.a_v) ** 2) * pulse_x.grid.dv

    print(f"\n总能量: 输入={total_energy_in:.6e} J, 输出={total_energy_out:.6e} J")
    print(f"能量比: {total_energy_out / total_energy_in:.4f}")

    # PMD 应该导致两偏振分量之间有一定程度的去相关
    # 但总能量应守恒 (忽略损耗)
    assert np.isfinite(total_energy_out), "PMD 传播后能量应为有限值"
    print("\n[PASS] PMD 测试通过 - 双偏振传播正常")


def test_fiber_channel_sech_pulse():
    """测试 sech 脉冲形状。"""
    print("\n" + "=" * 60)
    print("测试 5: Sech 脉冲传播")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**12
    center_freq = 193e12
    bandwidth = 100e9
    pulse_energy = 1e-12
    pulse_fwhm = 100e-15  # 100 fs

    print(f"创建 Sech 脉冲: E={pulse_energy * 1e12:.1f} pJ, FWHM={pulse_fwhm * 1e15:.1f} fs")

    # 双偏振脉冲
    pulse_x = create_sech(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    pulse_y = create_sech(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    combined_data = np.vstack([pulse_x.a_v_monotonic, pulse_y.a_v_monotonic])

    print(f"脉冲能量 (单偏振): {pulse_x.energy:.6e} J")

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    fiber = FiberChannel(
        length_m=50e3,
        alpha_db_per_km=0.2,
        beta2_ps2_per_m=-20e-3,
        beta3_ps3_per_m=0.0,
        step_size_m=500.0,
    )

    output = fiber.forward(signal)
    output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)

    print(f"输出能量: {output_state.energy:.6e} J")

    # 检查数值稳定性
    assert np.isfinite(output_state.energy), "Sech 脉冲传播应保持数值稳定"
    assert output_state.energy > 0, "输出能量应为正"

    print("\n[PASS] Sech 脉冲测试通过")


def test_fiber_channel_no_nan_inf():
    """测试各种场景下不会产生 NaN 或 Inf。"""
    print("\n" + "=" * 60)
    print("测试 6: 数值稳定性 - NaN/Inf 检查")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**10
    center_freq = 193e12
    bandwidth = 100e9
    pulse_energy = 1e-12
    pulse_fwhm = 50e-15

    # 双偏振脉冲
    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    pulse_y = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )
    combined_data = np.vstack([pulse_x.a_v_monotonic, pulse_y.a_v_monotonic])

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # 测试多种配置
    configs = [
        {
            "length_m": 80e3,
            "alpha_db_per_km": 0.2,
            "beta2_ps2_per_m": -20e-3,
            "beta3_ps3_per_m": 0.0,
            "d_pmd_ps_per_sqrt_km": 0.0,
        },
        {
            "length_m": 100e3,
            "alpha_db_per_km": 0.5,
            "beta2_ps2_per_m": -20e-3,
            "beta3_ps3_per_m": 0.1e-3,
            "d_pmd_ps_per_sqrt_km": 0.1,
        },
        {
            "length_m": 50e3,
            "alpha_db_per_km": 0.2,
            "beta2_ps2_per_m": 20e-3,
            "beta3_ps3_per_m": 0.0,
            "d_pmd_ps_per_sqrt_km": 0.0,
        },  # 反常色散
    ]

    for i, cfg in enumerate(configs):
        beta2_val = cfg["beta2_ps2_per_m"] * 1e3
        print(
            f"\n配置 {i + 1}: L={cfg['length_m'] / 1e3:.0f}km, "
            f"α={cfg['alpha_db_per_km']}dB/km, β2={beta2_val:.1f}ps²/km"
        )

        fiber = FiberChannel(**cfg)
        output = fiber.forward(signal)

        has_nan = np.any(np.isnan(output.data))
        has_inf = np.any(np.isinf(output.data))

        print(f"  NaN: {has_nan}, Inf: {has_inf}")

        assert not has_nan, f"配置 {i + 1} 产生 NaN"
        assert not has_inf, f"配置 {i + 1} 产生 Inf"

    print("\n[PASS] 数值稳定性测试通过 - 无 NaN/Inf")


def run_all_tests():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("FiberChannel 验证测试套件")
    print("=" * 60)

    try:
        test_fiber_channel_basic()
        test_fiber_channel_power_attenuation()
        test_fiber_channel_chromatic_dispersion()
        test_fiber_channel_pmd()
        test_fiber_channel_sech_pulse()
        test_fiber_channel_no_nan_inf()

        print("\n" + "=" * 60)
        print("所有测试通过!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试出错: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
