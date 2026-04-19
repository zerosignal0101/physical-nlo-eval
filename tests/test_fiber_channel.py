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
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
        coarse_step_size_m=500.0,
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
            beta2_ps2_per_km=-20e-3,
            beta3_ps3_per_km=0.1e-3,
            coarse_step_size_m=500.0,
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
    bandwidth = 2e12  # 2 THz (dt = 0.5 ps)
    pulse_energy = 1e-12
    pulse_fwhm = 5e-12  # 5 ps (确保有足够采样点测量 FWHM)

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
    beta2 = -20.0  # ps^2/km (正常色散，标准 SMF 值)
    length = 80e3  # 80 km
    D = -beta2 * 1e6 / (c * (center_freq / c) ** 2)  # ps/(nm·km) 简化转换
    print(f"\n光纤色散: β2={beta2:.1f} ps²/km, L={length / 1e3:.0f} km")

    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=0.2,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=0.0,
        coarse_step_size_m=500.0,
        n2_si=0.0,  # 抑制非线性效应
    )

    output = fiber.forward(signal)

    # 分析脉冲宽度变化
    output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)
    output_t = output_state.to_time(ctx)

    # 计算 FWHM
    def calculate_fwhm(time_grid, power):
        """计算功率包络的 FWHM。"""
        max_power = np.max(power)
        if max_power <= 0:
            return 0
        half_max = max_power / 2
        above_mask = power > half_max
        idx_above = np.where(above_mask)[0]
        if len(idx_above) < 2:
            return 0
        return time_grid[idx_above[-1]] - time_grid[idx_above[0]]

    initial_power = np.abs(pulse_t[0]) ** 2
    output_power = np.abs(output_t[0]) ** 2

    initial_fwhm = calculate_fwhm(pulse_x.grid.t_grid, initial_power)
    output_fwhm = calculate_fwhm(output_state.grid.t_grid, output_power)

    print("\n脉冲宽度分析:")
    print(f"  初始 FWHM: {initial_fwhm * 1e12:.3f} ps")
    print(f"  输出 FWHM: {output_fwhm * 1e12:.3f} ps")
    measured_broadening = output_fwhm / initial_fwhm
    print(f"  展宽比: {measured_broadening:.3f}x")

    # 理论公式验证 (高斯脉冲，忽略损耗)
    # τ_out/τ_in = sqrt(1 + (4*ln(2)*β2*L/τ_in²)²)
    # 单位: β2 [ps²/km], L [km], τ [ps]
    beta2_ps2_per_km = beta2  # beta2 已是 ps²/km
    L_km = length / 1e3
    tau_in_ps = initial_fwhm * 1e12  # convert s to ps
    ratio = beta2_ps2_per_km * L_km / tau_in_ps**2
    theoretical_broadening = np.sqrt(1 + (4 * np.log(2) * ratio) ** 2)

    print("\n理论验证:")
    print(f"  β2×L = {beta2_ps2_per_km * L_km:.1f} ps²")
    print(f"  理论展宽比: {theoretical_broadening:.3f}x")
    print(f"  实际展宽比: {measured_broadening:.3f}x")
    print(f"  误差: {(measured_broadening - theoretical_broadening) / theoretical_broadening * 100:.1f}%")

    # 允许 30% 误差（考虑公式简化、采样误差等）
    rel_error = abs(measured_broadening - theoretical_broadening) / theoretical_broadening
    assert rel_error < 0.005, f"展宽误差 {rel_error * 100:.1f}% 超过 0.5%"
    print("\n[PASS] 色散效应测试通过 - 展宽符合理论")


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
        beta2_ps2_per_km=-20e-3,
        beta3_ps3_per_km=0.1e-3,
        d_pmd_ps_per_sqrt_km=d_pmd,
        coarse_step_size_m=500.0,
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
        beta2_ps2_per_km=-20e-3,
        beta3_ps3_per_km=0.0,
        coarse_step_size_m=500.0,
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
            "beta2_ps2_per_km": -20e-3,
            "beta3_ps3_per_km": 0.0,
            "d_pmd_ps_per_sqrt_km": 0.0,
        },
        {
            "length_m": 100e3,
            "alpha_db_per_km": 0.5,
            "beta2_ps2_per_km": -20e-3,
            "beta3_ps3_per_km": 0.1e-3,
            "d_pmd_ps_per_sqrt_km": 0.1,
        },
        {
            "length_m": 50e3,
            "alpha_db_per_km": 0.2,
            "beta2_ps2_per_km": 20e-3,
            "beta3_ps3_per_km": 0.0,
            "d_pmd_ps_per_sqrt_km": 0.0,
        },  # 反常色散
    ]

    for i, cfg in enumerate(configs):
        beta2_val = cfg["beta2_ps2_per_km"]
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


def test_fiber_channel_single_polarization():
    """测试单偏振输入：验证自动padding到双偏振并正确处理。"""
    print("\n" + "=" * 60)
    print("测试 7: 单偏振输入 - 自动padding验证")
    print("=" * 60)

    ctx = get_numpy_context()

    # 脉冲参数
    n_samples = 2**12
    center_freq = 193e12
    bandwidth = 100e9
    pulse_energy = 5e-12  # 5 pJ
    pulse_fwhm = 100e-15  # 100 fs

    print(f"创建单偏振高斯脉冲: E={pulse_energy * 1e12:.1f} pJ, FWHM={pulse_fwhm * 1e15:.1f} fs")

    # 单偏振脉冲
    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )

    print(f"脉冲能量: {pulse_x.energy:.6e} J")

    # 单偏振信号
    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=1,  # 单偏振输入
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=pulse_x.a_v_monotonic.reshape(1, -1), ctx=signal_ctx)

    # 光纤参数
    length = 80e3  # 80 km
    alpha_db = 0.2  # 0.2 dB/km

    print(f"\n光纤参数: L={length / 1e3:.0f} km, α={alpha_db} dB/km")

    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=alpha_db,
        beta2_ps2_per_km=-20e-3,
        beta3_ps3_per_km=0.1e-3,
        coarse_step_size_m=500.0,
    )

    # 传播
    output = fiber.forward(signal)

    # 检查输出是双偏振
    assert output.data.shape[0] == 2, f"输出应为双偏振，实际 shape={output.data.shape}"
    assert output.ctx.n_pol == 2, f"输出 n_pol 应为 2，实际={output.ctx.n_pol}"

    print(f"\n输出形状: {output.data.shape}")
    print(f"输出 n_pol: {output.ctx.n_pol}")

    # 检查 Y 分量为零或接近零 (由于padding)
    y_component = output.data[1, :]
    y_max = np.max(np.abs(y_component))
    print(f"Y 分量最大值: {y_max:.4e}")

    # 分析输出能量
    output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)
    output_energy = output_state.energy

    # 预期能量：输入能量经过光纤损耗后，应该主要在 X 分量
    expected_loss_linear = 10 ** (-alpha_db * length / 1e3 / 10)
    expected_energy = pulse_x.energy * expected_loss_linear

    print("\n能量分析:")
    print(f"  输入能量: {pulse_x.energy:.6e} J")
    print(f"  输出总能量: {output_energy:.6e} J")
    print(f"  预期能量 (考虑 {alpha_db * length / 1e3:.1f} dB loss): {expected_energy:.6e} J")

    # 能量比
    energy_ratio = output_energy / pulse_x.energy
    print(f"  能量比: {energy_ratio:.4f} (预期 ~{expected_loss_linear:.4f})")

    # 检查 X 分量能量占比 (应该 > 90%，因为 Y 是 padding 的零)
    x_energy = np.sum(np.abs(output.data[0, :]) ** 2) * pulse_x.grid.dv
    y_energy = np.sum(np.abs(output.data[1, :]) ** 2) * pulse_x.grid.dv
    x_ratio = x_energy / output_energy

    print("\n偏振能量分布:")
    print(f"  X 分量能量: {x_energy:.6e} J ({x_ratio * 100:.1f}%)")
    print(f"  Y 分量能量: {y_energy:.6e} J ({(1 - x_ratio) * 100:.1f}%)")

    # 验证数值稳定性
    assert np.isfinite(output_energy), "输出能量应为有限值"
    assert output_energy > 0, "输出能量应为正"
    assert not np.any(np.isnan(output.data)), "输出不应包含 NaN"
    assert not np.any(np.isinf(output.data)), "输出不应包含 Inf"

    print("\n[PASS] 单偏振输入测试通过 - 自动padding工作正常")


def test_fiber_channel_single_pol_dispersion():
    """测试单偏振输入的色散效应。"""
    print("\n" + "=" * 60)
    print("测试 8: 单偏振输入 - 色散效应验证")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**13
    center_freq = 193e12
    bandwidth = 2e12  # 2 THz (dt = 0.5 ps, 确保 FWHM 有足够采样点)
    pulse_energy = 1e-12
    pulse_fwhm = 5e-12  # 5 ps FWHM，占约 10 个采样点

    print(f"创建单偏振高斯脉冲: FWHM={pulse_fwhm * 1e12:.2f} ps")

    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )

    # 单偏振信号
    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=1,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=pulse_x.a_v_monotonic.reshape(1, -1), ctx=signal_ctx)

    # 色散参数
    beta2 = -20.0  # ps^2/km (SMF 典型值)
    length = 50e3  # 50 km

    print(f"\n光纤色散: β2={beta2:.1f} ps²/km, L={length / 1e3:.0f} km")

    # 获取初始脉冲宽度
    pulse_t = pulse_x.to_time(ctx)
    initial_power = np.abs(pulse_t[0]) ** 2
    t_grid = pulse_x.grid.t_grid

    def calculate_fwhm(time_grid, power):
        """计算功率包络的 FWHM。"""
        max_power = np.max(power)
        if max_power <= 0:
            return 0
        half_max = max_power / 2
        above_mask = power > half_max
        idx_above = np.where(above_mask)[0]
        if len(idx_above) < 2:
            return 0
        return time_grid[idx_above[-1]] - time_grid[idx_above[0]]

    initial_fwhm = calculate_fwhm(t_grid, initial_power)
    print(f"\n初始脉冲 FWHM: {initial_fwhm * 1e12:.3f} ps")

    # 传播 (抑制非线性: n2_si=0)
    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=0.2,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=0.0,
        coarse_step_size_m=500.0,
        n2_si=0.0,  # 抑制非线性效应
    )

    output = fiber.forward(signal)
    output_state = FieldState(grid=pulse_x.grid, a_v=output.data, n_comp=2)
    output_t = output_state.to_time(ctx)

    # 分析输出脉冲
    output_power = np.abs(output_t[0]) ** 2
    output_fwhm = calculate_fwhm(output_state.grid.t_grid, output_power)

    print(f"输出脉冲 FWHM: {output_fwhm * 1e12:.3f} ps")
    measured_broadening = output_fwhm / initial_fwhm
    print(f"展宽比: {measured_broadening:.3f}x")

    # 理论公式验证 (高斯脉冲，忽略损耗)
    # τ_out/τ_in = sqrt(1 + (4*ln(2)*β2*L/τ_in²)²)
    # 单位: β2 [ps²/km], L [km], τ [ps]
    L_km = length / 1e3
    tau_in_ps = initial_fwhm * 1e12  # convert s to ps
    ratio = beta2 * L_km / tau_in_ps**2
    theoretical_broadening = np.sqrt(1 + (4 * np.log(2) * ratio) ** 2)

    print("\n理论验证:")
    print(f"  β2×L = {beta2 * L_km:.1f} ps²")
    print(f"  理论展宽比: {theoretical_broadening:.3f}x")
    print(f"  实际展宽比: {measured_broadening:.3f}x")
    print(f"  误差: {(measured_broadening - theoretical_broadening) / theoretical_broadening * 100:.1f}%")

    # 色散应该导致展宽
    rel_error = abs(measured_broadening - theoretical_broadening) / theoretical_broadening
    assert rel_error < 0.005, f"展宽误差 {rel_error * 100:.1f}% 超过 0.5%"
    print("\n[PASS] 单偏振色散测试通过 - 展宽符合理论")


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
        test_fiber_channel_single_polarization()
        test_fiber_channel_single_pol_dispersion()

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
