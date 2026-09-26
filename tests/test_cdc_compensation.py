"""ChromaticDispersionComp 色散补偿测试脚本。

测试色散补偿模块对脉冲展宽的恢复效果，验证二阶 (GVD) 和三阶色散 (TOD) 补偿。
"""

import numpy as np
from pynlo.backends import get_numpy_context
from pynlo.physics.field import create_gaussian

from physics_nlo_eval.channel.fiber import FiberChannel
from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp


def calculate_fwhm(time_grid, power):
    """计算功率包络的 FWHM，自动处理脉冲位置偏移。

    以峰值为中心，在两侧找到半功率边界来测量宽度。
    """
    max_power = np.max(power)
    if max_power <= 0:
        return 0

    peak_idx = np.argmax(power)
    half_max = max_power / 2

    # 向左搜索第一个低于半最大值的点
    left_idx = 0
    for i in range(peak_idx - 1, -1, -1):
        if power[i] < half_max:
            left_idx = i + 1
            break

    # 向右搜索第一个低于半最大值的点
    right_idx = len(power) - 1
    for i in range(peak_idx + 1, len(power)):
        if power[i] < half_max:
            right_idx = i - 1
            break

    if right_idx <= left_idx:
        return 0
    return time_grid[right_idx] - time_grid[left_idx]


def test_cdc_beta2_compensation():
    """测试二阶色散 (GVD/β2) 补偿效果。"""
    print("\n" + "=" * 60)
    print("测试 1: 二阶色散 (β2) 补偿效果")
    print("=" * 60)

    ctx = get_numpy_context()

    # 脉冲参数 - 使用较宽脉冲便于观察展宽
    n_samples = 2**13
    center_freq = 193e12  # 193 THz (C-band)
    bandwidth = 2e12  # 2 THz
    pulse_energy = 1e-12  # 1 pJ
    pulse_fwhm = 5e-12  # 5 ps FWHM

    print(f"创建高斯脉冲: FWHM={pulse_fwhm * 1e12:.2f} ps, 带宽={bandwidth * 1e-12:.1f} THz")

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
    # 获取时间域数据 (FiberChannel 期望时域输入)
    pulse_t_x = pulse_x.to_time(ctx)[0]  # [0] to get 1D array from (1, n)
    pulse_t_y = pulse_y.to_time(ctx)[0]
    combined_data = np.vstack([pulse_t_x, pulse_t_y])

    # 获取初始时间网格和脉冲宽度
    initial_power = np.abs(pulse_t_x) ** 2
    t_grid = pulse_x.grid.t_grid
    initial_fwhm = calculate_fwhm(t_grid, initial_power)
    print(f"初始脉冲 FWHM: {initial_fwhm * 1e12:.3f} ps")

    # 信号参数
    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # 光纤参数 - SMF 典型值
    beta2 = -20.0  # ps^2/km (正常色散)
    beta3 = 0.0  # 无三阶色散
    length = 80e3  # 80 km

    print(f"\n光纤色散: β2={beta2:.1f} ps²/km, β3={beta3:.1f} ps³/km, L={length / 1e3:.0f} km")

    # 创建光纤 (抑制非线性)
    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=0.0,  # 无损耗以便精确测量展宽
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
        coarse_step_size_m=500.0,
        n2_si=0.0,  # 抑制非线性
        compute_ctx_name="cupy",
    )

    # 传播
    output = fiber.forward(signal)
    # FiberChannel 输出时域数据，直接使用
    output_power = np.abs(output.data[0]) ** 2
    output_fwhm = calculate_fwhm(t_grid, output_power)

    print(f"\n光纤传播后 FWHM: {output_fwhm * 1e12:.3f} ps")
    broadening_ratio = output_fwhm / initial_fwhm
    print(f"展宽比: {broadening_ratio:.3f}x")

    # 理论展宽比
    L_km = length / 1e3
    tau_in_ps = initial_fwhm * 1e12
    ratio = beta2 * L_km / tau_in_ps**2
    theoretical_broadening = np.sqrt(1 + (4 * np.log(2) * ratio) ** 2)
    print(f"理论展宽比: {theoretical_broadening:.3f}x")

    # 应用 CDC 补偿
    cdc = ChromaticDispersionComp(
        fiber_length_km=length / 1e3,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
    )
    compensated = cdc.forward(output)
    # CDC 输出也是时域数据
    compensated_power = np.abs(compensated.data[0]) ** 2
    compensated_fwhm = calculate_fwhm(t_grid, compensated_power)

    print(f"\nCDC 补偿后 FWHM: {compensated_fwhm * 1e12:.3f} ps")
    recovery_ratio = compensated_fwhm / initial_fwhm
    print(f"恢复比 (补偿后/初始): {recovery_ratio:.3f}x")
    print(f"残余展宽: {(recovery_ratio - 1) * 100:.2f}%")

    # 验证：补偿后脉冲宽度应接近原始宽度 (允许 5% 误差)
    rel_error = abs(recovery_ratio - 1.0)
    assert rel_error < 0.05, f"β2 补偿误差 {rel_error * 100:.1f}% 超过 5%"
    print("\n[PASS] β2 补偿测试通过 - 脉冲宽度恢复")


def test_cdc_beta3_compensation():
    """测试三阶色散 (TOD/β3) 补偿效果。"""
    print("\n" + "=" * 60)
    print("测试 2: 三阶色散 (β3) 补偿效果")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**14
    center_freq = 193e12
    bandwidth = 5e12  # 5 THz 更大带宽以便观察 TOD 效应
    pulse_energy = 1e-12
    pulse_fwhm = 3e-12  # 3 ps 更短脉冲对 TOD 更敏感

    print(f"创建高斯脉冲: FWHM={pulse_fwhm * 1e12:.2f} ps, 带宽={bandwidth * 1e-12:.1f} THz")

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
    # 获取时间域数据
    pulse_t_x = pulse_x.to_time(ctx)[0]  # [0] to get 1D array from (1, n)
    pulse_t_y = pulse_y.to_time(ctx)[0]
    combined_data = np.vstack([pulse_t_x, pulse_t_y])

    initial_power = np.abs(pulse_t_x) ** 2
    t_grid = pulse_x.grid.t_grid
    initial_fwhm = calculate_fwhm(t_grid, initial_power)
    print(f"初始脉冲 FWHM: {initial_fwhm * 1e12:.3f} ps")

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # 光纤参数 - 主要 β3
    beta2 = 0.0  # 无二阶色散
    beta3 = 0.5e-3  # ps^3/km (典型 SMF 的 TOD)
    length = 100e3  # 100 km

    print(f"\n光纤色散: β2={beta2:.1f} ps²/km, β3={beta3 * 1e3:.2f} ps³/km, L={length / 1e3:.0f} km")

    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=0.0,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
        coarse_step_size_m=500.0,
        n2_si=0.0,
        compute_ctx_name="cupy",
    )

    output = fiber.forward(signal)
    # FiberChannel 输出时域数据，直接使用
    output_power = np.abs(output.data[0]) ** 2
    output_fwhm = calculate_fwhm(t_grid, output_power)

    print(f"\n光纤传播后 FWHM: {output_fwhm * 1e12:.3f} ps")
    broadening_ratio = output_fwhm / initial_fwhm
    print(f"展宽比: {broadening_ratio:.3f}x")

    # 应用 CDC 补偿
    cdc = ChromaticDispersionComp(
        fiber_length_km=length / 1e3,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
    )
    compensated = cdc.forward(output)
    # CDC 输出也是时域数据
    compensated_power = np.abs(compensated.data[0]) ** 2
    compensated_fwhm = calculate_fwhm(t_grid, compensated_power)

    print(f"\nCDC 补偿后 FWHM: {compensated_fwhm * 1e12:.3f} ps")
    recovery_ratio = compensated_fwhm / initial_fwhm
    print(f"恢复比 (补偿后/初始): {recovery_ratio:.3f}x")
    print(f"残余展宽: {(recovery_ratio - 1) * 100:.2f}%")

    # TOD 效应比 β2 更复杂，脉冲形状可能变形
    # 允许较大误差 (15%) 因为 TOD 补偿后脉冲形状可能有畸变
    rel_error = abs(recovery_ratio - 1.0)
    assert rel_error < 0.15, f"β3 补偿误差 {rel_error * 100:.1f}% 超过 15%"
    print("\n[PASS] β3 补偿测试通过 - TOD 效应被补偿")


def test_cdc_beta2_beta3_combined():
    """测试 β2 + β3 混合色散补偿效果。"""
    print("\n" + "=" * 60)
    print("测试 3: β2 + β3 混合色散补偿")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**14
    center_freq = 193e12
    bandwidth = 3e12
    pulse_energy = 1e-12
    pulse_fwhm = 4e-12  # 4 ps

    print(f"创建高斯脉冲: FWHM={pulse_fwhm * 1e12:.2f} ps")

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
    # 获取时间域数据
    pulse_t_x = pulse_x.to_time(ctx)[0]  # [0] to get 1D array from (1, n)
    pulse_t_y = pulse_y.to_time(ctx)[0]
    combined_data = np.vstack([pulse_t_x, pulse_t_y])

    initial_power = np.abs(pulse_t_x) ** 2
    t_grid = pulse_x.grid.t_grid
    initial_fwhm = calculate_fwhm(t_grid, initial_power)
    print(f"初始脉冲 FWHM: {initial_fwhm * 1e12:.3f} ps")

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    # SMF 典型色散值
    beta2 = -20.0  # ps^2/km
    beta3 = 0.1e-3  # ps^3/km
    length = 80e3  # 80 km

    print(f"\n光纤色散: β2={beta2:.1f} ps²/km, β3={beta3 * 1e3:.2f} ps³/km, L={length / 1e3:.0f} km")

    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=0.0,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
        coarse_step_size_m=500.0,
        n2_si=0.0,
        compute_ctx_name="cupy",
    )

    output = fiber.forward(signal)
    # FiberChannel 输出时域数据，直接使用
    output_power = np.abs(output.data[0]) ** 2
    output_fwhm = calculate_fwhm(t_grid, output_power)

    print(f"\n光纤传播后 FWHM: {output_fwhm * 1e12:.3f} ps")
    broadening_ratio = output_fwhm / initial_fwhm
    print(f"展宽比: {broadening_ratio:.3f}x")

    # CDC 补偿
    cdc = ChromaticDispersionComp(
        fiber_length_km=length / 1e3,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
    )
    compensated = cdc.forward(output)
    # CDC 输出也是时域数据
    compensated_power = np.abs(compensated.data[0]) ** 2
    compensated_fwhm = calculate_fwhm(t_grid, compensated_power)

    print(f"\nCDC 补偿后 FWHM: {compensated_fwhm * 1e12:.3f} ps")
    recovery_ratio = compensated_fwhm / initial_fwhm
    print(f"恢复比 (补偿后/初始): {recovery_ratio:.3f}x")
    print(f"残余展宽: {(recovery_ratio - 1) * 100:.2f}%")

    rel_error = abs(recovery_ratio - 1.0)
    assert rel_error < 0.05, f"β2+β3 补偿误差 {rel_error * 100:.1f}% 超过 5%"
    print("\n[PASS] β2+β3 混合色散补偿测试通过")


def test_cdc_metadata_extraction():
    """测试 CDC 自动从信号 metadata 提取色散参数。"""
    print("\n" + "=" * 60)
    print("测试 4: CDC 自动从 metadata 提取色散参数")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**13
    center_freq = 193e12
    bandwidth = 2e12
    pulse_energy = 1e-12
    pulse_fwhm = 5e-12

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
    # 获取时间域数据
    pulse_t_x = pulse_x.to_time(ctx)[0]  # [0] to get 1D array from (1, n)
    pulse_t_y = pulse_y.to_time(ctx)[0]
    combined_data = np.vstack([pulse_t_x, pulse_t_y])

    initial_power = np.abs(pulse_t_x) ** 2
    t_grid = pulse_x.grid.t_grid
    initial_fwhm = calculate_fwhm(t_grid, initial_power)
    print(f"初始脉冲 FWHM: {initial_fwhm * 1e12:.3f} ps")

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=2,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    signal = Signal(data=combined_data, ctx=signal_ctx)

    beta2 = -20.0
    beta3 = 0.0
    length = 80e3

    print(f"\n光纤色散: β2={beta2:.1f} ps²/km, L={length / 1e3:.0f} km")

    fiber = FiberChannel(
        length_m=length,
        alpha_db_per_km=0.0,
        beta2_ps2_per_km=beta2,
        beta3_ps3_per_km=beta3,
        coarse_step_size_m=500.0,
        n2_si=0.0,
        compute_ctx_name="cupy",
    )

    output = fiber.forward(signal)
    # FiberChannel 输出时域数据，直接使用
    output_power = np.abs(output.data[0]) ** 2

    print(f"\n光纤传播后峰值功率: {np.max(output_power):.6f}")
    print(f"初始峰值功率: {np.max(initial_power):.6f}")

    # 验证 metadata 包含色散参数
    print("\n信号 metadata 检查:")
    print(f"  fiber_length_m: {output.ctx.meta.get('fiber_length_m', 'N/A')}")
    print(f"  fiber_beta2_ps2_per_km: {output.ctx.meta.get('fiber_beta2_ps2_per_km', 'N/A')}")
    print(f"  fiber_beta3_ps3_per_km: {output.ctx.meta.get('fiber_beta3_ps3_per_km', 'N/A')}")

    # 创建 CDC 但不指定参数 - 应从 metadata 自动获取
    cdc = ChromaticDispersionComp()  # 不传参数
    compensated = cdc.forward(output)
    # CDC 输出也是时域数据
    compensated_power = np.abs(compensated.data[0]) ** 2
    compensated_fwhm = calculate_fwhm(t_grid, compensated_power)

    # 使用 FWHM 验证补偿效果
    recovery_ratio = compensated_fwhm / initial_fwhm
    print(f"\nCDC 补偿后 FWHM: {compensated_fwhm * 1e12:.3f} ps")
    print(f"恢复比 (补偿后/初始): {recovery_ratio:.3f}x")

    # 验证补偿后的元数据
    print("\n补偿后 metadata 检查:")
    print(f"  cdc_compensated_length_km: {compensated.ctx.meta.get('cdc_compensated_length_km', 'N/A')}")
    print(f"  cdc_beta2_ps2_per_km: {compensated.ctx.meta.get('cdc_beta2_ps2_per_km', 'N/A')}")
    print(f"  cdc_beta3_ps3_per_km: {compensated.ctx.meta.get('cdc_beta3_ps3_per_km', 'N/A')}")

    rel_error = abs(recovery_ratio - 1.0)
    assert rel_error < 0.05, f"自动提取参数补偿误差 {rel_error * 100:.1f}% 超过 5%"
    print("\n[PASS] metadata 自动提取测试通过")


def test_cdc_zero_dispersion():
    """测试零色散情况 - CDC 应抛出异常。"""
    print("\n" + "=" * 60)
    print("测试 5: 零色散情况处理")
    print("=" * 60)

    ctx = get_numpy_context()

    n_samples = 2**12
    center_freq = 193e12
    bandwidth = 100e9
    pulse_energy = 1e-12
    pulse_fwhm = 5e-12

    pulse_x = create_gaussian(
        n=n_samples,
        v_center=center_freq,
        bandwidth=bandwidth,
        v0=center_freq,
        e_p=pulse_energy,
        t_fwhm=pulse_fwhm,
        ctx=ctx,
    )

    signal_ctx = SignalContext(
        sample_rate=bandwidth,
        center_freq=center_freq,
        n_pol=1,
        sps=1,
        sym_rate=bandwidth,
        bits_per_symbol=1,
    )
    # 使用时间域数据
    pulse_t = pulse_x.to_time(ctx)
    signal = Signal(data=pulse_t.reshape(1, -1), ctx=signal_ctx)

    # 尝试补偿零色散
    cdc = ChromaticDispersionComp(
        fiber_length_km=80e3,
        beta2_ps2_per_km=0.0,
        beta3_ps3_per_km=0.0,
    )

    try:
        cdc.forward(signal)  # 应抛出 ValueError
        print("[FAIL] 应抛出 ValueError")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        print(f"正确抛出 ValueError: {e}")
        print("\n[PASS] 零色散情况测试通过")


def run_all_tests():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("ChromaticDispersionComp 色散补偿验证测试套件")
    print("=" * 60)

    try:
        test_cdc_beta2_compensation()
        test_cdc_beta3_compensation()
        test_cdc_beta2_beta3_combined()
        test_cdc_metadata_extraction()
        test_cdc_zero_dispersion()

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
