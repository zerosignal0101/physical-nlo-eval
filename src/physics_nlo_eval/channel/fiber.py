"""光纤信道模拟，集成 PyNLO。"""

from typing import Any

import numpy as np
from pynlo.backends import get_cupy_context, get_numpy_context
from pynlo.physics.equations.manakov import ManakovEquation
from pynlo.physics.field import FieldState
from pynlo.physics.grid import Grid
from pynlo.physics.solvers import Propagator
from pynlo.physics.solvers.adaptive import AdaptiveController
from pynlo.physics.solvers.rk4ip import RK4IPStepper
from scipy.constants import c

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class FiberChannel(BaseBlock):
    """光纤信道模拟模块，封装 PyNLO 的 Manakov 方程求解。

    模拟光纤中的色散 (CD)、偏振模色散 (PMD)、损耗 (Loss)、噪声 (Noise)
    和非线性效应 (Nonlinearity)。

    Parameters
    ----------
    length_m : float
        光纤总长度 (米)。
    alpha_db_per_km : float | np.ndarray
        光纤损耗 (dB/km)。支持标量或数组 (沿光纤长度变化)。
    beta2_ps2_per_m : float
        二阶色散系数 (ps^2/km)。
    beta3_ps3_per_m : float
        三阶色散系数 (ps^3/km)。
    v_comoving : float | None, optional
        共速参考频率 (Hz)。为 None 时使用频率网格中心频率。
    d_pmd_ps_per_sqrt_km : float, optional
        PMD 系数 (ps/sqrt(km))。如果为 0，则不模拟 PMD。
    step_size_m : float, optional
        PyNLO 求解器的步长 (米)。
    n2_si : float, optional
        光纤非线性折射率 n2 (m^2/W)。如果 gamma 已给出，则此参数可忽略。
    a_eff_um2 : float, optional
        光纤有效面积 (um^2)。如果 gamma 已给出，则此参数可忽略。
    slope_a_eff_um2_per_nm : float, optional
        有效面积随波长的斜率 (μm^2/nm)。
    ref_lambda_nm : float, optional
        参考波长 (nm)。
    raman_r3 : np.ndarray | None, optional
        拉曼响应函数，如果为 None 则不模拟拉曼效应。
    """

    def __init__(
        self,
        length_m: float,
        alpha_db_per_km: float | np.ndarray,
        beta2_ps2_per_m: float,
        beta3_ps3_per_m: float,
        v_comoving: float | None = None,
        d_pmd_ps_per_sqrt_km: float = 0.0,
        step_size_m: float = 1000.0,
        n2_si: float = 2.6e-20,  # Fused silica typical n2
        a_eff_um2: float = 80.0,  # SMF typical A_eff
        slope_a_eff_um2_per_nm: float = 0.05,  # SMF typical A_eff
        ref_lambda_nm: float = 1550.0,  # SMF typical reference wavelength
        raman_r3: np.ndarray | None = None,
        compute_ctx_name: str = "numpy",
    ) -> None:
        """初始化 FiberChannel。"""
        super().__init__(name="FiberChannel")
        if not length_m > 0:
            raise ValueError("Fiber length must be positive.")
        if not step_size_m > 0:
            raise ValueError("Step size must be positive.")

        self.length_m = length_m
        self.alpha_db_per_km = alpha_db_per_km
        self.beta2_ps2_per_m = beta2_ps2_per_m
        self.beta3_ps3_per_m = beta3_ps3_per_m
        self.v_comoving = v_comoving
        self.d_pmd_ps_per_sqrt_km = d_pmd_ps_per_sqrt_km
        self.step_size_m = step_size_m
        self.n2_si = n2_si
        self.a_eff_um2 = a_eff_um2
        self.slope_a_eff_um2_per_nm = slope_a_eff_um2_per_nm
        self.ref_lambda_nm = ref_lambda_nm
        self.raman_r3 = raman_r3
        if compute_ctx_name == "numpy":
            self.compute_ctx = get_numpy_context()
        elif compute_ctx_name == "cupy":
            self.compute_ctx = get_cupy_context()
        else:
            raise ValueError("ComputeCtx name must be 'numpy' or 'cupy'.")

        # 将单位转换为 SI 标准以匹配 PyNLO
        self.alpha_np_inv_m = self.alpha_db_per_km * 1e-3 * np.log(10) / 10  # dB/km to Np/m
        self.beta2_si = self.beta2_ps2_per_m * 1e-24  # ps^2/m to s^2/m
        self.beta3_si = self.beta3_ps3_per_m * 1e-36  # ps^3/m to s^3/m
        self.a_eff_si = self.a_eff_um2 * 1e-12  # μm^2 to m^2
        self.slope_a_eff_si = self.slope_a_eff_um2_per_nm * 1e-3  # μm^2/nm to dimensionless
        self.ref_lambda_si = self.ref_lambda_nm * 1e-9  # nm to m

        # PMD (使用 2km 固定值)，确保 z_grid 至少有两个点
        n_z_points = max(2, int(self.length_m / 2000.0))
        self.z_grid = np.linspace(0, self.length_m, n_z_points)
        self.delta_tau_si = self.d_pmd_ps_per_sqrt_km * self.compute_ctx.array.sqrt(2000.0 / 1e3) * 1e-12  # s

    def _setup_pynlo_environment(self, signal: Signal) -> tuple[Any, Any, Any]:
        """根据输入信号设置 PyNLO 的 Grid 和 ManakovEquation。"""
        num_samples = signal.num_samples
        v_center = signal.ctx.center_freq  # Hz
        bandwidth = signal.ctx.sample_rate  # Hz, Nyquist bandwidth

        # PyNLO grid uses frequency (v)
        # Assuming signal.ctx.sample_rate is the full bandwidth for FFT
        grid = Grid.create(n=num_samples, v_center=v_center, bandwidth=bandwidth, v_comoving=self.v_comoving)

        # Calculate beta_comoving (dispersion coefficients)
        # PyNLO expects beta_comoving matching grid.v_grid
        omega_relative = 2 * np.pi * (grid.v_grid - grid.v_comoving_exact)  # Relative angular frequency
        beta_comoving_val = self.beta2_si / 2 * omega_relative**2 + self.beta3_si / 6 * omega_relative**3

        # Calculate gamma_val
        lambdas_grid = c / grid.v_grid
        a_eff_val = self.a_eff_si + self.slope_a_eff_si * (lambdas_grid - self.ref_lambda_si)
        gamma_val = (2 * np.pi * grid.v_grid * self.n2_si) / (c * a_eff_val)

        # Compute DGD per scattering step
        if self.delta_tau_si != 0.0 and signal.ctx.n_pol == 2:
            omega_fft_order = 2 * np.pi * np.fft.fftfreq(num_samples, d=grid.dt)
            omega_gpu = self.compute_ctx.array.asarray(omega_fft_order)  # Angular frequency in FFT order
            self.pmd_dgd_op_x = self.compute_ctx.array.exp(-1j * omega_gpu * (self.delta_tau_si / 2))
            self.pmd_dgd_op_y = self.compute_ctx.array.exp(1j * omega_gpu * (self.delta_tau_si / 2))
        else:
            self.pmd_dgd_op_x = None
            self.pmd_dgd_op_y = None

        equation = ManakovEquation.from_fiber(
            grid=grid,
            beta_comoving=beta_comoving_val,
            gamma=gamma_val,
            alpha_comoving=self.alpha_np_inv_m,
            r3=self.raman_r3,
            manakov_factor=8 / 9,  # Default for Manakov
            ctx=self.compute_ctx,
        )

        initial_state = FieldState(grid=grid, a_v=signal.data, n_comp=signal.ctx.n_pol)

        return grid, equation, initial_state

    def generate_random_unitary(self) -> np.ndarray:
        """Generate random 2x2 unitary matrix for PMD."""
        theta = np.arcsin(np.sqrt(np.random.rand()))
        phi = 2 * np.pi * np.random.rand()
        psi = 2 * np.pi * np.random.rand()

        r00 = np.cos(theta) * np.exp(1j * phi)
        r01 = np.sin(theta) * np.exp(1j * psi)
        r10 = -np.sin(theta) * np.exp(-1j * psi)
        r11 = np.cos(theta) * np.exp(-1j * phi)

        return self.compute_ctx.array.asarray([[r00, r01], [r10, r11]])

    def apply_pmd_scat(self, z: float, state: FieldState) -> None:
        """Apply one PMD scattering event.

        Parameters
        ----------
        z : float
            Current propagation distance (m).
        state : FieldState
            Current field state.
        """
        # Random unitary rotation
        R = self.generate_random_unitary()

        # Apply rotation: A' = R @ A
        A_omega_vec = state.a_v
        A_rot = self.compute_ctx.array.zeros_like(A_omega_vec)
        A_rot[0] = R[0, 0] * A_omega_vec[0] + R[0, 1] * A_omega_vec[1]
        A_rot[1] = R[1, 0] * A_omega_vec[0] + R[1, 1] * A_omega_vec[1]

        # Apply DGD
        Ax_out = A_rot[0] * self.pmd_dgd_op_x
        Ay_out = A_rot[1] * self.pmd_dgd_op_y

        state.a_v = self.compute_ctx.array.stack([Ax_out, Ay_out])

    def forward(self, opt_signal: Signal) -> Signal:
        """
        通过光纤信道传播光信号。

        Parameters
        ----------
        opt_signal : Signal
            输入的初始光信号。数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            经过光纤传播后的输出光信号。
        """
        # Ensure signal is complex and has polarization components
        if not np.iscomplexobj(opt_signal.data):
            raise ValueError("Input optical signal must be complex.")
        if opt_signal.data.ndim != 2 or opt_signal.data.shape[0] != opt_signal.ctx.n_pol:
            raise ValueError("Input optical signal data must be 2D (n_pol, N_samples).")

        # 1. 设置 PyNLO 环境
        grid, equation, initial_field_state = self._setup_pynlo_environment(opt_signal)

        # 2. 初始化求解器
        stepper = RK4IPStepper(equation)
        controller = AdaptiveController(tol=1e-6)
        recorder = None  # None if not recording intermediate steps
        propagator = Propagator(enable_progress=False)

        # 3. 传播循环
        if self.pmd_dgd_op_x is not None and self.pmd_dgd_op_y is not None:
            callbacks = [self.apply_pmd_scat]
        else:
            callbacks = None
        final_field_state = propagator.run(
            initial_state=initial_field_state,
            z_grid=self.z_grid,
            equation=equation,
            stepper=stepper,
            controller=controller,
            recorder=recorder,
            callbacks=callbacks,
            ctx=self.compute_ctx,
        )

        # 4. 获取最终场状态并转换为 Signal 对象
        final_data = final_field_state.a_v[-1]  # Extract freq domain field

        # Update SignalContext with channel effects (e.g., accumulated CD/PMD for tracking)
        new_ctx = SignalContext(
            sample_rate=opt_signal.ctx.sample_rate,
            center_freq=opt_signal.ctx.center_freq,
            n_pol=opt_signal.ctx.n_pol,
            sps=opt_signal.ctx.sps,
            sym_rate=opt_signal.ctx.sym_rate,
            bits_per_symbol=opt_signal.ctx.bits_per_symbol,
            is_ofdm=opt_signal.ctx.is_ofdm,
            meta={
                **(opt_signal.ctx.meta or {}),
                "fiber_length_m": self.length_m,
                "fiber_alpha_db_per_km": self.alpha_db_per_km,
                "fiber_beta2_ps2_per_m": self.beta2_ps2_per_m,
                "fiber_beta3_ps3_per_m": self.beta3_ps3_per_m,
                "fiber_d_pmd_ps_per_sqrt_km": self.d_pmd_ps_per_sqrt_km,
                "fiber_a_eff_um2": self.a_eff_um2,
                "fiber_slope_a_eff_um2_per_nm": self.slope_a_eff_um2_per_nm,
            },
        )
        return Signal(data=final_data.astype(np.complex64), ctx=new_ctx)


class EDFA(BaseBlock):
    """掺铒光纤放大器 (EDFA) 模块。"""

    def __init__(self, gain_db: float, nf_db: float) -> None:
        """
        初始化 EDFA。

        Parameters
        ----------
        gain_db : float
            放大器的增益 (dB)。
        nf_db : float
            放大器的噪声系数 (dB)。
        """
        super().__init__(name=f"EDFA(Gain={gain_db}dB, NF={nf_db}dB)")
        self.gain_db = gain_db
        self.nf_db = nf_db
        self.gain_linear = 10 ** (self.gain_db / 10)
        self.nf_linear = 10 ** (self.nf_db / 10)

    def forward(self, opt_signal: Signal) -> Signal:
        """
        放大光信号并添加 ASE 噪声。

        Parameters
        ----------
        opt_signal : Signal
            输入的衰减光信号。

        Returns
        -------
        Signal
            输出的放大光信号，包含 ASE 噪声。
        """
        if not np.iscomplexobj(opt_signal.data):
            raise ValueError("Input optical signal must be complex.")

        # 放大信号
        amplified_signal_data = opt_signal.data * np.sqrt(self.gain_linear)

        # 计算 ASE 噪声功率 (简化模型)
        # P_ASE = G * h * nu * B_opt * N_sp * 2 (for two polarizations)
        # G = self.gain_linear
        # h = 6.626e-34 J.s (Planck's constant)
        # c = 2.9979e8 m/s (Speed of light)
        # nu = opt_signal.ctx.center_freq (optical frequency)
        # B_opt = optical bandwidth (typically electrical_bandwidth * 2 or more)
        # N_sp = F_n * G / (G - 1) * F_n_prime, where F_n is NF_linear (simplified N_sp ~ NF_linear)
        # For simplicity, assume B_opt is related to sample_rate

        # Using a simplified formula for ASE variance directly based on NF and gain
        # N_ase = F_n * h * nu * B_e * G (single polarization, electrical bandwidth B_e)
        # For complex noise, we need variance for real and imag parts: var = N_ase / 2

        h = 6.62607015e-34  # Planck's constant (J·s)
        nu = opt_signal.ctx.center_freq  # Optical frequency (Hz)

        # Noise bandwidth for ASE, typically related to electrical bandwidth (sample_rate)
        # Or a specific optical filter bandwidth. Let's use sample_rate for now.
        noise_bandwidth = opt_signal.ctx.sample_rate  # Hz (electrical bandwidth)

        # For our scale (signal power in mW), let's estimate noise power in mW.
        noise_variance_per_sample = (
            0.5 * self.nf_linear * self.gain_linear * h * nu * noise_bandwidth * 1e3
        )  # Convert to mW unit
        # Divide by 2 because complex noise has real and imaginary parts
        noise_std_dev = np.sqrt(noise_variance_per_sample / 2)

        # Generate complex Gaussian noise
        ase_noise = (
            np.random.normal(0, noise_std_dev, amplified_signal_data.shape)
            + 1j * np.random.normal(0, noise_std_dev, amplified_signal_data.shape)
        ).astype(np.complex64)

        output_signal_data = amplified_signal_data + ase_noise

        new_ctx = SignalContext(
            sample_rate=opt_signal.ctx.sample_rate,
            center_freq=opt_signal.ctx.center_freq,
            n_pol=opt_signal.ctx.n_pol,
            sps=opt_signal.ctx.sps,
            sym_rate=opt_signal.ctx.sym_rate,
            bits_per_symbol=opt_signal.ctx.bits_per_symbol,
            is_ofdm=opt_signal.ctx.is_ofdm,
            meta={
                **(opt_signal.ctx.meta or {}),
                "edfa_gain_db": self.gain_db,
                "edfa_nf_db": self.nf_db,
            },
        )
        return Signal(data=output_signal_data, ctx=new_ctx)
