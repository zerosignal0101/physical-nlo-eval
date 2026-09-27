"""光纤信道模拟，集成 PyNLO。"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    from pynlo.backends import get_cupy_context, get_numpy_context
    from pynlo.physics.equations.manakov import ManakovEquation
    from pynlo.physics.field import FieldState
    from pynlo.physics.grid import Grid
    from pynlo.physics.solvers import Propagator
    from pynlo.physics.solvers.adaptive import AdaptiveController
    from pynlo.physics.solvers.rk4ip import RK4IPStepper
except ImportError:  # PyNLO 为可选依赖；缺失时回退到自带的 NumPy SSFM 求解器。
    get_cupy_context = get_numpy_context = None
    ManakovEquation = FieldState = Grid = Propagator = AdaptiveController = RK4IPStepper = None

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
    beta2_ps2_per_km : float
        二阶色散系数 (ps^2/km)。
    beta3_ps3_per_km : float
        三阶色散系数 (ps^3/km)。
    v_comoving : float | None, optional
        共速参考频率 (Hz)。为 None 时使用频率网格中心频率。
    d_pmd_ps_per_sqrt_km : float, optional
        PMD 系数 (ps/sqrt(km))。如果为 0，则不模拟 PMD。
    coarse_step_size_m : float, optional
        PyNLO 粗步长 (米)。
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
        beta2_ps2_per_km: float,
        beta3_ps3_per_km: float,
        v_comoving: float | None = None,
        d_pmd_ps_per_sqrt_km: float = 0.0,
        coarse_step_size_m: float = 1000.0,
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
        if not coarse_step_size_m > 0:
            raise ValueError("Step size must be positive.")

        self.length_m = length_m
        self.alpha_db_per_km = alpha_db_per_km
        self.beta2_ps2_per_km = beta2_ps2_per_km
        self.beta3_ps3_per_km = beta3_ps3_per_km
        self.v_comoving = v_comoving
        self.d_pmd_ps_per_sqrt_km = d_pmd_ps_per_sqrt_km
        self.coarse_step_size_m = coarse_step_size_m
        self.n2_si = n2_si
        self.a_eff_um2 = a_eff_um2
        self.slope_a_eff_um2_per_nm = slope_a_eff_um2_per_nm
        self.ref_lambda_nm = ref_lambda_nm
        self.raman_r3 = raman_r3
        # compute_ctx 允许为 None：PyNLO 不可用时退化为自带的 NumPy SSFM 求解器。
        if compute_ctx_name == "numpy" and get_numpy_context is not None:
            self.compute_ctx = get_numpy_context()
        elif compute_ctx_name == "cupy" and get_cupy_context is not None:
            self.compute_ctx = get_cupy_context()
        elif compute_ctx_name == "numpy":
            self.compute_ctx = None
        elif compute_ctx_name == "cupy":
            warnings.warn(
                "PyNLO/CuPy is unavailable; falling back to the NumPy split-step solver.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.compute_ctx = None
        else:
            raise ValueError("ComputeCtx name must be 'numpy' or 'cupy'.")

        # 将单位转换为 SI 标准以匹配 PyNLO
        self.alpha_np_inv_m = self.alpha_db_per_km * 1e-3 * np.log(10) / 10  # dB/km to Np/m
        self.beta2_si = self.beta2_ps2_per_km * 1e-27  # ps^2/km to s^2/m
        self.beta3_si = self.beta3_ps3_per_km * 1e-39  # ps^3/km to s^3/m
        self.a_eff_si = self.a_eff_um2 * 1e-12  # μm^2 to m^2
        self.slope_a_eff_si = self.slope_a_eff_um2_per_nm * 1e-3  # μm^2/nm to dimensionless
        self.ref_lambda_si = self.ref_lambda_nm * 1e-9  # nm to m

        # PMD (使用 coarse_step_size_m 作为粗步长大小)，确保 z_grid 至少有两个点
        n_z_points = max(2, int(self.length_m / self.coarse_step_size_m))
        self.z_grid = np.linspace(0, self.length_m, n_z_points)
        # 用宿主 np.sqrt：compute_ctx 可能为 None（无 PyNLO 回退路径）。
        self.delta_tau_si = (
            self.d_pmd_ps_per_sqrt_km * np.sqrt(self.coarse_step_size_m / 1e3) * 1e-12
        )  # s

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

        # 转换到频率域单调序，构建 initial_state
        data_a_v_monotonic = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(signal.data), axis=-1)) * grid.dt
        if signal.ctx.n_pol == 1:
            data_a_v_monotonic = self.compute_ctx.array.stack(
                [
                    self.compute_ctx.array.asarray(data_a_v_monotonic[0]),
                    self.compute_ctx.array.zeros_like(data_a_v_monotonic[0]),
                ]
            )
        else:
            data_a_v_monotonic = self.compute_ctx.array.asarray(data_a_v_monotonic)
        initial_state = FieldState(grid=grid, a_v=data_a_v_monotonic, n_comp=2)

        return grid, equation, initial_state

    def generate_random_unitary(self):
        """Generate random 2x2 unitary matrix for PMD."""
        theta = np.arcsin(np.sqrt(np.random.rand()))
        phi = 2 * np.pi * np.random.rand()
        psi = 2 * np.pi * np.random.rand()

        r00 = np.cos(theta) * np.exp(1j * phi)
        r01 = np.sin(theta) * np.exp(1j * psi)
        r10 = -np.sin(theta) * np.exp(-1j * psi)
        r11 = np.cos(theta) * np.exp(-1j * phi)

        unitary = np.array([[r00, r01], [r10, r11]], dtype=np.complex128)
        return self.compute_ctx.array.asarray(unitary)

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

        # 无 PyNLO 上下文时走自带的 NumPy 分步傅里叶求解器。
        if self.compute_ctx is None:
            return self._forward_numpy(opt_signal)

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
        final_simulation_result = propagator.run(
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
        final_data = final_simulation_result.a_t[-1]  # Extract time domain field

        # Update SignalContext with channel effects (e.g., accumulated CD/PMD for tracking)
        new_ctx = SignalContext(
            sample_rate=opt_signal.ctx.sample_rate,
            center_freq=opt_signal.ctx.center_freq,
            n_pol=2,  # Manakov 方程输出为 X Y 偏振
            sps=opt_signal.ctx.sps,
            sym_rate=opt_signal.ctx.sym_rate,
            bits_per_symbol=opt_signal.ctx.bits_per_symbol,
            is_ofdm=opt_signal.ctx.is_ofdm,
            meta={
                **(opt_signal.ctx.meta or {}),
                "fiber_length_m": self.length_m,
                "fiber_alpha_db_per_km": self.alpha_db_per_km,
                "fiber_beta2_ps2_per_km": self.beta2_ps2_per_km,
                "fiber_beta3_ps3_per_km": self.beta3_ps3_per_km,
                "fiber_d_pmd_ps_per_sqrt_km": self.d_pmd_ps_per_sqrt_km,
                "fiber_a_eff_um2": self.a_eff_um2,
                "fiber_slope_a_eff_um2_per_nm": self.slope_a_eff_um2_per_nm,
            },
        )
        return Signal(data=final_data.astype(np.complex64), ctx=new_ctx)

    def _forward_numpy(self, opt_signal: Signal) -> Signal:
        """用自带的对称分步傅里叶 (SSFM) 求解器传播信号。

        这是 PyNLO/CuPy 不可用时的回退路径，输出元数据中带
        ``fiber_solver="numpy-ssfm"`` 以便下游识别。

        Notes
        -----
        与 PyNLO 路径存在两处已知的模型差异，切换求解器会带来数值差异：

        1. 非线性系数按中心频率取标量 γ，而 PyNLO 路径用逐频点 γ(ω)
           （含 ``slope_a_eff_um2_per_nm`` 的有效面积色散项），因此本路径
           不使用 ``slope_a_eff_um2_per_nm``；
        2. PMD 采用每步独立抽取的随机幺正旋转 + DGD 相移近似，而非
           PyNLO 的散射算子回调。
        """
        data = opt_signal.data.astype(np.complex128, copy=True)
        original_n_pol = data.shape[0]
        if original_n_pol == 1:
            # 单偏振输入补零为双偏振，与 Manakov 求解器的维度约定一致。
            data = np.vstack((data, np.zeros_like(data)))
        n_samples = data.shape[-1]
        n_steps = max(1, int(np.ceil(self.length_m / self.coarse_step_size_m)))
        dz = self.length_m / n_steps
        # 频率轴必须与 PyNLO 的网格约定一致（inclusive span，间隔 fs/(n-1)），
        # 否则本回退路径产出的信号与 ChromaticDispersionComp 不匹配——那是个
        # 正比于 ω²·L 的相位误差，n 小时足以让色散补偿完全失效。详见
        # dsp/time_domain.py 中 ChromaticDispersionComp 的说明。
        freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
        omega = 2 * np.pi * np.fft.fftfreq(n_samples, d=1 / opt_signal.ctx.sample_rate) * freq_scale
        beta = self.beta2_si * omega**2 / 2 + self.beta3_si * omega**3 / 6
        alpha = np.asarray(self.alpha_np_inv_m)
        if alpha.ndim and alpha.size not in {1, n_samples}:
            raise ValueError("Frequency-dependent alpha must have one value per signal sample.")
        # 色散项的符号必须与 PyNLO 一致：实测 PyNLO 施加的相位为
        # +|β2|·ω²·L/2（β2<0），而 beta = β2_si·ω²/2 本身为负，故取 -1j*beta。
        # 写成 +1j*beta 会传播出反向色散，既与 PyNLO 不符，也无法被 CDC 补偿。
        half_linear = np.exp((-1j * beta - alpha / 2) * dz / 2)
        center_gamma = 2 * np.pi * opt_signal.ctx.center_freq * self.n2_si / (c * self.a_eff_si)
        if np.all(alpha == 0):
            nonlinear_length = dz
        else:
            alpha_mean = float(np.mean(alpha))
            nonlinear_length = -np.expm1(-alpha_mean * dz) / alpha_mean

        if self.raman_r3 is not None:
            raman = np.asarray(self.raman_r3)
            if raman.shape[-1] != n_samples:
                warnings.warn(
                    "Raman response length does not match the grid; instantaneous Kerr response used.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                raman = None
        else:
            raman = None

        rng = np.random.default_rng()
        for _ in range(n_steps):
            data = np.fft.ifft(np.fft.fft(data, axis=-1) * half_linear, axis=-1)
            power = np.sum(np.abs(data) ** 2, axis=0)
            if raman is not None:
                power = np.real(np.fft.ifft(np.fft.fft(power) * np.fft.fft(raman))) / n_samples
            data *= np.exp(1j * (8 / 9) * center_gamma * nonlinear_length * power)[None, :]
            if self.d_pmd_ps_per_sqrt_km > 0:
                theta = np.arcsin(np.sqrt(rng.random()))
                phi, psi = 2 * np.pi * rng.random(2)
                rotation = np.array(
                    [
                        [np.cos(theta) * np.exp(1j * phi), np.sin(theta) * np.exp(1j * psi)],
                        [-np.sin(theta) * np.exp(-1j * psi), np.cos(theta) * np.exp(-1j * phi)],
                    ]
                )
                spectrum = rotation @ np.fft.fft(data, axis=-1)
                dgd = self.d_pmd_ps_per_sqrt_km * np.sqrt(dz / 1e3) * 1e-12
                spectrum[0] *= np.exp(-0.5j * omega * dgd)
                spectrum[1] *= np.exp(0.5j * omega * dgd)
                data = np.fft.ifft(spectrum, axis=-1)
            data = np.fft.ifft(np.fft.fft(data, axis=-1) * half_linear, axis=-1)

        ctx = opt_signal.ctx.copy()
        ctx.n_pol = 2
        ctx.meta.update(
            {
                "fiber_length_m": self.length_m,
                "fiber_alpha_db_per_km": self.alpha_db_per_km,
                "fiber_beta2_ps2_per_km": self.beta2_ps2_per_km,
                "fiber_beta3_ps3_per_km": self.beta3_ps3_per_km,
                "fiber_d_pmd_ps_per_sqrt_km": self.d_pmd_ps_per_sqrt_km,
                "fiber_a_eff_um2": self.a_eff_um2,
                # 与 PyNLO 路径保持键一致；注意本回退路径并不使用该斜率项。
                "fiber_slope_a_eff_um2_per_nm": self.slope_a_eff_um2_per_nm,
                "fiber_solver": "numpy-ssfm",
            }
        )
        return Signal(data.astype(np.complex64), ctx)


class EDFA(BaseBlock):
    """掺铒光纤放大器 (EDFA) 模块。

    ASE 噪声按量子极限模型 ``0.5 * NF * h * nu * (G - 1) * B`` 计算，
    因此单位增益 (0 dB) 时 ASE 功率严格为零。
    """

    def __init__(
        self,
        gain_db: float,
        nf_db: float,
        noise_bandwidth_hz: float | None = None,
        seed: int | None = None,
    ) -> None:
        """
        初始化 EDFA。

        Parameters
        ----------
        gain_db : float
            放大器的增益 (dB)。
        nf_db : float
            放大器的噪声系数 (dB)。
        noise_bandwidth_hz : float | None, optional
            等效光噪声带宽 (Hz)。为 None 时取信号的采样率作为复包络带宽。
        seed : int | None, optional
            指定后 ASE 噪声由该种子驱动的独立随机数发生器产生，结果可复现；
            为 None 时沿用全局 ``numpy.random`` 状态（便于配合
            ``np.random.seed`` 获得确定性结果）。
        """
        super().__init__(name=f"EDFA(Gain={gain_db}dB, NF={nf_db}dB)")
        self.gain_db = gain_db
        self.nf_db = nf_db
        if noise_bandwidth_hz is not None and noise_bandwidth_hz <= 0:
            raise ValueError("ASE noise bandwidth must be positive.")
        self.noise_bandwidth_hz = noise_bandwidth_hz
        self.gain_linear = 10 ** (self.gain_db / 10)
        self.nf_linear = 10 ** (self.nf_db / 10)
        self.rng = np.random.default_rng(seed) if seed is not None else None

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

        # 单个偏振的 ASE 光功率：P_ASE = 0.5 * NF * h * nu * (G - 1) * B，
        # 其中自发辐射因子 n_sp ~= NF/2。用 (G - 1) 而非 G 是关键：放大器不
        # 可能放大自身自发辐射之外的量子噪声，且单位增益时必须无 ASE 输出
        # （旧实现用 G，G=1 时会凭空注入噪声）。
        h = 6.62607015e-34  # Planck's constant (J·s)
        nu = opt_signal.ctx.center_freq  # Optical frequency (Hz)
        noise_bandwidth = self.noise_bandwidth_hz or opt_signal.ctx.sample_rate  # Hz

        ase_power_per_pol = (
            0.5 * self.nf_linear * h * nu * max(self.gain_linear - 1.0, 0.0) * noise_bandwidth
        )  # W
        # 复噪声的实部与虚部各自的标准差
        noise_std_dev = np.sqrt(ase_power_per_pol / 2)  # sqrt(W)

        # Generate complex Gaussian noise
        # 指定了 seed 就用独立 RNG（可复现）；否则沿用全局 numpy.random 状态。
        generator = self.rng if self.rng is not None else np.random
        ase_noise = (
            generator.normal(0, noise_std_dev, amplified_signal_data.shape)
            + 1j * generator.normal(0, noise_std_dev, amplified_signal_data.shape)
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
                "edfa_ase_bandwidth_hz": noise_bandwidth,
                "edfa_ase_power_per_pol_w": ase_power_per_pol,
            },
        )
        return Signal(data=output_signal_data, ctx=new_ctx)


class IdealAmplifier(BaseBlock):
    """理想光放大器 - 按入纤功率恢复信号电平并添加ASE噪声。

    自动从信号的 meta 中读取 IQModulator 存储的入纤功率 (tx_launch_power_dbm_per_pol)，
    将信号放大恢复到该功率电平。

    支持三种信号形态:
    - **宽带 WDM** (n_pol, N): 从 meta 读取 wdm_channels_info，频域逐信道增益
    - **堆叠 WDM** (n_ch, n_pol, N): 逐通道时域增益 (WDMDemux 输出)
    - **单通道** (n_pol, N): 整体时域增益

    与 ISRS-GN 模型 (network-nlo-eval) 保持同步，使用相同的 ASE 噪声计算公式：
    P_ASE = nf * h * nu * B * (G - 1)

    Parameters
    ----------
    nf_db : float
        放大器的噪声系数 (dB)。默认为 5.5 dB (C-band EDFA 典型值)。
    default_power_dbm : float | None
        默认入纤功率 (dBm)。当信号 meta 中没有存储入纤功率时使用。
        默认为 None，此时如果缺少入纤功率信息将报错。
    """

    def __init__(self, nf_db: float = 5.5, default_power_dbm: float | None = None) -> None:
        """初始化 IdealAmplifier。"""
        super().__init__(name=f"IdealAmplifier(NF={nf_db}dB)")
        self.nf_db = nf_db
        self.default_power_dbm = default_power_dbm
        self.nf_linear = 10 ** (nf_db / 10)

    def _get_channel_target_power_w(self, signal: Signal, ch_idx: int, is_stacked_wdm: bool) -> float:
        """从信号 meta 中提取各通道的目标功率 (入纤功率)。"""
        if is_stacked_wdm:
            wdm_info = signal.ctx.meta.get("wdm_channels_info", [])
            if wdm_info and len(wdm_info) > ch_idx:
                original_ctx = wdm_info[ch_idx].get("original_ctx")
                if original_ctx and original_ctx.meta:
                    launch_power_dbm = original_ctx.meta.get("tx_launch_power_dbm_per_pol")
                    if launch_power_dbm is not None:
                        return 2 * (1e-3 * 10 ** (launch_power_dbm / 10))

            launch_power_dbm = signal.ctx.meta.get("tx_launch_power_dbm_per_pol")
            if launch_power_dbm is not None:
                return 2 * (1e-3 * 10 ** (launch_power_dbm / 10))
        else:
            launch_power_dbm = signal.ctx.meta.get("tx_launch_power_dbm_per_pol")
            if launch_power_dbm is not None:
                return 2 * (1e-3 * 10 ** (launch_power_dbm / 10))

        if self.default_power_dbm is not None:
            return 2 * (1e-3 * 10 ** (self.default_power_dbm / 10))

        raise ValueError(
            "IdealAmplifier: 信号 meta 中未找到入纤功率 (tx_launch_power_dbm_per_pol)。"
            "请确保信号已经过 IQModulator 处理，或在 IdealAmplifier 构造函数中指定 default_power_dbm。"
        )

    def _get_channel_info(self, signal: Signal) -> tuple[int, bool, list[float], list[float]]:
        """从信号中提取各通道信息。"""
        is_stacked_wdm = signal.ctx.meta.get("is_stacked_wdm_signal", False)

        if is_stacked_wdm:
            n_channels = signal.data.shape[0]
            n_pol = signal.data.shape[1]

            wdm_info = signal.ctx.meta.get("wdm_channels_info", [])
            if wdm_info and len(wdm_info) == n_channels:
                reference_freq = signal.ctx.center_freq
                channel_freqs = []
                for ch_info in wdm_info:
                    freq_offset = ch_info.get("freq_offset_hz", 0.0)
                    channel_freqs.append(reference_freq + freq_offset)
            else:
                channel_bw = signal.ctx.meta.get("original_sample_rate_hz", signal.ctx.sample_rate)
                total_bw = n_channels * channel_bw
                start_freq = signal.ctx.center_freq - total_bw / 2
                channel_freqs = [start_freq + i * channel_bw for i in range(n_channels)]

            channel_powers = []
            for ch_idx in range(n_channels):
                ch_data = signal.data[ch_idx]
                power = np.mean(np.abs(ch_data) ** 2) * n_pol
                channel_powers.append(power)

            return n_channels, True, channel_freqs, channel_powers
        else:
            n_channels = 1
            n_pol = signal.data.shape[0]
            channel_freqs = [signal.ctx.center_freq]
            total_power = np.mean(np.abs(signal.data) ** 2) * n_pol
            channel_powers = [total_power]
            return n_channels, False, channel_freqs, channel_powers

    # ------------------------------------------------------------------
    # 宽带 WDM 频域放大 (避免 demux/remux 相位破坏)
    # ------------------------------------------------------------------

    def _measure_broadband_channel_powers(
        self, signal: Signal, wdm_info: list, rs_hz: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """FFT 测量宽带 WDM 中各通道功率 (频率域积分)。

        Returns
        -------
        X_shifted : (n_pol, N)  fftshift 后的频谱
        freqs : (N,) 相对频率 (Hz)
        channel_powers : (n_ch,) 各通道总功率 (W, 两偏振合计)
        """
        _, N = signal.data.shape
        sr = signal.ctx.sample_rate

        X_shifted = np.fft.fftshift(np.fft.fft(signal.data, axis=-1), axes=-1)
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / sr))

        half_bw = rs_hz / 2
        channel_powers = np.empty(len(wdm_info))
        for i, ch_info in enumerate(wdm_info):
            f_off = ch_info["freq_offset_hz"]
            mask = np.abs(freqs - f_off) <= half_bw
            # Parseval: P = sum_k |X[k]|^2 / N^2 (两偏振合计)
            channel_powers[i] = np.sum(np.abs(X_shifted[:, mask]) ** 2) / (N * N)

        return X_shifted, freqs, channel_powers

    def _build_smooth_gain_mask(
        self, freqs: np.ndarray, wdm_info: list, sqrt_gains: np.ndarray, rs_hz: float
    ) -> np.ndarray:
        """构造频域平滑增益掩码.

        信号带宽内 (含 RRC 滚降) 按最近信道指派增益,
        信道间重叠区高斯平滑, 带外增益=1.0 (不放大噪声).
        """
        N = len(freqs)
        roll_off = 0.1  # RRC 滚降系数, 与发射端匹配
        signal_half_bw = rs_hz / 2 * (1 + roll_off)  # 27.5 GHz

        gain_mask = np.ones(N, dtype=np.float64)
        ch_centers = np.array([ch["freq_offset_hz"] for ch in wdm_info])

        # 各信道信号带宽内设置增益
        for i, ch in enumerate(wdm_info):
            f_c = ch["freq_offset_hz"]
            band = np.abs(freqs - f_c) <= signal_half_bw
            gain_mask[band] = sqrt_gains[i]

        # 仅在 WDM 信号带宽内做平滑 (消除信道间突变, 不影响带外)
        min_f_sig = ch_centers[0] - signal_half_bw
        max_f_sig = ch_centers[-1] + signal_half_bw
        signal_band = (freqs >= min_f_sig) & (freqs <= max_f_sig)

        # 自适应平滑 sigma: 根据信道间最小间隔调整
        # 避免 sigma 过大使过渡区延伸入信道通带
        sigma_hz = 0.02 * rs_hz  # 默认: 1 GHz (50 GBaud)
        if len(ch_centers) > 1:
            gaps_hz = ch_centers[1:] - ch_centers[:-1]
            min_gap_between_bands_hz = min(gaps_hz) - 2.0 * signal_half_bw
            if min_gap_between_bands_hz > 0:
                sigma_hz = min(sigma_hz, min_gap_between_bands_hz / 4.0)
        sigma_bins = max(1.0, sigma_hz / (freqs[1] - freqs[0]))
        from scipy.ndimage import gaussian_filter1d

        if np.any(signal_band):
            idx_start = int(np.argmax(signal_band))
            idx_end = int(len(freqs) - np.argmax(signal_band[::-1]) - 1)
            band_gain = gain_mask[idx_start : idx_end + 1].copy()
            band_gain = gaussian_filter1d(band_gain, sigma=sigma_bins, mode="nearest")
            gain_mask[idx_start : idx_end + 1] = band_gain

        return gain_mask

    def _forward_wdm_broadband(self, opt_signal: Signal) -> Signal:
        """频域逐信道放大 + ASE 噪声 (宽带 WDM, 避免 demux/remux)."""
        n_pol, N = opt_signal.data.shape
        sr = opt_signal.ctx.sample_rate
        rs = opt_signal.ctx.sym_rate
        h = 6.62607015e-34
        wdm_info: list = opt_signal.ctx.meta.get("wdm_channels_info", [])
        n_ch = len(wdm_info)

        # 1. FFT 测量各通道功率
        X_shifted, freqs, channel_powers = self._measure_broadband_channel_powers(
            opt_signal, wdm_info, rs
        )

        # 2. 提取各通道目标功率 (从 wdm_channels_info 中 original_ctx.meta)
        channel_targets = np.empty(n_ch)
        for i, ch_info in enumerate(wdm_info):
            orig_ctx = ch_info.get("original_ctx")
            if orig_ctx and orig_ctx.meta:
                p_dbm = orig_ctx.meta.get("tx_launch_power_dbm_per_pol")
                if p_dbm is not None:
                    channel_targets[i] = 2 * 1e-3 * 10 ** (p_dbm / 10)
                    continue
            if self.default_power_dbm is not None:
                channel_targets[i] = 2 * 1e-3 * 10 ** (self.default_power_dbm / 10)
            else:
                raise ValueError(
                    f"IdealAmplifier: 通道 {i} 无 target power 信息。"
                )

        # 3. 计算增益
        gains_linear = np.where(channel_powers > 0, channel_targets / channel_powers, 1.0)

        # 4. 构建平滑增益掩码
        gain_mask = self._build_smooth_gain_mask(freqs, wdm_info, np.sqrt(gains_linear), rs)

        # 5. 频域放大
        Y_shifted = X_shifted * gain_mask[np.newaxis, :]

        # 6. 频域 ASE 噪声 (每偏振独立)
        #    PSD per pol: S_ase(f) = 0.5 * nf * h * nu(f) * (G(f)-1)
        #    ASE 仅限于 WDM 信道频带内 (Rs 带宽), 带外不产生 ASE
        half_bw = rs / 2
        ase_mask = np.zeros(N, dtype=bool)
        for ch_info in wdm_info:
            f_off = ch_info["freq_offset_hz"]
            ase_mask |= np.abs(freqs - f_off) <= half_bw

        S_ase = np.maximum(
            0.5 * self.nf_linear * h * (opt_signal.ctx.center_freq + freqs) * (gain_mask**2 - 1.0),
            0.0,
        )
        S_ase[~ase_mask] = 0.0  # 仅信道频带内产生 ASE
        shape = np.sqrt(S_ase * sr / 2)

        for pol in range(n_pol):
            n_t = np.random.randn(N) + 1j * np.random.randn(N)  # E[|n|^2] = 2
            n_fd = np.fft.fftshift(np.fft.fft(n_t))  # E[|N_fd|^2] = 2N
            Y_shifted[pol] += n_fd * shape

        # 7. IFFT 回时域
        output_data = np.fft.ifft(np.fft.ifftshift(Y_shifted, axes=-1), axis=-1)

        new_ctx = SignalContext(
            sample_rate=sr,
            center_freq=opt_signal.ctx.center_freq,
            n_pol=n_pol,
            sps=opt_signal.ctx.sps,
            sym_rate=rs,
            bits_per_symbol=opt_signal.ctx.bits_per_symbol,
            is_ofdm=opt_signal.ctx.is_ofdm,
            meta={
                **(opt_signal.ctx.meta or {}),
                "ideal_amp_nf_db": self.nf_db,
                "ideal_amp_gains_linear": gains_linear.tolist(),
            },
        )
        return Signal(data=output_data.astype(np.complex64), ctx=new_ctx)

    # ------------------------------------------------------------------
    # 主入口: 根据信号形态调度
    # ------------------------------------------------------------------

    def forward(self, opt_signal: Signal) -> Signal:
        """将信号放大恢复到入纤功率并添加 ASE 噪声。

        自动检测信号形态:
        - 宽带 WDM (wdm_channels_info 且非 stacked) → 频域逐信道增益
        - 堆叠 WDM (is_stacked_wdm_signal)          → 逐通道时域增益
        - 单通道                                      → 整体时域增益
        """
        if not np.iscomplexobj(opt_signal.data):
            raise ValueError("Input optical signal must be complex.")

        is_stacked = opt_signal.ctx.meta.get("is_stacked_wdm_signal", False)
        has_wdm_info = opt_signal.ctx.meta.get("wdm_channels_info") is not None

        if has_wdm_info and not is_stacked:
            # 宽带 WDM: 频域逐信道增益 (避免 demux/remux 相位破坏)
            return self._forward_wdm_broadband(opt_signal)

        # --- 以下为原逻辑: 堆叠 WDM 或单通道 ---
        n_channels, is_stacked_wdm, channel_freqs, channel_powers = self._get_channel_info(opt_signal)

        h = 6.62607015e-34
        rs_hz = opt_signal.ctx.sym_rate

        output_data_list = []
        gains_linear = []

        for ch_idx in range(n_channels):
            p_in_w = channel_powers[ch_idx]
            p_target_w = self._get_channel_target_power_w(
                opt_signal, ch_idx, is_stacked_wdm if is_stacked_wdm else False
            )

            if p_in_w > 0:
                gain_linear = p_target_w / p_in_w
            else:
                gain_linear = 0.0
            gains_linear.append(gain_linear)

            ch_data = opt_signal.data[ch_idx] if is_stacked_wdm else opt_signal.data

            amplified_ch_data = ch_data * np.sqrt(gain_linear)

            nu = channel_freqs[ch_idx]
            G_minus_1 = max(gain_linear - 1.0, 0.0)

            ase_variance_per_pol = 0.5 * self.nf_linear * h * nu * rs_hz * G_minus_1
            ase_std_dev = np.sqrt(ase_variance_per_pol)

            ase_shape = amplified_ch_data.shape
            ase_noise = (
                np.random.normal(0, ase_std_dev, ase_shape)
                + 1j * np.random.normal(0, ase_std_dev, ase_shape)
            ).astype(np.complex64)

            output_data_list.append(amplified_ch_data + ase_noise)

        if is_stacked_wdm:
            output_data = np.stack(output_data_list, axis=0)
        else:
            output_data = output_data_list[0]

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
                "ideal_amp_nf_db": self.nf_db,
                "ideal_amp_gains_linear": gains_linear,
            },
        )
        return Signal(data=output_data.astype(np.complex64), ctx=new_ctx)
