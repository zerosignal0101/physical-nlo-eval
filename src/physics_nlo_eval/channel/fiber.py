"""光纤信道模拟，集成 PyNLO。"""

from typing import Any

import numpy as np

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
    alpha_db_per_km : float
        光纤损耗 (dB/km)。
    beta2_ps2_per_km : float
        二阶色散系数 (ps^2/km)。
    gamma_inv_w_inv_km : float
        非线性系数 (W^-1 km^-1)。
    d_pmd_ps_per_sqrt_km : float, optional
        PMD 系数 (ps/sqrt(km))。如果为 0，则不模拟 PMD。
    step_size_m : float, optional
        PyNLO 求解器的步长 (米)。
    n2_si : float, optional
        光纤非线性折射率 n2 (m^2/W)。如果 gamma 已给出，则此参数可忽略。
    a_eff_um2 : float, optional
        光纤有效面积 (um^2)。如果 gamma 已给出，则此参数可忽略。
    raman_r3 : np.ndarray | None, optional
        拉曼响应函数，如果为 None 则不模拟拉曼效应。
    """

    def __init__(
        self,
        length_m: float,
        alpha_db_per_km: float,
        beta2_ps2_per_km: float,
        gamma_inv_w_inv_km: float,
        d_pmd_ps_per_sqrt_km: float = 0.0,
        step_size_m: float = 1000.0,
        n2_si: float = 2.6e-20,  # Fused silica typical n2
        a_eff_um2: float = 80.0,  # SMF typical A_eff
        raman_r3: np.ndarray | None = None,
    ) -> None:
        """初始化 FiberChannel。"""
        super().__init__(name="FiberChannel")
        if not length_m > 0:
            raise ValueError("Fiber length must be positive.")
        if not step_size_m > 0:
            raise ValueError("Step size must be positive.")

        self.length_m = length_m
        self.alpha_db_per_km = alpha_db_per_km
        self.beta2_ps2_per_km = beta2_ps2_per_km
        self.gamma_inv_w_inv_km = gamma_inv_w_inv_km
        self.d_pmd_ps_per_sqrt_km = d_pmd_ps_per_sqrt_km
        self.step_size_m = step_size_m
        self.n2_si = n2_si
        self.a_eff_um2 = a_eff_um2
        self.raman_r3 = raman_r3

        # 将单位转换为 SI 标准以匹配 PyNLO
        self.alpha_np_inv_m = self.alpha_db_per_km * 1e-3 * np.log(10) / 10  # dB/km to Np/m
        self.beta2_si = self.beta2_ps2_per_km * 1e-24  # ps^2/km to s^2/m
        self.gamma_si = self.gamma_inv_w_inv_km * 1e-3  # W^-1 km^-1 to W^-1 m^-1

    def _setup_pynlo_environment(self, signal: Signal) -> tuple[Any, Any, Any]:
        """根据输入信号设置 PyNLO 的 Grid 和 ManakovEquation。"""
        # Placeholder for PyNLO imports
        # You will need to import these specific classes from pynlo
        # For now, using 'Any' to satisfy MyPy without actual pynlo imports
        Grid = Any  # Replace with actual pynlo.core.grid.Grid
        ManakovEquation = Any  # Replace with actual pynlo.physics.equations.manakov.ManakovEquation
        FieldState = Any  # Replace with actual pynlo.core.field.FieldState

        N = signal.num_samples
        v_center = signal.ctx.center_freq  # Hz
        bandwidth = signal.ctx.sample_rate  # Hz, Nyquist bandwidth

        # PyNLO grid typically uses frequency (v) or angular frequency (omega)
        # Assuming signal.ctx.sample_rate is the full bandwidth for FFT
        # v_comoving is usually 0 for center frequency (relative frequency).
        grid = Grid.create(N=N, v_center=v_center, bandwidth=bandwidth, v_comoving=0.0)

        # Calculate beta_comoving (dispersion coefficients)
        # PyNLO expects beta_comoving matching grid.v_grid
        # omega = 2 * np.pi * grid.v_grid
        # beta_comoving = self.beta2_si / 2 * omega**2
        # For simplicity, if only beta2, it can be passed directly as a scalar or 1D array
        # if using custom dispersion profiles.
        # When `from_fiber` is used, it internally calculates dispersion from beta_comoving.
        # For default ManakovEquation, simpler parameters are possible.

        # As per PyNLO docs: beta_comoving should be 1/m, matched to grid.v_grid order
        # grid.v_grid is relative frequency. For `ManakovEquation.from_fiber`, it will
        # calculate the dispersion from beta2/beta3 given in the constructor.
        # If we specify beta_comoving directly, we need to map the coefficients correctly.
        # Let's simplify and pass beta2 directly as a parameter to equation if the constructor allows it.
        # Or, manually construct beta_comoving:
        omega_relative = 2 * np.pi * (grid.v_grid - grid.v_center)  # Relative angular frequency
        beta_comoving_val = self.beta2_si / 2 * omega_relative**2

        equation = ManakovEquation.from_fiber(
            grid=grid,
            beta_comoving=beta_comoving_val,
            gamma=self.gamma_si,
            alpha_comoving=self.alpha_np_inv_m,
            r3=self.raman_r3,
            manakov_factor=8 / 9,  # Default for Manakov
            # ctx=None, # For numpy context
        )

        initial_state = FieldState(grid=grid, a_v=signal.data, n_comp=signal.ctx.n_pol)

        return grid, equation, initial_state

    def _apply_pmd(self, field_state: Any, step_len: float, grid: Any) -> Any:
        """在频域应用偏振模色散 (PMD)。"""
        # Placeholder for PMD implementation
        # This typically involves applying a random unitary matrix in the frequency domain
        # for a given DGD (Differential Group Delay)
        # Based on self.d_pmd_ps_per_sqrt_km and step_len
        # For PyNLO: operate on field_state.a_v (freq domain field)
        # You'll need to define how to generate the random DGD and rotation matrix.
        # Example (simplified conceptual):
        # DGD = self.d_pmd_ps_per_sqrt_km * np.sqrt(step_len / 1000) * 1e-12 # s
        # # Generate random rotation matrix for PMD (e.g., from J. Opt. Soc. Am. B 21, 1944 (2004))
        # # Apply to field_state.a_v
        return field_state

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
        # Placeholder for PyNLO objects
        # stepper = Any  # pynlo.core.stepper.RK4IPStepper
        # controller = Any  # pynlo.core.controller.AdaptiveController
        # recorder = Any  # pynlo.core.recorder.FullRecorder

        # Ensure signal is complex and has polarization components
        if not np.iscomplexobj(opt_signal.data):
            raise ValueError("Input optical signal must be complex.")
        if opt_signal.data.ndim != 2 or opt_signal.data.shape[0] != opt_signal.ctx.n_pol:
            raise ValueError("Input optical signal data must be 2D (n_pol, N_samples).")

        # 1. 设置 PyNLO 环境
        # grid, equation, initial_field_state = self._setup_pynlo_environment(opt_signal)

        # 2. 初始化求解器
        # stepper = RK4IPStepper(equation)
        # controller = AdaptiveController(tol=1e-7)
        # recorder = FullRecorder() # Or None if not recording intermediate steps

        # 3. 传播循环 (手动步进以支持 PMD)
        num_steps = int(self.length_m / self.step_size_m)
        current_field_state = opt_signal.data  # Placeholder for actual PyNLO field state

        # For actual PyNLO:
        # current_field_state = initial_field_state
        # for _ in range(num_steps):
        #     current_field_state = stepper.step(current_field_state, self.step_size_m, controller)
        #     if self.d_pmd_ps_per_sqrt_km > 0:
        #         current_field_state = self._apply_pmd(current_field_state, self.step_size_m, grid)

        print(f"Propagating {self.length_m / 1000:.1f} km fiber... (PyNLO placeholder)")
        # Simulate some loss and noise effect for placeholder
        # Attenuation
        attenuation_factor = 10 ** (-self.alpha_db_per_km * self.length_m / 1000 / 10)
        current_field_state = current_field_state * attenuation_factor

        # Add some random complex noise (AWGN placeholder)
        noise_power = 1e-6  # Example noise power
        noise = np.random.normal(0, np.sqrt(noise_power / 2), current_field_state.shape) + 1j * np.random.normal(
            0, np.sqrt(noise_power / 2), current_field_state.shape
        )
        current_field_state = current_field_state + noise.astype(np.complex64)

        # 4. 获取最终场状态并转换为 Signal 对象
        # final_data = current_field_state.a_v # Extract freq domain field, or ifft to time domain

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
                "fiber_beta2_ps2_per_km": self.beta2_ps2_per_km,
                "fiber_gamma_inv_w_inv_km": self.gamma_inv_w_inv_km,
                "fiber_d_pmd_ps_per_sqrt_km": self.d_pmd_ps_per_sqrt_km,
            },
        )
        # For PyNLO, you might need to convert the field from frequency domain back to time domain
        # using ifft before passing it out.
        return Signal(data=current_field_state.astype(np.complex64), ctx=new_ctx)


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

        # Noise power density (single-sided, per polarization)
        # P_ase_per_pol_w = self.nf_linear * h * nu * self.gain_linear # * B_e

        # Let's use a common approximation for ASE noise power for complex field:
        # Noise variance per polarization, per sample
        # N0_ase = self.nf_linear * h * nu * noise_bandwidth / (opt_signal.ctx.sps)
        # Assuming the signal's power unit is mW.
        # So we need to ensure the noise power also matches the unit.

        # A more robust way: Noise Figure (NF) is F = (SNR_in / SNR_out)
        # N_out = G * N_in + P_ase => P_ase = N_out - G*N_in
        # For vacuum noise input, P_ase = NF * h * nu * B_opt

        # For complex noise: sigma_n^2 = P_ase / (2 * N_samples) * (unit conversion)
        # Let's simplify and generate a noise level based on an estimated optical noise power

        # P_ase_linear_per_sample = self.nf_linear * h * nu * (opt_signal.ctx.sample_rate / opt_signal.ctx.sps)
        # This P_ase_linear is in Watts per sample.
        # If signal.data is in sqrt(mW), then sqrt(P_ase_linear * 1e3) will be the noise amplitude scale.

        # A common simplified way: P_ase_total_per_pol = N_sp * h * nu * Bandwidth
        # (where N_sp is spontaneous emission factor)
        # And N_sp ~ NF_linear / 2 (if G >> 1).
        # P_ase_total_per_pol = (self.nf_linear / 2) * h * nu * noise_bandwidth # Total power in Watts
        # The complex noise has variance P_ase_total_per_pol / 2 for real/imag parts.

        # For our scale (signal power in mW), let's estimate noise power in mW.
        noise_variance_per_sample = self.nf_linear * h * nu * noise_bandwidth * 1e3  # Convert to mW unit
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
