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
        if compute_ctx_name == "numpy":
            self.compute_ctx = get_numpy_context()
        elif compute_ctx_name == "cupy":
            self.compute_ctx = get_cupy_context()
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
        self.delta_tau_si = (
            self.d_pmd_ps_per_sqrt_km * self.compute_ctx.array.sqrt(self.coarse_step_size_m / 1e3) * 1e-12
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

        # For our scale (signal power in W), let's estimate noise power in W.
        noise_variance_per_sample = (
            0.5 * self.nf_linear * self.gain_linear * h * nu * noise_bandwidth
        )  # Noise power in W
        # Divide by 2 because complex noise has real and imaginary parts
        # and we want std dev (sqrt of variance) for each component
        noise_std_dev = np.sqrt(noise_variance_per_sample / 2)  # sqrt(W)

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


class IdealAmplifier(BaseBlock):
    """理想光放大器 - 按入纤功率恢复信号电平并添加ASE噪声。

    自动从信号的 meta 中读取 IQModulator 存储的入纤功率 (tx_launch_power_dbm_per_pol)，
    将信号放大恢复到该功率电平。对于 WDM 堆叠信号，逐通道恢复各自的人纤功率。

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
        """从信号 meta 中提取各通道的目标功率 (入纤功率)。

        Parameters
        ----------
        signal : Signal
            输入信号。
        ch_idx : int
            通道索引。
        is_stacked_wdm : bool
            是否为堆叠 WDM 信号。

        Returns
        -------
        float
            目标功率 (W, 总功率两个偏振合计)。
        """
        if is_stacked_wdm:
            # 从 wdm_channels_info 中获取各通道的原始上下文
            wdm_info = signal.ctx.meta.get("wdm_channels_info", [])
            if wdm_info and len(wdm_info) > ch_idx:
                original_ctx = wdm_info[ch_idx].get("original_ctx")
                if original_ctx and original_ctx.meta:
                    launch_power_dbm = original_ctx.meta.get("tx_launch_power_dbm_per_pol")
                    if launch_power_dbm is not None:
                        # per pol 转为总功率 (两个偏振)
                        return 2 * (1e-3 * 10 ** (launch_power_dbm / 10))

            # 如果堆叠信号没有原始上下文信息，使用顶层 meta
            launch_power_dbm = signal.ctx.meta.get("tx_launch_power_dbm_per_pol")
            if launch_power_dbm is not None:
                return 2 * (1e-3 * 10 ** (launch_power_dbm / 10))
        else:
            # 单通道信号
            launch_power_dbm = signal.ctx.meta.get("tx_launch_power_dbm_per_pol")
            if launch_power_dbm is not None:
                return 2 * (1e-3 * 10 ** (launch_power_dbm / 10))

        # 回退到默认功率
        if self.default_power_dbm is not None:
            return 2 * (1e-3 * 10 ** (self.default_power_dbm / 10))

        raise ValueError(
            "IdealAmplifier: 信号 meta 中未找到入纤功率 (tx_launch_power_dbm_per_pol)。"
            "请确保信号已经过 IQModulator 处理，或在 IdealAmplifier 构造函数中指定 default_power_dbm。"
        )

    def _get_channel_info(self, signal: Signal) -> tuple[int, bool, list[float], list[float]]:
        """从信号中提取各通道信息。

        Returns
        -------
        tuple
            (n_channels, is_stacked_wdm, channel_freqs_hz, channel_powers_w)
        """
        is_stacked_wdm = signal.ctx.meta.get("is_stacked_wdm_signal", False)

        if is_stacked_wdm:
            # 堆叠 WDM 信号: (n_channels, n_pol, N_samples)
            n_channels = signal.data.shape[0]
            n_pol = signal.data.shape[1]

            # 获取各通道的绝对频率
            wdm_info = signal.ctx.meta.get("wdm_channels_info", [])
            if wdm_info and len(wdm_info) == n_channels:
                # 计算各通道的绝对频率
                reference_freq = signal.ctx.center_freq  # WDM 参考中心频率
                channel_freqs = []
                for ch_info in wdm_info:
                    freq_offset = ch_info.get("freq_offset_hz", 0.0)
                    channel_freqs.append(reference_freq + freq_offset)
            else:
                # 如果没有 wdm_info，使用等间隔假设
                channel_bw = signal.ctx.meta.get("original_sample_rate_hz", signal.ctx.sample_rate)
                total_bw = n_channels * channel_bw
                start_freq = signal.ctx.center_freq - total_bw / 2
                channel_freqs = [start_freq + i * channel_bw for i in range(n_channels)]

            # 计算各通道的平均功率 (两个偏振合计)
            channel_powers = []
            for ch_idx in range(n_channels):
                ch_data = signal.data[ch_idx]  # (n_pol, N_samples)
                power = np.mean(np.abs(ch_data) ** 2) * n_pol  # 总功率 (两个偏振)
                channel_powers.append(power)

            return n_channels, True, channel_freqs, channel_powers
        else:
            # 单通道信号: (n_pol, N_samples) 或宽带 WDM (n_pol, N_samples)
            n_channels = 1
            n_pol = signal.data.shape[0]

            # 中心频率作为信号频率
            channel_freqs = [signal.ctx.center_freq]

            # 计算信号平均功率
            total_power = np.mean(np.abs(signal.data) ** 2) * n_pol
            channel_powers = [total_power]

            return n_channels, False, channel_freqs, channel_powers

    def forward(self, opt_signal: Signal) -> Signal:
        """将信号放大恢复到入纤功率并添加 ASE 噪声。

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

        n_channels, is_stacked_wdm, channel_freqs, channel_powers = self._get_channel_info(opt_signal)

        # 物理常数
        h = 6.62607015e-34  # Planck's constant (J·s)
        rs_hz = opt_signal.ctx.sym_rate  # 符号速率 (Hz)

        # 对每个通道计算增益并添加噪声
        output_data_list = []
        gains_linear = []

        for ch_idx in range(n_channels):
            p_in_w = channel_powers[ch_idx]
            p_target_w = self._get_channel_target_power_w(opt_signal, ch_idx, is_stacked_wdm)

            # 计算所需增益 (线性值)
            if p_in_w > 0:
                gain_linear = p_target_w / p_in_w
            else:
                gain_linear = 0.0
            gains_linear.append(gain_linear)

            # 获取该通道的复数数据
            if is_stacked_wdm:
                ch_data = opt_signal.data[ch_idx]  # (n_pol, N_samples)
            else:
                ch_data = opt_signal.data  # (n_pol, N_samples)

            # 放大信号
            amplified_ch_data = ch_data * np.sqrt(gain_linear)

            # 计算 ASE 噪声方差 (ISRS-GN 公式)
            # sigma_ase^2 = nf * h * nu * rs * (G - 1)
            # 对于复数信号，需要除以 2 (实部/虚部分别承担噪声)
            nu = channel_freqs[ch_idx]
            G_minus_1 = max(gain_linear - 1.0, 0.0)

            ase_variance_per_pol = 0.5 * self.nf_linear * h * nu * rs_hz * G_minus_1
            ase_std_dev = np.sqrt(ase_variance_per_pol)  # sqrt(W)

            # 生成复数高斯噪声并添加到放大后的信号
            shape = amplified_ch_data.shape
            ase_noise = (np.random.normal(0, ase_std_dev, shape) + 1j * np.random.normal(0, ase_std_dev, shape)).astype(
                np.complex64
            )

            output_data_list.append(amplified_ch_data + ase_noise)

        # 组装输出信号
        if is_stacked_wdm:
            output_data = np.stack(output_data_list, axis=0)  # (n_channels, n_pol, N_samples)
        else:
            output_data = output_data_list[0]

        # 构建输出上下文 - 保留入纤功率信息供下一级使用
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
