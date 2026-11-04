"""分布式拉曼放大 (DRA) 的波形级参考模型。

本模块提供一个**独立的 SSFM 波形传播器**（``DistributedRamanFiber``），
用于给解析 DRA 模型（如 ``network-nlo-eval`` 的 Raman ODE）提供逐通道
波形参考。它复用 :class:`~physics_nlo_eval.channel.fiber.FiberChannel`
的 NumPy 对称分步傅里叶求解器，并在每个 SSFM 步施加频率相关的拉曼增益。

第一版的物理假设（详见 ``docs/dra-waveform.md``）：

* 泵浦**未耗尽**：泵浦功率沿 z 解析衰减，不受信号和泵浦间相互作用影响；
* 泵浦之间无耦合、无泵浦-泵浦拉曼转移；
* 信号拉曼增益采用 Gaussian 近似谱 ``g_R(f_p - f_s)``；
* 信号自身的损耗、色散、Kerr 非线性仍由原有 SSFM 处理；
* 通道间 ISRS 默认不建模；若传入 ``raman_r3``，则由原有 SSFM 的
  时域拉曼响应额外提供 ISRS；
* ASE 为可选近似模型，默认关闭，且与 EDFA ASE 分开记录。
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.constants import c, h

from physics_nlo_eval.channel.fiber import FiberChannel
from physics_nlo_eval.channel.network import PumpSpec
from physics_nlo_eval.core.signal import Signal

DEFAULT_RAMAN_PEAK_SHIFT_HZ = 13.2e12
# 与 RamanPump 的高斯近似保持一致：sigma = 2.12 THz，换算成 FWHM。
DEFAULT_RAMAN_SIGMA_HZ = 2.12e12
DEFAULT_RAMAN_FWHM_HZ = DEFAULT_RAMAN_SIGMA_HZ * np.sqrt(8 * np.log(2))
DEFAULT_RAMAN_PEAK_GAIN_M_PER_W = 6e-14


def raman_gain_spectrum(
    shift_hz: np.ndarray | float,
    peak_shift_hz: float = DEFAULT_RAMAN_PEAK_SHIFT_HZ,
    fwhm_hz: float = DEFAULT_RAMAN_FWHM_HZ,
) -> np.ndarray:
    """返回归一化的 Gaussian 拉曼增益谱。

    Parameters
    ----------
    shift_hz : ndarray | float
        泵浦频率减去信号频率 (Hz)。正值表示信号位于泵浦低频侧（Stokes）。
    peak_shift_hz : float, optional
        拉曼增益峰对应的频移 (Hz)。
    fwhm_hz : float, optional
        拉曼增益谱的半高全宽 (Hz)。

    Returns
    -------
    ndarray
        归一化增益谱，峰值 1。负频移侧的增益按 Gaussian 尾部自然趋于零。
    """
    if fwhm_hz <= 0 or peak_shift_hz <= 0:
        raise ValueError("Raman peak shift and FWHM must be positive.")
    sigma_hz = fwhm_hz / np.sqrt(8 * np.log(2))
    shift = np.asarray(shift_hz, dtype=float)
    return np.exp(-0.5 * ((shift - peak_shift_hz) / sigma_hz) ** 2)


class DistributedRamanFiber(FiberChannel):
    """分布式拉曼放大光纤，只使用 NumPy SSFM 路径。

    与 :class:`FiberChannel` 的区别：

    * 强制使用自带的 NumPy 对称分步傅里叶求解器，绝不走 PyNLO 的
      Manakov 传播器（后者没有逐 z 的泵浦增益钩子）；
    * 在每个 SSFM 步按泵浦局部功率、有效面积和频率相关的拉曼增益谱
      施加信号增益；
    * 泵浦功率沿 z 解析衰减（未耗尽近似），支持任意数量的前向/后向泵。

    Parameters
    ----------
    length_m : float
        光纤长度 (米)。
    alpha_db_per_km : float | ndarray
        信号损耗 (dB/km)。
    beta2_ps2_per_km : float
        二阶色散 (ps^2/km)。
    beta3_ps3_per_km : float
        三阶色散 (ps^3/km)。
    pumps : PumpSpec | list[PumpSpec] | None, optional
        拉曼泵浦列表。为 None 或空列表时不施加拉曼增益，此时传播结果与
        ``FiberChannel`` 的 NumPy 路径逐位一致。
    raman_peak_shift_hz : float, optional
        Gaussian 拉曼谱峰值频移 (Hz)。
    raman_fwhm_hz : float, optional
        Gaussian 拉曼谱半高全宽 (Hz)。
    raman_peak_gain_m_per_w : float, optional
        拉曼增益系数峰值 (m/W)，与 ``RamanPump`` 的默认值保持一致。
    pump_model : str, optional
        第一版仅支持 ``"undepleted"``。
    add_ase : bool, optional
        是否注入近似的分布式拉曼 ASE。默认关闭。
    ase_spontaneous_factor : float, optional
        ASE 自发辐射因子 ``n_sp``，默认为量子极限 1.0。
    seed : int | None, optional
        ASE 随机数种子。仅在 ``add_ase=True`` 时使用。
    v_comoving, d_pmd_ps_per_sqrt_km, coarse_step_size_m, n2_si, a_eff_um2
        与 :class:`FiberChannel` 同名参数含义一致。
    slope_a_eff_um2_per_nm, ref_lambda_nm, raman_r3
        NumPy 路径按中心频率取标量 γ，因此不使用 ``slope_a_eff_um2_per_nm``；
        ``raman_r3`` 可选地在 DRA 增益之外叠加时域 ISRS。
    """

    def __init__(
        self,
        length_m: float,
        alpha_db_per_km: float | np.ndarray,
        beta2_ps2_per_km: float,
        beta3_ps3_per_km: float,
        pumps: PumpSpec | list[PumpSpec] | None = None,
        *,
        v_comoving: float | None = None,
        d_pmd_ps_per_sqrt_km: float = 0.0,
        coarse_step_size_m: float = 1000.0,
        n2_si: float = 2.6e-20,
        a_eff_um2: float = 80.0,
        slope_a_eff_um2_per_nm: float = 0.05,
        ref_lambda_nm: float = 1550.0,
        raman_r3: np.ndarray | None = None,
        raman_peak_shift_hz: float = DEFAULT_RAMAN_PEAK_SHIFT_HZ,
        raman_fwhm_hz: float = DEFAULT_RAMAN_FWHM_HZ,
        raman_peak_gain_m_per_w: float = DEFAULT_RAMAN_PEAK_GAIN_M_PER_W,
        pump_model: str = "undepleted",
        add_ase: bool = False,
        ase_spontaneous_factor: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """初始化分布式拉曼光纤。"""
        super().__init__(
            length_m=length_m,
            alpha_db_per_km=alpha_db_per_km,
            beta2_ps2_per_km=beta2_ps2_per_km,
            beta3_ps3_per_km=beta3_ps3_per_km,
            v_comoving=v_comoving,
            d_pmd_ps_per_sqrt_km=d_pmd_ps_per_sqrt_km,
            coarse_step_size_m=coarse_step_size_m,
            n2_si=n2_si,
            a_eff_um2=a_eff_um2,
            slope_a_eff_um2_per_nm=slope_a_eff_um2_per_nm,
            ref_lambda_nm=ref_lambda_nm,
            raman_r3=raman_r3,
            compute_ctx_name="numpy",
        )
        # DRA 必须走本仓的 SSFM，PyNLO 的 Manakov 传播器没有逐 z 泵浦钩子。
        self.compute_ctx = None
        self.name = "DistributedRamanFiber"

        if pump_model != "undepleted":
            raise ValueError("The first DRA version only supports pump_model='undepleted'.")
        if raman_peak_shift_hz <= 0 or raman_fwhm_hz <= 0 or raman_peak_gain_m_per_w < 0:
            raise ValueError("Raman peak shift/FWHM must be positive and gain coefficient non-negative.")
        if ase_spontaneous_factor < 0:
            raise ValueError("ASE spontaneous factor must be non-negative.")

        if pumps is None:
            self.pumps: list[PumpSpec] = []
        elif isinstance(pumps, PumpSpec):
            self.pumps = [pumps]
        else:
            self.pumps = list(pumps)

        self.raman_peak_shift_hz = raman_peak_shift_hz
        self.raman_fwhm_hz = raman_fwhm_hz
        self.raman_peak_gain_m_per_w = raman_peak_gain_m_per_w
        self.pump_model = pump_model
        self.add_ase = add_ase
        self.ase_spontaneous_factor = ase_spontaneous_factor
        self._ase_rng = np.random.default_rng(seed)

        if self.raman_r3 is not None:
            warnings.warn(
                "DistributedRamanFiber: raman_r3 enables time-domain ISRS in addition to "
                "pump-induced Raman gain; the reference dataset leaves it None.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def n_pumps(self) -> int:
        """泵浦数量。"""
        return len(self.pumps)

    @staticmethod
    def _pump_alpha_np_per_m(pump: PumpSpec) -> float:
        """把泵浦损耗从 dB/km 转换为 Np/m。"""
        return pump.attenuation_db_per_km * 1e-3 * np.log(10) / 10

    @staticmethod
    def _pump_frequency_hz(pump: PumpSpec) -> float:
        """泵浦光频率 (Hz)。"""
        return c / (pump.wavelength_nm * 1e-9)

    def _pump_effective_length_m(self, pump: PumpSpec) -> float:
        """未耗尽泵浦的有效长度 (m)。前向与后向泵的积分相同。"""
        alpha = self._pump_alpha_np_per_m(pump)
        if alpha == 0.0:
            return self.length_m
        return -np.expm1(-alpha * self.length_m) / alpha

    def pump_power_w(self, z_m: float | np.ndarray) -> np.ndarray:
        """返回沿光纤位置的泵浦功率。

        Parameters
        ----------
        z_m : float | ndarray
            位置 (米)，0 为信号输入端，``length_m`` 为输出端。

        Returns
        -------
        ndarray
            形状 ``(n_pumps,)`` 或 ``(n_pumps, *z.shape)`` 的泵浦功率 (W)。
            前向泵在 z=0 发射，后向泵在 z=L 发射。
        """
        z = np.asarray(z_m, dtype=float)
        if not self.pumps:
            return np.zeros((0, *z.shape), dtype=float)

        profiles = []
        for pump in self.pumps:
            alpha = self._pump_alpha_np_per_m(pump)
            if pump.direction == "forward":
                attenuation = np.exp(-alpha * z)
            else:
                attenuation = np.exp(-alpha * (self.length_m - z))
            profiles.append(pump.power_w * attenuation)
        return np.stack(profiles, axis=0)

    def raman_gain_rate(self, z_m: float, optical_freq_hz: np.ndarray) -> np.ndarray:
        """返回位置 ``z_m`` 处的信号功率增益率 (1/m)。

        Parameters
        ----------
        z_m : float
            位置 (米)。
        optical_freq_hz : ndarray
            信号光频率 (Hz)。

        Returns
        -------
        ndarray
            与 ``optical_freq_hz`` 同形状的功率增益率，单位 Np/m。
            未耗尽近似下正比于该位置所有泵浦功率之和。
        """
        freqs = np.asarray(optical_freq_hz, dtype=float)
        rate = np.zeros_like(freqs, dtype=float)
        if not self.pumps:
            return rate

        powers = self.pump_power_w(z_m)
        for pump, power_w in zip(self.pumps, powers, strict=True):
            shift = self._pump_frequency_hz(pump) - freqs
            spectral = raman_gain_spectrum(
                shift,
                peak_shift_hz=self.raman_peak_shift_hz,
                fwhm_hz=self.raman_fwhm_hz,
            )
            rate += self.raman_peak_gain_m_per_w * spectral * float(power_w) / self.a_eff_si
        return rate

    def on_off_gain(self, optical_freq_hz: np.ndarray | float) -> np.ndarray:
        """未耗尽近似下的泵浦开关增益 (线性功率比)。

        该量只包含拉曼增益，不包含信号自身损耗、色散或 Kerr 非线性，
        因此可直接与 ``RamanPump._gain`` 的频率形状对照。
        """
        freqs = np.asarray(optical_freq_hz, dtype=float)
        exponent = np.zeros_like(freqs, dtype=float)
        for pump in self.pumps:
            shift = self._pump_frequency_hz(pump) - freqs
            spectral = raman_gain_spectrum(
                shift,
                peak_shift_hz=self.raman_peak_shift_hz,
                fwhm_hz=self.raman_fwhm_hz,
            )
            exponent += (
                self.raman_peak_gain_m_per_w
                * spectral
                * pump.power_w
                * self._pump_effective_length_m(pump)
                / self.a_eff_si
            )
        return np.exp(exponent)

    def _make_step_gain(
        self, center_freq_hz: float
    ) -> Callable[[float, float, np.ndarray], np.ndarray]:
        """构造逐 SSFM 步的幅度增益回调。"""
        def step_gain(z_center_m: float, dz_m: float, omega_rad_per_s: np.ndarray) -> np.ndarray:
            optical_freq_hz = center_freq_hz + omega_rad_per_s / (2 * np.pi)
            rate = self.raman_gain_rate(z_center_m, optical_freq_hz)
            return np.exp(0.5 * rate * dz_m)

        return step_gain

    def forward(self, opt_signal: Signal) -> Signal:
        """通过带分布式拉曼放大的光纤传播信号。

        Parameters
        ----------
        opt_signal : Signal
            输入光信号，形状 ``(n_pol, N_samples)``，振幅单位 sqrt(W)。

        Returns
        -------
        Signal
            传播后的光信号。元数据中 ``fiber_solver="numpy-ssfm-dra"``，
            并记录泵浦配置、开关增益范围和 ASE 模型。
        """
        if not np.iscomplexobj(opt_signal.data):
            raise ValueError("Input optical signal must be complex.")
        if opt_signal.data.ndim != 2 or opt_signal.data.shape[0] != opt_signal.ctx.n_pol:
            raise ValueError("Input optical signal data must be 2D (n_pol, N_samples).")

        step_gain = self._make_step_gain(opt_signal.ctx.center_freq) if self.pumps else None
        output = self._forward_numpy(opt_signal, step_gain=step_gain)

        if self.add_ase and self.pumps:
            output = self._add_raman_ase(output, opt_signal)

        n_samples = opt_signal.num_samples
        sample_rate = opt_signal.ctx.sample_rate
        freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
        rel_freq = np.fft.fftfreq(n_samples, d=1 / sample_rate) * freq_scale
        on_off = self.on_off_gain(opt_signal.ctx.center_freq + rel_freq)
        on_off_db = 10 * np.log10(np.maximum(on_off, 1e-300))

        output.ctx.meta.update(
            {
                "fiber_solver": "numpy-ssfm-dra",
                "dra_model": self.pump_model,
                "dra_raman_spectrum": "gaussian",
                "dra_raman_peak_shift_hz": self.raman_peak_shift_hz,
                "dra_raman_fwhm_hz": self.raman_fwhm_hz,
                "dra_raman_peak_gain_m_per_w": self.raman_peak_gain_m_per_w,
                "dra_n_pumps": self.n_pumps,
                "dra_pumps": [
                    {
                        "wavelength_nm": pump.wavelength_nm,
                        "power_w": pump.power_w,
                        "direction": pump.direction,
                        "attenuation_db_per_km": pump.attenuation_db_per_km,
                    }
                    for pump in self.pumps
                ],
                "dra_on_off_gain_db_min": float(np.min(on_off_db)),
                "dra_on_off_gain_db_max": float(np.max(on_off_db)),
                "dra_channel_isrs": "raman_r3" if self.raman_r3 is not None else "not_modelled",
            }
        )
        if not (self.add_ase and self.pumps):
            output.ctx.meta.update({"dra_ase_model": "none", "dra_ase_power_per_pol_w": 0.0})

        return output

    # ------------------------------------------------------------------
    # 可选 ASE
    # ------------------------------------------------------------------

    def _add_raman_ase(self, propagated: Signal, input_signal: Signal) -> Signal:
        """在输出端注入未耗尽近似的分布式拉曼 ASE。

        模型为每个偏振独立计算输出 ASE 功率谱密度::

            S_ase(f) = h f n_sp * sum_p ∫_0^L γ_p(f, z) G(z→L, f) dz

        其中 ``γ_p = g_R(f_p-f) P_p(z)/A_eff`` 是泵浦 p 的局部增益率，
        ``G(z→L,f) = exp(∫_z^L [-α(f) + Σ_p γ_p] dz')`` 是信号从 z 到
        输出端的净增益。该模型忽略泵浦耗尽、泵浦噪声传递和后向 ASE，
        仅用于可选的波形噪声注入；参考数据集默认不启用。
        """
        n_samples = input_signal.num_samples
        n_pol = propagated.data.shape[0]
        sample_rate = input_signal.ctx.sample_rate
        freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
        rel_freq = np.fft.fftfreq(n_samples, d=1 / sample_rate) * freq_scale
        optical_freq = input_signal.ctx.center_freq + rel_freq
        df_hz = sample_rate / n_samples

        n_steps = max(1, int(np.ceil(self.length_m / self.coarse_step_size_m)))
        dz = self.length_m / n_steps
        z_centers = (np.arange(n_steps) + 0.5) * dz

        alpha = np.asarray(self.alpha_np_inv_m, dtype=float)
        if alpha.ndim == 0:
            alpha_array = np.full(n_samples, float(alpha))
        elif alpha.size == 1:
            alpha_array = np.full(n_samples, float(alpha.ravel()[0]))
        elif alpha.size == n_samples:
            alpha_array = alpha
        else:
            raise ValueError("Frequency-dependent alpha must have one value per signal sample.")

        gamma_pumps: list[np.ndarray] = []
        gamma_total = np.zeros((n_steps, n_samples), dtype=float)
        for pump_index, pump in enumerate(self.pumps):
            powers = self.pump_power_w(z_centers)[pump_index]
            shift = self._pump_frequency_hz(pump) - optical_freq
            spectral = raman_gain_spectrum(
                shift,
                peak_shift_hz=self.raman_peak_shift_hz,
                fwhm_hz=self.raman_fwhm_hz,
            )
            gamma = (
                self.raman_peak_gain_m_per_w
                * spectral[None, :]
                * powers[:, None]
                / self.a_eff_si
            )
            gamma_pumps.append(gamma)
            gamma_total += gamma

        # 从输出端反向累加净增益指数，得到 G(z→L, f)。
        net_rate = gamma_total - alpha_array[None, :]
        tail_integral = np.cumsum(net_rate[::-1], axis=0)[::-1] * dz
        gain_to_output = np.exp(tail_integral)

        source = np.zeros(n_samples, dtype=float)
        for gamma in gamma_pumps:
            source += np.sum(gamma * gain_to_output, axis=0) * dz
        ase_psd_per_pol = h * optical_freq * self.ase_spontaneous_factor * source
        ase_psd_per_pol = np.maximum(ase_psd_per_pol, 0.0)

        # 频域注入复高斯噪声：E[|N_f|^2] = N^2 S df，对应时域 PSD S(f)。
        sigma_bin = np.sqrt(n_samples**2 * ase_psd_per_pol * df_hz / 2.0)
        noise_spectrum = sigma_bin[None, :] * (
            self._ase_rng.standard_normal((n_pol, n_samples))
            + 1j * self._ase_rng.standard_normal((n_pol, n_samples))
        )
        noise_time = np.fft.ifft(noise_spectrum, axis=-1)
        data = propagated.data.astype(np.complex128) + noise_time
        ase_power_per_pol_w = float(np.sum(ase_psd_per_pol * df_hz))

        ctx = propagated.ctx.copy()
        ctx.meta.update(
            {
                "dra_ase_model": "undepleted-raman-ase",
                "dra_ase_power_per_pol_w": ase_power_per_pol_w,
                "dra_ase_bandwidth_hz": sample_rate,
                "dra_ase_spontaneous_factor": self.ase_spontaneous_factor,
            }
        )
        return Signal(data.astype(np.complex64), ctx)


__all__ = [
    "DEFAULT_RAMAN_FWHM_HZ",
    "DEFAULT_RAMAN_PEAK_GAIN_M_PER_W",
    "DEFAULT_RAMAN_PEAK_SHIFT_HZ",
    "DEFAULT_RAMAN_SIGMA_HZ",
    "DistributedRamanFiber",
    "raman_gain_spectrum",
]
