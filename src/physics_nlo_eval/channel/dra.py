"""分布式拉曼放大 (DRA) 的波形级参考模型。

本模块提供一个**独立的 SSFM 波形传播器**（``DistributedRamanFiber``），
用于给解析 DRA 模型（如 ``network-nlo-eval`` 的 Raman ODE）提供逐通道
波形参考。它复用 :class:`~physics_nlo_eval.channel.fiber.FiberChannel`
的 NumPy 对称分步傅里叶求解器，并在每个 SSFM 步施加频率相关的拉曼增益。

物理假设（详见 ``docs/dra-waveform.md``）：

* 支持两种泵浦模型：``"undepleted"``（默认）泵浦功率沿 z 解析衰减，不受
  信号和泵浦间相互作用影响；``"depleted"`` 在 SSFM 步进之外用 RK4 积分
  泵浦-信号功率耦合方程，泵浦按局部信号功率（两个偏振合计的带内总功率）
  被耗尽，信号增益使用局部泵浦功率，后向泵用不动点迭代求解边值问题；
  耗尽耦合系数默认取信号中心频率，也可选 ``depletion_coupling="band_average"``
  用带内功率谱加权平均（见 ``docs/dra-waveform.md`` §13）；
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
#: 支持的泵浦模型。
PUMP_MODELS = ("undepleted", "depleted")
#: 耗尽模型里泵浦-信号耦合系数的取法。
DEPLETION_COUPLINGS = ("center", "band_average")

# NumPy 2.0 把 ``trapz`` 改名为 ``trapezoid``；本仓仍兼容 1.x。
_trapezoid = getattr(np, "trapezoid", np.trapz)


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
    * ``pump_model="undepleted"`` 时泵浦功率沿 z 解析衰减，支持任意数量的
      前向/后向泵；
    * ``pump_model="depleted"`` 时先求解泵浦-信号功率耦合方程，再用求得的
      泵浦剖面施加增益；信号侧仍按频域逐点增益处理。

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
        ``"undepleted"``（默认）或 ``"depleted"``。后者把泵浦作为状态量，
        按局部信号功率逐步耗尽；默认值保持历史数值路径不变。
    depleted_step_m : float, optional
        耗尽模型求解泵浦-信号耦合方程的 RK4 步长 (m)。默认 25 m，与
        ``coarse_step_size_m`` 无关，保证泵浦剖面积分足够精细。
    depleted_max_iterations : int, optional
        后向泵边值问题的不动点迭代上限。纯前向泵一次前向积分即可收敛。
    depleted_tolerance : float, optional
        不动点迭代的相对收敛阈值（相对各泵浦标称功率与输入信号功率）。
    depletion_coupling : str, optional
        耗尽方程里泵浦-信号耦合系数 ``γ_p`` 的取法：

        * ``"center"``（默认）：取信号中心频率的 ``γ_p``。这是历史口径，
          与既有参考数据逐位一致；
        * ``"band_average"``：取信号带内功率谱加权的 ``γ_p`` 平均值，使泵浦
          耗尽量与逐频率施加的信号增益使用同一平均耦合。窄带/CW 时两者
          趋于一致；宽带时中心频率可能落在拉曼峰上或远离峰值，两者差别
          可达 1–2 dB（见 ``docs/dra-waveform.md`` 的剖面诊断一节）。

        该参数只改变泵浦 ODE 的驱动系数，信号增益仍逐频率施加。
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
        depleted_step_m: float = 25.0,
        depleted_max_iterations: int = 8,
        depleted_tolerance: float = 1e-8,
        depletion_coupling: str = "center",
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

        if pump_model not in PUMP_MODELS:
            raise ValueError(
                f"pump_model must be one of {PUMP_MODELS}, got {pump_model!r}."
            )
        if raman_peak_shift_hz <= 0 or raman_fwhm_hz <= 0 or raman_peak_gain_m_per_w < 0:
            raise ValueError("Raman peak shift/FWHM must be positive and gain coefficient non-negative.")
        if ase_spontaneous_factor < 0:
            raise ValueError("ASE spontaneous factor must be non-negative.")
        if depleted_step_m <= 0:
            raise ValueError("depleted_step_m must be positive.")
        if depleted_max_iterations < 1:
            raise ValueError("depleted_max_iterations must be at least 1.")
        if depleted_tolerance <= 0:
            raise ValueError("depleted_tolerance must be positive.")
        if depletion_coupling not in DEPLETION_COUPLINGS:
            raise ValueError(
                f"depletion_coupling must be one of {DEPLETION_COUPLINGS}, "
                f"got {depletion_coupling!r}."
            )

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
        self.depleted_step_m = depleted_step_m
        self.depleted_max_iterations = depleted_max_iterations
        self.depleted_tolerance = depleted_tolerance
        self.depletion_coupling = depletion_coupling
        self.add_ase = add_ase
        self.ase_spontaneous_factor = ase_spontaneous_factor
        self._ase_rng = np.random.default_rng(seed)
        # 耗尽模型的求解结果，``forward()`` 时按输入信号功率刷新。
        self._depleted_pump_z: np.ndarray | None = None
        self._depleted_pump_power: np.ndarray | None = None
        self._depleted_signal_power: np.ndarray | None = None
        self._depletion_diagnostics: dict | None = None

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

    def _undepleted_pump_power_w(self, z_m: float | np.ndarray) -> np.ndarray:
        """未耗尽近似下的泵浦功率剖面。"""
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

        Notes
        -----
        ``pump_model="undepleted"`` 时返回解析衰减剖面。``"depleted"`` 时返回
        :meth:`forward` 中按最近一次输入信号求得的耦合剖面；由于泵浦耗尽
        依赖信号功率，调用 ``forward()`` 之前没有定义，会显式报错。
        """
        if (
            self.pump_model == "depleted"
            and self.pumps
            and self._depleted_pump_z is not None
            and self._depleted_pump_power is not None
        ):
            z = np.asarray(z_m, dtype=float)
            profiles = [
                np.asarray(np.interp(z, self._depleted_pump_z, row)).reshape(z.shape)
                for row in self._depleted_pump_power
            ]
            return np.stack(profiles, axis=0)
        if self.pump_model == "depleted" and self.pumps:
            raise RuntimeError(
                "Depleted pump profiles depend on the signal; call forward() before pump_power_w()."
            )
        return self._undepleted_pump_power_w(z_m)

    def _pump_coupling_rates(self, signal_freq_hz: float) -> np.ndarray:
        """泵浦到信号中心频率的功率耦合系数 (1/(W·m))。

        返回 ``γ_p = g_peak · g̃_R(f_p − f_s) / A_eff``。耗尽模型用该系数
        同时决定信号增益和泵浦耗尽量，因此二者严格互为光子数守恒的一对。
        """
        if not self.pumps:
            return np.zeros(0, dtype=float)
        pump_freqs = np.array([self._pump_frequency_hz(pump) for pump in self.pumps])
        spectral = raman_gain_spectrum(
            pump_freqs - float(signal_freq_hz),
            peak_shift_hz=self.raman_peak_shift_hz,
            fwhm_hz=self.raman_fwhm_hz,
        )
        return self.raman_peak_gain_m_per_w * spectral / self.a_eff_si

    def _pump_coupling_rates_band(
        self, band_freqs_hz: np.ndarray, band_weights: np.ndarray
    ) -> np.ndarray:
        """信号带内功率加权平均的泵浦-信号耦合系数 (1/(W·m))。

        ``band_weights`` 归一化为 1（信号功率谱权重），返回
        ``Σ_k w_k · g_peak · g̃_R(f_p − f_k) / A_eff``。权重为空或全零时
        退回中心频率口径。
        """
        if not self.pumps:
            return np.zeros(0, dtype=float)
        freqs = np.asarray(band_freqs_hz, dtype=float)
        weights = np.asarray(band_weights, dtype=float)
        if (
            freqs.size == 0
            or weights.shape != freqs.shape
            or not np.any(weights > 0.0)
        ):
            return self._pump_coupling_rates(float(np.mean(freqs)) if freqs.size else 0.0)
        weights = weights / float(np.sum(weights))
        pump_freqs = np.array([self._pump_frequency_hz(pump) for pump in self.pumps])
        spectral = raman_gain_spectrum(
            pump_freqs[:, None] - freqs[None, :],
            peak_shift_hz=self.raman_peak_shift_hz,
            fwhm_hz=self.raman_fwhm_hz,
        )
        average = spectral @ weights
        return self.raman_peak_gain_m_per_w * average / self.a_eff_si

    def _solve_depleted_profiles(
        self,
        input_power_w: float,
        signal_freq_hz: float,
        *,
        band_freqs_hz: np.ndarray | None = None,
        band_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """求解泵浦-信号总功率耦合方程，返回泵浦剖面与能量账。

        状态量是各泵浦功率；信号只保留一个标量剖面——带内总功率（两个
        偏振合计）。沿 +z 的耦合方程::

            dP_s/dz = −α_s P_s + P_s · Σ_p γ_p P_p
            dP_p/dz = −s_p · [α_p P_p + (f_p/f_s) γ_p P_p P_s],  s_p = ±1

        ``s_p=+1`` 为前向泵（在 z=0 注入），``s_p=−1`` 为后向泵（在 z=L
        注入）。``γ_p`` 由 ``depletion_coupling`` 决定：``"center"`` 取信号
        中心频率，忽略 WDM 带宽内拉曼谱的缓慢变化；``"band_average"`` 用
        ``band_weights`` 对带内频点做功率加权平均（见
        :meth:`_pump_coupling_rates_band`）。光子数因子 ``f_p/f_s`` 始终用
        中心频率（带内变化 <1%，属于二阶效应）。信号损耗 ``α_s`` 取
        ``alpha_np_inv_m`` 的均值（参考数据只有标量损耗）。泵浦之间没有
        拉曼转移项。

        前向泵从 z=0 正向积分；后向泵从 z=L 反向积分。二者通过不动点
        迭代闭合边值问题：先用当前信号剖面积分后向泵，再用更新后的泵浦
        剖面正向积分信号与前向泵。纯前向泵时第一次迭代即收敛。
        """
        n_steps = max(1, int(np.ceil(self.length_m / self.depleted_step_m)))
        dz = self.length_m / n_steps
        z = np.linspace(0.0, self.length_m, n_steps + 1)

        pump_freqs = np.array([self._pump_frequency_hz(pump) for pump in self.pumps])
        alphas = np.array([self._pump_alpha_np_per_m(pump) for pump in self.pumps])
        if self.depletion_coupling == "band_average" and band_freqs_hz is not None:
            gamma = self._pump_coupling_rates_band(band_freqs_hz, band_weights)
        else:
            gamma = self._pump_coupling_rates(signal_freq_hz)
        # 泵浦功率的损耗项按光子数守恒放大 (f_p/f_s)：信号多出的光子来自泵浦。
        depletion = gamma * pump_freqs / float(signal_freq_hz)
        forward_mask = np.array([pump.direction == "forward" for pump in self.pumps], dtype=bool)
        fwd_idx = np.flatnonzero(forward_mask)
        bwd_idx = np.flatnonzero(~forward_mask)
        signs = np.where(forward_mask, 1.0, -1.0)
        alpha_s = float(np.mean(np.asarray(self.alpha_np_inv_m, dtype=float)))

        profile = self._undepleted_pump_power_w(z)
        # 初值：以未耗尽剖面推一个信号功率剖面，供后向泵迭代使用。
        rate = gamma @ profile - alpha_s
        ps = input_power_w * np.exp(
            np.concatenate(([0.0], np.cumsum(0.5 * (rate[:-1] + rate[1:]) * dz)))
        )

        def backward_rhs(zz: float, y: np.ndarray) -> np.ndarray:
            signal = np.interp(zz, z, ps)
            return -signs[bwd_idx] * (
                alphas[bwd_idx] * y + depletion[bwd_idx] * y * signal
            )

        def forward_rhs(zz: float, y: np.ndarray) -> np.ndarray:
            signal = y[0]
            forward_powers = y[1:]
            total = np.zeros(self.n_pumps, dtype=float)
            total[fwd_idx] = forward_powers
            if bwd_idx.size:
                total[bwd_idx] = [np.interp(zz, z, profile[i]) for i in bwd_idx]
            d_signal = (gamma @ total - alpha_s) * signal
            d_pumps = -signs[fwd_idx] * (
                alphas[fwd_idx] * forward_powers
                + depletion[fwd_idx] * forward_powers * signal
            )
            return np.concatenate(([d_signal], d_pumps))

        nominal = np.maximum(np.array([pump.power_w for pump in self.pumps], dtype=float), 1e-30)
        converged = False
        iterations = 0
        for iteration in range(1, self.depleted_max_iterations + 1):
            iterations = iteration
            previous_profile = profile.copy()
            previous_ps = ps

            if bwd_idx.size:
                y = profile[bwd_idx, -1].copy()
                updated = np.empty_like(profile[bwd_idx])
                updated[:, -1] = y
                for k in range(n_steps, 0, -1):
                    z0 = z[k]
                    h = -dz
                    k1 = backward_rhs(z0, y)
                    k2 = backward_rhs(z0 + h / 2, y + 0.5 * h * k1)
                    k3 = backward_rhs(z0 + h / 2, y + 0.5 * h * k2)
                    k4 = backward_rhs(z0 + h, y + h * k3)
                    y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                    updated[:, k - 1] = y
                profile[bwd_idx] = updated

            y = np.concatenate(([input_power_w], profile[fwd_idx, 0]))
            updated_y = np.empty((1 + fwd_idx.size, n_steps + 1))
            updated_y[:, 0] = y
            for k in range(n_steps):
                z0 = z[k]
                k1 = forward_rhs(z0, y)
                k2 = forward_rhs(z0 + dz / 2, y + 0.5 * dz * k1)
                k3 = forward_rhs(z0 + dz / 2, y + 0.5 * dz * k2)
                k4 = forward_rhs(z0 + dz, y + dz * k3)
                y = y + dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                updated_y[:, k + 1] = y
            ps = updated_y[0]
            profile[fwd_idx] = updated_y[1:]

            pump_error = float(np.max(np.abs(profile - previous_profile) / nominal[:, None]))
            signal_error = float(np.max(np.abs(ps - previous_ps)) / max(input_power_w, 1e-30))
            if max(pump_error, signal_error) < self.depleted_tolerance:
                converged = True
                break

        passive_loss = _trapezoid(alphas[:, None] * profile, z, axis=1)
        transfer = _trapezoid(depletion[:, None] * profile * ps[None, :], z, axis=1)
        signal_gain = float(
            _trapezoid((gamma[:, None] * profile).sum(axis=0) * ps, z)
        )
        pumps_diag = []
        for index, pump in enumerate(self.pumps):
            launch = float(profile[index, 0] if pump.direction == "forward" else profile[index, -1])
            residual = float(profile[index, -1] if pump.direction == "forward" else profile[index, 0])
            pumps_diag.append(
                {
                    "direction": pump.direction,
                    "launch_power_w": launch,
                    "residual_power_w": residual,
                    "passive_loss_w": float(passive_loss[index]),
                    "transferred_to_signal_w": float(transfer[index]),
                }
            )
        diagnostics = {
            "pumps": pumps_diag,
            "signal_raman_gain_w": signal_gain,
            "iterations": iterations,
            "converged": converged,
            "step_m": dz,
        }
        return z, profile, ps, diagnostics

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
            正比于该位置所有泵浦功率之和；泵浦功率取自
            :meth:`pump_power_w`（未耗尽为解析剖面，耗尽为耦合求解剖面）。
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
        """泵浦开关增益 (线性功率比)。

        该量只包含拉曼增益，不包含信号自身损耗、色散或 Kerr 非线性，
        因此可直接与 ``RamanPump._gain`` 的频率形状对照。

        ``"undepleted"`` 时按解析有效长度计算；``"depleted"`` 时对
        :meth:`forward` 求得的泵浦剖面做数值积分，即给定该输入信号下
        的小信号开关增益（泵浦耗尽已体现在剖面里）。
        """
        freqs = np.asarray(optical_freq_hz, dtype=float)
        exponent = np.zeros_like(freqs, dtype=float)
        if not self.pumps:
            return np.exp(exponent)

        if self.pump_model == "depleted":
            if self._depleted_pump_z is None or self._depleted_pump_power is None:
                raise RuntimeError(
                    "Depleted on-off gain depends on the signal; call forward() first."
                )
            effective_length = _trapezoid(self._depleted_pump_power, self._depleted_pump_z, axis=1)
            for pump, length_m in zip(self.pumps, effective_length, strict=True):
                shift = self._pump_frequency_hz(pump) - freqs
                spectral = raman_gain_spectrum(
                    shift,
                    peak_shift_hz=self.raman_peak_shift_hz,
                    fwhm_hz=self.raman_fwhm_hz,
                )
                exponent += (
                    self.raman_peak_gain_m_per_w * spectral * float(length_m) / self.a_eff_si
                )
            return np.exp(exponent)

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
        """构造逐 SSFM 步的幅度增益回调。

        拉曼谱形只随频率变化，泵浦功率只随 z 变化，因此把每个泵浦的
        ``g_peak · g̃_R(f_p − f)`` 在第一次调用时算好缓存；逐 z 只需乘上
        局部泵浦功率。乘法顺序与 :meth:`raman_gain_rate` 完全一致，结果
        逐位相同。
        """
        # 只缓存最近一次的频率轴；``forward()`` 的每一步都传入同一个数组对象。
        cached_omega: np.ndarray | None = None
        cached_spectra: np.ndarray | None = None

        def step_gain(z_center_m: float, dz_m: float, omega_rad_per_s: np.ndarray) -> np.ndarray:
            nonlocal cached_omega, cached_spectra
            if cached_omega is not omega_rad_per_s:
                optical_freq_hz = center_freq_hz + omega_rad_per_s / (2 * np.pi)
                cached_spectra = np.stack(
                    [
                        self.raman_peak_gain_m_per_w
                        * raman_gain_spectrum(
                            self._pump_frequency_hz(pump) - optical_freq_hz,
                            peak_shift_hz=self.raman_peak_shift_hz,
                            fwhm_hz=self.raman_fwhm_hz,
                        )
                        for pump in self.pumps
                    ]
                )
                cached_omega = omega_rad_per_s
            rate = np.zeros_like(cached_spectra[0], dtype=float)
            for spectrum, power_w in zip(
                cached_spectra, self.pump_power_w(z_center_m), strict=True
            ):
                rate += (spectrum * float(power_w)) / self.a_eff_si
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

        if self.pumps and self.pump_model == "depleted":
            # 信号带内总功率（两个偏振合计），单位 W；耗尽模型据此更新泵浦剖面。
            input_power_w = float(np.sum(np.mean(np.abs(opt_signal.data.astype(np.complex128)) ** 2, axis=1)))
            band_freqs_hz: np.ndarray | None = None
            band_weights: np.ndarray | None = None
            if self.depletion_coupling == "band_average":
                # 用输入信号的功率谱做带内加权：泵浦消耗与逐频率增益同口径。
                n_samples = opt_signal.num_samples
                sample_rate = opt_signal.ctx.sample_rate
                freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
                band_freqs_hz = opt_signal.ctx.center_freq + np.fft.fftfreq(
                    n_samples, d=1.0 / sample_rate
                ) * freq_scale
                spectrum = np.mean(
                    np.abs(np.fft.fft(opt_signal.data.astype(np.complex128), axis=-1)) ** 2,
                    axis=0,
                )
                total = float(np.sum(spectrum))
                band_weights = spectrum / total if total > 0.0 else None
            (
                self._depleted_pump_z,
                self._depleted_pump_power,
                self._depleted_signal_power,
                self._depletion_diagnostics,
            ) = self._solve_depleted_profiles(
                input_power_w,
                opt_signal.ctx.center_freq,
                band_freqs_hz=band_freqs_hz,
                band_weights=band_weights,
            )

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
        if self.pump_model == "depleted" and self.pumps and self._depletion_diagnostics is not None:
            output.ctx.meta.update(
                {
                    "dra_pump_model_detail": (
                        "coupled signal/pump power ODE along z (RK4), signal represented by its "
                        "in-band total power; depletion coupling "
                        + (
                            "taken at the centre frequency"
                            if self.depletion_coupling == "center"
                            else "is the signal-PSD weighted band average"
                        )
                        + "; backward pumps solved by fixed-point iteration; "
                        "no pump-pump Raman transfer"
                    ),
                    "dra_depletion_coupling": self.depletion_coupling,
                    "dra_pump_energy_balance": self._depletion_diagnostics,
                    "dra_depletion_converged": bool(self._depletion_diagnostics["converged"]),
                    "dra_depletion_iterations": int(self._depletion_diagnostics["iterations"]),
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
    "DEPLETION_COUPLINGS",
    "PUMP_MODELS",
    "DistributedRamanFiber",
    "raman_gain_spectrum",
]
