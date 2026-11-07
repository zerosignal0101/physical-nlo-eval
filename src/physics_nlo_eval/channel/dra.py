"""分布式拉曼放大 (DRA) 的波形级参考模型。

本模块提供一个**独立的 SSFM 波形传播器**（``DistributedRamanFiber``），
用于给解析 DRA 模型（如 ``network-nlo-eval`` 的 Raman ODE）提供逐通道
波形参考。它复用 :class:`~physics_nlo_eval.channel.fiber.FiberChannel`
的 NumPy 对称分步傅里叶求解器，并在每个 SSFM 步施加频率相关的拉曼增益。

物理假设（详见 ``docs/dra-waveform.md``）：

* 支持两种泵浦模型：``"undepleted"``（默认）泵浦功率沿 z 解析衰减，不受
  信号和泵浦间相互作用影响；``"depleted"`` 在 SSFM 步进之外用 RK4 积分
  泵浦-信号功率耦合方程，泵浦按局部信号功率被耗尽，信号增益使用局部泵浦
  功率，后向泵用不动点迭代求解边值问题。耗尽模型的信号驱动有两种口径
  （``depletion_model``）：``"scalar"``（默认）用带内总功率标量，
  ``"spectral"`` 用逐频箱功率分布，与 SSFM 的逐频率增益同口径；标量口径
  还可选 ``depletion_coupling="band_average"`` 用带内功率谱加权平均
  （见 ``docs/dra-waveform.md`` §13）；
* 泵浦之间默认无耦合；``pump_pump_raman=True``（只对 ``"depleted"`` 且
  ``depletion_model="spectral"`` 有效）时按成对光子数守恒加入泵浦-泵浦
  拉曼转移（见 ``docs/dra-waveform.md`` §13.8/§13.9）；
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
#: 耗尽方程里信号驱动项的取法。
#:
#: * ``"scalar"``：历史口径，信号压成一个带内总功率标量；
#: * ``"spectral"``：逐频箱保留信号功率分布，泵浦 ODE 与 SSFM 的逐频率
#:   增益同口径。
DEPLETION_MODELS = ("scalar", "spectral")
#: 标量耗尽模型里泵浦-信号耦合系数的取法（``"spectral"`` 不使用）。
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


def signal_spectrum_bins(
    waveform: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把输入波形的 FFT 功率谱聚合成频箱，供谱域耗尽模型使用。

    频率轴与 SSFM 完全一致
    （``center_freq + fftfreq(n, 1/fs) · n/(n−1)``），功率谱取 ``|FFT|²``
    沿偏振轴的均值。随后把频率轴等宽切成 ``n_bins`` 个箱，只保留
    有功率的箱；每箱用**功率加权平均频率**代表，权重归一化到 1。

    Parameters
    ----------
    waveform : ndarray
        输入波形，形状 ``(n_pol, n_samples)``。
    sample_rate_hz : float
        采样率 (Hz)。
    center_freq_hz : float
        中心光频率 (Hz)。
    n_bins : int
        频箱数量，必须为正。

    Returns
    -------
    bin_freqs_hz : ndarray
        每个有效频箱的功率加权平均频率 (Hz)，形状 ``(n_bins_eff,)``。
    bin_weights : ndarray
        归一化到 1 的功率权重，形状 ``(n_bins_eff,)``。总信号功率为
        ``input_power_w · bin_weights``。

    Notes
    -----
    ``input_power_w = sum(mean(|A|², axis=1))``；FFT 功率谱只用来给出
    **形状**（各箱相对权重），因此 Parseval 常数不影响归一化。空箱
    （功率严格为 0）会被丢弃；全零信号退化为中心频率单箱。
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")
    data = np.asarray(waveform)
    n_samples = data.shape[-1]
    freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
    freqs = float(center_freq_hz) + np.fft.fftfreq(
        n_samples, d=1.0 / float(sample_rate_hz)
    ) * freq_scale
    spectrum = np.mean(np.abs(np.fft.fft(data, axis=-1)) ** 2, axis=0)
    total = float(np.sum(spectrum))
    if total <= 0.0 or n_samples < 2:
        return np.array([float(center_freq_hz)]), np.array([1.0])

    fmin = float(np.min(freqs))
    fmax = float(np.max(freqs))
    if n_bins == 1 or fmax <= fmin:
        weighted = float(np.dot(freqs, spectrum)) / total
        return np.array([weighted]), np.array([1.0])

    edges = np.linspace(fmin, fmax, n_bins + 1)
    # 内边界用 digitize；最大频率必须归入最后一箱。
    index = np.clip(
        np.digitize(freqs, edges[1:-1], right=False), 0, n_bins - 1
    )
    power = np.bincount(index, weights=spectrum, minlength=n_bins)
    occupied = power > 0.0
    if not np.any(occupied):
        return np.array([float(center_freq_hz)]), np.array([1.0])
    weighted = np.bincount(index, weights=spectrum * freqs, minlength=n_bins)
    bin_freqs = weighted[occupied] / power[occupied]
    bin_weights = power[occupied] / float(np.sum(power[occupied]))
    return bin_freqs, bin_weights


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
        标量耗尽模型里泵浦-信号耦合系数 ``γ_p`` 的取法：

        * ``"center"``（默认）：取信号中心频率的 ``γ_p``。这是历史口径，
          与既有参考数据逐位一致；
        * ``"band_average"``：取信号带内功率谱加权的 ``γ_p`` 平均值，使泵浦
          耗尽量与逐频率施加的信号增益使用同一平均耦合。窄带/CW 时两者
          趋于一致；宽带时中心频率可能落在拉曼峰上或远离峰值，两者差别
          可达 1–2 dB（见 ``docs/dra-waveform.md`` 的剖面诊断一节）。

        该参数只对 ``depletion_model="scalar"`` 有意义；``"spectral"`` 时
        直接由逐频箱的 ``γ_{p,b}`` 驱动，``depletion_coupling`` 被忽略，
        元数据里记为 ``None`` 并显式注明。
    depletion_model : str, optional
        泵浦耗尽方程的**信号驱动口径**：

        * ``"scalar"``（默认）：信号压成一个带内总功率标量 ``P_s(z)``，
          泵浦耗尽量用中心频率或带内平均耦合（见 ``depletion_coupling``）。
          与历史参考数据逐位一致；
        * ``"spectral"``：把输入信号功率谱聚合成 ``spectral_bins`` 个频箱，
          每个频箱独立演化 ``P_{s,b}(z)``，泵浦行按
          ``Σ_b (f_p/f_b) γ_{p,b} P_{s,b}`` 耗尽。这样泵浦 ODE 与 SSFM 的
          逐频率增益严格同口径，是启用 ``pump_pump_raman`` 的前提。

        只对 ``pump_model="depleted"`` 生效。
    spectral_bins : int, optional
        谱域耗尽模型的频箱数，默认 256，必须为正。箱按频率等宽划分，
        每箱以功率加权平均频率代表，空箱丢弃（权重为 0）。
    pump_pump_raman : bool, optional
        是否在耗尽模型里计入泵浦-泵浦拉曼转移。默认 ``False``，此时泵浦 ODE
        与历史参考数据逐位一致。只对 ``pump_model="depleted"`` 生效；与
        ``pump_model="undepleted"`` 同时使用会显式报错（未耗尽剖面是解析
        衰减，没有可以加转移项的状态量），而不是静默忽略。另要求
        ``depletion_model="spectral"``：标量信号驱动会低估强增益信道对
        泵浦的耗尽，打开泵浦-泵浦转移后会把这一缺陷放大（见
        ``docs/dra-waveform.md`` §13.8/§13.9）。

        每一对泵 ``(i, j)``（``f_i > f_j``）用成对光子数守恒的形式::

            R_ij = g_R(f_i − f_j) / A_eff · P_i · P_j
            dP_i += −(f_i / f_j) · R_ij      # 高频泵失去一个泵浦光子
            dP_j += +R_ij                    # 低频泵获得功率

        ``g_R`` 与信号增益共用同一 Gaussian 谱参数与 ``a_eff_si``。前向/后向
        泵的 RK4 都包含该项；交叉项（前向↔后向）在每方向用对方的上一轮
        不动点剖面，保持原有迭代结构。该项只改变泵浦剖面，信号仍按局部
        泵浦功率逐频率取增益。
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
        depletion_model: str = "scalar",
        spectral_bins: int = 256,
        pump_pump_raman: bool = False,
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
        if depletion_model not in DEPLETION_MODELS:
            raise ValueError(
                f"depletion_model must be one of {DEPLETION_MODELS}, "
                f"got {depletion_model!r}."
            )
        if int(spectral_bins) < 1:
            raise ValueError("spectral_bins must be a positive integer.")
        if pump_pump_raman and pump_model != "depleted":
            raise ValueError(
                "pump_pump_raman=True requires pump_model='depleted': the undepleted "
                "pump profile is analytic and has no pump state to couple."
            )
        if pump_pump_raman and depletion_model != "spectral":
            raise ValueError(
                "pump_pump_raman=True requires depletion_model='spectral': the scalar "
                "depletion drive only knows the in-band total signal power and "
                "under-estimates pump depletion when per-channel gain slopes are steep. "
                "Enable the per-bin ('spectral') depletion drive first."
            )
        if depletion_model == "spectral" and depletion_coupling != "center":
            warnings.warn(
                "depletion_model='spectral' drives the pump ODE per frequency bin; "
                "depletion_coupling is ignored (the per-bin gamma is used instead).",
                RuntimeWarning,
                stacklevel=2,
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
        self.depletion_model = depletion_model
        self.spectral_bins = int(spectral_bins)
        self.pump_pump_raman = bool(pump_pump_raman)
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

    def _pump_coupling_rates_spectrum(
        self, bin_freqs_hz: np.ndarray
    ) -> np.ndarray:
        """逐频箱的泵浦-信号功率耦合系数 (1/(W·m))，形状 ``(n_pumps, n_bins)``。

        返回 ``γ_{p,b} = g_peak · g̃_R(f_p − f_b) / A_eff``。这是谱域耗尽
        模型的核心：每个频箱用自己的耦合系数，与 SSFM 在 ``f_b`` 处逐频率
        施加的增益完全同口径。
        """
        freqs = np.asarray(bin_freqs_hz, dtype=float)
        if not self.pumps:
            return np.zeros((0, freqs.size), dtype=float)
        pump_freqs = np.array([self._pump_frequency_hz(pump) for pump in self.pumps])
        spectral = raman_gain_spectrum(
            pump_freqs[:, None] - freqs[None, :],
            peak_shift_hz=self.raman_peak_shift_hz,
            fwhm_hz=self.raman_fwhm_hz,
        )
        return self.raman_peak_gain_m_per_w * spectral / self.a_eff_si

    def _pump_pump_coupling(self) -> np.ndarray | None:
        """泵浦-泵浦拉曼转移的耦合矩阵 (1/(W·m))。

        返回 ``(n_pumps, n_pumps)`` 矩阵 ``C``，使得前向口径的泵浦功率导数
        为 ``dP_i/dz = P_i · Σ_j C_ij P_j``。对每一对 ``i≠j``：

        * ``f_j > f_i``：``C_ij = +g_R(f_j − f_i)/A_eff``（低频泵从高频泵
          获得功率）；
        * ``f_j < f_i``：``C_ij = −(f_i/f_j)·g_R(f_i − f_j)/A_eff``（高频泵
          失去一个泵浦光子，功率减少按频率比放大）。

        因此 ``d(P_i/f_i) + d(P_j/f_j) = 0``：峰值增益 ``g_R`` 用与信号增益
        相同的 Gaussian 谱参数（``raman_peak_shift_hz``/``raman_fwhm_hz``/
        ``raman_peak_gain_m_per_w``）与同一个 ``a_eff_si``，差值
        ``(f_i − f_j)/f_j · g_R P_i P_j`` 以声子形式离开光场。

        ``pump_pump_raman=False`` 或泵浦少于两个时返回 ``None``，调用方据此
        完全跳过该项（保持旧数值路径逐位不变）。
        """
        if not self.pump_pump_raman or self.n_pumps < 2:
            return None
        pump_freqs = np.array([self._pump_frequency_hz(pump) for pump in self.pumps])
        delta = np.abs(pump_freqs[:, None] - pump_freqs[None, :])
        gain = (
            self.raman_peak_gain_m_per_w
            * raman_gain_spectrum(
                delta,
                peak_shift_hz=self.raman_peak_shift_hz,
                fwhm_hz=self.raman_fwhm_hz,
            )
            / self.a_eff_si
        )
        # 源频率更高（f_j > f_i）时接收波获得功率，否则按 f_i/f_j 付出功率。
        source_higher = pump_freqs[None, :] > pump_freqs[:, None]
        coupling = np.where(
            source_higher, gain, -(pump_freqs[:, None] / pump_freqs[None, :]) * gain
        )
        np.fill_diagonal(coupling, 0.0)
        return coupling

    def _solve_depleted_profiles(
        self,
        input_power_w: float,
        signal_freq_hz: float,
        *,
        band_freqs_hz: np.ndarray | None = None,
        band_weights: np.ndarray | None = None,
        bin_freqs_hz: np.ndarray | None = None,
        bin_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """求解泵浦-信号耦合方程，返回泵浦剖面与能量账。

        按 ``depletion_model`` 分派：

        * ``"scalar"``（默认）走本节描述的标量带内总功率 ODE，数值路径与
          历史参考数据逐位一致；
        * ``"spectral"`` 走 :meth:`_solve_depleted_profiles_spectral`，
          信号由 ``bin_freqs_hz`` / ``bin_weights`` 描述的逐频箱功率驱动。

        标量模型的信号只保留一个剖面——带内总功率（两个偏振合计）。
        沿 +z 的耦合方程::

            dP_s/dz = −α_s P_s + P_s · Σ_p γ_p P_p
            dP_p/dz = −s_p · [α_p P_p + (f_p/f_s) γ_p P_p P_s],  s_p = ±1

        ``s_p=+1`` 为前向泵（在 z=0 注入），``s_p=−1`` 为后向泵（在 z=L
        注入）。``γ_p`` 由 ``depletion_coupling`` 决定：``"center"`` 取信号
        中心频率，忽略 WDM 带宽内拉曼谱的缓慢变化；``"band_average"`` 用
        ``band_weights`` 对带内频点做功率加权平均（见
        :meth:`_pump_coupling_rates_band`）。光子数因子 ``f_p/f_s`` 始终用
        中心频率（带内变化 <1%，属于二阶效应）。信号损耗 ``α_s`` 取
        ``alpha_np_inv_m`` 的均值（参考数据只有标量损耗）。

        打开 ``pump_pump_raman`` 时泵浦行再加成对转移项（``f_i > f_j``）::

            R_ij = g_R(f_i − f_j) / A_eff · P_i P_j
            dP_i/dz += −(f_i/f_j) R_ij,  dP_j/dz += +R_ij

        即 ``d(P_i/f_i) + d(P_j/f_j) = 0``，差额 ``(f_i − f_j)/f_j · R_ij``
        作为声子能量离开光场。实现上把它写成"沿传播方向的泵浦损耗率"
        ``−P_i · Σ_j C_ij P_j``（``C_ij`` 见 :meth:`_pump_pump_coupling`），
        再与其他两项一起乘 ``−s_p``：前向泵得到 ``+dP_i/dz``；后向泵沿 −z
        传播，同一份增益对应 ``dP_i/dz`` 取反。

        前向泵从 z=0 正向积分；后向泵从 z=L 反向积分。二者通过不动点
        迭代闭合边值问题：先用当前信号剖面积分后向泵，再用更新后的泵浦
        剖面正向积分信号与前向泵。纯前向泵时第一次迭代即收敛。泵浦-泵浦
        的交叉项（前向泵 ↔ 后向泵）在两个方向各用对方的上一轮剖面，保持
        这一不动点结构。
        """
        if self.depletion_model == "spectral":
            if bin_freqs_hz is None or bin_weights is None:
                raise ValueError(
                    "depletion_model='spectral' requires bin_freqs_hz and bin_weights; "
                    "use signal_spectrum_bins() to build them from the input waveform."
                )
            return self._solve_depleted_profiles_spectral(
                input_power_w,
                np.asarray(bin_freqs_hz, dtype=float),
                np.asarray(bin_weights, dtype=float),
            )

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

        # 泵浦-泵浦耦合矩阵；``None`` 时整项跳过（旧数值路径逐位不变）。
        # RHS 里的贡献写成“沿传播方向的损耗率” −P_i·Σ_j C_ij P_j，与 α、耗尽
        # 两项同号，一起乘 −s_p。
        pp_coupling = self._pump_pump_coupling()
        if pp_coupling is not None:
            pp_rows_fwd = pp_coupling[fwd_idx]
            pp_rows_bwd = pp_coupling[bwd_idx]

        def backward_rhs(zz: float, y: np.ndarray) -> np.ndarray:
            signal = np.interp(zz, z, ps)
            d_pumps = alphas[bwd_idx] * y + depletion[bwd_idx] * y * signal
            if pp_coupling is not None:
                total = np.zeros(self.n_pumps, dtype=float)
                total[bwd_idx] = y
                if fwd_idx.size:
                    total[fwd_idx] = [np.interp(zz, z, profile[i]) for i in fwd_idx]
                d_pumps = d_pumps - y * (pp_rows_bwd @ total)
            return -signs[bwd_idx] * d_pumps

        def forward_rhs(zz: float, y: np.ndarray) -> np.ndarray:
            signal = y[0]
            forward_powers = y[1:]
            total = np.zeros(self.n_pumps, dtype=float)
            total[fwd_idx] = forward_powers
            if bwd_idx.size:
                total[bwd_idx] = [np.interp(zz, z, profile[i]) for i in bwd_idx]
            d_signal = (gamma @ total - alpha_s) * signal
            d_pumps = (
                alphas[fwd_idx] * forward_powers
                + depletion[fwd_idx] * forward_powers * signal
            )
            if pp_coupling is not None:
                d_pumps = d_pumps - forward_powers * (pp_rows_fwd @ total)
            return np.concatenate(([d_signal], -signs[fwd_idx] * d_pumps))

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
        # 泵浦-泵浦能量账：逐对算 dP_i/dz 的贡献，再按符号拆成“送出/获得”。
        # contrib[i, j] = C_ij · P_i · P_j（i 从 j 处得到的 dP_i/dz 份额）。
        if pp_coupling is None:
            pp_sent = np.zeros(self.n_pumps)
            pp_received = np.zeros(self.n_pumps)
        else:
            contrib = pp_coupling[:, :, None] * profile[:, None, :] * profile[None, :, :]
            pp_sent = _trapezoid(np.maximum(-contrib, 0.0).sum(axis=1), z, axis=1)
            pp_received = _trapezoid(np.maximum(contrib, 0.0).sum(axis=1), z, axis=1)
        pp_net = pp_received - pp_sent
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
                    "pump_pump_sent_w": float(pp_sent[index]),
                    "pump_pump_received_w": float(pp_received[index]),
                    "pump_pump_net_w": float(pp_net[index]),
                }
            )
        diagnostics = {
            "pumps": pumps_diag,
            "signal_raman_gain_w": signal_gain,
            "iterations": iterations,
            "converged": converged,
            "step_m": dz,
            "pump_pump_raman": self.pump_pump_raman,
            # Σ 逐泵净收益 < 0：差额以声子能量离开光场。
            "pump_pump_net_total_w": float(np.sum(pp_net)),
            # 新增诊断键；标量路径的数值不变。
            "signal_out_w": float(ps[-1]),
            "spectral_bins": None,
            "depletion_model": "scalar",
        }
        return z, profile, ps, diagnostics

    def _solve_depleted_profiles_spectral(
        self,
        input_power_w: float,
        bin_freqs_hz: np.ndarray,
        bin_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """求解逐频箱的泵浦-信号耦合方程，返回泵浦剖面与能量账。

        信号状态是 ``(n_bins,)`` 的功率向量，泵浦状态是 ``(n_pumps,)``。
        沿 +z 的耦合方程::

            dP_{s,b}/dz = (Σ_p γ_{p,b} P_p − α_s) P_{s,b}
            dP_p/dz     = −s_p [α_p P_p
                                 + P_p Σ_b (f_p/f_b) γ_{p,b} P_{s,b}
                                 + P_p Σ_j C_{pj} P_j]

        ``γ_{p,b}`` 见 :meth:`_pump_coupling_rates_spectrum`；``C`` 与标量
        模型共用 :meth:`_pump_pump_coupling`（成对光子数守恒，符号/频率
        因子不变）。初值用未耗尽泵浦剖面和逐箱速率积分得到，因此后向泵
        第一次迭代就带上了信号的频谱分布，而不是退回标量初值。

        能量账与标量模型口径一致：``transferred_to_signal_w[p]`` 是
        ``∫ P_p Σ_b (f_p/f_b) γ_{p,b} P_{s,b} dz``；另有
        ``signal_photon_gain_per_s`` 与 ``pump_photon_transfer_per_s``
        用于核对逐箱光子数守恒。
        """
        bin_freqs = np.asarray(bin_freqs_hz, dtype=float)
        bin_weights = np.asarray(bin_weights, dtype=float)
        if bin_freqs.ndim != 1 or bin_freqs.shape != bin_weights.shape or bin_freqs.size == 0:
            raise ValueError("bin_freqs_hz and bin_weights must be non-empty 1D arrays of equal length.")
        if np.any(bin_freqs <= 0.0):
            raise ValueError("bin_freqs_hz must be positive.")
        total_weight = float(np.sum(bin_weights))
        if total_weight <= 0.0:
            raise ValueError("bin_weights must sum to a positive value.")
        bin_weights = bin_weights / total_weight
        n_bins = int(bin_freqs.size)

        n_steps = max(1, int(np.ceil(self.length_m / self.depleted_step_m)))
        dz = self.length_m / n_steps
        z = np.linspace(0.0, self.length_m, n_steps + 1)

        pump_freqs = np.array([self._pump_frequency_hz(pump) for pump in self.pumps])
        alphas = np.array([self._pump_alpha_np_per_m(pump) for pump in self.pumps])
        gamma = self._pump_coupling_rates_spectrum(bin_freqs)  # (n_pumps, n_bins)
        # 泵浦行按光子数守恒放大 (f_p/f_b)，逐箱计算。
        depletion_rate = gamma * (pump_freqs[:, None] / bin_freqs[None, :])
        forward_mask = np.array(
            [pump.direction == "forward" for pump in self.pumps], dtype=bool
        )
        fwd_idx = np.flatnonzero(forward_mask)
        bwd_idx = np.flatnonzero(~forward_mask)
        signs = np.where(forward_mask, 1.0, -1.0)
        alpha_s = float(np.mean(np.asarray(self.alpha_np_inv_m, dtype=float)))

        profile = self._undepleted_pump_power_w(z)
        # 初值：未耗尽泵浦剖面 + 逐箱速率，积分出完整的 ps(z)，供后向泵使用。
        rate = gamma.T @ profile - alpha_s  # (n_bins, n_steps+1)
        exponent = np.concatenate(
            (
                np.zeros((n_bins, 1)),
                np.cumsum(0.5 * (rate[:, :-1] + rate[:, 1:]) * dz, axis=1),
            ),
            axis=1,
        )
        ps0 = (input_power_w * bin_weights)[:, None]
        ps = ps0 * np.exp(exponent)

        pp_coupling = self._pump_pump_coupling()
        if pp_coupling is not None:
            pp_rows_fwd = pp_coupling[fwd_idx]
            pp_rows_bwd = pp_coupling[bwd_idx]
        # 后向泵 + 泵浦-泵浦转移时，从 z=L 反向积分 dP/dz=+λP 是强不稳定的：
        # 直接替换整条剖面会让不动点出现周期 2 振荡。这里对后向泵剖面做
        # 欠松弛；没有泵浦-泵浦项时保持原样（收敛很快）。
        backward_relaxation = 0.5 if pp_coupling is not None else 1.0

        def interpolate_signal(zz: float) -> np.ndarray:
            """把 ``ps``（逐箱）线性插值到 ``zz``。"""
            index = int(np.searchsorted(z, zz, side="right") - 1)
            index = min(max(index, 0), n_steps - 1)
            weight = (zz - z[index]) / dz
            return ps[:, index] * (1.0 - weight) + ps[:, index + 1] * weight

        def backward_rhs(zz: float, y: np.ndarray) -> np.ndarray:
            signal = interpolate_signal(zz)
            d_pumps = alphas[bwd_idx] * y + y * (depletion_rate[bwd_idx] @ signal)
            if pp_coupling is not None:
                total = np.zeros(self.n_pumps, dtype=float)
                total[bwd_idx] = y
                if fwd_idx.size:
                    total[fwd_idx] = [np.interp(zz, z, profile[i]) for i in fwd_idx]
                d_pumps = d_pumps - y * (pp_rows_bwd @ total)
            return -signs[bwd_idx] * d_pumps

        def forward_rhs(zz: float, y: np.ndarray) -> np.ndarray:
            signal = y[:n_bins]
            forward_powers = y[n_bins:]
            total = np.zeros(self.n_pumps, dtype=float)
            total[fwd_idx] = forward_powers
            if bwd_idx.size:
                total[bwd_idx] = [np.interp(zz, z, profile[i]) for i in bwd_idx]
            d_signal = (gamma.T @ total - alpha_s) * signal
            d_pumps = (
                alphas[fwd_idx] * forward_powers
                + forward_powers * (depletion_rate[fwd_idx] @ signal)
            )
            if pp_coupling is not None:
                d_pumps = d_pumps - forward_powers * (pp_rows_fwd @ total)
            return np.concatenate((d_signal, -signs[fwd_idx] * d_pumps))

        nominal = np.maximum(
            np.array([pump.power_w for pump in self.pumps], dtype=float), 1e-30
        )
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
                if backward_relaxation >= 1.0:
                    profile[bwd_idx] = updated
                else:
                    profile[bwd_idx] = (
                        (1.0 - backward_relaxation) * previous_profile[bwd_idx]
                        + backward_relaxation * updated
                    )

            y = np.concatenate((input_power_w * bin_weights, profile[fwd_idx, 0]))
            updated_y = np.empty((n_bins + fwd_idx.size, n_steps + 1))
            updated_y[:, 0] = y
            for k in range(n_steps):
                z0 = z[k]
                k1 = forward_rhs(z0, y)
                k2 = forward_rhs(z0 + dz / 2, y + 0.5 * dz * k1)
                k3 = forward_rhs(z0 + dz / 2, y + 0.5 * dz * k2)
                k4 = forward_rhs(z0 + dz, y + dz * k3)
                y = y + dz * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                updated_y[:, k + 1] = y
            ps = updated_y[:n_bins]
            profile[fwd_idx] = updated_y[n_bins:]

            pump_error = float(
                np.max(np.abs(profile - previous_profile) / nominal[:, None])
            )
            signal_error = float(
                np.max(np.abs(ps - previous_ps)) / max(input_power_w, 1e-30)
            )
            if max(pump_error, signal_error) < self.depleted_tolerance:
                converged = True
                break
        passive_loss = _trapezoid(alphas[:, None] * profile, z, axis=1)
        transfer = _trapezoid(profile * (depletion_rate @ ps), z, axis=1)
        gain_per_bin = gamma.T @ profile  # (n_bins, n_steps+1)
        signal_gain = float(_trapezoid(np.sum(gain_per_bin * ps, axis=0), z))
        signal_photon_gain = float(
            _trapezoid(np.sum(gain_per_bin * ps / bin_freqs[:, None], axis=0), z)
        )
        pump_photon_transfer = float(np.sum(transfer / pump_freqs))
        if pp_coupling is None:
            pp_sent = np.zeros(self.n_pumps)
            pp_received = np.zeros(self.n_pumps)
        else:
            contrib = pp_coupling[:, :, None] * profile[:, None, :] * profile[None, :, :]
            pp_sent = _trapezoid(np.maximum(-contrib, 0.0).sum(axis=1), z, axis=1)
            pp_received = _trapezoid(np.maximum(contrib, 0.0).sum(axis=1), z, axis=1)
        pp_net = pp_received - pp_sent
        pumps_diag = []
        for index, pump in enumerate(self.pumps):
            launch = float(
                profile[index, 0] if pump.direction == "forward" else profile[index, -1]
            )
            residual = float(
                profile[index, -1] if pump.direction == "forward" else profile[index, 0]
            )
            pumps_diag.append(
                {
                    "direction": pump.direction,
                    "launch_power_w": launch,
                    "residual_power_w": residual,
                    "passive_loss_w": float(passive_loss[index]),
                    "transferred_to_signal_w": float(transfer[index]),
                    "pump_pump_sent_w": float(pp_sent[index]),
                    "pump_pump_received_w": float(pp_received[index]),
                    "pump_pump_net_w": float(pp_net[index]),
                }
            )
        diagnostics = {
            "pumps": pumps_diag,
            "signal_raman_gain_w": signal_gain,
            "iterations": iterations,
            "converged": converged,
            "step_m": dz,
            "pump_pump_raman": self.pump_pump_raman,
            "pump_pump_net_total_w": float(np.sum(pp_net)),
            "signal_out_w": float(np.sum(ps[:, -1])),
            "spectral_bins": n_bins,
            "depletion_model": "spectral",
            "signal_photon_gain_per_s": signal_photon_gain,
            "pump_photon_transfer_per_s": pump_photon_transfer,
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
            bin_freqs_hz: np.ndarray | None = None
            bin_weights: np.ndarray | None = None
            if self.depletion_model == "spectral":
                # 逐频箱功率分布：与 SSFM 逐频率增益同口径，驱动泵浦耗尽。
                bin_freqs_hz, bin_weights = signal_spectrum_bins(
                    opt_signal.data.astype(np.complex128),
                    opt_signal.ctx.sample_rate,
                    opt_signal.ctx.center_freq,
                    self.spectral_bins,
                )
            elif self.depletion_coupling == "band_average":
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
                bin_freqs_hz=bin_freqs_hz,
                bin_weights=bin_weights,
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
            if self.depletion_model == "spectral":
                detail = (
                    "coupled signal/pump power ODE along z (RK4), signal represented by "
                    f"{self._depletion_diagnostics['spectral_bins']} frequency bins with "
                    "per-bin depletion coupling; depletion_coupling is not used; "
                    "backward pumps solved by fixed-point iteration; "
                    + (
                        "pump-pump Raman transfer included (pairwise photon-number "
                        "conserving)"
                        if self.pump_pump_raman
                        else "no pump-pump Raman transfer"
                    )
                )
                coupling_meta = None
            else:
                detail = (
                    "coupled signal/pump power ODE along z (RK4), signal represented by its "
                    "in-band total power; depletion coupling "
                    + (
                        "taken at the centre frequency"
                        if self.depletion_coupling == "center"
                        else "is the signal-PSD weighted band average"
                    )
                    + "; backward pumps solved by fixed-point iteration; "
                    + (
                        "pump-pump Raman transfer included (pairwise photon-number "
                        "conserving)"
                        if self.pump_pump_raman
                        else "no pump-pump Raman transfer"
                    )
                )
                coupling_meta = self.depletion_coupling
            output.ctx.meta.update(
                {
                    "dra_pump_model_detail": detail,
                    "dra_depletion_model": self.depletion_model,
                    "dra_spectral_bins": self._depletion_diagnostics["spectral_bins"],
                    "dra_depletion_coupling": coupling_meta,
                    "dra_pump_pump_raman": bool(self.pump_pump_raman),
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
    "DEPLETION_MODELS",
    "PUMP_MODELS",
    "DistributedRamanFiber",
    "raman_gain_spectrum",
    "signal_spectrum_bins",
]
