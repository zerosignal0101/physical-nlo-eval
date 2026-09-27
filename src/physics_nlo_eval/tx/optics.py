"""实现电光调制和偏振复用。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class IQModulator(BaseBlock):
    """IQ 调制器，将归一化数字基带信号转换为具有物理功率的光信号。

    PyNLO 的 Manakov 方程要求输入信号振幅单位为 sqrt(W)。
    此模块假设输入数字信号的平均功率为 1，将其缩放至指定的发射功率。

    Parameters
    ----------
    launch_power_dbm_per_pol : float
        每个偏振的发射光功率。
        对于 DP 信号，总功率是该值的 2 倍。
        合理范围通常为 -5 ~ 5 dBm (对应 0.3 ~ 3 mW)。
    wavelength_nm : float
        激光器的中心波长 (纳米)。
    """

    def __init__(
        self,
        launch_power_dbm_per_pol: float = 0.0,  # 默认 0 dBm = 1 mW
        wavelength_nm: float = 1550.0,
    ) -> None:
        """初始化 IQModulator。"""
        super().__init__(name="IQModulator")
        self.launch_power_dbm_per_pol = launch_power_dbm_per_pol
        self.wavelength_nm = wavelength_nm

        # Constants
        self.c = 2.99792458e8  # Speed of light (m/s)

    def forward(self, electrical_signal: Signal) -> Signal:
        """
        将基带电信号转换为物理单位的光信号包络。

        Parameters
        ----------
        electrical_signal : Signal
            输入的归一化基带电信号。期望平均功率约为 1。

        Returns
        -------
        Signal
            输出的光信号。振幅单位为 sqrt(W)，可直接输入 FiberChannel。
            SignalContext 会更新 center_freq 和 meta 中的功率信息。
        """
        if electrical_signal.data.ndim != 2:
            raise ValueError("Electrical signal data must be 2D (n_pol, N_samples).")
        if not np.iscomplexobj(electrical_signal.data):
            raise ValueError("Electrical signal data must be complex for IQ modulation.")

        n_pol = electrical_signal.ctx.n_pol

        # 1. 计算目标线性功率
        # P_linear(W) = 10^(P_dBm/10) * 1e-3
        target_power_per_pol_w = 10 ** (self.launch_power_dbm_per_pol / 10) * 1e-3
        amplitude_scale = np.sqrt(target_power_per_pol_w)

        # 2. 归一化输入信号至单位平均功率，然后缩放至目标功率
        # 这确保了无论前端 Mapper 输出功率是否严格为1，物理功率都准确
        current_avg_power = np.mean(np.abs(electrical_signal.data) ** 2, axis=1, keepdims=True)
        current_avg_power[current_avg_power == 0] = 1.0  # 防止除零

        # 物理光场 E_phys = E_digital * sqrt(P_target / P_digital_avg)
        optical_field = electrical_signal.data * (amplitude_scale / np.sqrt(current_avg_power))

        # 3. 更新 SignalContext
        optical_center_freq = self.c / (self.wavelength_nm * 1e-9)  # Hz
        new_ctx = SignalContext(
            sample_rate=electrical_signal.ctx.sample_rate,
            center_freq=optical_center_freq,  # 光载波绝对频率
            n_pol=n_pol,
            sps=electrical_signal.ctx.sps,
            sym_rate=electrical_signal.ctx.sym_rate,
            bits_per_symbol=electrical_signal.ctx.bits_per_symbol,
            is_ofdm=electrical_signal.ctx.is_ofdm,
            meta={
                **(electrical_signal.ctx.meta or {}),
                "tx_launch_power_dbm_per_pol": self.launch_power_dbm_per_pol,
                "tx_wavelength_nm": self.wavelength_nm,
                "tx_power_per_pol_w": target_power_per_pol_w,
            },
        )
        # 保留复数精度，以匹配光纤仿真的需求
        return Signal(data=optical_field.astype(np.complex128), ctx=new_ctx)


class PolarizationMultiplexer(BaseBlock):
    """偏振复用器 (Dual Polarization)

    将两个单偏振光信号组合成一个双偏振信号。
    此模块通常在 IQModulator 之后，或由 IQModulator 内部完成。

    两种用法：

    - 直接调用 :meth:`combine` 合并两个 SP 信号；
    - 在 Pipeline 中把第二个 ``Signal`` 放入第一个信号的
      ``ctx.meta["orthogonal_polarization"]``，由 :meth:`forward` 取出合并。

    已经是双偏振的信号原样透传。
    """

    def __init__(self) -> None:
        """初始化 PolarizationMultiplexer。"""
        super().__init__(name="PolarizationMultiplexer")

    def forward(self, optical_signal: Signal) -> Signal:
        """
        处理光信号。

        如果输入已经是双偏振，则直接返回。
        如果输入是单偏振，则尝试从 ``ctx.meta["orthogonal_polarization"]``
        取出正交偏振信号并合并；取不到则报错。

        Parameters
        ----------
        optical_signal : Signal
            输入的单偏振或双偏振光信号。

        Returns
        -------
        Signal
            输出的双偏振光信号。

        Raises
        ------
        ValueError
            如果输入信号不是双偏振且无法进行偏振复用。
        """
        # 同时校验 n_pol 与数据行数：仅声明 n_pol=2 但只有一行数据的畸形信号
        # 以前会被静默透传，现在必须走 combine 路径。
        if optical_signal.ctx.n_pol == 2 and optical_signal.data.shape[0] == 2:
            return optical_signal
        second = optical_signal.ctx.meta.get("orthogonal_polarization")
        if isinstance(second, Signal):
            return self.combine(optical_signal, second)
        raise ValueError(
            "Cannot perform polarization multiplexing on a single-polarization signal from this module. "
            "Provide the second polarization via combine(x, y) or "
            "signal.ctx.meta['orthogonal_polarization']."
        )

    def combine(self, x_pol: Signal, y_pol: Signal) -> Signal:
        """
        将两个兼容的单偏振信号合并为一个双偏振信号。

        Parameters
        ----------
        x_pol, y_pol : Signal
            形状为 (1, N_samples) 且 ``n_pol == 1`` 的单偏振信号。
            两者的采样率、中心频率、sps 和 OFDM 标志必须一致。

        Returns
        -------
        Signal
            形状为 (2, N_samples)、``n_pol == 2`` 的双偏振信号。

        Raises
        ------
        ValueError
            输入形状/偏振态不合法，长度不一致，或上下文参数不兼容。
        """
        for name, value in (("x_pol", x_pol), ("y_pol", y_pol)):
            if value.data.ndim != 2 or value.data.shape[0] != 1 or value.ctx.n_pol != 1:
                raise ValueError(f"{name} must have shape (1, samples) and n_pol=1.")
        if x_pol.data.shape[-1] != y_pol.data.shape[-1]:
            raise ValueError("Polarization signals must have the same sample count.")
        for attr in ("sample_rate", "center_freq", "sps", "is_ofdm"):
            if getattr(x_pol.ctx, attr) != getattr(y_pol.ctx, attr):
                raise ValueError(f"Polarization signals have incompatible {attr}.")
        ctx = x_pol.ctx.copy()
        ctx.n_pol = 2
        ctx.meta.pop("orthogonal_polarization", None)
        ctx.meta["polarization_multiplexed"] = True
        return Signal(np.concatenate((x_pol.data, y_pol.data), axis=0), ctx)
