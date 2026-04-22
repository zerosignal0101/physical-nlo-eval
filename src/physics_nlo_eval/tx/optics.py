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
    为了简化，我们假设 IQModulator 已经处理了偏振复用（输出 N_pol, N_samples）。
    此模块仅作为一个概念上的占位符。
    """

    def __init__(self) -> None:
        """初始化 PolarizationMultiplexer。"""
        super().__init__(name="PolarizationMultiplexer")

    def forward(self, optical_signal: Signal) -> Signal:
        """
        处理光信号。

        如果输入已经是双偏振，则直接返回。
        如果输入是单偏振，此模块无法将其变为双偏振，会报错。

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
        if optical_signal.ctx.n_pol == 2:
            return optical_signal
        # Realistically, this should take two SP signals and combine them.
        # Given our current Signal structure, a single Signal already has n_pol.
        # This implies that a DP signal is generated directly by Mapper/IQModulator.
        # So this module mainly serves to validate if the signal is indeed DP.
        raise ValueError(
            "Cannot perform polarization multiplexing on a single-polarization signal from this module. "
            "Ensure your Mapper/IQModulator is configured for DP output if needed."
        )
