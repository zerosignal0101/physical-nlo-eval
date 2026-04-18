"""实现电光调制和偏振复用。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class IQModulator(BaseBlock):
    """IQ 调制器，将基带电信号转换为光信号。

    模拟理想的马赫曾德调制器 (Mach-Zehnder Modulator) 组合。

    Parameters
    ----------
    optical_power_per_pol_mw : float
        每个偏振的输出光功率 (毫瓦)。
        用于归一化输出光信号的幅度。
    wavelength_nm : float
        激光器的中心波长 (纳米)。
    """

    def __init__(
        self,
        optical_power_per_pol_mw: float = 10.0,
        wavelength_nm: float = 1550.0,
    ) -> None:
        """初始化 IQModulator。"""
        super().__init__(name="IQModulator")
        self.optical_power_per_pol_mw = optical_power_per_pol_mw
        self.wavelength_nm = wavelength_nm

        # Constants
        self.c = 2.99792458e8  # Speed of light (m/s)
        self.h = 6.62607015e-34  # Planck's constant (J·s)

    def forward(self, electrical_signal: Signal) -> Signal:
        """
        将基带电信号转换为光信号。

        假设输入 `electrical_signal` 为理想的复数基带电信号，
        其幅度已经代表了光场的复数包络。

        Parameters
        ----------
        electrical_signal : Signal
            输入的基带电信号。数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            输出的光信号。数据形状为 (n_pol, N_samples)。
            SignalContext 会更新 center_freq 为光载波频率。
        """
        if electrical_signal.data.ndim != 2:
            raise ValueError("Electrical signal data must be 2D (n_pol, N_samples).")
        if not np.iscomplexobj(electrical_signal.data):
            raise ValueError("Electrical signal data must be complex for IQ modulation.")

        n_pol = electrical_signal.ctx.n_pol
        num_samples = electrical_signal.num_samples

        # 确保偏振数量匹配
        if n_pol != electrical_signal.data.shape[0]:
            raise ValueError(
                f"SignalContext n_pol ({n_pol}) does not match data shape ({electrical_signal.data.shape[0]})."
            )

        # 计算光子能量和对应幅度归一化因子
        # 1 mW = 1e-3 J/s
        # P = |E|^2 / (2 * Z_0)
        # 这里我们假设 E 是光场复振幅，我们关注的是其归一化的功率，
        # 并将其归一化到期望的 mW 级别。
        # 如果 electrical_signal.data 的平均功率为 1，则输出光信号的平均功率为 optical_power_per_pol_mw。
        current_avg_power = np.mean(np.abs(electrical_signal.data) ** 2, axis=1, keepdims=True)
        # Avoid division by zero for zero-power signals
        current_avg_power[current_avg_power == 0] = 1.0

        # Scale factor to achieve desired optical power
        # target_power_linear = self.optical_power_per_pol_mw * 1e-3  # Convert mW to Watts
        # optical_field = electrical_signal.data * np.sqrt(target_power_linear / current_avg_power)
        # ^ This is for average power. Peak power might be different.
        # For simplicity, let's normalize the signal so its average power in the simulation units
        # corresponds to the optical_power_per_pol_mw.
        # If 'data' is already normalized to avg power 1 by Mapper, then just scale by sqrt(power_mW).

        # A_eff is usually 80 um^2 for SMF, n_eff ~ 1.45
        # The relationship between field amplitude |E| and power P is P = 0.5 * n_eff * epsilon_0 * c * |E|^2 * A_eff
        # This is for converting Pynlo units.
        # Here, we simplify to scale the input electrical field to achieve the desired optical power.
        # Assuming electrical_signal.data is already average power normalized to 1 by Mapper.
        # So, we just scale it by sqrt(power_mW_per_pol).
        scale_factor = np.sqrt(self.optical_power_per_pol_mw)  # In mW, for Pynlo compatibility
        optical_field = electrical_signal.data * scale_factor

        # 更新 SignalContext
        optical_center_freq = self.c / (self.wavelength_nm * 1e-9)  # Hz
        new_ctx = SignalContext(
            sample_rate=electrical_signal.ctx.sample_rate,
            center_freq=optical_center_freq,  # 光载波频率
            n_pol=n_pol,
            sps=electrical_signal.ctx.sps,
            sym_rate=electrical_signal.ctx.sym_rate,
            bits_per_symbol=electrical_signal.ctx.bits_per_symbol,
            is_ofdm=electrical_signal.ctx.is_ofdm,
            meta={
                **(electrical_signal.ctx.meta or {}),
                "tx_optical_power_per_pol_mw": self.optical_power_per_pol_mw,
                "tx_wavelength_nm": self.wavelength_nm,
            },
        )
        return Signal(data=optical_field.astype(np.complex64), ctx=new_ctx)


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
