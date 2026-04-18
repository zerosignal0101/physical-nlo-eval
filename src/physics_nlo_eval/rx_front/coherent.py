"""实现相干混合和光电探测。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class CoherentReceiver(BaseBlock):
    """相干接收机模块。

    模拟相干混合、理想光电探测器和本地振荡器 (LO) 相关的损伤。

    Parameters
    ----------
    lo_power_dbm : float, optional
        本地振荡器 (LO) 的光功率 (dBm)。
    lo_linewidth_hz : float, optional
        本地振荡器 (LO) 的激光线宽 (Hz)。用于生成 LO 相位噪声。
    lo_freq_offset_hz : float, optional
        本地振荡器 (LO) 相对于信号光载波的频率偏移 (Hz)。
    responsivity_a_per_w : float, optional
        光电探测器的响应度 (A/W)。
    """

    def __init__(
        self,
        lo_power_dbm: float = 13.0,
        lo_linewidth_hz: float = 100e3,
        lo_freq_offset_hz: float = 0.0,
        responsivity_a_per_w: float = 1.0,  # Ideal photodiode
    ) -> None:
        """初始化 CoherentReceiver。"""
        super().__init__(name="CoherentReceiver")
        self.lo_power_dbm = lo_power_dbm
        self.lo_linewidth_hz = lo_linewidth_hz
        self.lo_freq_offset_hz = lo_freq_offset_hz
        self.responsivity_a_per_w = responsivity_a_per_w

        self.lo_power_linear_w = 10 ** (self.lo_power_dbm / 10) * 1e-3  # Convert dBm to Watts
        # For simplicity, assume LO field amplitude is sqrt(Power) in simulation units (sqrt(mW))
        # This will be adjusted when converting input optical signal power from mW to W.

    def _generate_lo_phase_noise(self, num_samples: int, sample_rate: float) -> np.ndarray:
        """
        生成 LO 的相位噪声 (维纳过程)。

        Parameters
        ----------
        num_samples : int
            噪声序列的长度。
        sample_rate : float
            采样率 (Hz)。

        Returns
        -------
        np.ndarray
            LO 相位噪声数组 (弧度)。
        """
        # Wiener process (random walk) for phase noise
        # Variance of phase increment is 2 * pi * linewidth * dt
        dt = 1 / sample_rate
        phase_increment_std = np.sqrt(2 * np.pi * self.lo_linewidth_hz * dt)
        phase_noise = np.cumsum(np.random.normal(0, phase_increment_std, num_samples))
        return phase_noise

    def forward(self, rx_opt_signal: Signal) -> Signal:
        """
        执行相干混合和光电探测。

        Parameters
        ----------
        rx_opt_signal : Signal
            输入的接收光信号。数据形状为 (n_pol, N_samples)。
            光信号的幅度应在 sqrt(mW) 级别，LO 功率也应与此单位保持一致。

        Returns
        -------
        Signal
            输出的基带电信号。数据形状为 (n_pol, N_samples)，复数。
            SignalContext 会更新 center_freq 为 0 (基带)。
        """
        if not np.iscomplexobj(rx_opt_signal.data):
            raise ValueError("Input optical signal must be complex.")
        if rx_opt_signal.data.ndim != 2 or rx_opt_signal.data.shape[0] != rx_opt_signal.ctx.n_pol:
            raise ValueError("Input optical signal data must be 2D (n_pol, N_samples).")

        n_pol = rx_opt_signal.ctx.n_pol
        num_samples = rx_opt_signal.num_samples
        sample_rate = rx_opt_signal.ctx.sample_rate

        # 1. 生成 LO 信号 (理想 CW + 频率偏移 + 相位噪声)
        t = np.arange(num_samples) / sample_rate

        # LO frequency offset term
        lo_freq_offset_term = np.exp(-1j * 2 * np.pi * self.lo_freq_offset_hz * t)

        # LO phase noise
        lo_phase_noise = self._generate_lo_phase_noise(num_samples, sample_rate)
        lo_phase_noise_term = np.exp(-1j * lo_phase_noise)

        # Total LO complex signal (assuming LO field is real, scaled by sqrt(LO_Power))
        # Note: Input optical signal data is typically in units of sqrt(mW).
        # We convert LO power from W to mW for consistency.
        lo_amplitude_per_pol = np.sqrt(self.lo_power_linear_w * 1e3)  # sqrt(mW)
        lo_signal = lo_amplitude_per_pol * lo_freq_offset_term * lo_phase_noise_term

        # 2. 相干混合和光电探测 (理想 BPD)
        # Detected current I_out = 2 * R * sqrt(P_sig * P_lo) * cos(delta_phi + delta_omega * t)
        # Which is proportional to (E_sig * E_lo^*) after mixing.
        # So, the complex baseband output is roughly E_sig * E_lo^*

        # Here, we assume the input signal is E_sig, and LO is E_lo
        # Output is E_sig * E_lo^* (plus DC term and other noise)
        # We simplify to `electrical_field = E_sig * E_lo_conjugate`.
        # The LO signal generated above is `lo_amplitude * exp(-j * (phi_offset + phi_noise))`.
        # Its conjugate is `lo_amplitude * exp(j * (phi_offset + phi_noise))`.

        # Apply LO to each polarization
        baseband_electrical_data = np.empty_like(rx_opt_signal.data, dtype=np.complex64)
        for pol_idx in range(n_pol):
            # E_sig * E_lo_conj
            baseband_electrical_data[pol_idx, :] = rx_opt_signal.data[pol_idx, :] * lo_signal.conj()  # type: ignore

        # The output of coherent receiver is essentially a complex-valued electrical signal
        # whose amplitude is proportional to the optical field.
        # We can scale it to represent a voltage or current, but for DSP, the relative amplitude matters.
        # The factor `2 * R * sqrt(P_lo)` is usually applied.
        # For simplicity, we just use the complex field product for DSP.
        # A rough scaling: assuming input field is in sqrt(mW), output will be in mW.
        # Then, apply Responsivity R to get Amps. We will keep it unit-less for now (complex baseband).

        # Update SignalContext
        new_ctx = SignalContext(
            sample_rate=sample_rate,
            center_freq=0.0,  # 基带电信号
            n_pol=n_pol,
            sps=rx_opt_signal.ctx.sps,
            sym_rate=rx_opt_signal.ctx.sym_rate,
            bits_per_symbol=rx_opt_signal.ctx.bits_per_symbol,
            is_ofdm=rx_opt_signal.ctx.is_ofdm,
            meta={
                **(rx_opt_signal.ctx.meta or {}),
                "rx_lo_power_dbm": self.lo_power_dbm,
                "rx_lo_linewidth_hz": self.lo_linewidth_hz,
                "rx_lo_freq_offset_hz": self.lo_freq_offset_hz,
            },
        )
        return Signal(data=baseband_electrical_data, ctx=new_ctx)
