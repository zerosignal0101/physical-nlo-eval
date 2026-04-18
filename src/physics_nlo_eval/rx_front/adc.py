"""模数转换器 (ADC) 模块。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class ADC(BaseBlock):
    """模数转换器 (ADC) 模块。

    模拟 ADC 的有限分辨率、采样和可能的削波效应。

    Parameters
    ----------
    resolution_bits : int, optional
        ADC 的分辨率 (比特数)。例如 8-bit, 10-bit。
    sampling_rate_hz : float | None, optional
        ADC 的采样率 (Hz)。如果为 None，则使用输入信号的采样率。
    clip_ratio : float, optional
        ADC 的削波比。定义了满量程电压与信号 RMS 值之比。
        例如，1.0 表示信号 RMS 值等于满量程的 1/clip_ratio。
    """

    def __init__(
        self,
        resolution_bits: int = 8,
        sampling_rate_hz: float | None = None,
        clip_ratio: float = 1.0,
    ) -> None:
        """初始化 ADC。"""
        super().__init__(name="ADC")
        if not resolution_bits > 0:
            raise ValueError("Resolution bits must be positive.")
        if not clip_ratio > 0:
            raise ValueError("Clip ratio must be positive.")

        self.resolution_bits = resolution_bits
        self.sampling_rate_hz = sampling_rate_hz
        self.clip_ratio = clip_ratio

        self.num_levels = 2**resolution_bits
        self.quantization_step = 1.0 / (self.num_levels - 1)  # Assuming signal scaled to [-0.5, 0.5] or [0, 1]

    def _quantize(self, data: np.ndarray) -> np.ndarray:
        """
        对信号进行量化。

        Parameters
        ----------
        data : np.ndarray
            浮点信号数据。

        Returns
        -------
        np.ndarray
            量化后的整数或浮点信号数据。
        """
        # Scale data to [-1, 1] based on its peak or RMS for clipping, then quantize.
        # For simplicity, let's normalize by current max amplitude to fit within ADC range
        # and then apply quantization.

        # Calculate RMS for clipping
        data_rms = np.sqrt(np.mean(np.abs(data) ** 2))

        # Max voltage swing for the ADC. If clip_ratio=1, RMS is at full scale.
        # This is a simplification. Usually, full_scale_voltage = clip_ratio * RMS_signal_voltage.
        # Let's assume input signal is normalized by its RMS to fit range.

        if data_rms > 1e-12:  # Avoid division by zero
            scaled_data = data / data_rms * self.clip_ratio  # If clip_ratio = 1, RMS is 1.

            # Clip values if they exceed the ADC's max range (e.g., [-1, 1])
            clipped_data = np.clip(scaled_data.real, -1.0, 1.0) + 1j * np.clip(scaled_data.imag, -1.0, 1.0)
        else:
            clipped_data = data.copy()

        # Quantization: map values from [-1, 1] to discrete levels
        # Assuming signed quantization.
        # Shift to [0, 2^B-1], round, then shift back.

        # Quantize real and imaginary parts separately
        # Normalize to [0, 1] for quantization levels: (clipped_data.real + 1) / 2
        quantized_real = (
            np.round(((clipped_data.real + 1.0) / 2.0) * (self.num_levels - 1)) / (self.num_levels - 1) * 2.0 - 1.0
        )
        quantized_imag = (
            np.round(((clipped_data.imag + 1.0) / 2.0) * (self.num_levels - 1)) / (self.num_levels - 1) * 2.0 - 1.0
        )

        return (quantized_real + 1j * quantized_imag).astype(np.complex64)

    def forward(self, analog_signal: Signal) -> Signal:
        """
        执行模数转换。

        Parameters
        ----------
        analog_signal : Signal
            输入的模拟电信号 (或高采样率数字信号)。数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            输出的数字信号。数据形状为 (n_pol, N_downsampled_samples)。
            SignalContext 会更新 sample_rate。
        """
        if analog_signal.data.ndim != 2:
            raise ValueError("Analog signal data must be 2D (n_pol, N_samples).")

        current_sample_rate = analog_signal.ctx.sample_rate
        target_sample_rate = self.sampling_rate_hz if self.sampling_rate_hz is not None else current_sample_rate

        if target_sample_rate > current_sample_rate:
            raise ValueError(
                f"Target sampling rate ({target_sample_rate} Hz) cannot be higher "
                f"than input sampling rate ({current_sample_rate} Hz) for downsampling."
            )

        # Resampling / Downsampling
        if target_sample_rate < current_sample_rate:
            resampling_factor = int(current_sample_rate / target_sample_rate)
            if current_sample_rate % target_sample_rate != 0:
                print(
                    f"Warning: Input sample rate ({current_sample_rate}) is not an integer multiple of "
                    f"target sample rate ({target_sample_rate}). Simple downsampling will truncate."
                )

            downsampled_data = analog_signal.data[:, ::resampling_factor]
        else:
            downsampled_data = analog_signal.data.copy()

        # Quantization
        quantized_data = self._quantize(downsampled_data)

        # Update SignalContext
        new_ctx = SignalContext(
            sample_rate=target_sample_rate,
            center_freq=analog_signal.ctx.center_freq,  # Still baseband
            n_pol=analog_signal.ctx.n_pol,
            sps=analog_signal.ctx.sps,  # sps might change if this is for symbol-rate sampling
            sym_rate=analog_signal.ctx.sym_rate,
            bits_per_symbol=analog_signal.ctx.bits_per_symbol,
            is_ofdm=analog_signal.ctx.is_ofdm,
            meta={
                **(analog_signal.ctx.meta or {}),
                "adc_resolution_bits": self.resolution_bits,
                "adc_sampling_rate_hz": target_sample_rate,
                "adc_clip_ratio": self.clip_ratio,
            },
        )
        # If target_sample_rate is now symbol_rate, update sps to 1
        # if abs(target_sample_rate - analog_signal.ctx.sym_rate) < 1e-6:
        #     new_ctx.sps = 1
        return Signal(data=quantized_data, ctx=new_ctx)
