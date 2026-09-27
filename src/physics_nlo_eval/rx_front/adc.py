"""模数转换器 (ADC) 模块。"""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

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

        Notes
        -----
        量化在**物理幅度域**进行：输出的幅度尺度与输入一致，不做归一化。
        这一点很重要——旧实现先按 RMS 归一化再量化，返回值被隐式重新缩放到
        单位 RMS，下游无法从输出还原输入的真实幅度。
        """
        # 满量程 = clip_ratio × 分量 RMS。这里用实部/虚部拼接后的 RMS（即分量
        # RMS），而非复包络 RMS，二者相差 sqrt(2) 倍。
        component_rms = np.sqrt(np.mean(np.concatenate((data.real.ravel(), data.imag.ravel())) ** 2))
        if component_rms <= np.finfo(float).eps:
            # 全零（或次正规）输入：量化无意义，原样返回，避免除零。
            return data.astype(np.complex64, copy=True)
        full_scale = self.clip_ratio * component_rms
        step = 2 * full_scale / (self.num_levels - 1)

        def quantize_component(component: np.ndarray) -> np.ndarray:
            clipped = np.clip(component, -full_scale, full_scale)
            return np.round((clipped + full_scale) / step) * step - full_scale

        return (quantize_component(data.real) + 1j * quantize_component(data.imag)).astype(np.complex64)

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
        # 用有理多相重采样代替整数抽取：旧实现对非整数倍率直接按 int() 截断抽取，
        # 既无抗混叠滤波，也会静默改变实际采样率。Fraction(...).limit_denominator
        # 把任意倍率近似为有理数，再由 resample_poly 做带抗混叠的多相滤波。
        if target_sample_rate < current_sample_rate:
            ratio = Fraction(target_sample_rate / current_sample_rate).limit_denominator(10_000)
            downsampled_data = resample_poly(analog_signal.data, ratio.numerator, ratio.denominator, axis=-1)
        else:
            downsampled_data = analog_signal.data.copy()

        # Quantization
        quantized_data = self._quantize(downsampled_data)

        # Update SignalContext
        new_ctx = SignalContext(
            sample_rate=target_sample_rate,
            center_freq=analog_signal.ctx.center_freq,  # Still baseband
            n_pol=analog_signal.ctx.n_pol,
            # 采样率变了，sps 必须按新采样率重算；沿用输入的 sps 会让
            # SignalContext.__post_init__ 推出一个错误的 sym_rate。
            sps=target_sample_rate / analog_signal.ctx.sym_rate if analog_signal.ctx.sym_rate > 0 else 0,
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
