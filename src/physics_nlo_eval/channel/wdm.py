"""波分复用 (WDM) 合波与分波模块。"""

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


@dataclass
class ChannelSpec:
    """单个 WDM 通道的物理层规范。

    Attributes
    ----------
    channel_id : int
        通道的唯一标识符 (通常对应阵列索引)。
    freq_offset_hz : float
        相对于 WDM 参考中心频率的频偏. 正数表示高频端，负数表示低频端。
        这允许非对称、非均匀的灵活栅格配置。
    """

    channel_id: int
    freq_offset_hz: float


@dataclass
class WDMPlan:
    """整个 WDM 系统的频率规划。

    Attributes
    ----------
    reference_freq_hz : float
        WDM 复用信号的绝对参考中心频率 (例如光载波中心 193.1e12 Hz)。
    channels : list[ChannelSpec]
        包含所有通道规范的列表。
    """

    reference_freq_hz: float = 193.1e12
    channels: list[ChannelSpec] = field(default_factory=list)

    def get_channel(self, channel_id: int) -> ChannelSpec:
        """根据 channel_id 获取通道规范。"""
        for ch in self.channels:
            if ch.channel_id == channel_id:
                return ch
        raise ValueError(f"Channel ID {channel_id} not found in WDMPlan.")


class WDMMux(BaseBlock):
    """波分复用合波器 (支持灵活栅格)。

    将多个物理功率已确定的单通道基带信号，上变频到 WDMPlan 指定的灵活栅格频率，
    并相加合并为一个宽带时域信号。

    Parameters
    ----------
    wdm_plan : WDMPlan
        灵活栅格规划，包含参考频率和各通道频偏。
    target_sample_rate_hz : float | None, optional
        合并后的目标总采样率。必须覆盖所有通道的总带宽。
        若为 None，则根据通道的最大/最小频偏及单通道带宽自动推算。
    """

    def __init__(
        self,
        wdm_plan: WDMPlan,
        target_sample_rate_hz: float | None = None,
    ) -> None:
        super().__init__(name="WDMMux")
        self.wdm_plan = wdm_plan
        self.target_sample_rate_hz = target_sample_rate_hz

    def _resample_to_target(self, data: np.ndarray, current_sr: float, target_sr: float) -> np.ndarray:
        """将信号重采样到目标采样率。"""
        if current_sr == target_sr:
            return data

        ratio = Fraction(target_sr / current_sr).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator

        if up > 10000:
            raise RuntimeError(
                f"Resampling ratio too large ({up}/{down}). "
                f"Check if current_sr ({current_sr}) and target_sr ({target_sr}) are consistent."
            )

        g = gcd(int(up), int(down))
        up_simplified = int(up // g)
        down_simplified = int(down // g)

        return resample_poly(data, up_simplified, down_simplified, axis=-1)

    def combine(self, signals: list[Signal]) -> Signal:
        """合并多个通道信号。

        Parameters
        ----------
        signals : list[Signal]
            各通道的 Signal 列表。列表顺序必须与 wdm_plan.channels 的顺序一致。
            信号振幅必须已经是物理单位 (sqrt(W))，由 IQModulator 设定。

        Returns
        -------
        Signal
            合并后的宽带复用 Signal，振幅单位 sqrt(W)。
        """
        if not signals:
            raise ValueError("Signal list cannot be empty.")
        if len(signals) != len(self.wdm_plan.channels):
            raise ValueError("Number of signals must match number of channels in WDMPlan.")

        n_pol = signals[0].ctx.n_pol
        single_channel_bw = signals[0].ctx.sample_rate  # 基带带宽

        # 1. 确定总采样率
        if self.target_sample_rate_hz is None:
            max_freq = max(ch.freq_offset_hz for ch in self.wdm_plan.channels) + single_channel_bw / 2
            min_freq = min(ch.freq_offset_hz for ch in self.wdm_plan.channels) - single_channel_bw / 2
            total_bandwidth = max_freq - min_freq
            target_sr = total_bandwidth * 2.5  # 留足够余量防止频谱混叠
        else:
            target_sr = self.target_sample_rate_hz

        # 2. 计算统一长度
        base_num_symbols = max(sig.num_samples for sig in signals)
        target_num_samples = int(np.ceil(base_num_symbols * (target_sr / signals[0].ctx.sample_rate)))
        t = np.arange(target_num_samples) / target_sr

        muxed_data_accumulator = np.zeros((n_pol, target_num_samples), dtype=np.complex128)
        wdm_channels_info = []

        for i, (sig, ch_spec) in enumerate(zip(signals, self.wdm_plan.channels, strict=True)):
            if sig.ctx.n_pol != n_pol:
                raise ValueError(f"Channel {i} has inconsistent n_pol.")

            # 重采样到全带宽
            resampled_data = self._resample_to_target(sig.data, sig.ctx.sample_rate, target_sr)
            if resampled_data.ndim == 1:
                resampled_data = resampled_data[np.newaxis, :]

            # 截断或补零对齐
            curr_len = resampled_data.shape[-1]
            if curr_len > target_num_samples:
                resampled_data = resampled_data[:, :target_num_samples]
            elif curr_len < target_num_samples:
                pad_width = target_num_samples - curr_len
                resampled_data = np.pad(resampled_data, ((0, 0), (0, pad_width)), "constant")

            # 上变频 (基于 WDMPlan 的频偏)
            phasor = np.exp(1j * 2 * np.pi * ch_spec.freq_offset_hz * t)
            shifted_data = resampled_data * phasor[np.newaxis, :]

            # 累加到合波信号
            muxed_data_accumulator += shifted_data

            wdm_channels_info.append(
                {
                    "channel_id": ch_spec.channel_id,
                    "freq_offset_hz": ch_spec.freq_offset_hz,
                    "original_sample_rate_hz": sig.ctx.sample_rate,
                    "original_ctx": sig.ctx.copy(),
                    "original_num_samples": sig.num_samples,
                }
            )

        # 3. 构建复用信号上下文
        new_ctx = SignalContext(
            sample_rate=target_sr,
            center_freq=self.wdm_plan.reference_freq_hz,  # 记录绝对参考中心频率
            n_pol=n_pol,
            sps=target_sr / signals[0].ctx.sym_rate,
            sym_rate=signals[0].ctx.sym_rate,
            bits_per_symbol=signals[0].ctx.bits_per_symbol,
            is_ofdm=signals[0].ctx.is_ofdm,
            meta={
                **(signals[0].ctx.meta or {}),
                "wdm_plan": self.wdm_plan,  # 传递完整的 Plan 供 Demux 使用
                "wdm_channels_info": wdm_channels_info,
                "wdm_n_channels": len(signals),
            },
        )

        return Signal(data=muxed_data_accumulator.astype(np.complex128), ctx=new_ctx)

    def forward(self, signal: Signal) -> Signal:
        """Pipeline 模式的 WDM 合波接口."""
        channels_to_mux = signal.ctx.meta.get("channels_to_mux")
        if channels_to_mux is None:
            raise ValueError("WDMMux in Pipeline requires 'channels_to_mux' in signal.meta.")
        all_signals = [signal] + channels_to_mux
        return self.combine(all_signals)


class WDMDemux(BaseBlock):
    """波分复用分波器 (支持灵活栅格)。

    从宽带复用信号中提取各个通道。借助 meta 中的 WDMPlan 载波频偏信息，
    执行下变频、FIR 低通滤波与降采样。

    Parameters
    ----------
    filter_type : str, optional
        通道滤波类型，支持 'fir' 和 'ideal_rectangular'.
    roll_off : float, optional
        RRC 滚降系数, 用于计算 LPF 截止频率 = sym_rate * (1 + roll_off) / 2.
        默认为 0.1。若设为 None 则使用 original_sr / 2 (旧行为).
    """

    def __init__(self, filter_type: str = "fir", roll_off: float | None = 0.1) -> None:
        super().__init__(name="WDMDemux")
        if filter_type not in ["fir", "ideal_rectangular"]:
            raise ValueError(f"Unsupported filter type: {filter_type}.")
        self.filter_type = filter_type
        self.roll_off = roll_off

    def _fir_lowpass(self, data: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
        from scipy.signal import filtfilt, firwin

        n_samples = data.shape[-1]
        num_taps = min(101, n_samples // 4)
        if num_taps % 2 == 0:
            num_taps -= 1
        if num_taps < 7:
            num_taps = 7

        b = firwin(num_taps, cutoff_hz, fs=sample_rate, window="hann")
        filtered = np.empty_like(data)
        for p in range(data.shape[0]):
            filtered[p] = filtfilt(b, 1, data[p], padtype="even")
        return filtered

    def _ideal_lowpass(self, data: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
        n_samples = data.shape[-1]
        freqs = np.fft.fftfreq(n_samples, d=1.0 / sample_rate)
        mask = np.abs(freqs) <= cutoff_hz

        filtered = np.empty_like(data)
        for p in range(data.shape[0]):
            spectrum = np.fft.fft(data[p])
            filtered[p] = np.fft.ifft(spectrum * mask)
        return filtered

    def _downsample_to_target(self, data: np.ndarray, current_sr: float, target_sr: float) -> np.ndarray:
        if current_sr == target_sr:
            return data

        ratio = Fraction(target_sr / current_sr).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator

        if up > 10000:
            raise RuntimeError(
                f"Resampling ratio too large ({up}/{down}). "
                f"Check if current_sr ({current_sr}) and target_sr ({target_sr}) are consistent."
            )

        g = gcd(int(up), int(down))
        return resample_poly(data, int(up // g), int(down // g), axis=-1)

    def forward(self, signal: Signal) -> Signal:
        """解复用宽带信号，输出多通道堆叠数据与上下文。"""
        if signal.data.ndim != 2:
            raise ValueError("Input WDM signal must be 2D (n_pol, N_total_samples).")

        wdm_info = signal.ctx.meta.get("wdm_channels_info")
        if not wdm_info:
            raise ValueError("WDM channel info not found in signal.ctx.meta.")

        n_pol = signal.data.shape[0]
        n_channels = len(wdm_info)
        current_sr = signal.ctx.sample_rate
        t = np.arange(signal.data.shape[-1]) / current_sr

        demuxed_data_list = []
        demuxed_ctx_list = []

        for ch_info in wdm_info:
            f_offset = ch_info["freq_offset_hz"]
            original_sr = ch_info["original_sample_rate_hz"]
            original_ctx = ch_info["original_ctx"]
            original_num_samples = ch_info["original_num_samples"]

            # 1. 下变频 (基于 WDMPlan 的频偏)
            phasor = np.exp(-1j * 2 * np.pi * f_offset * t)
            downconverted_data = signal.data * phasor[np.newaxis, :]

            # 2. 低通滤波 (截断至信号带宽: sym_rate * (1+roll_off) / 2)
            if self.roll_off is not None:
                cutoff_hz = original_ctx.sym_rate * (1 + self.roll_off) / 2.0
            else:
                cutoff_hz = original_sr / 2.0
            if self.filter_type == "fir":
                filtered_data = self._fir_lowpass(downconverted_data, cutoff_hz, current_sr)
            else:
                filtered_data = self._ideal_lowpass(downconverted_data, cutoff_hz, current_sr)

            # 3. 降采样回原采样率
            resampled_data = self._downsample_to_target(filtered_data, current_sr, original_sr)

            # 4. 严格恢复原始符号长度
            curr_len = resampled_data.shape[-1]
            if curr_len > original_num_samples:
                resampled_data = resampled_data[:, :original_num_samples]
            elif curr_len < original_num_samples:
                pad_width = original_num_samples - curr_len
                resampled_data = np.pad(resampled_data, ((0, 0), (0, pad_width)), "constant")

            demuxed_data_list.append(resampled_data.astype(np.complex128))
            demuxed_ctx_list.append(original_ctx)

        # 堆叠所有通道数据: (n_channels, n_pol, N_samples)
        stacked_data = np.stack(demuxed_data_list, axis=0)

        # 构建输出的堆叠 Signal
        new_meta = {**(signal.ctx.meta or {}), "demuxed_contexts": demuxed_ctx_list, "is_stacked_wdm_signal": True}
        new_ctx = SignalContext(
            sample_rate=wdm_info[0]["original_sample_rate_hz"],
            center_freq=wdm_info[0]["original_ctx"].center_freq,
            n_pol=n_pol,
            sps=wdm_info[0]["original_ctx"].sps,
            sym_rate=wdm_info[0]["original_ctx"].sym_rate,
            bits_per_symbol=wdm_info[0]["original_ctx"].bits_per_symbol,
            is_ofdm=wdm_info[0]["original_ctx"].is_ofdm,
            meta=new_meta,
        )

        return Signal(data=stacked_data, ctx=new_ctx)


class ChannelExtractor(BaseBlock):
    """从解复用后的堆叠 WDM 信号中提取指定单通道。"""

    def __init__(self, channel_idx: int) -> None:
        super().__init__(name=f"ChannelExtractor(idx={channel_idx})")
        self.channel_idx = channel_idx

    def forward(self, signal: Signal) -> Signal:
        """从堆叠 WDM 信号中提取指定通道."""
        if not signal.ctx.meta.get("is_stacked_wdm_signal"):
            raise ValueError("Input signal is not a stacked WDM signal from WDMDemux.")
        if signal.data.ndim != 3 or signal.data.shape[0] <= self.channel_idx:
            raise ValueError(f"Invalid channel_idx {self.channel_idx} for data shape {signal.data.shape}.")

        extracted_data = signal.data[self.channel_idx]
        extracted_ctx = signal.ctx.meta["demuxed_contexts"][self.channel_idx]

        final_meta = {
            **(signal.ctx.meta),
            **(extracted_ctx.meta),
            "extracted_channel_idx": self.channel_idx,
        }
        extracted_ctx.meta = final_meta

        return Signal(data=extracted_data, ctx=extracted_ctx)
