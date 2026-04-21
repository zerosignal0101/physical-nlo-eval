"""波分复用 (WDM) 合波与分波模块。"""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class WDMMux(BaseBlock):
    """波分复用合波器。

    将多个单通道基带信号上变频到各自的光载波频率，并相加合并为一个宽带时域信号。
    由于合波需要多输入，建议在 Pipeline 外部使用 `combine()` 方法生成复用信号；
    若需在 Pipeline 内部使用，可通过输入 Signal 的 meta 传递通道列表。

    Parameters
    ----------
    channel_spacing_hz : float
        通道间隔。
    center_freq_hz : float, optional
        WDM 复用信号的中心参考频率。各通道频偏将相对于此计算。
    target_sample_rate_hz : float | None, optional
        合并后的目标总采样率。必须覆盖所有通道的总带宽。
        若为 None，则根据通道数量和单通道带宽自动推算。
    """

    def __init__(
        self,
        channel_spacing_hz: float,
        center_freq_hz: float = 0.0,
        target_sample_rate_hz: float | None = None,
    ) -> None:
        super().__init__(name="WDMMux")
        self.channel_spacing_hz = channel_spacing_hz
        self.center_freq_hz = center_freq_hz
        self.target_sample_rate_hz = target_sample_rate_hz

    def _resample_to_target(self, signal: Signal, target_sr: float) -> np.ndarray:
        """将信号重采样到目标采样率。"""
        if signal.ctx.sample_rate == target_sr:
            return signal.data

        ratio = Fraction(target_sr / signal.ctx.sample_rate).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator

        # 如果比例太大，说明量纲依然有问题，抛出异常保护内存
        if up > 10000:
            raise RuntimeError(
                f"Resampling ratio too large ({up}/{down}). "
                f"Check if signal.ctx.sample_rate ({signal.ctx.sample_rate}) "
                f"is in the same unit as target_sr ({target_sr})."
            )

        # 寻找最大公约数以简化多相滤波器计算
        from math import gcd

        g = gcd(int(up), int(down))
        up_simplified = int(up // g)
        down_simplified = int(down // g)

        return resample_poly(signal.data, up_simplified, down_simplified, axis=-1)

    def combine(self, signals: list[Signal]) -> Signal:
        """合并多个通道信号 (推荐在 Pipeline 外部调用)。

        Parameters
        ----------
        signals : list[Signal]
            各通道的基带 Signal 列表。

        Returns
        -------
        Signal
            合并后的宽带复用 Signal，其 data 形状为。
        """
        if not signals:
            raise ValueError("Signal list cannot be empty.")

        n_pol = signals[0].ctx.n_pol
        # 1. 确定总采样率
        if self.target_sample_rate_hz is None:
            single_channel_bw = signals[0].ctx.sample_rate
            total_bandwidth = (len(signals) - 1) * self.channel_spacing_hz + single_channel_bw
            # 采样率至少为总带宽的 2 倍 (Nyquist)，通常取稍大一点以留余量
            target_sr = total_bandwidth * 2.0
        else:
            target_sr = self.target_sample_rate_hz

        # 2. 分配频率偏移 (以 center_freq_hz 为中心，对称分布)
        n_channels = len(signals)
        # 假设通道编号 0 到 N-1，中心频率偏移为: (i - (N-1)/2) * spacing
        offsets_hz = [(i - (n_channels - 1) / 2.0) * self.channel_spacing_hz for i in range(n_channels)]

        muxed_data = np.zeros((n_pol, 0), dtype=np.complex64)
        wdm_channels_info = []

        # 计算统一长度 (根据 target_sr)
        # 找到一个基准长度，确保所有通道重采样后对齐
        base_num_symbols = max(sig.num_samples for sig in signals)
        target_num_samples = int(np.ceil(base_num_symbols * (target_sr / signals[0].ctx.sample_rate)))

        t = np.arange(target_num_samples) / target_sr

        muxed_data_accumulator = np.zeros((n_pol, target_num_samples), dtype=np.complex128)

        for i, sig in enumerate(signals):
            if sig.ctx.n_pol != n_pol:
                raise ValueError(f"Channel {i} has inconsistent n_pol.")

            # 重采样到全带宽
            resampled_data = self._resample_to_target(sig, target_sr)
            # 确保 resampled_data 是 (n_pol, N)
            if resampled_data.ndim == 1:
                resampled_data = resampled_data[np.newaxis, :]

            # 截断或补零以确保长度严格对齐
            curr_len = resampled_data.shape[-1]
            if curr_len > target_num_samples:
                resampled_data = resampled_data[:, :target_num_samples]
            elif curr_len < target_num_samples:
                pad_width = target_num_samples - curr_len
                resampled_data = np.pad(resampled_data, ((0, 0), (0, pad_width)), "constant")

            # 上变频 (频移)
            f_offset = offsets_hz[i]
            phasor = np.exp(1j * 2 * np.pi * f_offset * t)
            shifted_data = resampled_data * phasor[np.newaxis, :]

            # 累加到合波信号
            muxed_data_accumulator += shifted_data

            # 记录通道信息，供 Demux 使用
            wdm_channels_info.append(
                {
                    "f_offset_hz": f_offset,
                    "original_sample_rate_hz": sig.ctx.sample_rate,
                    "original_ctx": sig.ctx.copy(),
                    "original_num_samples": sig.num_samples,
                }
            )

        # 3. 构建复用信号上下文
        new_ctx = SignalContext(
            sample_rate=target_sr,
            center_freq=self.center_freq_hz,
            n_pol=n_pol,
            sps=target_sr / signals[0].ctx.sym_rate,  # 宽带信号的等效 sps
            sym_rate=signals[0].ctx.sym_rate,
            bits_per_symbol=signals[0].ctx.bits_per_symbol,
            is_ofdm=signals[0].ctx.is_ofdm,
            meta={
                **(signals[0].ctx.meta or {}),
                "wdm_channels_info": wdm_channels_info,
                "wdm_n_channels": n_channels,
                "wdm_channel_spacing_hz": self.channel_spacing_hz,
            },
        )

        return Signal(data=muxed_data_accumulator.astype(np.complex64), ctx=new_ctx)

    def forward(self, signal: Signal) -> Signal:
        """Pipeline 内部调用接口。

        如果 signal.meta 中包含 'channels_to_mux' (list[Signal])，则执行合并；
        否则抛出异常，提示应使用 combine() 方法。
        """
        channels_to_mux = signal.ctx.meta.get("channels_to_mux")
        if channels_to_mux is None:
            raise ValueError(
                "WDMMux in Pipeline requires 'channels_to_mux' in signal.meta, "
                "or use WDMMux.combine() outside the Pipeline."
            )
        # 将当前 signal 作为第一个通道，与 meta 中的其他通道合并
        all_signals = [signal] + channels_to_mux
        return self.combine(all_signals)


class WDMDemux(BaseBlock):
    """波分复用分波器。

    从宽带复用信号中提取各个通道。借助 meta 中的载波频偏信息，
    执行下变频、FIR 低通滤波与降采样。
    输出的 Signal.data 形状为，各通道独立的上下文存于 meta 中。

    Parameters
    ----------
    filter_type : str, optional
        通道滤波类型，支持 'fir' (线性相位 FIR, 零相移) 和
        'ideal_rectangular' (频域硬截断, 仅供对比).
    """

    def __init__(self, filter_type: str = "fir") -> None:
        super().__init__(name="WDMDemux")
        if filter_type not in ["fir", "ideal_rectangular"]:
            raise ValueError(f"Unsupported filter type: {filter_type}. Use 'fir' or 'ideal_rectangular'.")
        self.filter_type = filter_type

    def _fir_lowpass(self, data: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
        """线性相位 FIR 低通滤波 (使用 filtfilt 实现零相移).

        相比 FFT 矩形硬截断, FIR 滤波器具有:
        - 线性相位响应, 无相位畸变
        - 无 Gibbs 现象
        - 良好的旁瓣抑制
        """
        from scipy.signal import filtfilt, firwin

        n_samples = data.shape[-1]
        # 滤波器阶数: 需要足够长以实现陡峭过渡带, 但不能超过信号长度
        # 限制为信号长度的 1/4, 并确保为奇数 (线性相位 FIR 要求)
        num_taps = min(101, n_samples // 4)
        if num_taps % 2 == 0:
            num_taps -= 1
        if num_taps < 7:
            num_taps = 7  # 最小滤波器长度

        # 设计线性相位 FIR 低通滤波器 (Hann 窗)
        b = firwin(num_taps, cutoff_hz, fs=sample_rate, window="hann")

        # 对每个偏振分量进行零相移滤波
        filtered = np.empty_like(data)
        for p in range(data.shape[0]):
            # filtfilt = 双向滤波, 零相移, 但有效滤波器长度翻倍
            # 边界使用 reflect 模式减少瞬态效应
            filtered[p] = filtfilt(b, 1, data[p], padtype="even")
        return filtered

    def _ideal_lowpass(self, data: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
        """频域理想矩形低通滤波 (对比用, 会产生 Gibbs 现象)."""
        n_samples = data.shape[-1]
        freqs = np.fft.fftfreq(n_samples, d=1.0 / sample_rate)
        mask = np.abs(freqs) <= cutoff_hz

        filtered = np.empty_like(data)
        for p in range(data.shape[0]):
            spectrum = np.fft.fft(data[p])
            filtered[p] = np.fft.ifft(spectrum * mask)
        return filtered

    def _downsample_to_target(self, data: np.ndarray, current_sr: float, target_sr: float) -> np.ndarray:
        """将信号降采样回原通道采样率。"""
        if current_sr == target_sr:
            return data
        up = target_sr
        down = current_sr
        from math import gcd

        g = gcd(int(up), int(down))
        return resample_poly(data, int(up // g), int(down // g), axis=-1)

    def forward(self, signal: Signal) -> Signal:
        """解复用宽带信号，输出多通道堆叠数据与上下文。

        Parameters
        ----------
        signal : Signal
            宽带复用信号，data 形状为。

        Returns
        -------
        Signal
            解复用后的信号。data 形状为。
            meta["demuxed_contexts"] 包含各通道原始的 SignalContext。
        """
        if signal.data.ndim != 2:
            raise ValueError("Input WDM signal must be 2D (n_pol, N_total_samples).")

        wdm_info = signal.ctx.meta.get("wdm_channels_info")
        if not wdm_info:
            raise ValueError("WDM channel info not found in signal.ctx.meta. WDMMux must be used first.")

        n_pol = signal.data.shape[0]
        n_channels = len(wdm_info)
        current_sr = signal.ctx.sample_rate
        t = np.arange(signal.data.shape[-1]) / current_sr

        demuxed_data_list = []
        demuxed_ctx_list = []

        for _i, ch_info in enumerate(wdm_info):
            f_offset = ch_info["f_offset_hz"]
            original_sr = ch_info["original_sample_rate_hz"]
            original_ctx = ch_info["original_ctx"]
            original_num_samples = ch_info["original_num_samples"]

            # 1. 下变频
            phasor = np.exp(-1j * 2 * np.pi * f_offset * t)
            downconverted_data = signal.data * phasor[np.newaxis, :]

            # 2. 低通滤波 (截断至原单通道带宽)
            cutoff_hz = original_sr / 2.0
            if self.filter_type == "fir":
                filtered_data = self._fir_lowpass(downconverted_data, cutoff_hz, current_sr)
            else:
                filtered_data = self._ideal_lowpass(downconverted_data, cutoff_hz, current_sr)

            # 3. 降采样回原采样率
            resampled_data = self._downsample_to_target(filtered_data, current_sr, original_sr)

            # 4. 截断/补零以严格恢复原始符号长度 (消除重采样可能带来的1~2样本误差)
            curr_len = resampled_data.shape[-1]
            if curr_len > original_num_samples:
                resampled_data = resampled_data[:, :original_num_samples]
            elif curr_len < original_num_samples:
                pad_width = original_num_samples - curr_len
                resampled_data = np.pad(resampled_data, ((0, 0), (0, pad_width)), "constant")

            demuxed_data_list.append(resampled_data.astype(np.complex64))
            demuxed_ctx_list.append(original_ctx)

        # 堆叠所有通道数据: (n_channels, n_pol, N_samples)
        stacked_data = np.stack(demuxed_data_list, axis=0)

        # 构建输出的堆叠 Signal
        new_meta = {**(signal.ctx.meta or {}), "demuxed_contexts": demuxed_ctx_list, "is_stacked_wdm_signal": True}
        # 堆叠信号的 Context 以第一个通道为基准，主要起载体作用
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
    """从解复用后的堆叠 WDM 信号中提取指定单通道。

    将形状为 的信号转换为标准的，
    从而无缝对接后续的单载波 DSP Pipeline (如 PhaseNoiseCompensation, Demapper 等)。

    Parameters
    ----------
    channel_idx : int
        要提取的通道索引 (0 到 n_channels-1)。
    """

    def __init__(self, channel_idx: int) -> None:
        super().__init__(name=f"ChannelExtractor(idx={channel_idx})")
        self.channel_idx = channel_idx

    def forward(self, signal: Signal) -> Signal:
        """Extract a single channel from a stacked WDM signal."""
        if not signal.ctx.meta.get("is_stacked_wdm_signal"):
            raise ValueError("Input signal is not a stacked WDM signal from WDMDemux.")
        if signal.data.ndim != 3 or signal.data.shape[0] <= self.channel_idx:
            raise ValueError(f"Invalid channel_idx {self.channel_idx} for data shape {signal.data.shape}.")

        # 提取指定通道的数据
        extracted_data = signal.data[self.channel_idx]  # shape: (n_pol, N_samples)

        # 恢复该通道独立的 Context
        extracted_ctx = signal.ctx.meta["demuxed_contexts"][self.channel_idx]

        # 继承全局可能新增的元信息 (如光纤传输参数)，但不覆盖通道特有属性
        final_meta = {
            **(signal.ctx.meta),  # 保留 WDM 全局信息
            **(extracted_ctx.meta),  # 覆盖恢复单通道特有信息
            "extracted_channel_idx": self.channel_idx,
        }
        extracted_ctx.meta = final_meta

        return Signal(data=extracted_data, ctx=extracted_ctx)
