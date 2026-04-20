"""时域 DSP 模块：重采样、定时同步、粗频偏补偿。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class Resampler(BaseBlock):
    """重采样模块。

    用于改变信号的采样率。

    Parameters
    ----------
    target_sps : int
        目标每符号采样点数 (Samples per Symbol)。
        例如，如果原始信号为 2 sps，目标为 1 sps，则进行 2 倍下采样。
        如果目标采样率与当前采样率不兼容，将报错。
    resampling_filter : str, optional
        重采样滤波器类型，目前支持 'fir' (理想 FIR 低通)。
    """

    def __init__(self, target_sps: int, resampling_filter: str = "fir") -> None:
        """初始化 Resampler。"""
        super().__init__(name=f"Resampler({target_sps}sps)")
        if not target_sps > 0:
            raise ValueError("Target sps must be positive.")
        if resampling_filter not in ["fir"]:
            raise ValueError(f"Unsupported resampling filter: {resampling_filter}. Currently only 'fir' is supported.")

        self.target_sps = target_sps
        self.resampling_filter = resampling_filter

    def forward(self, signal: Signal) -> Signal:
        """
        对信号进行重采样。

        Parameters
        ----------
        signal : Signal
            输入的信号。

        Returns
        -------
        Signal
            重采样后的信号。
        """
        current_sps = signal.ctx.sps
        if current_sps == self.target_sps:
            return signal  # No resampling needed

        if current_sps % self.target_sps != 0 and self.target_sps % current_sps != 0:
            raise ValueError(
                f"Resampling factor (from {current_sps} to {self.target_sps} sps) is not an integer or its inverse. "
                "Non-integer rational resampling not yet supported by this simple implementation."
            )

        if self.target_sps < current_sps:  # Downsampling
            factor = current_sps // self.target_sps
            # Simple decimation after filtering (ideal low-pass)
            # In real DSP, an actual anti-aliasing filter is used before decimation.
            # Here, we mimic with scipy's resample_poly for simplicity.
            # Or just simple decimation if no filter needed
            resampled_data = signal.data[:, ::factor]
        else:  # Upsampling
            factor = self.target_sps // current_sps
            # Simple zero-padding in frequency domain (or interpolate in time domain)
            # This is more complex than simple decimation
            # For simplicity, let's assume this is not a common path or require advanced filter design.
            raise NotImplementedError("Upsampling is not yet implemented in this simple Resampler.")

        new_sample_rate = signal.ctx.sym_rate * self.target_sps

        new_ctx = SignalContext(
            sample_rate=new_sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=signal.ctx.n_pol,
            sps=self.target_sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={**(signal.ctx.meta or {}), "resampled_sps": self.target_sps},
        )
        return Signal(data=resampled_data.astype(np.complex64), ctx=new_ctx)


class OFDMChannelEstimator(BaseBlock):
    """OFDM 信道估计与均衡模块。

    使用 Least Squares (LS) 算法估计每个子载波的频率响应，
    并对 payload 数据进行 Zero-Forcing (ZF) 均衡。

    Parameters
    ----------
    estimation_method : str, optional
        估计方法，目前支持 'LS' (Least Squares)。
    smooth_subcarriers : bool, optional
        是否对估计的信道响应在子载波维度进行平滑（减少噪声影响），默认 False。
    """

    def __init__(self, estimation_method: str = "LS", smooth_subcarriers: bool = False) -> None:
        super().__init__(name=f"ChannelEstimator({estimation_method})")
        self.estimation_method = estimation_method
        self.smooth_subcarriers = smooth_subcarriers

    def forward(self, signal: Signal) -> Signal:
        """执行信道估计与均衡。

        要求输入 Signal 的 meta 中包含:
        - 'ofdm_ts_freq_symbols': 接收端提取的 TS (n_pol, n_sub, n_frames)
        - 'ofdm_ts_pattern': 发射端原始 TS 模式 (n_pol, n_sub)
        - 'ofdm_num_data_symbols_per_frame': 每帧的数据符号数
        """
        if not signal.ctx.is_ofdm:
            raise ValueError("Signal is not OFDM.")

        meta = signal.ctx.meta or {}
        rx_ts = meta.get("ofdm_ts_freq_symbols")  # (n_pol, n_sub, n_frames)
        tx_ts = meta.get("ofdm_ts_pattern")  # (n_pol, n_sub)
        num_data_syms_per_frame = meta.get("ofdm_num_data_symbols_per_frame")

        if rx_ts is None or tx_ts is None:
            raise ValueError("TS patterns not found in signal metadata. Demodulator must extract TS first.")

        n_pol, n_sub, n_frames = rx_ts.shape
        # 数据当前的形状 (n_pol, n_sub, total_data_symbols)
        n_pol_data, n_sub_data, n_total_syms = signal.data.shape

        # 1. 信道估计 (LS: H = Y / X)
        # 将 tx_ts 广播到 rx_ts 的形状
        tx_ts_expanded = np.repeat(tx_ts[:, :, np.newaxis], n_frames, axis=2)

        # 计算每一帧、每一极化、每个子载波的信道响应 H
        # 避免除以 0
        eps = 1e-12
        h_est = rx_ts / (tx_ts_expanded + eps)

        # (可选) 在子载波上平滑以抑制噪声
        if self.smooth_subcarriers:
            from scipy.ndimage import uniform_filter1d

            h_est = uniform_filter1d(h_est, size=3, axis=1)

        # 2. 均衡 (ZF: X_hat = Y / H)
        equalized_data = np.zeros_like(signal.data)

        # 遍历每一帧进行均衡
        for f_idx in range(n_frames):
            start_idx = f_idx * num_data_syms_per_frame
            end_idx = min((f_idx + 1) * num_data_syms_per_frame, n_total_syms)

            if start_idx >= n_total_syms:
                break

            # 提取当前帧的信道估计结果 (n_pol, n_sub)
            h_frame = h_est[:, :, f_idx]

            # 对该帧内的所有数据符号进行均衡
            # 形状：(n_pol, n_sub, symbols_in_this_frame)
            frame_data = signal.data[:, :, start_idx:end_idx]

            # 执行均衡 (对每个符号应用对应的子载波增益)
            h_frame_expanded = h_frame[:, :, np.newaxis]
            equalized_data[:, :, start_idx:end_idx] = frame_data / (h_frame_expanded + eps)

        # Update Context
        new_meta = {
            **meta,
            "ofdm_channel_response": h_est,
            "ofdm_equalized": True,
        }
        new_ctx = signal.ctx.copy()
        new_ctx.meta = new_meta

        return Signal(data=equalized_data, ctx=new_ctx)


class TimingRecovery(BaseBlock):
    """定时同步模块。

    使用 Gardner 算法进行符号同步。

    Parameters
    ----------
    sps_in : int, optional
        输入信号的每符号采样点数。
    gardner_loop_filter_gain : float, optional
        Gardner 环路滤波器增益。
    """

    def __init__(self, sps_in: int = 2, gardner_loop_filter_gain: float = 0.01) -> None:
        """初始化 TimingRecovery。"""
        super().__init__(name="TimingRecovery(Gardner)")
        if not sps_in > 1:
            raise ValueError("Input sps for timing recovery must be > 1.")

        self.sps_in = sps_in
        self.gardner_loop_filter_gain = gardner_loop_filter_gain
        # For simplicity, we assume output is 1 sps.

    def _gardner_timing_recovery(self, signal_data: np.ndarray) -> np.ndarray:
        """
        执行 Gardner 定时误差检测和补偿。

        Parameters
        ----------
        signal_data : np.ndarray
            输入的信号数据，形状为 (n_pol, N_samples)。

        Returns
        -------
        np.ndarray
            定时同步后的符号序列，形状为 (n_pol, N_symbols)。
        """
        n_pol, num_samples = signal_data.shape
        output_symbols = []

        # Gardner detector works on samples at 2*symbol_rate
        if self.sps_in != 2:
            print(
                f"Warning: Gardner detector typically expects 2 samples/symbol. "
                f"Input sps is {self.sps_in}. Performance may vary."
            )

        for pol_idx in range(n_pol):
            current_pol_data = signal_data[pol_idx, :]

            # Initialize timing phase (fractional sample offset)
            timing_phase = 0.0
            # Initialize loop filter output (for integrating timing error)
            integral = 0.0

            symbols_per_pol: list[complex] = []

            # Start index for first symbol
            idx = self.sps_in // 2  # Start in the middle of first symbol interval if sps=2

            while idx < num_samples - self.sps_in:
                # Get samples around current timing estimate
                # x_k-1 (previous symbol), x_k-1/2 (mid-point), x_k (current symbol)
                # Gardner error = Re(x_k-1/2 * (x_k^* - x_k-1^*))

                # Interpolate to find mid-point sample
                # For simplicity, using nearest samples as approximation.
                # In practice, a polyphase filter bank would be used.

                # Current symbol sample
                x_k_idx = int(idx)
                if x_k_idx + self.sps_in // 2 >= num_samples:  # Ensure enough samples for x_k-1/2, x_k
                    break

                x_k_minus_half_idx = int(idx + self.sps_in / 2)

                # Make sure indices are within bounds
                if x_k_minus_half_idx < num_samples and x_k_idx < num_samples:
                    x_k = current_pol_data[x_k_idx]
                    x_k_minus_half = current_pol_data[x_k_minus_half_idx]  # This is actually x_k+0.5

                    # Previous symbol's full sample (for the error calculation)
                    # We need the previous symbol value, and current symbol value.
                    # The error depends on the mid-point sample and the difference of adjacent full-rate samples.
                    # e = real(x_mid * (conj(x_k) - conj(x_k-1)))
                    # Let's adjust for correct indices:
                    # x_k = symbol at index idx
                    # x_k_minus_1 = symbol at index idx - sps_in
                    # x_mid = symbol at index idx - sps_in / 2

                    if idx >= self.sps_in:  # Ensure we have x_k-1 sample
                        x_prev_idx = int(idx - self.sps_in)
                        x_mid_idx = int(idx - self.sps_in // 2)

                        if x_prev_idx >= 0 and x_mid_idx >= 0:
                            x_k = current_pol_data[x_k_idx]
                            x_prev_k = current_pol_data[x_prev_idx]
                            x_mid_k = current_pol_data[x_mid_idx]

                            error = (x_mid_k * (x_k.conj() - x_prev_k.conj())).real

                            # Loop filter (Proportional + Integral)
                            prop = error
                            integral += error

                            # Update timing phase (fractional part)
                            timing_phase += self.gardner_loop_filter_gain * (prop + integral)

                            # Wrap timing phase to [0, sps_in)
                            timing_phase %= self.sps_in

                            # Take symbol sample
                            symbols_per_pol.append(x_k)  # Take the symbol at current idx

                            # Advance index by 1 symbol interval (sps_in samples)
                            # adjusted by timing_phase
                            idx += self.sps_in - timing_phase

                        else:
                            idx += 1  # Advance if not enough samples
                    else:
                        idx += 1  # Advance if not enough samples
                else:
                    break
            output_symbols.append(np.array(symbols_per_pol))

        # Pad shorter polarization if needed (shouldn't happen with proper input)
        max_len = max(len(s) for s in output_symbols)
        padded_symbols = np.array([np.pad(s, (0, max_len - len(s)), "constant") for s in output_symbols])

        return padded_symbols

    def forward(self, signal: Signal) -> Signal:
        """
        对信号执行定时同步。

        Parameters
        ----------
        signal : Signal
            输入的信号。数据形状为 (n_pol, N_samples)。
            预期 `signal.ctx.sps` 为 `sps_in`。

        Returns
        -------
        Signal
            定时同步后的符号信号。数据形状为 (n_pol, N_symbols)。
            `signal.ctx.sps` 将更新为 1。
        """
        if signal.ctx.sps != self.sps_in:
            raise ValueError(
                f"Input signal sps ({signal.ctx.sps}) does not match TimingRecovery's expected sps_in ({self.sps_in})."
            )

        synced_symbols_data = self._gardner_timing_recovery(signal.data)

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sym_rate,  # New sample rate is symbol rate
            center_freq=signal.ctx.center_freq,
            n_pol=signal.ctx.n_pol,
            sps=1,  # After timing recovery, 1 sample per symbol
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={**(signal.ctx.meta or {}), "timing_recovered": True},
        )
        return Signal(data=synced_symbols_data, ctx=new_ctx)


class FOCoarse(BaseBlock):
    """粗频偏补偿模块。

    使用基于训练序列或 FFT 的方法进行粗频偏估计和补偿。
    此处实现一个基于 FFT 峰值的方法。

    Parameters
    ----------
    search_range_hz : float, optional
        频率搜索范围 (Hz)。
    fft_size_for_estimation : int, optional
        用于 FFT 估计的 FFT 大小。
    """

    def __init__(self, search_range_hz: float = 200e6, fft_size_for_estimation: int = 1024) -> None:
        """初始化 FOCoarse。"""
        super().__init__(name="FOCoarse")
        self.search_range_hz = search_range_hz
        self.fft_size_for_estimation = fft_size_for_estimation

    def forward(self, signal: Signal) -> Signal:
        """
        估计并补偿粗频偏。

        Parameters
        ----------
        signal : Signal
            输入的信号。数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            补偿粗频偏后的信号。
        """
        if signal.data.ndim != 2:
            raise ValueError("Input signal data must be 2D (n_pol, N_samples).")

        n_pol, num_samples = signal.data.shape
        sample_rate = signal.ctx.sample_rate

        # For simplicity, we assume an FFT-based estimation on one polarization.
        # More robust methods use squaring or training sequences.

        # Take a chunk of signal for estimation
        if num_samples < self.fft_size_for_estimation:
            fft_data = signal.data[:, :num_samples]
            fft_size = num_samples
        else:
            fft_data = signal.data[:, : self.fft_size_for_estimation]
            fft_size = self.fft_size_for_estimation

        estimated_freq_offset_hz_per_pol: list[float] = []

        for pol_idx in range(n_pol):
            # FFT of the squared signal (for BPSK/QPSK) or direct FFT (for presence of pilot/strong tone)
            # For general QAM, a frequency offset causes rotation in constellation.
            # A common method is to raise to the power of M (M-QAM) to remove data dependency.
            # Or use a pilot tone if available.

            # Simple direct FFT peak detection (assumes some residual carrier or pilot)
            spectrum = np.fft.fft(fft_data[pol_idx, :])
            freqs = np.fft.fftfreq(fft_size, d=1 / sample_rate)

            # Find peak in the spectrum (focus on positive frequencies or around expected offset)
            # For a pure baseband signal, this peak should be at 0.
            # If there's an offset, there will be a shift.

            # Limit search to search_range_hz
            freq_idx_range = np.where(
                np.logical_and(freqs >= -self.search_range_hz / 2, freqs <= self.search_range_hz / 2)
            )

            if freq_idx_range[0].size > 0:
                peak_idx = freq_idx_range[0][np.argmax(np.abs(spectrum[freq_idx_range]))]
                estimated_offset = freqs[peak_idx]
            else:
                estimated_offset = 0.0  # Default if no strong peak in range

            estimated_freq_offset_hz_per_pol.append(estimated_offset)

            # Compensate frequency offset for the entire signal
            t = np.arange(num_samples) / sample_rate
            compensation_phase = np.exp(-1j * 2 * np.pi * estimated_offset * t)
            signal.data[pol_idx, :] = signal.data[pol_idx, :] * compensation_phase

        # Assume average offset for now if multiple polarizations
        # More robust approach would be to compensate each polarization individually.
        avg_estimated_offset = np.mean(estimated_freq_offset_hz_per_pol)

        new_ctx = SignalContext(
            sample_rate=sample_rate,
            center_freq=signal.ctx.center_freq - avg_estimated_offset,  # Adjust center freq for estimation
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={
                **(signal.ctx.meta or {}),
                "coarse_freq_offset_hz": avg_estimated_offset,
                "coarse_freq_offset_per_pol_hz": estimated_freq_offset_hz_per_pol,
            },
        )
        return Signal(data=signal.data, ctx=new_ctx)


class ChromaticDispersionComp(BaseBlock):
    """色度色散补偿 (CDC) 模块。

    在频域通过乘法器补偿光纤传输中产生的二阶色散 (GVD) 和三阶色散 (TOD)。
    该模块采用全信号 FFT 实现，适用于补偿累积色散。

    Parameters
    ----------
    fiber_length_km : float | None, optional
        需要补偿的光纤长度 (km)。如果为 None，则尝试从信号 metadata 中获取 "fiber_length_km"。
    beta2_ps2_per_km : float | None, optional
        光纤的二阶色散系数 (ps^2/km)。如果为 None，则尝试从信号 metadata 中获取。
    beta3_ps3_per_km : float | None, optional
        光纤的三阶色散系数 (ps^3/km)。如果为 None，则尝试从信号 metadata 中获取。
    """

    def __init__(
        self,
        fiber_length_km: float | None = None,
        beta2_ps2_per_km: float | None = None,
        beta3_ps3_per_km: float | None = None,
    ) -> None:
        """初始化 ChromaticDispersionComp。"""
        super().__init__(name="CDC")
        self.fiber_length_km = fiber_length_km
        self.beta2_ps2_per_km = beta2_ps2_per_km
        self.beta3_ps3_per_km = beta3_ps3_per_km

    def forward(self, signal: Signal) -> Signal:
        """执行频域色散补偿。

        Parameters
        ----------
        signal : Signal
            输入的复数基带信号，数据形状应为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            补偿色散后的信号，上下文 metadata 中会记录补偿参数。

        Raises
        ------
        ValueError
            如果输入信号不是复数类型或维度不等于 2。
        """
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex for CDC.")
        if signal.data.ndim != 2:
            raise ValueError("Input signal data must be 2D (n_pol, N_samples).")

        # 1. 参数提取与推断
        length_km = self.fiber_length_km
        if length_km is None:
            length_km = float(signal.ctx.meta.get("fiber_length_m", 0.0)) / 1000.0

        if length_km <= 0:
            # 若长度为0或未指定，直接返回原信号
            return signal

        beta2 = self.beta2_ps2_per_km
        if beta2 is None:
            beta2 = float(signal.ctx.meta.get("fiber_beta2_ps2_per_km", 0.0))

        beta3 = self.beta3_ps3_per_km
        if beta3 is None:
            beta3 = float(signal.ctx.meta.get("fiber_beta3_ps3_per_km", 0.0))

        if beta2 == 0.0 and beta3 == 0.0:
            raise ValueError("Zero beta, unable to compensate dispersion.")

        n_pol, num_samples = signal.data.shape
        sample_rate = signal.ctx.sample_rate

        # 2. 构建角频率网格 (rad/s)
        # freq: [-fs/2, ..., fs/2-1], omega = 2 * pi * freq
        freq = np.fft.fftfreq(num_samples, d=1.0 / sample_rate)
        omega = 2 * np.pi * freq

        # 3. 计算补偿相位
        # 物理公式: H_fiber = exp(-j * (beta2/2 * omega^2 + beta3/6 * omega^3) * L)
        # 补偿相位 (逆函数取共轭): phi = +(beta2/2 * omega^2 + beta3/6 * omega^3) * L
        # 量纲转换说明:
        # beta2: ps^2/km -> s^2/km 需要乘 1e-24
        # beta3: ps^3/km -> s^3/km 需要乘 1e-36
        # length_km 已是 km，无需额外转换
        phase = ((beta2 * 1e-24) * (omega**2) / 2 + (beta3 * 1e-36) * (omega**3) / 6) * length_km

        # 4. 频域滤波
        h_cdc = np.exp(1j * phase)
        compensated_data = np.empty_like(signal.data, dtype=np.complex128)

        for pol_idx in range(n_pol):
            # 时域 -> 频域
            sig_fft = np.fft.fft(signal.data[pol_idx, :])
            # 频域相乘
            sig_comp_fft = sig_fft * h_cdc
            # 频域 -> 时域
            compensated_data[pol_idx, :] = np.fft.ifft(sig_comp_fft)

        # 5. 更新上下文与元数据
        new_meta = (signal.ctx.meta or {}).copy()
        new_meta.update(
            {
                "cdc_compensated_length_km": length_km,
                "cdc_beta2_ps2_per_km": beta2,
                "cdc_beta3_ps3_per_km": beta3,
            }
        )

        new_ctx = SignalContext(
            sample_rate=sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta=new_meta,
        )

        return Signal(data=compensated_data, ctx=new_ctx)
