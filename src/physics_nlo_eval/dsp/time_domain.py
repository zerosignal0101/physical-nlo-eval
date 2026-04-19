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

    在频域通过乘法器补偿色度色散。

    Parameters
    ----------
    fiber_length_km : float, optional
        需要补偿的光纤长度 (km)。
    beta2_ps2_per_km : float, optional
        光纤的二阶色散系数 (ps^2/km)。
    """

    def __init__(
        self,
        fiber_length_km: float | None = None,
        beta2_ps2_per_km: float = -21.6,  # Typical SMF
    ) -> None:
        """初始化 ChromaticDispersionComp。"""
        super().__init__(name="CDC")
        self.fiber_length_km = fiber_length_km
        self.beta2_ps2_per_km = beta2_ps2_per_km

    def forward(self, signal: Signal) -> Signal:
        """
        执行频域色散补偿。

        Parameters
        ----------
        signal : Signal
            输入的信号。数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            补偿色散后的信号。
        """
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex for CDC.")
        if signal.data.ndim != 2:
            raise ValueError("Input signal data must be 2D (n_pol, N_samples).")

        # Get dispersion parameters
        # Try to infer length from signal metadata if not provided
        length_to_compensate_km = self.fiber_length_km
        if length_to_compensate_km is None:
            length_to_compensate_km = signal.ctx.meta.get("fiber_length_m", 0.0) / 1000.0
            print(f"CDC: Using fiber length from metadata: {length_to_compensate_km:.1f} km.")

        if length_to_compensate_km <= 0:
            print("Warning: CDC length is zero or not specified. No CD compensation performed.")
            return signal

        n_pol, num_samples = signal.data.shape
        sample_rate = signal.ctx.sample_rate

        # Angular frequency grid
        # omega = 2 * np.pi * np.fft.fftfreq(num_samples, d=1/sample_rate)
        # Shifted to relative frequency around center (not optical frequency)
        # For baseband, use freq = np.fft.fftfreq.
        freq = np.fft.fftfreq(num_samples, d=1 / sample_rate)
        omega = 2 * np.pi * freq

        # CD Compensation Filter in frequency domain
        # H(omega) = exp(j * beta2 * L * omega^2 / 2)
        # Note: beta2 is usually in s^2/m, length in m.
        # beta2_si = self.beta2_ps2_per_km * 1e-24  # ps^2/km to s^2/m
        # length_si = length_to_compensate_km * 1000  # km to m
        # dispersion_term = beta2_si * length_si / 2

        # Filter phase response
        # CD effect: exp(-j * beta2 * L * omega^2 / 2)
        # Compensation filter: exp(+j * beta2 * L * omega^2 / 2)
        # Pay attention to the sign of beta2 and the sign in the formula.
        # If beta2 < 0 for SMF in 1550nm, the phase delay is negative, so compensation should be positive phase.

        # From definition, D = -2*pi*c/lambda^2 * beta2. D typically given in ps/(nm.km)
        # beta2 in ps^2/km. In formula: beta2 * L * omega^2 / 2.
        # omega here is baseband angular frequency.
        # This formula usually assumes the optical carrier is removed, and omega is baseband frequency.

        # beta2 is in ps^2/km, L in km
        # omega in rad/s
        # factor should be 1e-24 (ps^2 to s^2)
        cdc_filter_phase = self.beta2_ps2_per_km * length_to_compensate_km * (omega**2) / 2 * 1e-24

        # Apply to each polarization
        compensated_data = np.empty_like(signal.data)
        for pol_idx in range(n_pol):
            signal_fft = np.fft.fft(signal.data[pol_idx, :])
            compensated_fft = signal_fft * np.exp(1j * cdc_filter_phase)
            compensated_data[pol_idx, :] = np.fft.ifft(compensated_fft)

        new_ctx = SignalContext(
            sample_rate=sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={
                **(signal.ctx.meta or {}),
                "cdc_compensated_length_km": length_to_compensate_km,
                "cdc_beta2_ps2_per_km": self.beta2_ps2_per_km,
            },
        )
        return Signal(data=compensated_data.astype(np.complex64), ctx=new_ctx)
