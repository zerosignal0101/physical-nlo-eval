"""频域 DSP 模块：FFT, 精频偏补偿。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class OFDM_Demodulator(BaseBlock):
    """OFDM 解调模块。

    执行去除循环前缀 (CP)、FFT 和符号提取。

    Parameters
    ----------
    n_subcarriers : int
        OFDM 子载波数量。
    cp_len : int
        循环前缀的长度。
    """

    def __init__(self, n_subcarriers: int, cp_len: int) -> None:
        """初始化 OFDM_Demodulator。"""
        super().__init__(name="OFDM_Demodulator")
        if not n_subcarriers > 0:
            raise ValueError("Number of subcarriers must be positive.")
        if not cp_len >= 0:
            raise ValueError("CP length must be non-negative.")

        self.n_subcarriers = n_subcarriers
        self.cp_len = cp_len
        self.fft_size = n_subcarriers  # Time-domain length of an OFDM symbol (excluding CP)
        self.ofdm_symbol_len = n_subcarriers + cp_len  # Length including CP

    def forward(self, signal: Signal) -> Signal:
        """
        对 OFDM 时域信号进行解调。

        Parameters
        ----------
        signal : Signal
            输入的 OFDM 时域信号。数据形状为 (n_pol, N_ofdm_samples)。
            `signal.ctx.is_ofdm` 应该为 True。

        Returns
        -------
        Signal
            输出的 OFDM 频域符号。数据形状为 (n_pol, N_subcarriers, N_symbols)。
            `signal.ctx.sps` 将更新为 1。
        """
        if not signal.ctx.is_ofdm:
            raise ValueError("Input signal is not marked as OFDM. Cannot perform OFDM demodulation.")
        if signal.data.ndim != 2:
            raise ValueError("Input signal data must be 2D (n_pol, N_samples).")

        n_pol, num_total_samples = signal.data.shape

        # Identify TS locations if available in metadata to exclude from data symbols
        ts_locations = signal.ctx.meta.get("ofdm_ts_locations", [])

        # Calculate number of OFDM symbols (including potential TS symbols)
        num_ofdm_symbols_total = num_total_samples // self.ofdm_symbol_len
        if num_total_samples % self.ofdm_symbol_len != 0:
            print(
                f"Warning: Total samples ({num_total_samples}) not perfectly divisible by "
                f"OFDM symbol length ({self.ofdm_symbol_len}). Some samples will be truncated."
            )

        demodulated_symbols_list: list[np.ndarray] = []

        for pol_idx in range(n_pol):
            pol_data = signal.data[pol_idx, :]
            pol_freq_symbols: list[np.ndarray] = []

            for sym_idx in range(num_ofdm_symbols_total):
                start_sample = sym_idx * self.ofdm_symbol_len
                end_sample = start_sample + self.ofdm_symbol_len

                if end_sample > num_total_samples:
                    break  # Reached end of signal

                ofdm_symbol_time = pol_data[start_sample:end_sample]

                # Skip if this is a TS symbol (handled separately by Channel Estimation)
                is_ts_symbol = False
                for _frame_idx, _ts_start_in_frame in ts_locations:
                    # ts_start_sample_global = (
                    #     _frame_idx
                    #     * (num_ofdm_symbols_per_frame * ofdm_symbol_len + ts_len_with_cp)
                    #     + _ts_start_in_frame
                    # )
                    # A more robust check is needed here to skip TS symbols from the data stream.
                    # For simplicity now, we assume TS is at fixed location, like first symbol of each frame.
                    # If the first symbol of a "frame" (logical grouping) is a TS
                    # This check needs to be more sophisticated based on `ofdm_ts_locations`
                    pass  # Placeholder: Implement actual TS skipping

                if not is_ts_symbol:
                    # Remove CP
                    ofdm_symbol_no_cp = ofdm_symbol_time[self.cp_len :]

                    # FFT
                    freq_domain_symbol = (
                        np.fft.fft(ofdm_symbol_no_cp) / self.n_subcarriers
                    )  # Scale by N for power preservation
                    pol_freq_symbols.append(freq_domain_symbol)

            # Stack symbols for this polarization: (N_subcarriers, N_symbols_per_pol)
            if pol_freq_symbols:
                demodulated_symbols_list.append(np.stack(pol_freq_symbols, axis=1))
            else:
                demodulated_symbols_list.append(np.array([]).reshape(self.n_subcarriers, 0))  # Empty if no symbols

        # Stack polarizations: (n_pol, N_subcarriers, N_symbols)
        # Handle potentially different numbers of symbols per polarization if TS skipping is asymmetric
        max_symbols = max(s.shape[1] for s in demodulated_symbols_list)
        padded_demod_symbols = np.array(
            [np.pad(s, ((0, 0), (0, max_symbols - s.shape[1])), "constant") for s in demodulated_symbols_list]
        )

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sym_rate,  # The effective symbol rate per subcarrier
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=1,  # 1 symbol per subcarrier (frequency domain)
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=True,  # Still OFDM, but now in frequency domain
            meta={
                **(signal.ctx.meta or {}),
                "ofdm_demodulated": True,
                "ofdm_symbols_shape": padded_demod_symbols.shape,
            },
        )
        return Signal(data=padded_demod_symbols.astype(np.complex64), ctx=new_ctx)


class FOFine(BaseBlock):
    """精细频偏补偿模块。

    通常在频域进行，用于补偿残余的频率偏移。
    在 OFDM 中，这可能涉及导频辅助或基于数据的估计。
    此处实现一个简单的基于导频 (如果可用) 或数据循环的估计。

    Parameters
    ----------
    method : str, optional
        精细频偏估计方法，例如 'pilot_assisted' (如果导频可用), 'data_aided'。
    """

    def __init__(self, method: str = "data_aided") -> None:
        """初始化 FOFine。"""
        super().__init__(name=f"FOFine({method})")
        if method not in ["data_aided", "pilot_assisted"]:
            raise ValueError(f"Unsupported fine frequency offset method: {method}.")
        self.method = method

    def forward(self, signal: Signal) -> Signal:
        """
        估计并补偿精细频偏。

        Parameters
        ----------
        signal : Signal
            输入的信号。数据形状为 (n_pol, N_samples) (单载波) 或 (n_pol, N_subcarriers, N_symbols) (OFDM)。

        Returns
        -------
        Signal
            补偿精细频偏后的信号。
        """
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex.")

        # Determine signal type and apply appropriate FO fine compensation
        if signal.ctx.is_ofdm:
            return self._ofdm_fine_freq_offset_comp(signal)
        return self._sc_fine_freq_offset_comp(signal)

    def _sc_fine_freq_offset_comp(self, signal: Signal) -> Signal:
        """单载波 (SC) 信号的精细频偏补偿。"""
        n_pol, num_symbols = signal.data.shape
        compensated_data = np.empty_like(signal.data)

        # Data-aided method: Viterbi & Viterbi (V&V) or Decision-directed (DD)
        # For simplicity, a simple power-of-M approach might be used for QPSK/16QAM
        # For M-QAM, raise to power of 4 or 8 and detect phase rotation.

        # Placeholder for data-aided method (e.g., squaring loop for QPSK)
        # A more general approach is needed for higher-order QAM.
        for pol_idx in range(n_pol):
            # For 4-QAM, raise to power of 4 to remove modulation
            # freq_offset_estimation = np.mean(signal.data[pol_idx, :]**4) # This will give average phase error.
            # A more detailed approach involves filtering and finding the phase of a tone.

            # Simple average phase error detection (assuming minimal phase change over segment)
            # This is not a robust frequency offset estimator, more of a phase tracker.
            avg_phase_error = np.angle(np.mean(signal.data[pol_idx, :] * np.roll(signal.data[pol_idx, :], -1).conj()))

            # Convert phase error to frequency offset
            # freq_offset = avg_phase_error / (2 * np.pi * (1/signal.ctx.sym_rate)) # Approx.

            # Apply a slow compensation (e.g., track phase over time)
            # For simplicity, apply a constant phase rotation
            compensated_data[pol_idx, :] = signal.data[pol_idx, :] * np.exp(
                -1j * avg_phase_error / 2
            )  # Example: small compensation

        print("Warning: Single-carrier fine frequency offset compensation is a simplified placeholder.")

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={
                **(signal.ctx.meta or {}),
                "fine_freq_offset_compensated_sc": True,
            },
        )
        return Signal(data=compensated_data, ctx=new_ctx)

    def _ofdm_fine_freq_offset_comp(self, signal: Signal) -> Signal:
        """OFDM 信号的精细频偏补偿。"""
        n_pol, n_subcarriers, n_symbols = signal.data.shape
        compensated_data = signal.data.copy()  # Operate in place

        # OFDM fine FO compensation: Usually relies on pilot tones (common phase error) or virtual carriers.
        # If method is 'pilot_assisted', need pilot subcarrier locations and expected values from metadata.
        # If method is 'data_aided', use differential detection or similar.

        # For simplicity, estimate common phase error across all subcarriers and symbols
        # This is more like common phase error (CPE) compensation, rather than inter-carrier interference (ICI)

        total_phase_error_per_pol: list[float] = []

        for pol_idx in range(n_pol):
            # Example: differential detection in frequency domain for phase rotation
            # Angle of average of (symbol[k] * conj(symbol[k-1]))
            # This is more for phase noise tracking.

            # Assume a common phase error (CPE) for all subcarriers within an OFDM symbol
            # Estimate CPE for each symbol by averaging phase of data symbols or using pilot symbols.

            if self.method == "pilot_assisted":
                # Requires 'pilot_subcarriers' and 'expected_pilot_values' in metadata
                # Placeholder logic:
                # pilots_rx = signal.data[pol_idx, pilot_subcarrier_indices, :]
                # pilots_tx = expected_pilot_values
                # error_per_symbol = np.angle(np.mean(pilots_rx * pilots_tx.conj(), axis=0))
                pass  # TODO: Implement pilot-assisted
            else:  # Data-aided (simplified)
                # For each symbol, estimate a common phase rotation
                # by averaging the phase of all received symbols.
                # This works if the constellation is symmetric or if there's a strong phase error.

                cpe_per_symbol = np.angle(np.mean(compensated_data[pol_idx, :, :], axis=0))  # Shape (N_symbols,)
                total_phase_error_per_pol.append(np.mean(cpe_per_symbol))  # Track total estimated phase error

                # Apply compensation for each symbol
                for sym_idx in range(n_symbols):
                    compensated_data[pol_idx, :, sym_idx] *= np.exp(-1j * cpe_per_symbol[sym_idx])

        print(f"Warning: OFDM fine frequency offset compensation ({self.method}) is a simplified placeholder.")

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={
                **(signal.ctx.meta or {}),
                "fine_freq_offset_compensated_ofdm": True,
                "fine_freq_offset_estimated_per_pol": total_phase_error_per_pol,
            },
        )
        return Signal(data=compensated_data, ctx=new_ctx)
