"""频域 DSP 模块：OFDM 解调, 公共相位误差补偿。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class OFDM_Demodulator(BaseBlock):
    """OFDM 解调模块。

    执行去除循环前缀 (CP)、FFT 和符号提取。
    现已支持标准 OFDM 和加窗 OFDM (W-OFDM) 解调。

    Parameters
    ----------
    n_subcarriers : int
        OFDM 子载波数量。
    cp_len : int
        循环前缀的长度。
    cs_len : int, optional
        W-OFDM 的循环后缀长度。如果为 0，则按标准 OFDM 处理。
    """

    def __init__(self, n_subcarriers: int, cp_len: int, cs_len: int = 0) -> None:
        """初始化 OFDM_Demodulator。"""
        super().__init__(name=f"OFDM_Demodulator(W-OFDM={'True' if cs_len > 0 else 'False'})")
        if not n_subcarriers > 0:
            raise ValueError("Number of subcarriers must be positive.")
        if not cp_len >= 0:
            raise ValueError("CP length must be non-negative.")
        if not cs_len >= 0:
            raise ValueError("CS length must be non-negative.")

        self.n_subcarriers = n_subcarriers
        self.cp_len = cp_len
        self.cs_len = cs_len
        self.fft_size = n_subcarriers  # Time-domain length of an OFDM symbol (excluding CP/CS)

        # Effective symbol length (what's expected per symbol from channel)
        # For standard OFDM: fft_size + cp_len
        # For Windowed OFDM with OLA: fft_size + cp_len (the effective block size sent)
        self.ofdm_symbol_len_from_channel = self.fft_size + self.cp_len

        # This is the length of the data portion after removing CP and CS (if applicable)
        self.payload_len_after_cp_cs = self.fft_size

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

        # Retrieve relevant TS and frame metadata
        meta = signal.ctx.meta or {}
        # Ensure 'ofdm_is_windowed' is consistent
        is_windowed_tx = meta.get("ofdm_is_windowed", False)
        if is_windowed_tx and self.cs_len == 0:
            print("Warning: Demodulating W-OFDM without CS length. Performance may degrade.")
        if not is_windowed_tx and self.cs_len > 0:
            print("Warning: Demodulating standard OFDM with CS length. Demodulation may be incorrect.")

        ts_locations = meta.get("ofdm_ts_locations", [])
        num_data_symbols_per_frame = meta.get("ofdm_num_data_symbols_per_frame", None)
        num_ts_symbols_per_frame = meta.get("ofdm_num_ts_symbols_per_frame", 1 if ts_locations else 0)

        # Build a set of global symbol indices that correspond to Training Sequences
        ts_global_indices = set()
        if ts_locations and num_data_symbols_per_frame is not None:
            total_symbols_per_full_frame = num_data_symbols_per_frame + num_ts_symbols_per_frame
            for frame_idx, ts_start_in_frame in ts_locations:
                for offset in range(num_ts_symbols_per_frame):
                    global_idx = frame_idx * total_symbols_per_full_frame + ts_start_in_frame + offset
                    ts_global_indices.add(global_idx)

        # Calculate number of OFDM symbols from the total samples received
        # For windowed OFDM, total samples is (num_symbols * effective_output_sym_len) + cs_len (for last tail)
        # So we need to consider how many full *effective* symbols we can extract.

        # For windowed OFDM (W-OFDM) with OLA, the last CS_LEN samples of the full signal might be incomplete.
        # We need to correctly segment based on `self.ofdm_symbol_len_from_channel`.

        num_effective_ofdm_symbols = num_total_samples // self.ofdm_symbol_len_from_channel
        if num_total_samples % self.ofdm_symbol_len_from_channel != 0:
            print(
                f"Warning: Total samples ({num_total_samples}) not perfectly divisible by "
                f"OFDM effective symbol length ({self.ofdm_symbol_len_from_channel}). Some samples will be truncated."
            )

        # Prepare containers: data symbols for signal.data, TS symbols for meta
        data_symbols_list: list[list[np.ndarray]] = [[] for _ in range(n_pol)]
        ts_symbols_list: list[list[np.ndarray]] = [[] for _ in range(n_pol)]

        # Pre-compute the receiver side window (for overlap-add/save, if implemented)
        # For simplicity, we just extract the non-overlapping part.
        # A more advanced W-OFDM receiver would perform overlap-and-add of the received windowed symbols.
        # Here, we assume the OLA from TX has been received, and we simply extract the useful `fft_size` part.

        for pol_idx in range(n_pol):
            pol_data = signal.data[pol_idx, :]

            for sym_idx in range(num_effective_ofdm_symbols):
                start_sample = sym_idx * self.ofdm_symbol_len_from_channel
                end_sample = start_sample + self.ofdm_symbol_len_from_channel

                if end_sample > num_total_samples:
                    break

                received_ofdm_block = pol_data[start_sample:end_sample]

                # Check if this symbol belongs to the Training Sequence (TS) indices
                is_ts_symbol = sym_idx in ts_global_indices

                # 1. Remove CP
                ofdm_symbol_no_cp = received_ofdm_block[self.cp_len :]  # Length: fft_size

                # 2. For Windowed OFDM, if CS_LEN > 0, the windowed symbol length was cp+fft+cs.
                # The effective output from TX has length cp+fft. So after removing CP, we have FFT_size.
                # If there was a trailing CS, it would have been part of the next symbol's CP in OLA.
                # So here, we just take the core FFT_size part.
                # No further removal of CS is needed if the received signal is already the result of OLA.

                # FFT to frequency domain
                freq_domain_symbol = (
                    np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ofdm_symbol_no_cp))) / self.n_subcarriers
                )

                if is_ts_symbol:
                    ts_symbols_list[pol_idx].append(freq_domain_symbol)
                else:
                    data_symbols_list[pol_idx].append(freq_domain_symbol)

        # Assemble data symbols: (n_pol, n_subcarriers, n_data_symbols)
        data_arrays = [np.stack(pol_list, axis=1) for pol_list in data_symbols_list if pol_list]
        max_data_symbols = max(s.shape[1] for s in data_arrays) if data_arrays else 0

        # Handle case where no data symbols were extracted (e.g., only TS)
        if not data_arrays:
            padded_data = np.zeros((n_pol, self.n_subcarriers, 0), dtype=np.complex64)
        else:
            padded_data = np.array(
                [np.pad(s, ((0, 0), (0, max_data_symbols - s.shape[1])), "constant") for s in data_arrays]
            )

        # Assemble TS symbols: (n_pol, n_subcarriers, n_ts_symbols) or None
        ts_freq_symbols = None
        if any(ts_symbols_list):
            ts_arrays = [np.stack(pol_list, axis=1) for pol_list in ts_symbols_list if pol_list]
            if ts_arrays:
                ts_freq_symbols = np.array(ts_arrays)

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sym_rate,  # The symbol rate is the *effective* symbol rate
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=1,  # After demodulation, sps is 1 (one sample per subcarrier symbol)
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=True,
            meta={
                **(meta),  # Preserve existing meta info, including 'ofdm_is_windowed'
                "ofdm_demodulated": True,
                "ofdm_ts_freq_symbols": ts_freq_symbols,
                "ofdm_payload_shape": padded_data.shape,
            },
        )
        return Signal(data=padded_data.astype(np.complex64), ctx=new_ctx)


class CommonPhaseNoiseCompensate(BaseBlock):
    """OFDM 频域公共相位误差 (CPE) 补偿模块。

    通过导频辅助估计并补偿每个 OFDM 符号的公共相位误差 (CPE)。

    Parameters
    ----------
    method : str, optional
        估计方法，当前仅支持 'pilot_assisted'。
    """

    def __init__(self, method: str = "pilot_assisted") -> None:
        """初始化 CommonPhaseNoiseCompensate。"""
        super().__init__(name=f"CommonPhaseNoiseCompensate({method})")
        if method not in ["pilot_assisted"]:
            raise ValueError(f"Unsupported CPE estimation method: {method}.")
        self.method = method

    def forward(self, signal: Signal) -> Signal:
        """估计并补偿 OFDM 频域信号的 CPE。

        Parameters
        ----------
        signal : Signal
            输入的 OFDM 频域信号。数据形状为 (n_pol, N_subcarriers, N_symbols)。

        Returns
        -------
        Signal
            补偿 CPE 后的 OFDM 频域信号。
        """
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex.")

        if not signal.ctx.is_ofdm:
            raise ValueError("CommonPhaseNoiseCompensate module only supports OFDM signal.")
        return self._ofdm_cpe_comp(signal)

    def _ofdm_cpe_comp(self, signal: Signal) -> Signal:
        """OFDM 信号的 CPE 估计与补偿。"""
        n_pol, n_subcarriers, n_symbols = signal.data.shape
        compensated_data = signal.data.copy()

        for pol_idx in range(n_pol):
            for sym_idx in range(n_symbols):
                pilot_indices = signal.ctx.meta.get("ofdm_pilot_subcarriers_indices")
                expected_pilots = signal.ctx.meta.get("ofdm_expected_pilot_symbols")

                if pilot_indices is None or expected_pilots is None:
                    # Not enough info to compensate, skip this symbol/pol
                    continue

                rx_pilots = signal.data[pol_idx, pilot_indices, sym_idx]

                # Adjust expected_pilots shape for current symbol if it's 3D (i.e., pilot symbols vary per symbol)
                if expected_pilots.ndim == 3:
                    tx_pilots = expected_pilots[pol_idx, :, sym_idx]
                else:  # Assume pilots are constant across symbols/polarizations (or 2D: n_pol, n_pilots)
                    if expected_pilots.ndim == 2:
                        tx_pilots = expected_pilots[pol_idx, :]
                    else:  # 1D array of pilots, assumes same for all pol/symbols
                        tx_pilots = expected_pilots

                # Ensure rx_pilots and tx_pilots have the same length for computation
                if len(rx_pilots) == 0 or len(tx_pilots) == 0 or len(rx_pilots) != len(tx_pilots):
                    continue  # Skip if no pilots or mismatch

                cpe = np.angle(np.mean(rx_pilots * tx_pilots.conj()))
                compensated_data[pol_idx, :, sym_idx] *= np.exp(-1j * cpe)

        # Update metadata to reflect compensation
        meta = signal.ctx.meta or {}
        if "ofdm_cpe_compensated" not in meta:
            meta["ofdm_cpe_compensated"] = True

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta=meta,  # Use the potentially updated meta dictionary
        )
        return Signal(data=compensated_data, ctx=new_ctx)


class OFDMToSymbolStream(BaseBlock):
    """OFDM 频域格式 → 流格式转换模块。

    将 OFDM 解调后的频域信号 (n_pol, n_sub, n_sym) 转换为流格式
    (n_pol, n_sym * n_sub)，以适配 Demapper 的输入期望。

    此模块应置于 CommonPhaseNoiseCompensate 之后使用，因为 CPE 需要在
    (n_pol, n_sub, n_sym) 格式上进行逐符号处理。

    Parameters
    ----------
    data_indices : np.ndarray | None, optional
        数据子载波索引。如果为 None，则包含所有子载波。
    """

    def __init__(self, data_indices: np.ndarray | None = None) -> None:
        """初始化 OFDMToSymbolStream。"""
        super().__init__(name="OFDMToSymbolStream")
        self.data_indices = data_indices

    def forward(self, signal: Signal) -> Signal:
        """将 OFDM 频域信号转换为 QAM 流格式。

        Parameters
        ----------
        signal : Signal
            输入的 OFDM 频域信号。数据形状为 (n_pol, n_sub, n_sym)。

        Returns
        -------
        Signal
            QAM 流格式信号。数据形状为 (n_pol, n_sym * n_sub)。
            `signal.ctx.sps` 保持不变。
        """
        if not signal.ctx.is_ofdm:
            raise ValueError("OFDMToQAMStream only supports OFDM signal.")
        if signal.data.ndim != 3:
            raise ValueError("Input signal data must be 3D (n_pol, n_sub, n_sym).")

        n_pol = signal.data.shape[0]

        if self.data_indices is not None:
            # 仅保留数据子载波
            output_data = signal.data[:, self.data_indices, :].transpose(0, 2, 1).reshape(n_pol, -1)
        else:
            # 包含所有子载波
            output_data = signal.data.transpose(0, 2, 1).reshape(n_pol, -1)

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=False,  # 转换后不再是 OFDM 格式
            meta={
                **(signal.ctx.meta or {}),
                "ofdm_to_qam_converted": True,
            },
        )
        return Signal(data=output_data.astype(np.complex64), ctx=new_ctx)
