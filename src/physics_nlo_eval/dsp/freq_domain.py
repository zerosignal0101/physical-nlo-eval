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

        # Retrieve relevant TS and frame metadata
        meta = signal.ctx.meta or {}
        ts_locations = meta.get("ofdm_ts_locations", [])
        num_data_symbols = meta.get("ofdm_num_data_symbols_per_frame", None)
        num_ts_symbols = meta.get("ofdm_num_ts_symbols_per_frame", 1 if ts_locations else 0)

        # Build a set of global symbol indices that correspond to Training Sequences
        ts_global_indices = set()
        if ts_locations and num_data_symbols is not None:
            total_symbols_per_frame = num_data_symbols + num_ts_symbols
            for frame_idx, ts_start_in_frame in ts_locations:
                for offset in range(num_ts_symbols):
                    global_idx = frame_idx * total_symbols_per_frame + ts_start_in_frame + offset
                    ts_global_indices.add(global_idx)

        # Calculate number of OFDM symbols (including potential TS symbols)
        num_ofdm_symbols_total = num_total_samples // self.ofdm_symbol_len
        if num_total_samples % self.ofdm_symbol_len != 0:
            print(
                f"Warning: Total samples ({num_total_samples}) not perfectly divisible by "
                f"OFDM symbol length ({self.ofdm_symbol_len}). Some samples will be truncated."
            )

        # Prepare containers: data symbols for signal.data, TS symbols for meta
        data_symbols_list: list[list[np.ndarray]] = [[] for _ in range(n_pol)]
        ts_symbols_list: list[list[np.ndarray]] = [[] for _ in range(n_pol)]

        for pol_idx in range(n_pol):
            pol_data = signal.data[pol_idx, :]

            for sym_idx in range(num_ofdm_symbols_total):
                start_sample = sym_idx * self.ofdm_symbol_len
                end_sample = start_sample + self.ofdm_symbol_len

                if end_sample > num_total_samples:
                    break

                ofdm_symbol_time = pol_data[start_sample:end_sample]

                # Check if this symbol belongs to the Training Sequence (TS) indices
                is_ts_symbol = sym_idx in ts_global_indices

                # Remove CP and FFT to frequency domain
                ofdm_symbol_no_cp = ofdm_symbol_time[self.cp_len :]
                freq_domain_symbol = (
                    np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ofdm_symbol_no_cp))) / self.n_subcarriers
                )

                if is_ts_symbol:
                    ts_symbols_list[pol_idx].append(freq_domain_symbol)
                else:
                    data_symbols_list[pol_idx].append(freq_domain_symbol)

        # Assemble data symbols: (n_pol, n_subcarriers, n_data_symbols)
        data_arrays = [np.stack(pol_list, axis=1) for pol_list in data_symbols_list]
        max_data_symbols = max(s.shape[1] for s in data_arrays) if data_arrays else 0
        padded_data = np.array(
            [np.pad(s, ((0, 0), (0, max_data_symbols - s.shape[1])), "constant") for s in data_arrays]
        )

        # Assemble TS symbols: (n_pol, n_subcarriers, n_ts_symbols) or None
        ts_freq_symbols = None
        if any(ts_symbols_list):
            ts_arrays = [np.stack(pol_list, axis=1) for pol_list in ts_symbols_list]
            ts_freq_symbols = np.array(ts_arrays)

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sym_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=1,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=True,
            meta={
                **meta,
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
                    continue

                rx_pilots = signal.data[pol_idx, pilot_indices, sym_idx]
                tx_pilots = expected_pilots[pol_idx, :, sym_idx] if expected_pilots.ndim == 3 else expected_pilots
                cpe = np.angle(np.mean(rx_pilots * tx_pilots.conj()))
                compensated_data[pol_idx, :, sym_idx] *= np.exp(-1j * cpe)

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
                "ofdm_cpe_compensated": True,
            },
        )
        return Signal(data=compensated_data, ctx=new_ctx)
