"""MIMO 偏振解复用和信道估计模块。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class MIMO_ChannelEstimation(BaseBlock):
    """MIMO 偏振解复用/信道估计模块。

    主要用于双偏振系统，通过训练序列 (TS) 或盲均衡算法
    估计和补偿偏振相关的信道损伤。

    Parameters
    ----------
    method : str, optional
        信道估计方法，例如 'LS' (最小二乘), 'MMSE' (最小均方误差), 'CMA' (常模算法)。
    ts_interval : int, optional
        训练序列的间隔 (OFDM 符号数)。如果 method 为 'LS' 或 'MMSE' 且信号为 OFDM，则需要。
    equalizer_length : int, optional
        均衡器抽头长度，对于 CMA 等时域均衡器使用。
    cma_mu : float, optional
        CMA 算法的步长 (step size)。
    """

    def __init__(
        self,
        method: str = "LS",
        ts_interval: int = 10,
        equalizer_length: int = 21,
        cma_mu: float = 1e-4,
    ) -> None:
        """初始化 MIMO_ChannelEstimation。"""
        super().__init__(name=f"MIMO_CE_PD({method})")
        if method not in ["LS", "MMSE", "CMA"]:
            raise ValueError(f"Unsupported MIMO channel estimation method: {method}.")
        self.method = method
        self.ts_interval = ts_interval
        self.equalizer_length = equalizer_length
        self.cma_mu = cma_mu

    def forward(self, signal: Signal) -> Signal:
        """
        执行 MIMO 信道估计和偏振解复用。

        Parameters
        ----------
        signal : Signal
            输入的信号。
            如果是 OFDM 信号，数据形状为 (n_pol, N_subcarriers, N_symbols)。
            如果是单载波信号，数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            解复用后的信号。
        """
        if signal.ctx.n_pol == 1:
            print("Warning: Input signal is single polarization. MIMO processing skipped.")
            return signal

        if signal.ctx.is_ofdm:
            return self._ofdm_mimo_processing(signal)
        return self._sc_mimo_processing(signal)

    def _ofdm_mimo_processing(self, signal: Signal) -> Signal:
        """OFDM 信号的 MIMO 处理 (信道估计和解复用)。"""
        n_pol, n_subcarriers, n_symbols = signal.data.shape
        demuxed_data = np.empty_like(signal.data)

        # Retrieve TX training sequence from metadata
        tx_ts_pattern = signal.ctx.meta.get("ofdm_ts_pattern")
        ts_locations = signal.ctx.meta.get("ofdm_ts_locations", [])  # (frame_idx, start_sample_in_frame)

        if tx_ts_pattern is None or not ts_locations:
            print(
                "Warning: No TX training sequence found in metadata for OFDM. "
                "LS/MMSE channel estimation may not be possible. Falling back to simple identity matrix."
            )
            # If no TS, then assume identity or use blind methods
            H_inv = np.zeros((n_pol, n_pol, n_subcarriers), dtype=np.complex64)
            for sc_idx in range(n_subcarriers):
                H_inv[:, :, sc_idx] = np.eye(n_pol)

            # Apply identity to all symbols
            for sym_idx in range(n_symbols):
                for sc_idx in range(n_subcarriers):
                    rx_vec = signal.data[:, sc_idx, sym_idx]  # Shape (n_pol,)
                    demuxed_data[:, sc_idx, sym_idx] = H_inv[:, :, sc_idx] @ rx_vec  # H_inv * Y
            return Signal(data=demuxed_data, ctx=signal.ctx)

        # Assuming tx_ts_pattern is (n_pol, n_subcarriers)
        # Assuming TS symbols are inserted periodically.

        # LS/MMSE for OFDM works per subcarrier.
        # Estimate H matrix for each subcarrier using received TS.

        estimated_H: np.ndarray = np.zeros((n_pol, n_pol, n_subcarriers), dtype=np.complex64)

        # For each subcarrier, we need RX TS and TX TS to estimate H
        # Y_ts = H * X_ts + N
        # If we have multiple TS symbols, average over them for better estimate.

        # Find actual received TS symbols (this needs to be robust)
        # For simplicity, assume first symbol of every `ts_interval` is a TS
        tx_ts_sym = tx_ts_pattern  # This is for one TS symbol

        num_ts_symbols = 0
        rx_ts_symbols_list: list[np.ndarray] = []  # List of (n_pol, n_subcarriers)

        for sym_idx in range(n_symbols):
            # This logic needs to align with how TS were inserted in OFDMModulator
            if (sym_idx % self.ts_interval == 0) and sym_idx * (
                signal.ctx.meta["ofdm_fft_size"] + signal.ctx.meta["ofdm_cp_len"]
            ) in [loc[1] for loc in ts_locations]:
                rx_ts_symbols_list.append(signal.data[:, :, sym_idx])
                num_ts_symbols += 1

        if num_ts_symbols == 0:
            print(
                "Warning: No TS symbols found in received OFDM data for channel estimation. "
                "Falling back to identity matrix."
            )
            H_inv = np.zeros((n_pol, n_pol, n_subcarriers), dtype=np.complex64)
            for sc_idx in range(n_subcarriers):
                H_inv[:, :, sc_idx] = np.eye(n_pol)
        else:
            rx_ts_avg = np.mean(np.stack(rx_ts_symbols_list, axis=2), axis=2)  # Average over TS symbols

            # Estimate H_hat per subcarrier (Y = H * X => H = Y * X^-1)
            # Y and X are 2x1 vectors for each subcarrier
            # H is 2x2 matrix for each subcarrier

            # For each subcarrier
            for sc_idx in range(n_subcarriers):
                y_sc = rx_ts_avg[:, sc_idx]  # Rx TS for this subcarrier: (2,)
                x_sc = tx_ts_sym[:, sc_idx]  # Tx TS for this subcarrier: (2,)

                # Construct Y_vec = H * X_vec
                # [y_x] = [h_xx h_xy] [x_x]
                # [y_y]   [h_yx h_yy] [x_y]
                # To solve for H, we need a "matrix" of X, or multiple (X, Y) pairs.
                # A single (X,Y) pair gives 2 equations for 4 unknowns.
                # Typically, TX transmits orthogonal pilots or known sequence over time/subcarriers.
                # Here, we assume tx_ts_sym is a known 2x1 vector.

                # Simplified LS estimation (assuming independent polarizations for pilots for this example)
                # If X_sc is a known vector, Y_sc = H * X_sc.
                # This needs to be done carefully. A more robust way is to use a 2x2 TX pilot matrix over two symbols.

                # If X_sc is actually a *matrix* of known pilots, then H = Y_sc @ np.linalg.inv(X_sc).
                # For DP, we usually use two distinct pilots for X and Y, e.g., [1, 0] and [0, 1] over two time slots.

                # Simplistic approximation: If x_sc is a single 2x1 vector, assume
                # H_xx = y_x / x_x, H_yy = y_y / x_y, H_xy = H_yx = 0 (no crosstalk)
                # This is a very poor estimate.

                # Let's assume tx_ts_sym[0, sc_idx] is X_TX, tx_ts_sym[1, sc_idx] is Y_TX
                # And we receive rx_ts_avg[0, sc_idx] as X_RX, rx_ts_avg[1, sc_idx] as Y_RX
                # For LS estimation, we need the TX "channel sounding" matrix X

                # Assuming `tx_ts_sym` is meant to be the *actual QAM symbols* transmitted on the TS (2, n_subcarriers)
                # To estimate H, we usually transmit X_tx = [[X_pilot, 0], [0, Y_pilot]] or similar orthogonal setup.

                # Placeholder: Assume the ideal case where X_tx is [1,0] and [0,1] for specific subcarriers or symbols
                # We can't form a 2x2 inverse from a single 2x1 X_sc.
                # For a 2x2 H matrix, we need two linearly independent inputs (e.g., [1,0] and [0,1])
                # on X and Y polarization to estimate the 4 components.

                # Let's assume `tx_ts_sym` is effectively a 2x2 matrix for each subcarrier for simpler illustration
                # E.g., [[tx_x_p1, tx_x_p2], [tx_y_p1, tx_y_p2]] over two time slots or two subcarriers
                # For this general framework, we'll keep a placeholder.

                # A common LS estimate for each subcarrier:
                # If we have (Y_ts_k, X_ts_k) for k=1...N_ts,
                # then H_hat = (sum Y_ts_k X_ts_k^H) (sum X_ts_k X_ts_k^H)^-1
                # Where Y_ts_k, X_ts_k are 2x1 vectors.

                # Simplest working LS for DP (assuming TX TS is the true symbols):
                # For each subcarrier, Y_rx = H_sc @ X_tx
                # We need to reshape tx_ts_sym and rx_ts_avg for this.
                # H_sc = Y_rx @ pinv(X_tx)

                # Assuming X_tx_matrix = [[tx_ts_sym[0,sc_idx], 0], [0, tx_ts_sym[1,sc_idx]]]
                # And Y_rx_matrix = [[rx_ts_avg[0,sc_idx], 0], [0, rx_ts_avg[1,sc_idx]]]
                # This isn't strictly correct for a full 2x2 H.

                # For now, a simple direct inversion per polarization for simplicity.
                # This implies no inter-polarization crosstalk.
                if np.abs(tx_ts_sym[0, sc_idx]) > 1e-6:
                    estimated_H[0, 0, sc_idx] = rx_ts_avg[0, sc_idx] / tx_ts_sym[0, sc_idx]
                else:
                    estimated_H[0, 0, sc_idx] = 1.0  # Default if TX is zero

                if np.abs(tx_ts_sym[1, sc_idx]) > 1e-6:
                    estimated_H[1, 1, sc_idx] = rx_ts_avg[1, sc_idx] / tx_ts_sym[1, sc_idx]
                else:
                    estimated_H[1, 1, sc_idx] = 1.0

                # Assume off-diagonals are zero for this simplified LS.
                # To get full H, you need to transmit two different (or orthogonal)
                # pilot symbols in the two polarizations over at least two distinct time/frequency instances.

        # Now, invert H to get H_inv (or use MMSE estimator)
        H_inv_matrix = np.zeros_like(estimated_H)
        for sc_idx in range(n_subcarriers):
            H_sc = estimated_H[:, :, sc_idx]
            if np.linalg.det(H_sc) > 1e-9:  # Check for invertibility
                H_inv_matrix[:, :, sc_idx] = np.linalg.inv(H_sc)
            else:
                H_inv_matrix[:, :, sc_idx] = np.eye(n_pol)  # Fallback to identity if singular

        # Apply H_inv to all data symbols (and potentially TS symbols if they need to be equalized)
        for sym_idx in range(n_symbols):
            for sc_idx in range(n_subcarriers):
                rx_vec = signal.data[:, sc_idx, sym_idx]  # (2,) vector
                demuxed_data[:, sc_idx, sym_idx] = H_inv_matrix[:, :, sc_idx] @ rx_vec  # H_inv * Y

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=True,
            meta={
                **(signal.ctx.meta or {}),
                "mimo_ofdm_demuxed": True,
                "ofdm_channel_estimated_H": estimated_H,
            },
        )
        return Signal(data=demuxed_data, ctx=new_ctx)

    def _sc_mimo_processing(self, signal: Signal) -> Signal:
        """单载波 (SC) 信号的 MIMO 处理 (盲均衡)。"""
        n_pol, num_samples = signal.data.shape
        demuxed_data = np.empty_like(signal.data)

        # For SC, CMA (Constant Modulus Algorithm) or MMA (Multi-Modulus Algorithm) is common.
        # This involves an adaptive equalizer (e.g., 2x2 FIR filter matrix).

        # Placeholder for CMA algorithm (simplified)
        # Initialize equalizer matrix (e.g., identity with zeros for off-diagonals)
        # Assuming an ideal 2x2 equalizer for now (no time domain taps, just 2x2 matrix)
        # For a full CMA, you need a 2x2 filter with length `equalizer_length`.
        # This implies a time-domain processing.

        # Simple 2x2 matrix processing (ignoring ISI for now)
        equalizer_matrix = np.eye(n_pol, dtype=np.complex64)  # Start with identity

        # CMA cost function: J = E[(|y_x|^2 - R)^2 + (|y_y|^2 - R)^2]
        # R is the constant modulus target (e.g., for QPSK, R=1 if normalized)
        R = 1.0  # Target modulus, assuming normalized QAM

        # Loop through symbols to adapt the equalizer (very simplified loop)
        # In practice, this would involve block-wise processing or symbol-by-symbol with filter taps.

        # Example: One-shot equalization with a single 2x2 matrix (ignoring ISI)
        # This is a highly simplified CMA, essentially just trying to de-rotate.
        # A full CMA needs an adaptive filter.

        if self.method == "CMA":
            # For a true CMA, you'd have W(z) matrix of FIR filters.
            # Y_hat = W @ X_rx
            # For simplicity here, let's treat it as a single 2x2 matrix

            # This is a very basic gradient update without full filter structure.
            # It will attempt to rotate and scale, but not truly remove ISI or PMD.
            for sample_idx in range(num_samples):
                # Input sample vector
                rx_vec = signal.data[:, sample_idx]  # (2,)

                # Output sample vector
                output_vec = equalizer_matrix @ rx_vec  # (2,)

                # CMA error for each polarization
                e_x = output_vec[0] * (np.abs(output_vec[0]) ** 2 - R)
                e_y = output_vec[1] * (np.abs(output_vec[1]) ** 2 - R)

                # Update equalizer matrix (simplified: update single 2x2 matrix)
                # J = |y_x|^2 - R + |y_y|^2 - R
                # W_new = W - mu * dJ/dW^*
                # dJ/dW^* ~ y_x * (y_x^2 - R) * x^*

                # Simplified update (treating as 2x1 input vector update, not full matrix)
                # This is more like two independent CMAs per pol, ignoring cross-polarization.
                # Full 2x2 CMA: W_k+1 = W_k - mu * (y_k y_k^H - R I) y_k x_k^H (this is complex)

                # A more common form of CMA update for a 2x2 filter:
                # E = (abs(y)^2 - R) * y
                # W_new = W_old - mu * E * x_conj

                # For simplified single-tap 2x2 matrix:
                error_vec = np.array([e_x, e_y], dtype=np.complex64)
                equalizer_matrix = equalizer_matrix - self.cma_mu * np.outer(error_vec, rx_vec.conj())

                demuxed_data[:, sample_idx] = output_vec
        else:  # Default if no CMA, or for "LS" in SC (which is rare, usually DD-LMS or CMA)
            # Just pass through with identity matrix
            demuxed_data = signal.data.copy()
            print(
                f"Warning: Single-carrier MIMO processing with method "
                f"'{self.method}' is a placeholder. "
                "No actual equalization performed."
            )

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
                "mimo_sc_demuxed": True,
                "sc_equalizer_method": self.method,
            },
        )
        return Signal(data=demuxed_data, ctx=new_ctx)
