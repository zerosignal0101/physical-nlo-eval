"""载波相位恢复模块。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class PhaseNoiseCompensation(BaseBlock):
    """载波相位噪声补偿模块。

    用于补偿残余的载波相位噪声。

    Parameters
    ----------
    method : str, optional
        相位噪声补偿方法，例如 'BPS' (盲相扫), 'V&V' (Viterbi-Viterbi), 'pilot_assisted'。
    n_test_phases : int, optional
        BPS 方法中测试的相位数量。
    bps_block_size : int, optional
        BPS 方法中处理的符号块大小。
    """

    def __init__(
        self,
        method: str = "BPS",
        n_test_phases: int = 32,
        bps_block_size: int = 128,
    ) -> None:
        """初始化 PhaseNoiseCompensation。"""
        super().__init__(name=f"PhaseNoiseComp({method})")
        if method not in ["BPS", "V&V", "pilot_assisted"]:
            raise ValueError(f"Unsupported phase noise compensation method: {method}.")

        self.method = method
        self.n_test_phases = n_test_phases
        self.bps_block_size = bps_block_size

    def forward(self, signal: Signal) -> Signal:
        """
        执行载波相位噪声补偿。

        Parameters
        ----------
        signal : Signal
            输入的信号。数据形状为 (n_pol, N_samples) (单载波)
            或 (n_pol, N_subcarriers, N_symbols) (OFDM)。

        Returns
        -------
        Signal
            补偿相位噪声后的信号。
        """
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex.")

        if signal.ctx.is_ofdm:
            return self._ofdm_phase_noise_comp(signal)
        return self._sc_phase_noise_comp(signal)

    def _sc_phase_noise_comp(self, signal: Signal) -> Signal:
        """单载波 (SC) 信号的相位噪声补偿。"""
        n_pol, num_symbols = signal.data.shape
        compensated_data = np.empty_like(signal.data)

        for pol_idx in range(n_pol):
            pol_data = signal.data[pol_idx, :]

            if self.method == "BPS":
                # Blind Phase Search (BPS)
                compensated_pol_data = np.zeros_like(pol_data)

                # Iterate over blocks
                for i in range(0, num_symbols, self.bps_block_size):
                    block_start = i
                    block_end = min(i + self.bps_block_size, num_symbols)
                    current_block = pol_data[block_start:block_end]

                    if current_block.size == 0:
                        continue

                    best_phase_error = 0.0
                    min_cost = np.inf

                    # Test multiple phases
                    test_phases = np.linspace(-np.pi / 4, np.pi / 4, self.n_test_phases)  # Typical for QPSK, 16QAM

                    for phase_offset in test_phases:
                        rotated_block = current_block * np.exp(1j * phase_offset)

                        # Cost function: sum of squared error to nearest constellation point
                        # This requires knowing the constellation (e.g., from ctx.bits_per_symbol)
                        # For simplicity, assume constellation is roughly on a circle of radius 1 (normalized QAM)
                        # Cost = sum( abs(rotated_block)**2 - R )^2 or sum(dist to ideal point)

                        # A common BPS cost function for QAM: sum of |Re(symbol)| - |Im(symbol)|
                        # This works for square QAM, as it tries to align symbols to axes.
                        cost = np.sum(np.abs(np.abs(rotated_block.real) - np.abs(rotated_block.imag)))

                        if cost < min_cost:
                            min_cost = cost
                            best_phase_error = phase_offset

                    compensated_pol_data[block_start:block_end] = current_block * np.exp(1j * best_phase_error)
                compensated_data[pol_idx, :] = compensated_pol_data

            elif self.method == "V&V":
                # Viterbi & Viterbi algorithm (usually for M-PSK/QAM)
                # Works on (Rx_k)^M to remove modulation, then track phase.
                # For 16QAM, M=4 is common (raise to power of 4).

                # Placeholder: This is more complex than BPS.
                # For simplicity, if not BPS, just pass through.
                compensated_data[pol_idx, :] = pol_data.copy()
                print("Warning: V&V method is a placeholder and not fully implemented.")

            else:  # Default or unsupported
                compensated_data[pol_idx, :] = pol_data.copy()

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
                "sc_phase_noise_compensated": True,
                "sc_phase_noise_method": self.method,
            },
        )
        return Signal(data=compensated_data, ctx=new_ctx)

    def _ofdm_phase_noise_comp(self, signal: Signal) -> Signal:
        """OFDM 信号的相位噪声补偿。"""
        n_pol, n_subcarriers, n_symbols = signal.data.shape
        compensated_data = signal.data.copy()

        # OFDM phase noise compensation often uses pilots (common phase error)
        # or differential detection.

        if self.method == "pilot_assisted":
            # Requires pilot subcarrier locations and expected values from metadata
            # For each OFDM symbol, estimate a common phase error (CPE)
            # using the pilots, then correct all subcarriers in that symbol.

            # Placeholder:
            # pilot_subcarriers_indices = signal.ctx.meta.get("ofdm_pilot_subcarriers_indices")
            # expected_pilot_symbols = signal.ctx.meta.get("ofdm_expected_pilot_symbols")

            # if pilot_subcarriers_indices is None or expected_pilot_symbols is None:
            #     print("Warning: Pilot data not found for pilot-assisted compensation. Skipping.")
            #     return signal

            # for pol_idx in range(n_pol):
            #     for sym_idx in range(n_symbols):
            #         rx_pilots = signal.data[pol_idx, pilot_subcarriers_indices, sym_idx]
            #         # Estimate CPE
            #         cpe = np.angle(np.mean(rx_pilots * expected_pilot_symbols.conj()))
            #         # Apply compensation
            #         compensated_data[pol_idx, :, sym_idx] *= np.exp(-1j * cpe)
            pass  # TODO: Implement pilot-assisted for OFDM
            print("Warning: OFDM pilot-assisted method is a placeholder and not implemented.")

        elif self.method == "BPS":
            # BPS for OFDM is similar to SC, but applied per OFDM symbol (across subcarriers)
            # or per block of symbols.
            # Estimate a single CPE per OFDM symbol (or block of symbols).
            for pol_idx in range(n_pol):
                for sym_idx in range(n_symbols):
                    current_symbol_subcarriers = signal.data[pol_idx, :, sym_idx]

                    if current_symbol_subcarriers.size == 0:
                        continue

                    best_phase_error = 0.0
                    min_cost = np.inf
                    test_phases = np.linspace(-np.pi / 4, np.pi / 4, self.n_test_phases)

                    for phase_offset in test_phases:
                        rotated_subcarriers = current_symbol_subcarriers * np.exp(1j * phase_offset)
                        cost = np.sum(np.abs(np.abs(rotated_subcarriers.real) - np.abs(rotated_subcarriers.imag)))

                        if cost < min_cost:
                            min_cost = cost
                            best_phase_error = phase_offset

                    compensated_data[pol_idx, :, sym_idx] = current_symbol_subcarriers * np.exp(1j * best_phase_error)

        else:  # V&V or other unsupported methods
            print("Warning: OFDM phase noise compensation method is a placeholder or not implemented.")
            compensated_data = signal.data.copy()

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
                "ofdm_phase_noise_compensated": True,
                "ofdm_phase_noise_method": self.method,
            },
        )
        return Signal(data=compensated_data, ctx=new_ctx)
