"""OFDM modulation."""

import numpy as np
from numpy.typing import NDArray


def ofdm_modulate(
    data_symbols: NDArray[np.complexfloating],
    n_fft: int,
    n_cp: int,
) -> NDArray[np.complexfloating]:
    """OFDM modulation.

    Parameters
    ----------
    data_symbols : NDArray
        Data symbols in frequency domain.
    n_fft : int
        FFT size.
    n_cp : int
        Cyclic prefix length (samples).

    Returns
    -------
    NDArray
        OFDM time-domain signal.
    """
    # IFFT to get time domain
    ofdm_symbols = np.fft.ifft(data_symbols, n=n_fft)

    # Add cyclic prefix
    cp = ofdm_symbols[-n_cp:]
    tx_signal = np.concatenate([cp, ofdm_symbols])

    return tx_signal


def ofdm_demodulate(
    signal: NDArray[np.complexfloating],
    n_fft: int,
    n_cp: int,
) -> NDArray[np.complexfloating]:
    """OFDM demodulation.

    Parameters
    ----------
    signal : NDArray
        Received OFDM signal.
    n_fft : int
        FFT size.
    n_cp : int
        Cyclic prefix length (samples).

    Returns
    -------
    NDArray
        Data symbols in frequency domain.
    """
    # Remove cyclic prefix
    ofdm_symbols = signal[n_cp : n_cp + n_fft]

    # FFT to get frequency domain
    data_symbols = np.fft.fft(ofdm_symbols, n=n_fft)

    return data_symbols
