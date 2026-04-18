"""BER and EVM metrics."""

import numpy as np
from numpy.typing import NDArray
from scipy import special


def calculate_evm(
    tx_symbols: NDArray[np.complexfloating],
    rx_symbols: NDArray[np.complexfloating],
) -> float:
    """Calculate Error Vector Magnitude (EVM).

    Parameters
    ----------
    tx_symbols : NDArray
        Transmitted symbols.
    rx_symbols : NDArray
        Received symbols.

    Returns
    -------
    float
        EVM as a percentage.
    """
    error = rx_symbols - tx_symbols
    rms_error = np.sqrt(np.mean(np.abs(error) ** 2))
    tx_power = np.sqrt(np.mean(np.abs(tx_symbols) ** 2))
    evm = rms_error / tx_power * 100

    return evm


def calculate_ber_qpsk(evm: float) -> float:
    """Calculate BER from EVM for QPSK.

    Parameters
    ----------
    evm : float
        EVM as percentage.

    Returns
    -------
    float
        Estimated BER.
    """
    # Convert EVM to SNR
    evm_linear = evm / 100
    snr = 1 / (evm_linear**2)

    # QPSK BER approximation
    ber = 0.5 * special.erfc(np.sqrt(snr / 2))

    return ber
