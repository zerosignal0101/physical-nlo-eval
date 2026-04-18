"""Carrier phase recovery (CPR) algorithms."""

import numpy as np
from numpy.typing import NDArray


def phase_recovery_viterbi(
    symbols: NDArray[np.complexfloating],
    m: int = 4,
) -> NDArray[np.complexfloating]:
    """Viterbi-Viterbi carrier phase recovery for PSK.

    Parameters
    ----------
    symbols : NDArray
        Received symbols (complex).
    m : int
        PSK order (4 for QPSK, 8 for 8PSK).

    Returns
    -------
    NDArray
        Phase-corrected symbols.
    """
    n_samples = len(symbols)

    # Mth power phase estimation
    z = np.zeros(n_samples, dtype=np.complexfloating)
    for n in range(m - 1, n_samples):
        product = np.prod(symbols[n - m + 1 : n + 1])
        z[n] = np.exp(1j * np.angle(product) / m)

    # Pilot-assisted phase recovery (simplified)
    corrected = symbols * np.exp(-1j * np.angle(z))

    return corrected
