"""Bit-to-symbol mapping."""

import numpy as np
from numpy.typing import NDArray


def qpsk_map(bits: NDArray[np.integer]) -> NDArray[np.complexfloating]:
    """Map bits to QPSK symbols.

    Parameters
    ----------
    bits : NDArray
        Input bits (0 or 1).

    Returns
    -------
    NDArray
        Complex QPSK symbols.
    """
    # Gray-coded QPSK mapping
    symbols = np.zeros(len(bits) // 2, dtype=np.complexfloating)

    for i in range(len(symbols)):
        b1, b2 = bits[2 * i], bits[2 * i + 1]
        if b1 == 0 and b2 == 0:
            symbols[i] = 1 + 1j
        elif b0 == 0 and b2 == 1:
            symbols[i] = 1 - 1j
        elif b1 == 1 and b2 == 0:
            symbols[i] = -1 + 1j
        else:
            symbols[i] = -1 - 1j

    return symbols / np.sqrt(2)


def qpsk_demap(symbols: NDArray[np.complexfloating]) -> NDArray[np.integer]:
    """Demap QPSK symbols to bits.

    Parameters
    ----------
    symbols : NDArray
        Received QPSK symbols.

    Returns
    -------
    NDArray
        Demodulated bits.
    """
    bits = np.zeros(len(symbols) * 2, dtype=np.int8)

    for i, s in enumerate(symbols):
        real_part = np.real(s)
        imag_part = np.imag(s)
        bits[2 * i] = 0 if real_part >= 0 else 1
        bits[2 * i + 1] = 0 if imag_part >= 0 else 1

    return bits
