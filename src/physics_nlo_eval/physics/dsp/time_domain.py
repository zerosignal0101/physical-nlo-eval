"""Time domain DSP operations."""

import numpy as np
from numpy.typing import NDArray


def match_filter_rrc(
    signal: NDArray[np.floating],
    sps: int,
    roll_off: float,
    span: int = 10,
) -> NDArray[np.floating]:
    """Root-raised cosine matched filter.

    Parameters
    ----------
    signal : NDArray
        Input signal (real).
    sps : int
        Samples per symbol.
    roll_off : float
        Roll-off factor (0 to 1).
    span : int
        Filter span in symbols.

    Returns
    -------
    NDArray
        Matched filtered signal.
    """
    t = np.arange(-span / 2, span / 2 + 1 / sps, 1 / sps)
    t[span * sps // 2] = 1e-10  # Avoid division by zero

    # RRC impulse response
    if roll_off == 0:
        rrc = np.sinc(t) * np.cos(np.pi * roll_off * t)
    else:
        numerator = np.sin(np.pi * t * (1 - roll_off)) + 4 * roll_off * t * np.cos(np.pi * t * (1 + roll_off))
        denominator = np.pi * t * (1 - (4 * roll_off * t) ** 2)
        rrc = numerator / denominator

    # Normalize
    rrc = rrc / np.sqrt(np.sum(rrc**2))

    # Apply filter
    filtered = np.convolve(signal, rrc, mode="same")

    return filtered
