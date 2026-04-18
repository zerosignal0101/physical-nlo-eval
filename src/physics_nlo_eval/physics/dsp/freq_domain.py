"""Frequency domain DSP operations."""

import numpy as np
from numpy.typing import NDArray

from physics_nlo_eval.core.constants import TWO_PI


def compensate_dispersion(
    signal: NDArray[np.complexfloating],
    beta2: float,
    L: float,
    fs: float,
) -> NDArray[np.complexfloating]:
    """Compensate chromatic dispersion in frequency domain.

    Parameters
    ----------
    signal : NDArray
        Input complex signal.
    beta2 : float
        Group velocity dispersion parameter (ps^2/km).
    L : float
        Propagation distance (km).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    NDArray
        Dispersion-compensated signal.
    """
    n_samples = len(signal)
    f = np.fft.fftfreq(n_samples, 1 / fs)

    # Dispersion transfer function
    H = np.exp(1j * beta2 * L * (TWO_PI * f) ** 2 / 2)

    # Frequency domain compensation
    signal_ft = np.fft.fft(signal)
    compensated_ft = signal_ft * H
    compensated = np.fft.ifft(compensated_ft)

    return compensated
