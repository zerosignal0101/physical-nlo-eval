"""Equalization algorithms."""

import numpy as np
from numpy.typing import NDArray

from physics_nlo_eval.core.types import NDArrayFloat


def apply_cma_equalizer(
    signal: NDArray[np.complexfloating],
    num_taps: int = 15,
    mu: float = 1e-3,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """Apply CMA (Constant Modulus Algorithm) equalizer.

    Parameters
    ----------
    signal : NDArray
        Input complex signal.
    num_taps : int
        Number of equalizer taps.
    mu : float
        Step size for LMS adaptation.

    Returns
    -------
    tuple[NDArray, NDArray]
        (Equalized signal, Error signal).
    """
    n_samples = len(signal)
    w = np.ones(num_taps, dtype=np.complexfloating)
    y = np.zeros(n_samples, dtype=np.complexfloating)
    e = np.zeros(n_samples, dtype=np.complexfloating)

    for n in range(num_taps, n_samples):
        x = signal[n - num_taps : n][::-1]
        y[n] = np.dot(w.conj(), x)
        e[n] = 1.0 - np.abs(y[n]) ** 2
        w = w + mu * e[n] * y[n] * x.conj()

    return y, e
