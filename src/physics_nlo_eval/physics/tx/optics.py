"""Optical transmitter components."""

import numpy as np
from numpy.typing import NDArray


def iq_modulator(
    i_signal: NDArray[np.floating],
    q_signal: NDArray[np.floating],
    v_pi: float,
    v_b: float,
) -> NDArray[np.complexfloating]:
    """IQ modulator for optical signals.

    Parameters
    ----------
    i_signal : NDArray
        In-phase drive signal.
    q_signal : NDArray
        Quadrature drive signal.
    v_pi : float
        Half-wave voltage.
    v_b : float
        Bias voltage.

    Returns
    -------
    NDArray
        Optical field (complex).
    """
    # MZM transfer function
    i_out = np.cos(np.pi * (i_signal + v_b) / (2 * v_pi))
    q_out = np.cos(np.pi * (q_signal + v_b) / (2 * v_pi))

    # Optical field
    optical_field = i_out + 1j * q_out

    return optical_field


def laserCW(
    power_dbm: float,
    wavelength_nm: float,
    n_samples: int,
    fs: float,
) -> NDArray[np.complexfloating]:
    """Generate continuous wave laser.

    Parameters
    ----------
    power_dbm : float
        Laser power (dBm).
    wavelength_nm : float
        Wavelength (nm).
    n_samples : int
        Number of samples.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    NDArray
        Complex optical field.
    """
    power_linear = 1e-3 * 10 ** (power_dbm / 10)
    amplitude = np.sqrt(power_linear)

    optical_field = amplitude * np.ones(n_samples, dtype=np.complexfloating)

    return optical_field
