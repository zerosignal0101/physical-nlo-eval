"""Pilot-based channel estimation."""

import numpy as np
from numpy.typing import NDArray


def insert_pilot_tone(
    signal: NDArray[np.complexfloating],
    pilot_freq: float,
    pilot_power: float,
    fs: float,
) -> NDArray[np.complexfloating]:
    """Insert pilot tone for channel estimation.

    Parameters
    ----------
    signal : NDArray
        Baseband signal.
    pilot_freq : float
        Pilot tone frequency offset (Hz).
    pilot_power : float
        Pilot tone power (linear).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    NDArray
        Signal with pilot tone.
    """
    n_samples = len(signal)
    t = np.arange(n_samples) / fs

    pilot = np.sqrt(pilot_power) * np.exp(1j * 2 * np.pi * pilot_freq * t)
    signal_with_pilot = signal + pilot

    return signal_with_pilot


def estimate_channel_pilot(
    received: NDArray[np.complexfloating],
    pilot_freq: float,
    fs: float,
) -> NDArray[np.complexfloating]:
    """Estimate channel response using pilot tone.

    Parameters
    ----------
    received : NDArray
        Received signal with pilot.
    pilot_freq : float
        Pilot tone frequency (Hz).
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    NDArray
        Estimated channel response.
    """
    n_samples = len(received)
    t = np.arange(n_samples) / fs

    # Extract pilot
    pilot_ref = np.exp(1j * 2 * np.pi * pilot_freq * t)
    channel_estimate = received / pilot_ref

    return channel_estimate
