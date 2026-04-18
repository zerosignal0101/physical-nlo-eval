"""Tests for physics module."""

import numpy as np
import pytest


def test_import_core() -> None:
    """Test core module imports."""
    from physics_nlo_eval.core.types import NDArrayFloat
    from physics_nlo_eval.core.constants import C, TWO_PI

    assert C > 0
    assert TWO_PI > 0


def test_import_signal() -> None:
    """Test signal module imports."""
    from physics_nlo_eval.physics.signal import OpticalSignal

    field = np.array([1 + 1j, -1 - 1j])
    signal = OpticalSignal(
        field=field,
        power_dbm=0.0,
        wavelength_nm=1550.0,
        symbol_rate=32e9,
    )

    assert signal.power_linear == 1e-3
