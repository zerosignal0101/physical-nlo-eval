"""Fast analytical fiber nonlinearity assessment.

The time-domain :class:`~physics_nlo_eval.channel.fiber.FiberChannel` is the
reference waveform simulator.  This module provides the missing system-level
counterpart: a deterministic, inexpensive Gaussian-noise (GN) estimate that is
useful for launch-power sweeps and route feasibility checks.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NLIChannel:
    """A rectangular WDM channel used by the analytical estimator."""

    center_freq_hz: float
    symbol_rate_hz: float
    launch_power_dbm: float
    channel_id: int | str | None = None

    def __post_init__(self) -> None:
        """Validate physical channel parameters."""
        if self.center_freq_hz <= 0 or self.symbol_rate_hz <= 0:
            raise ValueError("Channel frequency and symbol rate must be positive.")

    @property
    def launch_power_w(self) -> float:
        """Return total dual-polarization channel power in watts."""
        return 1e-3 * 10 ** (self.launch_power_dbm / 10)


@dataclass(frozen=True)
class FiberSpan:
    """Homogeneous fiber span parameters for the GN estimate."""

    length_km: float
    alpha_db_per_km: float = 0.2
    beta2_ps2_per_km: float = -21.7
    gamma_per_w_km: float = 1.3

    def __post_init__(self) -> None:
        """Validate span geometry and fiber coefficients."""
        if self.length_km <= 0:
            raise ValueError("Span length must be positive.")
        if self.alpha_db_per_km < 0 or self.gamma_per_w_km < 0:
            raise ValueError("Fiber loss and nonlinear coefficient must be non-negative.")


@dataclass(frozen=True)
class NonlinearityResult:
    """Per-channel nonlinear-interference assessment result."""

    channel_id: int | str | None
    signal_power_w: float
    nli_power_w: float
    snr_nli_db: float
    eta_per_w2: float
    sci_power_w: float
    xci_power_w: float


def _span_efficiency(span: FiberSpan, bandwidth_hz: float, spacing_hz: float) -> float:
    """Approximate GN link efficiency with loss and dispersion decorrelation."""
    alpha = span.alpha_db_per_km * np.log(10.0) / 10.0
    if alpha > 0:
        length_eff = -np.expm1(-alpha * span.length_km) / alpha
    else:
        length_eff = span.length_km
    beta2 = abs(span.beta2_ps2_per_km) * 1e-24  # s^2/km
    walkoff = 2 * np.pi**2 * beta2 * span.length_km * bandwidth_hz * spacing_hz
    return (16.0 / 27.0) * (span.gamma_per_w_km * length_eff) ** 2 / np.sqrt(1.0 + walkoff**2)


def assess_fiber_nonlinearity(
    channels: list[NLIChannel],
    spans: FiberSpan | list[FiberSpan],
    *,
    coherence_exponent: float = 0.0,
) -> list[NonlinearityResult]:
    """Estimate self- and cross-channel NLI with an incoherent GN model.

    Parameters
    ----------
    channels:
        WDM channels. Powers are total channel powers (both polarizations).
    spans:
        One span or an ordered list of spans. Amplification is assumed to
        restore channel power after every span.
    coherence_exponent:
        Optional partial-coherence correction.  Zero is incoherent span
        accumulation; values in ``[0, 1]`` scale equal spans as
        ``N ** (1 + coherence_exponent)``.

    Notes
    -----
    This closed-form rectangular-spectrum approximation is intended for rapid
    comparisons. Validate final designs with ``FiberChannel`` or a calibrated
    EGN implementation.
    """
    if not channels:
        raise ValueError("At least one WDM channel is required.")
    span_list = [spans] if isinstance(spans, FiberSpan) else list(spans)
    if not span_list:
        raise ValueError("At least one fiber span is required.")
    if not 0 <= coherence_exponent <= 1:
        raise ValueError("coherence_exponent must be between 0 and 1.")

    powers = np.asarray([ch.launch_power_w for ch in channels], dtype=float)
    results: list[NonlinearityResult] = []
    for i, cut in enumerate(channels):
        sci = 0.0
        xci = 0.0
        eta_total = 0.0
        for span in span_list:
            eta_sci = _span_efficiency(span, cut.symbol_rate_hz, cut.symbol_rate_hz)
            sci += eta_sci * powers[i] ** 3
            eta_total += eta_sci
            for j, interferer in enumerate(channels):
                if j == i:
                    continue
                spacing = max(abs(interferer.center_freq_hz - cut.center_freq_hz), cut.symbol_rate_hz / 2)
                eta_xci = 2.0 * _span_efficiency(span, interferer.symbol_rate_hz, spacing)
                xci += eta_xci * powers[i] * powers[j] ** 2
                eta_total += eta_xci * (powers[j] / powers[i]) ** 2

        if coherence_exponent and len(span_list) > 1:
            correction = len(span_list) ** coherence_exponent
            sci *= correction
            xci *= correction
            eta_total *= correction
        nli = sci + xci
        snr = np.inf if nli == 0 else 10 * np.log10(powers[i] / nli)
        results.append(
            NonlinearityResult(cut.channel_id, powers[i], nli, float(snr), eta_total, sci, xci)
        )
    return results


# Backward-friendly, discoverable name for callers looking for the repository function.
fiber_nonlinearity_assessment = assess_fiber_nonlinearity
