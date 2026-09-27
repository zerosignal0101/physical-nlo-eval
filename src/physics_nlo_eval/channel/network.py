"""Optical network elements: wavelength-selective ROADMs and Raman pumps."""

from dataclasses import dataclass

import numpy as np
from scipy.constants import h

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal


class ROADM(BaseBlock):
    """Reconfigurable optical add/drop multiplexer with a WSS passband model.

    The block operates on a broadband WDM waveform. Selected channel IDs pass
    through super-Gaussian filters; all other spectrum is blocked. Cascading
    ROADMs therefore captures passband narrowing.
    """

    def __init__(
        self,
        pass_channels: list[int] | None = None,
        *,
        passband_hz: float | None = None,
        insertion_loss_db: float = 5.0,
        filter_order: int = 4,
        express_attenuation_db: dict[int, float] | None = None,
    ) -> None:
        super().__init__(name="ROADM")
        if passband_hz is not None and passband_hz <= 0:
            raise ValueError("ROADM passband must be positive.")
        if insertion_loss_db < 0 or filter_order < 1:
            raise ValueError("Insertion loss must be non-negative and filter order positive.")
        self.pass_channels = None if pass_channels is None else set(pass_channels)
        self.passband_hz = passband_hz
        self.insertion_loss_db = insertion_loss_db
        self.filter_order = filter_order
        self.express_attenuation_db = express_attenuation_db or {}

    def forward(self, signal: Signal) -> Signal:
        """Filter selected WDM channels and apply insertion/express loss."""
        if signal.data.ndim != 2 or not np.iscomplexobj(signal.data):
            raise ValueError("ROADM requires a complex broadband signal shaped (n_pol, samples).")
        info = signal.ctx.meta.get("wdm_channels_info", [])
        if not info:
            raise ValueError("ROADM requires 'wdm_channels_info' from WDMMux.")
        n = signal.data.shape[-1]
        freqs = np.fft.fftfreq(n, d=1.0 / signal.ctx.sample_rate)
        transfer_power = np.zeros(n, dtype=float)
        passed: list[int] = []
        for entry in info:
            channel_id = int(entry["channel_id"])
            if self.pass_channels is not None and channel_id not in self.pass_channels:
                continue
            bandwidth = self.passband_hz or entry.get("original_sample_rate_hz", signal.ctx.sym_rate)
            offset = float(entry["freq_offset_hz"])
            response = np.exp(-np.log(2.0) * (2 * np.abs(freqs - offset) / bandwidth) ** (2 * self.filter_order))
            attenuation = self.insertion_loss_db + self.express_attenuation_db.get(channel_id, 0.0)
            transfer_power += response * 10 ** (-attenuation / 10)
            passed.append(channel_id)
        transfer_power = np.clip(transfer_power, 0.0, 1.0)
        spectrum = np.fft.fft(signal.data, axis=-1)
        data = np.fft.ifft(spectrum * np.sqrt(transfer_power)[None, :], axis=-1)
        ctx = signal.ctx.copy()
        ctx.meta.update(
            {"roadm_pass_channels": passed, "roadm_insertion_loss_db": self.insertion_loss_db,
             "roadm_filter_order": self.filter_order}
        )
        return Signal(data.astype(signal.data.dtype, copy=False), ctx)


@dataclass(frozen=True)
class PumpSpec:
    """Physical Raman pump specification."""

    wavelength_nm: float
    power_w: float
    direction: str = "backward"
    attenuation_db_per_km: float = 0.25

    def __post_init__(self) -> None:
        """Validate pump wavelength, power, loss, and direction."""
        if self.wavelength_nm <= 0 or self.power_w < 0 or self.attenuation_db_per_km < 0:
            raise ValueError("Pump wavelength must be positive; power and loss must be non-negative.")
        if self.direction not in {"forward", "backward"}:
            raise ValueError("Pump direction must be 'forward' or 'backward'.")


class RamanPump(BaseBlock):
    """Lumped equivalent of distributed Raman amplification from one or more pumps.

    Gain is evaluated across the sampled WDM band with a Gaussian approximation
    to the silica Raman-gain spectrum. This is suitable for link budgeting; use
    a measured ``r3`` response in ``FiberChannel`` for waveform-level ISRS.
    """

    def __init__(
        self,
        pumps: PumpSpec | list[PumpSpec],
        length_km: float,
        *,
        effective_area_um2: float = 80.0,
        peak_gain_coefficient_m_per_w: float = 6e-14,
        add_noise: bool = True,
        noise_figure_db: float = 1.5,
        max_gain_db: float = 30.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name="RamanPump")
        self.pumps = [pumps] if isinstance(pumps, PumpSpec) else list(pumps)
        if not self.pumps or length_km <= 0 or effective_area_um2 <= 0:
            raise ValueError("At least one pump and positive length/effective area are required.")
        self.length_km = length_km
        self.effective_area_m2 = effective_area_um2 * 1e-12
        self.gain_coefficient = peak_gain_coefficient_m_per_w
        self.add_noise = add_noise
        self.noise_figure = 10 ** (noise_figure_db / 10)
        self.max_gain_db = max_gain_db
        self.rng = np.random.default_rng(seed)

    def _gain(self, frequencies_hz: np.ndarray) -> np.ndarray:
        c0 = 299_792_458.0
        exponent = np.zeros_like(frequencies_hz, dtype=float)
        for pump in self.pumps:
            pump_freq = c0 / (pump.wavelength_nm * 1e-9)
            shift = pump_freq - frequencies_hz
            # Silica Raman peak near 13.2 THz, approximately 5 THz FWHM.
            spectral = np.exp(-0.5 * ((shift - 13.2e12) / 2.12e12) ** 2)
            alpha = pump.attenuation_db_per_km * np.log(10) / 10
            leff_km = self.length_km if alpha == 0 else -np.expm1(-alpha * self.length_km) / alpha
            exponent += self.gain_coefficient * pump.power_w * leff_km * 1e3 / self.effective_area_m2 * spectral
        return np.minimum(np.exp(exponent), 10 ** (self.max_gain_db / 10))

    def forward(self, signal: Signal) -> Signal:
        """Apply frequency-dependent Raman gain and optional spontaneous noise."""
        if signal.data.ndim != 2 or not np.iscomplexobj(signal.data):
            raise ValueError("RamanPump requires a complex signal shaped (n_pol, samples).")
        n = signal.data.shape[-1]
        rel_freq = np.fft.fftfreq(n, d=1 / signal.ctx.sample_rate)
        optical_freq = signal.ctx.center_freq + rel_freq
        gain = self._gain(optical_freq)
        spectrum = np.fft.fft(signal.data, axis=-1) * np.sqrt(gain)[None, :]
        data = np.fft.ifft(spectrum, axis=-1)
        if self.add_noise:
            noise_psd = 0.5 * self.noise_figure * h * optical_freq * np.maximum(gain - 1, 0)
            variance = float(np.mean(noise_psd) * signal.ctx.sample_rate)
            noise = np.sqrt(variance / 2) * (
                self.rng.standard_normal(data.shape) + 1j * self.rng.standard_normal(data.shape)
            )
            data += noise
        ctx = signal.ctx.copy()
        ctx.meta.update({"raman_gain_db_min": float(10 * np.log10(gain.min())),
                         "raman_gain_db_max": float(10 * np.log10(gain.max())),
                         "raman_pump_count": len(self.pumps)})
        return Signal(data.astype(signal.data.dtype, copy=False), ctx)
