# physics-nlo-eval

Optical Network Physical Layer Simulation — a modular, end-to-end coherent optical transmission simulation framework built on PyNLO.

## Features

### Transmitter (TX)
- **Mapper** — Bit-to-symbol mapping supporting DP/SP modulation formats: OOK, BPSK, QPSK, M-QAM (up to 256-QAM). Includes Gray coding, probability shaping (Maxwell-Boltzmann), quadrant differential encoding, and periodic pilot insertion.
- **OFDMModulator** — OFDM modulation with IFFT, cyclic prefix (CP) addition, pilot subcarrier insertion, and training sequence (TS) framing.
- **WindowedOFDMModulator (W-OFDM)** — Extends OFDM with raised-cosine time-domain windowing and overlap-and-add (OLA) using cyclic suffix (CS) for reduced out-of-band emission. ⚠️ Window function and OLA implementation are simplified; not validated against analytical OOBE reduction targets.
- **IQModulator** — Electrical-to-optical up-conversion with physical power scaling to sqrt(W) units, compatible with PyNLO's Manakov solver.
- **RRCFilter** — Root-raised cosine pulse shaping for Tx shaping and Rx matched filtering, with energy normalization.
- **PolarizationMultiplexer** — ⚠️ Conceptual placeholder. Validates DP signal presence but does not implement SP→DP multiplexing.

### WDM (Wavelength Division Multiplexing)
- **WDMPlan / ChannelSpec** — Flexible-grid frequency planning with per-channel frequency offset.
- **WDMMux** — Multi-channel up-conversion, resampling, and coherent combining into a broadband WDM signal.
- **WDMDemux** — Channel extraction via down-conversion, FIR or ideal-rectangular low-pass filtering, and downsampling. Supports roll-off-matched cutoff.
- **ChannelExtractor** — Single-channel extraction from stacked demux output.

### Fiber Channel
- **FiberChannel** — PyNLO-based Manakov equation solver. Models chromatic dispersion (GVD + TOD), polarization mode dispersion (PMD with random scattering), fiber loss, Kerr nonlinearity (SPM/XPM), and stimulated Raman scattering (ISRS). Supports GPU acceleration via CuPy.
- **EDFA** — Erbium-doped fiber amplifier with configurable gain and ASE noise. ⚠️ ASE model uses a simplified variance formula; noise bandwidth approximated by sample rate.
- **IdealAmplifier** — Amplifier that restores per-channel launch power. Supports three signal formats: single-channel, stacked WDM, and broadband WDM (frequency-domain per-channel gain with smooth transition masks). ASE noise follows P_ASE = nf · h · ν · B · (G − 1).

### Receiver Front-end
- **CoherentReceiver** — Coherent optical detection with local oscillator (LO) modeling: frequency offset, Wiener-process phase noise, and configurable LO power.
- **ADC** — Analog-to-digital conversion with finite resolution (quantization), clipping ratio, and downsampling. ⚠️ Only integer-factor downsampling; non-integer ratios produce truncation with a warning.

### DSP (Digital Signal Processing)
- **Resampler** — Polyphase FIR resampling via `scipy.signal.resample_poly`.
- **TimingRecovery** — Oerder & Meyer feed-forward timing synchronization using square-law clock tone extraction and cubic interpolation.
- **FOCoarse** — Coarse frequency offset estimation and compensation via FFT peak detection.
- **ChromaticDispersionComp (CDC)** — Frequency-domain CD compensation for GVD (β₂) and TOD (β₃), with automatic parameter inference from signal metadata.
- **OFDMChannelEstimator** — Per-subcarrier LS channel estimation and zero-forcing equalization for OFDM signals.
- **MIMO_ChannelEstimation** — Polarization demultiplexing. ⚠️ Simplified implement.
- **PhaseNoiseCompensation** — Carrier phase recovery for single-carrier signals. Implements three methods: Blind Phase Search (BPS), Viterbi & Viterbi (V&V) with pilot-assisted ambiguity/cycle-slip correction, and pilot-assisted interpolation.
- **CommonPhaseNoiseCompensate (CPE)** — OFDM common phase error estimation and compensation via pilot subcarriers.
- **OFDMToSymbolStream** — Converts OFDM 3D format `(n_pol, n_sub, n_sym)` to 2D stream format for Demapper compatibility.

### Metrics
- **Demapper** — Symbol-to-bit demapping with minimum Euclidean distance decision, pilot symbol removal, and differential decoding.
- **BERCalculator** — Bit error rate (BER) and error vector magnitude (EVM) computation, supporting OFDM and single-carrier formats, pilot removal, and header skipping.

### Core Framework
- **BaseBlock** — Abstract base class providing a unified `forward(signal)` interface.
- **Pipeline** — Sequential block execution chain with block insertion/removal/indexing.
- **Signal / SignalContext** — Unified data carrier wrapping `np.ndarray` with metadata (sample rate, center frequency, polarization count, SPS, modulation format, etc.). Supports both 2D `(n_pol, N_samples)` and 3D OFDM `(n_pol, n_sub, n_sym)` formats.

## Architecture

Simulation components are organized as modular `BaseBlock` subclasses connected via `Pipeline`. Data flows as `Signal` objects through a chain: **TX → Channel → RX Front-end → RX DSP → Metrics**.

## Dependencies

- Python ≥ 3.11, ≤ 3.13
- numpy 1.26.4, scipy, numba
- pydantic
- PyNLO (Manakov equation solver)
- click (CLI)

## Installation

```bash
poetry install
```

## Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .

# Type checking
poetry run mypy src/
```

## Tests

16 test files covering:
- Mapper/Demapper round-trip BER for all modulation formats, with AWGN validation against theoretical BER
- OFDM and Windowed-OFDM modulation/demodulation BER
- WDM mux/demux with timing recovery and phase noise
- Fiber channel numerical stability, CD walk-off, power attenuation
- Chromatic dispersion compensation (GVD + TOD)
- Timing recovery (Oerder-Meyer) accuracy with fractional delay
- Phase noise compensation (BPS vs V&V with pilot-assisted cycle-slip correction)
- EVM computation validation
- Multi-span fiber + IdealAmplifier power recovery and ASE accumulation
- Multi-span fiber with Raman (ISRS) scattering
- W-OFDM over nonlinear fiber channel with per-subcarrier BER
- QoT comparison: PyNLO simulation vs ISRS-GN analytical model

## License

GPL-3.0
