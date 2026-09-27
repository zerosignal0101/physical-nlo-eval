# physics-nlo-eval

Optical Network Physical Layer Simulation — a modular, end-to-end coherent optical transmission simulation framework built on PyNLO.

## Features

### Transmitter (TX)
- **Mapper** — Bit-to-symbol mapping supporting DP/SP modulation formats: OOK, BPSK, QPSK, M-QAM (up to 256-QAM). Includes Gray coding, probability shaping (Maxwell-Boltzmann), quadrant differential encoding, and periodic pilot insertion.
- **OFDMModulator** — OFDM modulation with IFFT, cyclic prefix (CP) addition, pilot subcarrier insertion, and training sequence (TS) framing.
- **WindowedOFDMModulator (W-OFDM)** — Extends OFDM with complementary raised-cosine time-domain windowing and constant-overlap-add (COLA) using a cyclic suffix (CS) for reduced out-of-band emission.
- **IQModulator** — Electrical-to-optical up-conversion with physical power scaling to sqrt(W) units, compatible with PyNLO's Manakov solver.
- **RRCFilter** — Root-raised cosine pulse shaping for Tx shaping and Rx matched filtering, with energy normalization.
- **PolarizationMultiplexer** — Combines two compatible SP fields into a DP signal through `combine(x_pol, y_pol)` and supports Pipeline metadata input.

### WDM (Wavelength Division Multiplexing)
- **WDMPlan / ChannelSpec** — Flexible-grid frequency planning with per-channel frequency offset.
- **WDMMux** — Multi-channel up-conversion, resampling, and coherent combining into a broadband WDM signal.
- **WDMDemux** — Channel extraction via down-conversion, FIR or ideal-rectangular low-pass filtering, and downsampling. Supports roll-off-matched cutoff.
- **ChannelExtractor** — Single-channel extraction from stacked demux output.

### Fiber Channel
- **FiberChannel** — Manakov split-step solver modeling chromatic dispersion (GVD + TOD), PMD, fiber loss, Kerr nonlinearity (SPM/XPM), and stimulated Raman scattering (ISRS). Uses PyNLO/CuPy when available (GPU acceleration on CUDA) and includes a self-contained NumPy fallback.
- **Fiber nonlinearity assessment** — Fast analytical GN approximation (`assess_fiber_nonlinearity`) reporting SCI, XCI, total NLI and NLI-limited SNR for WDM routes.
- **EDFA** — Erbium-doped fiber amplifier with configurable gain, noise figure, optical noise bandwidth, and reproducible ASE noise.
- **IdealAmplifier** — Amplifier that restores per-channel launch power. Supports three signal formats: single-channel, stacked WDM, and broadband WDM (frequency-domain per-channel gain with smooth transition masks). ASE noise follows P_ASE = nf · h · ν · B · (G − 1).
- **ROADM** — Wavelength-selective super-Gaussian passband, add/drop selection, insertion loss, and per-channel express attenuation.
- **RamanPump / PumpSpec** — Forward/backward distributed-pump link-budget model with wavelength-dependent silica Raman gain and spontaneous-noise contribution.

### Receiver Front-end
- **CoherentReceiver** — Coherent optical detection with local oscillator (LO) modeling: frequency offset, Wiener-process phase noise, and configurable LO power.
- **ADC** — Analog-to-digital conversion with finite resolution, RMS-referenced clipping, physical-amplitude-preserving quantization, and rational polyphase downsampling.

### DSP (Digital Signal Processing)
- **Resampler** — Polyphase FIR resampling via `scipy.signal.resample_poly`.
- **TimingRecovery** — Oerder & Meyer feed-forward timing synchronization using square-law clock tone extraction and cubic interpolation.
- **FOCoarse** — Coarse frequency offset estimation and compensation via FFT peak detection.
- **ChromaticDispersionComp (CDC)** — Frequency-domain CD compensation for GVD (β₂) and TOD (β₃), with automatic parameter inference from signal metadata.
- **OFDMChannelEstimator** — Per-subcarrier LS channel estimation and zero-forcing equalization for OFDM signals.
- **MIMO_ChannelEstimation** — Full 2×2 LS/MMSE polarization-channel estimation for OFDM and single carrier, plus adaptive butterfly-FIR CMA.
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

## Fast nonlinearity assessment

Define rectangular WDM channels and fiber spans in JSON, then run:

```bash
poetry run physics-nlo-eval assess --config examples/nli_assessment.json
```

The output reports signal power, SCI, XCI, total NLI, effective eta, and the
NLI-limited SNR for every channel. This GN approximation is intended for route
screening and power sweeps; use `FiberChannel` for waveform validation.

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
