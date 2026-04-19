# Cross-Layer Thinking Guide

> Think through data flow across layers before implementing.

---

## Purpose

Physical layer simulation involves multiple processing stages. This guide helps you think through:

- How data flows between stages
- Where transformations happen
- What state needs to be preserved
- How to structure the pipeline

---

## Layers in Physics Simulation

### 1. Signal Representation Layer

- `physics/signal.py` - Base signal representation
- What: Optical field, power, phase information
- Key types: Complex-valued arrays for I/Q components

### 2. Transmitter Layer

- `physics/tx/` - Modulation, pulse shaping, upconversion
- Input: Bit stream
- Output: Baseband complex signal

### 3. Channel Layer

- `physics/channel/` - Fiber propagation, amplification
- Input: Transmitted signal
- Output: Distorted signal (ASE, NLI, dispersion)

### 4. DSP Layer

- `physics/dsp/` - Equalization, carrier recovery
- Input: Received signal (after ADC)
- Output: Equalized symbols

### 5. Metrics Layer

- `physics/metrics/` - BER, EVM computation
- Input: Processed symbols
- Output: Performance metrics

---

## Thinking Triggers

Apply this guide when:

- [ ] Feature touches 3+ layers (tx, channel, dsp, metrics)
- [ ] Data format changes between layers (e.g., real to complex)
- [ ] Multiple consumers need the same intermediate result
- [ ] You're not sure where to put some logic

---

## Checklist

### Step 1: Map the Data Flow

```
Input bits
    ↓
TX: Mapper → OFDM/Pulse Shaping → Optical Modulator
    ↓
Channel: Fiber (GVD, Kerr) + EDFA (ASE)
    ↓
RX Front: Coherent Receiver → ADC
    ↓
DSP: CD Compensation → Equalization → CPR
    ↓
Demod: BER/EVM Calculation
```

### Step 2: Identify Interfaces

For each boundary, define:

- Data format (complex array, real array?)
- Dimensions (samples, symbols, channels?)
- Assumptions (power normalization?)

### Step 3: Consider State Management

- Does this stage need to know about previous stages?
- What needs to be passed through the pipeline?
- Are there feedback paths (e.g., CPR)?

### Step 4: Check for Cross-Cutting Concerns

- Logging/metrics at each stage?
- Error handling at boundaries?
- Numerical precision issues?

---

## Red Flags

- [ ] Module imports from non-adjacent layers
- [ ] "Helper" functions that know about multiple stages
- [ ] State passed through many layers
- [ ] Different units at different stages (dBm vs W, rad vs deg)

---

## Good Patterns

### Pipeline Pattern

```python
class ProcessingPipeline:
    def __init__(self):
        self.stages: list[ProcessingStage] = []

    def add(self, stage: ProcessingStage) -> None:
        self.stages.append(stage)

    def process(self, signal: Signal) -> Signal:
        for stage in self.stages:
            signal = stage.process(signal)
        return signal
```

### Configuration Objects

```python
class ChannelConfig(BaseModel):
    fiber_length_km: float
    attenuation_db_km: float
    dispersion_ps_nm_km: float
    edfa_gain_db: float
```

---

## Numerical Considerations

- **Sampling**: Ensure consistent sampling rates across stages
- **Normalization**: Normalize power to avoid overflow
- **Units**: Keep track of dB vs linear, rad vs deg
- **Complex vs Real**: Know when signal is complex baseband vs real passband
