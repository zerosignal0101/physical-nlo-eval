# Directory Structure

> How backend code is organized in this project.

---

## Overview

This is a **scientific computing / simulation project** for optical network physical layer evaluation. The codebase is organized around domain concepts rather than technical layers.

---

## Directory Layout

```
src/physics_nlo_eval/
├── __init__.py           # Package init (can contain __version__)
├── __main__.py           # CLI entry point using click
├── core/                 # Core types, constants
│   ├── types.py          # Type aliases (NDArrayFloat, etc.)
│   └── constants.py      # Physical constants
├── physics/              # Physical layer models
│   ├── signal.py        # Signal representation
│   ├── channel/         # Channel models (fiber, edfa)
│   │   ├── fiber.py
│   │   └── edfa.py
│   ├── dsp/             # Digital signal processing
│   │   ├── equalizer.py
│   │   ├── cpr.py
│   │   ├── time_domain.py
│   │   └── freq_domain.py
│   ├── metrics/         # Performance metrics (BER, EVM)
│   │   └── ber_evm.py
│   ├── tx/              # Transmitter models
│   │   ├── mapper.py
│   │   ├── ofdm.py
│   │   ├── optics.py
│   │   └── pilot.py
│   ├── rx_front/        # Receiver front-end
│   │   ├── coherent.py
│   │   └── adc.py
│   ├── base.py          # Base classes
│   └── pipeline.py      # Processing pipeline
```

---

## Module Organization

### Core Principles

1. **Domain-driven organization** - Modules are grouped by domain concept (physics, dsp, tx, rx)
2. **No traditional "layers"** - This is not an API service; it's a physics simulation library
3. **Clear entry points** - `__main__.py` for CLI, `__init__.py` for package API

### Key Patterns

- **Type aliases** - `core/types.py` defines `NDArrayFloat`, etc.
- **Config models** - Use Pydantic `BaseModel` with `Field()` descriptions
- **Data models** - Use `@dataclass` for plain data carriers
- **Signal processing** - Functional style with numpy arrays

---

## Naming Conventions

- Python files: `snake_case.py` (e.g., `fiber.py`, `equalizer.py`)
- Classes: `PascalCase` (e.g., `CoherentReceiver`, `FiberChannel`)
- Functions/methods: `snake_case` (e.g., `compensate_dispersion()`, `equalize_cpr()`)
- Type aliases: `PascalCase` suffix (e.g., `NDArrayFloat`)
- Constants: `UPPER_SNAKE_CASE` (in `core/constants.py`)
- Private members: `_leading_underscore`

---

## Examples

### Well-Organized Modules

- `physics/channel/fiber.py` - Fiber channel model with proper docstrings
- `physics/dsp/equalizer.py` - DSP equalization algorithms
- `physics/metrics/ber_evm.py` - Performance metrics calculations

### Adding a New Module

1. Place it in the appropriate domain directory
2. Use Pydantic `BaseModel` for configuration/data validation
3. Use `@dataclass` for plain data carriers
4. Use `numpy.typing.NDArray` type aliases from `core/types.py`
5. Add Chinese comments for domain-specific explanations

---

## Anti-Patterns

- **DO NOT** create "utils" or "helpers" directories - prefer organizing code in domain modules
- **DO NOT** use classes where simple functions suffice (e.g., pure calculation functions)
- **DO NOT** import `numpy` as `np` in type alias files - keep type aliases clean for static analysis
