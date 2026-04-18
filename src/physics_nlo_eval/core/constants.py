"""Physical constants for optical fiber communication."""

import numpy as np

# Speed of light in vacuum (m/s)
C = 299792458.0

# Planck constant (J·s)
H = 6.62607015e-34

# Elementary charge (C)
Q_E = 1.602176634e-19

# Boltzmann constant (J/K)
K_B = 1.380649e-23

# Reference frequency for C-band (THz)
FREQ_C_BAND_CENTER = 193.1e12

# Reference wavelength for C-band (nm)
WAVELENGTH_C_BAND_CENTER = 1550.0

# Speed of light in fiber (m/s) - approximate
C_FIBER = 2.0 * C / 3.0

# Two pi constant
TWO_PI = 2.0 * np.pi
