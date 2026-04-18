"""Signal representation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class OpticalSignal:
    """光信号表示。

    Attributes
    ----------
    field : NDArray
        光场复数表示 (E(t) = I + jQ)。
    power_dbm : float
        平均功率 (dBm)。
    wavelength_nm : float
        中心波长 (nm)。
    symbol_rate : float
        符号率 (Baud)。
    """

    field: NDArray[np.complexfloating]
    power_dbm: float
    wavelength_nm: float
    symbol_rate: float

    @property
    def power_linear(self) -> float:
        """功率 (线性单位, W)。"""
        return 1e-3 * 10 ** (self.power_dbm / 10)

    @property
    def samples_per_symbol(self) -> int | None:
        """采样率与符号率之比 (如果已知)。"""
        return getattr(self, "_sps", None)

    @samples_per_symbol.setter
    def samples_per_symbol(self, value: int) -> None:
        """设置采样率与符号率之比。"""
        self._sps = value
