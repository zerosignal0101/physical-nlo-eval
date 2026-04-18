"""Base classes for physics module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Signal:
    """信号数据类。

    Attributes
    ----------
    field : NDArray
        光场 (复数)。
    power_dbm : float
        功率 (dBm)。
    wavelength_nm : float
        波长 (nm)。
    """

    field: NDArray[np.complexfloating]
    power_dbm: float
    wavelength_nm: float

    @property
    def power_linear(self) -> float:
        """功率 (线性单位)。"""
        return 1e-3 * 10 ** (self.power_dbm / 10)


class ProcessingStage(ABC):
    """信号处理基类。"""

    @abstractmethod
    def process(self, signal: Signal) -> Signal:
        """处理信号。

        Parameters
        ----------
        signal : Signal
            输入信号。

        Returns
        -------
        Signal
            输出信号。
        """
        pass
