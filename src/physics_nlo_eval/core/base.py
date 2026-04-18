"""所有处理模块的基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from physics_nlo_eval.core.signal import Signal

# Signal is defined in signal.py, imported lazily to avoid circular import
T_Signal = TypeVar("T_Signal", bound="Signal")


class BaseBlock(ABC):
    """所有仿真处理模块的抽象基类。

    提供一个统一的 `forward` 接口，以便在 Pipeline 中进行链式调用。
    """

    def __init__(self, name: str | None = None) -> None:
        """
        初始化 BaseBlock。

        Parameters
        ----------
        name : str, optional
            模块的名称，用于日志或调试。如果为 None, 则使用类名。
        """
        self.name = name if name is not None else self.__class__.__name__

    @abstractmethod
    def forward(self, signal: T_Signal) -> T_Signal:
        """核心处理逻辑。

        子类必须实现此方法以定义其功能。

        Parameters
        ----------
        signal : Signal
            待处理的输入信号对象。

        Returns
        -------
        Signal
            处理后的输出信号对象。
        """
        raise NotImplementedError

    def __call__(self, signal: T_Signal) -> T_Signal:
        """允许模块像函数一样被调用，并包装 `forward` 方法。

        可以在此处添加统一的日志、性能测量或可视化探针等功能。

        Parameters
        ----------
        signal : Signal
            待处理的输入信号对象。

        Returns
        -------
        Signal
            处理后的输出信号对象。
        """
        # print(f"Processing with {self.name}...") # 示例：添加日志
        return self.forward(signal)
