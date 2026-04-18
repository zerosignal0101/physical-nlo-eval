"""定义仿真中使用的信号数据结构。"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SignalContext:
    """信号的上下文信息。

    包含了信号的关键元数据，在信号处理过程中可能被修改或更新。

    Attributes
    ----------
    sample_rate : float
        信号的采样率 (Hz)。
    center_freq : float
        信号的中心频率 (Hz)。
    n_pol : int
        信号的偏振态数量 (单偏振 = 1, 双偏振 = 2)。
    sps : int
        每符号采样点数 (Samples per symbol)。
    sym_rate : float
        符号速率 (Symbol rate, Baud)。
    bits_per_symbol : int
        每个符号承载的比特数 (对于 M-QAM, log2(M))。
    is_ofdm : bool
        指示信号是否为 OFDM 格式。
    meta : dict[str, Any]
        存储 Ground Truth, TS 位置等辅助信息的字典。
    """

    sample_rate: float
    center_freq: float  # Hz, 可能是光载波频率，也可能是基带LO频率
    n_pol: int = 2
    sps: int = 2
    sym_rate: float = 0.0  # Derived: sample_rate / sps
    bits_per_symbol: int = 0
    is_ofdm: bool = False
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        初始化后的处理。

        确保 sym_rate 被正确计算。
        """
        if self.sps > 0:
            self.sym_rate = self.sample_rate / self.sps
        if self.meta is None:
            self.meta = {}


class Signal:
    """统一的信号数据载体。

    封装了信号的原始数据 (NumPy 数组) 和相关的上下文信息 (`SignalContext`)。
    """

    def __init__(self, data: np.ndarray, ctx: SignalContext) -> None:
        """
        初始化 Signal 对象。

        Parameters
        ----------
        data : np.ndarray
            信号数据。通常形状为 (n_pol, N_samples) 或 (n_pol, N_subcarriers, N_symbols)。
            如果 `data` 是实数数组，则表示模拟电信号；如果是复数数组，则表示基带电信号或光信号。
        ctx : SignalContext
            信号的上下文信息。
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Signal data must be a numpy.ndarray.")
        if not isinstance(ctx, SignalContext):
            raise TypeError("Signal context must be a SignalContext object.")

        self.data: np.ndarray = data
        self.ctx: SignalContext = ctx

    def to_gpu(self) -> "Signal":
        """将信号数据移动到 GPU。

        Returns
        -------
        Signal
            数据已移动到 GPU 的新 Signal 对象。

        Notes
        -----
        此方法目前为占位符。实际实现将需要 `cupy` 或其他 GPU 库。
        """
        print("Warning: GPU functionality is a placeholder and not implemented.")
        # try:
        #     import cupy as cp
        #     return Signal(cp.asarray(self.data), self.ctx)
        # except ImportError:
        #     print("CuPy not found. Data remains on CPU.")
        return self

    def to_cpu(self) -> "Signal":
        """将信号数据移动到 CPU。

        Returns
        -------
        Signal
            数据已移动到 CPU 的新 Signal 对象。

        Notes
        -----
        此方法目前为占位符。实际实现将需要 `cupy` 或其他 GPU 库。
        """
        # if isinstance(self.data, cp.ndarray): # type: ignore
        #     return Signal(self.data.get(), self.ctx) # type: ignore
        return self

    @property
    def num_samples(self) -> int:
        """获取信号的总采样点数。

        Returns
        -------
        int
            信号的采样点数。
        """
        if self.data.ndim == 2:
            return self.data.shape[1]
        if self.data.ndim == 3:  # For OFDM (n_pol, n_subcarriers, n_symbols)
            return self.data.shape[1] * self.data.shape[2]  # Total symbols * subcarriers
        return self.data.shape[0]  # Fallback for 1D

    @property
    def duration(self) -> float:
        """获取信号的总持续时间 (秒)。

        Returns
        -------
        float
            信号的持续时间。
        """
        if self.ctx.sample_rate > 0:
            return self.num_samples / self.ctx.sample_rate
        return 0.0
