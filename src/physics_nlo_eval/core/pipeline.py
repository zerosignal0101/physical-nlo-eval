"""用于组织和执行仿真模块的管道。"""

from collections.abc import Iterable

from physics_nlo_eval.core.base import BaseBlock, T_Signal


class Pipeline(BaseBlock):
    """一个将多个处理模块按顺序连接起来的管道。

    数据将依次通过管道中的每个模块。

    Parameters
    ----------
    blocks : Iterable[BaseBlock]
        按顺序执行的处理模块列表。
    name : str, optional
        管道的名称，用于日志或调试。

    Examples
    --------
    >>> from physics_nlo_eval.core.signal import Signal, SignalContext
    >>> import numpy as np
    >>> class AddOne(BaseBlock):
    ...     def forward(self, signal: Signal) -> Signal:
    ...         signal.data += 1
    ...         return signal
    >>> ctx = SignalContext(sample_rate=1e9, center_freq=0)
    >>> initial_signal = Signal(np.array([[1.0]]), ctx)
    >>> pipeline = Pipeline([AddOne(), AddOne()])
    >>> result = pipeline(initial_signal)
    >>> print(result.data)
    [[3.]]
    """

    def __init__(self, blocks: Iterable[BaseBlock], name: str = "Pipeline") -> None:
        """初始化 Pipeline。"""
        super().__init__(name=name)
        self.blocks: list[BaseBlock] = list(blocks)
        if not all(isinstance(b, BaseBlock) for b in self.blocks):
            raise TypeError("All elements in the pipeline must be instances of BaseBlock.")

    def forward(self, signal: T_Signal) -> T_Signal:
        """按顺序通过管道中的所有模块处理信号。

        Parameters
        ----------
        signal : Signal
            待处理的输入信号。

        Returns
        -------
        Signal
            经过所有模块处理后的输出信号。
        """
        current_signal: T_Signal = signal
        for block in self.blocks:
            current_signal = block(current_signal)
        return current_signal

    def __getitem__(self, idx: int) -> BaseBlock:
        """
        允许通过索引访问管道中的模块。

        Parameters
        ----------
        idx : int
            模块的索引。

        Returns
        -------
        BaseBlock
            指定索引处的模块。
        """
        return self.blocks[idx]

    def __len__(self) -> int:
        """
        获取管道中的模块数量。

        Returns
        -------
        int
            模块数量。
        """
        return len(self.blocks)

    def add_block(self, block: BaseBlock) -> None:
        """
        向管道末尾添加一个新模块。

        Parameters
        ----------
        block : BaseBlock
            要添加的新模块。
        """
        if not isinstance(block, BaseBlock):
            raise TypeError("Added block must be an instance of BaseBlock.")
        self.blocks.append(block)

    def insert_block(self, index: int, block: BaseBlock) -> None:
        """
        在指定位置插入一个新模块。

        Parameters
        ----------
        index : int
            插入位置的索引。
        block : BaseBlock
            要插入的新模块。
        """
        if not isinstance(block, BaseBlock):
            raise TypeError("Inserted block must be an instance of BaseBlock.")
        self.blocks.insert(index, block)

    def remove_block(self, block_name: str) -> None:
        """
        根据模块名称移除管道中的模块。

        Parameters
        ----------
        block_name : str
            要移除模块的名称。

        Raises
        ------
        ValueError
            如果找不到指定名称的模块。
        """
        original_len = len(self.blocks)
        self.blocks = [block for block in self.blocks if block.name != block_name]
        if len(self.blocks) == original_len:
            raise ValueError(f"Block with name '{block_name}' not found in pipeline.")
