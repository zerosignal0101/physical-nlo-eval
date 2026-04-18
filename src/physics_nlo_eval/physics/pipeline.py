"""Processing pipeline for physics simulation."""

from physics_nlo_eval.physics.base import ProcessingStage, Signal


class Pipeline:
    """信号处理流水线。"""

    def __init__(self) -> None:
        """初始化流水线。"""
        self.stages: list[ProcessingStage] = []

    def add(self, stage: ProcessingStage) -> None:
        """添加处理阶段。

        Parameters
        ----------
        stage : ProcessingStage
            处理阶段实例。
        """
        self.stages.append(stage)

    def process(self, signal: Signal) -> Signal:
        """通过流水线处理信号。

        Parameters
        ----------
        signal : Signal
            输入信号。

        Returns
        -------
        Signal
            处理后的信号。
        """
        for stage in self.stages:
            signal = stage.process(signal)
        return signal
