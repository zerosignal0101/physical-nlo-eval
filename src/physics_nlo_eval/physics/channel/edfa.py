"""EDFA (Erbium-Doped Fiber Amplifier) model."""

from pydantic import BaseModel, Field


class EDFAConfig(BaseModel):
    """EDFA参数配置。

    Attributes
    ----------
    gain_db : float
        增益 (dB)。
    noise_figure_db : float
        噪声系数 (dB)。
    """

    gain_db: float = Field(..., description="增益 (dB)")
    noise_figure_db: float = Field(5.0, description="噪声系数 (dB)")


class EDFA:
    """EDFA放大器模型。"""

    def __init__(self, config: EDFAConfig) -> None:
        """初始化EDFA。

        Parameters
        ----------
        config : EDFAConfig
            EDFA配置参数。
        """
        self.config = config

    def amplify(self, signal, gain_db: float | None = None) -> tuple[signal, dict]:
        """放大信号。

        Parameters
        ----------
        signal : NDArrayFloat
            输入信号。
        gain_db : float, optional
            增益 (dB)，使用配置的增益如果为None。

        Returns
        -------
        tuple[NDArrayFloat, dict]
            (输出信号, 状态字典)。
        """
        raise NotImplementedError("EDFA amplification not yet implemented.")
