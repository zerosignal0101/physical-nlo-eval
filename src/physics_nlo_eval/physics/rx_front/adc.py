"""ADC (Analog-to-Digital Converter) model."""

from pydantic import BaseModel, Field

from physics_nlo_eval.core.types import NDArrayFloat


class ADCConfig(BaseModel):
    """ADC配置。

    Attributes
    ----------
    sampling_rate_hz : float
        采样率 (Hz)。
    resolution_bits : int
        ADC分辨率 (bits)。
    """

    sampling_rate_hz: float = Field(..., description="采样率 (Hz)")
    resolution_bits: int = Field(8, description="ADC分辨率 (bits)")


class ADC:
    """ADC模型。"""

    def __init__(self, config: ADCConfig) -> None:
        """初始化ADC。

        Parameters
        ----------
        config : ADCConfig
            ADC配置参数。
        """
        self.config = config
        self.full_scale = 2**config.resolution_bits

    def sample(
        self,
        signal: NDArrayFloat,
    ) -> NDArrayFloat:
        """采样模拟信号。

        Parameters
        ----------
        signal : NDArrayFloat
            模拟输入信号。

        Returns
        -------
        NDArrayFloat
            数字化采样信号。
        """
        raise NotImplementedError("ADC sampling not yet implemented.")
