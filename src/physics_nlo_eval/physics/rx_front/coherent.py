"""Coherent receiver model."""

from pydantic import BaseModel, Field

from physics_nlo_eval.core.types import NDArrayFloat


class CoherentReceiverConfig(BaseModel):
    """相干接收机配置。

    Attributes
    ----------
    line_width_hz : float
        激光线宽 (Hz)。
    local_oscillator_power_dbm : float
        本振功率 (dBm)。
    """

    line_width_hz: float = Field(100e3, description="激光线宽 (Hz)")
    local_oscillator_power_dbm: float = Field(10.0, description="本振功率 (dBm)")


class CoherentReceiver:
    """相干接收机模型。"""

    def __init__(self, config: CoherentReceiverConfig) -> None:
        """初始化相干接收机。

        Parameters
        ----------
        config : CoherentReceiverConfig
            接收机配置参数。
        """
        self.config = config

    def detect(
        self,
        signal: NDArrayFloat,
        lo_field: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """相干检测。

        Parameters
        ----------
        signal : NDArrayFloat
            接收到的光信号。
        lo_field : NDArrayFloat
            本振光场。

        Returns
        -------
        tuple[NDArrayFloat, NDArrayFloat]
            (I路信号, Q路信号)。
        """
        raise NotImplementedError("Coherent detection not yet implemented.")
