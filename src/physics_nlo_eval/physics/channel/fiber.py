"""Fiber channel model."""

from pydantic import BaseModel, Field


class FiberConfig(BaseModel):
    """光纤参数配置。

    Attributes
    ----------
    length_km : float
        光纤长度 (km)。
    attenuation_db_km : float
        衰减系数 (dB/km)。
    dispersion_ps_nm_km : float
        色散系数 (ps/nm/km)。
    nonlinear_index : float
        非线性折射率 (m^2/W)。
    effective_area : float
        有效模场面积 (μm^2)。
    """

    length_km: float = Field(..., description="光纤长度 (km)")
    attenuation_db_km: float = Field(0.2, description="衰减系数 (dB/km)")
    dispersion_ps_nm_km: float = Field(17.0, description="色散系数 (ps/nm/km)")
    nonlinear_index: float = Field(2.6e-20, description="非线性折射率 (m^2/W)")
    effective_area: float = Field(80.0, description="有效模场面积 (μm^2)")


class FiberChannel:
    """光纤信道模型。"""

    def __init__(self, config: FiberConfig) -> None:
        """初始化光纤信道。

        Parameters
        ----------
        config : FiberConfig
            光纤配置参数。
        """
        self.config = config

    def propagate(self, signal, distance_km: float) -> tuple[signal, dict]:
        """传播信号指定距离。

        Parameters
        ----------
        signal : NDArrayFloat
            输入信号。
        distance_km : float
            传播距离 (km)。

        Returns
        -------
        tuple[NDArrayFloat, dict]
            (输出信号, 状态字典)。
        """
        raise NotImplementedError("Fiber propagation not yet implemented.")
