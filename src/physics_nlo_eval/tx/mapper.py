"""实现比特到符号的映射，支持 QAM 和 PS-QAM。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class Mapper(BaseBlock):
    """比特到符号的映射模块。

    支持 QAM 调制，并可扩展支持概率整形 (PS-QAM)。

    Parameters
    ----------
    modulation : str
        调制格式，例如 'DP-16QAM', 'SP-64QAM'。
    power_norm : bool, optional
        是否进行功率归一化，默认为 True。
    prob_shaping : dict[str, float] | None, optional
        概率整形参数，如果为 None 则为均匀 QAM。
        目前仅为占位符，需要根据具体 PS 算法实现。
    """

    _constellations: dict[str, np.ndarray] = {}

    def __init__(
        self,
        modulation: str = "DP-16QAM",
        power_norm: bool = True,
        prob_shaping: dict[str, float] | None = None,
    ) -> None:
        """初始化 Mapper。"""
        super().__init__(name=f"Mapper({modulation})")
        self.modulation = modulation
        self.power_norm = power_norm
        self.prob_shaping = prob_shaping
        self.n_pol: int
        self.m_ary: int
        self.bits_per_symbol: int

        self._parse_modulation()
        self._generate_constellation()

    def _parse_modulation(self) -> None:
        """解析调制格式字符串，提取偏振数量和 M 值。"""
        parts = self.modulation.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid modulation string: {self.modulation}. Expected format 'DP-16QAM'.")

        pol_type, qam_m = parts[0], parts[1]

        if pol_type.upper() == "DP":
            self.n_pol = 2
        elif pol_type.upper() == "SP":
            self.n_pol = 1
        else:
            raise ValueError(f"Invalid polarization type: {pol_type}. Expected 'DP' or 'SP'.")

        if not qam_m.endswith("QAM"):
            raise ValueError(f"Invalid QAM type: {qam_m}. Expected '16QAM' or '64QAM'.")

        self.m_ary = int(qam_m[:-3])
        if not np.log2(self.m_ary).is_integer():
            raise ValueError(f"M must be a power of 2: {self.m_ary}.")
        self.bits_per_symbol = int(np.log2(self.m_ary))

    def _generate_constellation(self) -> None:
        """生成 QAM 星座图。"""
        key = f"{self.m_ary}QAM_norm={self.power_norm}"
        if key in Mapper._constellations:
            return

        sqrt_m = int(np.sqrt(self.m_ary))
        if not sqrt_m * sqrt_m == self.m_ary:
            raise ValueError(f"Only square QAM constellations are supported for M={self.m_ary}.")

        # 生成[-sqrt_m+1, ..., sqrt_m-1]的整数点
        levels = np.arange(-sqrt_m + 1, sqrt_m, 2)
        # 生成星座点 (IQ 坐标)
        constellation = (levels[np.newaxis, :] + 1j * levels[:, np.newaxis]).flatten()

        if self.power_norm:
            # 归一化平均功率为 1
            constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))

        Mapper._constellations[key] = constellation

    def _map_bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        将比特流映射到 QAM 符号。

        Parameters
        ----------
        bits : np.ndarray
            输入比特流，形状为 (N_bits,)。

        Returns
        -------
        np.ndarray
            QAM 符号数组，形状为 (N_symbols,)。
        """
        if bits.ndim != 1 or bits.dtype != bool:  # type: ignore [attr-defined]
            raise ValueError("Input bits must be a 1D boolean array.")

        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(
                f"Number of bits ({len(bits)}) must be a multiple of bits_per_symbol ({self.bits_per_symbol})."
            )

        constellation = Mapper._constellations[f"{self.m_ary}QAM_norm={self.power_norm}"]

        # 将比特流重塑为 (N_symbols, bits_per_symbol)
        bits_reshaped = bits.reshape(-1, self.bits_per_symbol)

        # Gray 码映射 (简化版，实际需要完整 Gray 码逻辑)
        # 这里为简便直接转换为整数索引，实际工程中应使用 Gray 码查找表
        indices = np.sum(bits_reshaped * (2 ** np.arange(self.bits_per_symbol - 1, -1, -1)), axis=1)

        # 应用概率整形 (占位符)
        if self.prob_shaping:
            print("Warning: Probability shaping is a placeholder and not implemented.")

        return constellation[indices]

    def forward(self, bits: np.ndarray) -> Signal:
        """
        将输入比特流映射为双偏振或单偏振的 QAM 符号。

        Parameters
        ----------
        bits : np.ndarray
            输入的比特流，形状为 (N_bits,)，数据类型为 bool 或 int (0/1)。

        Returns
        -------
        Signal
            包含映射后的 QAM 符号的信号对象。
            数据形状为 (n_pol, N_symbols)。
            SignalContext 会更新 bits_per_symbol 和 n_pol。
        """
        # Ensure bits are boolean
        if bits.dtype != bool:
            bits = bits.astype(bool)

        # 单偏振或双偏振的比特分配
        if self.n_pol == 1:
            symbols = self._map_bits_to_symbols(bits)
            data = symbols[np.newaxis, :]  # Shape (1, N_symbols)
        else:  # Dual polarization
            if len(bits) % 2 != 0:
                raise ValueError("For DP modulation, number of bits must be even for even distribution.")
            half_bits = len(bits) // 2
            symbols_x = self._map_bits_to_symbols(bits[:half_bits])
            symbols_y = self._map_bits_to_symbols(bits[half_bits:])
            data = np.stack([symbols_x, symbols_y], axis=0)  # Shape (2, N_symbols)

        # 创建或更新 SignalContext
        # 在 Mapper 阶段，sps 通常为 1 (符号速率)
        ctx = SignalContext(
            sample_rate=1.0,  # 符号速率，后续会被 upsampling 改变
            center_freq=0.0,  # 基带信号，中心频率为 0
            n_pol=self.n_pol,
            sps=1,
            bits_per_symbol=self.bits_per_symbol,
            meta={"modulation": self.modulation},
        )
        ctx.sym_rate = ctx.sample_rate / ctx.sps  # Update sym_rate

        return Signal(data=data.astype(np.complex64), ctx=ctx)
