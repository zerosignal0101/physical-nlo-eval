"""实现比特到符号的映射，支持 OOK, BPSK, QPSK, QAM 及 PS-QAM。"""

import re

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


def _gray_to_binary(n: np.ndarray) -> np.ndarray:
    """将整数数组从格雷码索引转换为二进制整数索引。"""
    mask = n >> 1
    res = n.copy()
    # 假设最大支持 4096-QAM，循环次数足够
    for _ in range(12):
        mask = res >> 1
        if not np.any(mask):
            break
        res = res ^ mask
    return res


class Mapper(BaseBlock):
    """比特到符号的映射模块。

    支持 OOK, BPSK, QPSK, QAM 调制，支持格雷映射、差分编码以及概率整形 (PS-QAM)。

    Parameters
    ----------
    modulation : str
        调制格式，例如 'DP-16QAM', 'SP-QPSK', 'DP-OOK'。
    power_norm : bool, optional
        是否进行功率归一化，默认为 True。
    diff_coding : bool, optional
        是否启用象限差分编码，默认为 False。
    prob_shaping : dict[str, float] | None, optional
        概率整形参数。例如 {"nu": 0.1} (麦克斯韦-玻尔兹曼分布因子)。
    """

    def __init__(
        self,
        modulation: str = "DP-16QAM",
        power_norm: bool = True,
        diff_coding: bool = False,
        prob_shaping: dict[str, float] | None = None,
    ) -> None:
        super().__init__(name=f"Mapper({modulation})")
        self.modulation = modulation.upper()
        self.power_norm = power_norm
        self.diff_coding = diff_coding
        self.prob_shaping = prob_shaping

        self.n_pol: int
        self.m_ary: int
        self.bits_per_symbol: int
        self.ps_probs: np.ndarray | None = None
        self.lut: np.ndarray  # 查找表 (Look-Up Table)

        self._parse_modulation()
        self._generate_constellation_and_lut()

    def _parse_modulation(self) -> None:
        """解析调制格式字符串。"""
        match = re.match(r"^(DP|SP)-(OOK|BPSK|QPSK|(\d+)QAM)$", self.modulation)
        if not match:
            raise ValueError(f"Invalid modulation string: {self.modulation}.")

        pol_type, mod_type, qam_m = match.groups()
        self.n_pol = 2 if pol_type == "DP" else 1

        if mod_type == "OOK":
            self.m_ary = 2
        elif mod_type == "BPSK":
            self.m_ary = 2
        elif mod_type == "QPSK":
            self.m_ary = 4
        else:
            self.m_ary = int(qam_m)

        self.bits_per_symbol = int(np.log2(self.m_ary))
        if 2**self.bits_per_symbol != self.m_ary:
            raise ValueError(f"M-ary ({self.m_ary}) must be a power of 2.")

    def _generate_constellation_and_lut(self) -> None:
        """生成支持格雷映射的星座图查找表，并计算概率整形。"""
        # 生成基础星座点
        if self.modulation.endswith("OOK"):
            constellation = np.array([0.0, 1.0], dtype=np.complex128)
        elif self.modulation.endswith("BPSK"):
            constellation = np.array([-1.0, 1.0], dtype=np.complex128)
        else:  # QPSK / QAM
            sqrt_m = int(np.sqrt(self.m_ary))
            if sqrt_m * sqrt_m != self.m_ary:
                raise ValueError(f"Only square QAM is supported for M={self.m_ary}.")

            k_per_dim = self.bits_per_symbol // 2
            # 生成 1D 格雷映射 PAM
            idx = np.arange(sqrt_m)
            bin_val = _gray_to_binary(idx)
            pam_levels = 2 * bin_val - (sqrt_m - 1)

            # 生成 2D 查找表 (前一半比特为 I，后一半为 Q)
            i_grid, q_grid = np.meshgrid(pam_levels, pam_levels, indexing="ij")
            constellation = (i_grid + 1j * q_grid).flatten()

        # 计算概率整形分布 (Maxwell-Boltzmann)
        if self.prob_shaping and "nu" in self.prob_shaping:
            nu = self.prob_shaping["nu"]
            # P(x) = exp(-nu * |x|^2)
            energies = np.abs(constellation) ** 2
            probs = np.exp(-nu * energies)
            self.ps_probs = probs / np.sum(probs)
        else:
            self.ps_probs = np.ones(self.m_ary) / self.m_ary

        # 功率归一化 (考虑 PS 导致的概率分布变化)
        if self.power_norm:
            avg_power = np.sum(self.ps_probs * (np.abs(constellation) ** 2))
            constellation /= np.sqrt(avg_power)

        self.lut = constellation

    def _diff_encode(self, symbol_indices: np.ndarray) -> np.ndarray:
        """基于象限的差分编码 (适用于 QPSK/QAM)。"""
        if self.m_ary < 4:
            return symbol_indices  # OOK/BPSK 暂不处理象限差分

        k_per_dim = self.bits_per_symbol // 2
        # 提取象限比特 (MSB of I and MSB of Q in Gray mapped QAM)
        # 在我们的 LUT 中，最高位决定 I 的符号，中间位决定 Q 的符号
        i_sign_bit = (symbol_indices >> (self.bits_per_symbol - 1)) & 1
        q_sign_bit = (symbol_indices >> (k_per_dim - 1)) & 1

        quad_bits = (i_sign_bit << 1) | q_sign_bit
        encoded_quad = np.zeros_like(quad_bits)

        # 差分累加器
        current_phase = 0
        for i in range(len(quad_bits)):
            current_phase = (current_phase + quad_bits[i]) % 4
            encoded_quad[i] = current_phase

        # 将差分后的象限比特重新放回 symbol_indices
        mask = ~((1 << (self.bits_per_symbol - 1)) | (1 << (k_per_dim - 1)))
        new_indices = (
            (symbol_indices & mask)
            | ((encoded_quad >> 1) << (self.bits_per_symbol - 1))
            | ((encoded_quad & 1) << (k_per_dim - 1))
        )
        return new_indices

    def _map_bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """将比特流映射到符号。"""
        if len(bits) % self.bits_per_symbol != 0:
            raise ValueError(f"Bits length must be a multiple of {self.bits_per_symbol}.")

        bits_reshaped = bits.reshape(-1, self.bits_per_symbol)

        # 极速比特转整数
        powers_of_two = 1 << np.arange(self.bits_per_symbol - 1, -1, -1)
        indices = np.sum(bits_reshaped * powers_of_two, axis=1)

        if self.diff_coding:
            indices = self._diff_encode(indices)

        if self.prob_shaping:
            # 真实系统的 Distribution Matcher (DM) 会在此处根据 self.ps_probs
            # 将均匀分布的比特流重塑为特定概率的振幅。
            # 这里保留架构位置，暂以直接查表作为 Placeholder。
            pass

        return self.lut[indices]

    def forward(self, bits: np.ndarray) -> Signal:
        """将比特流映射为复符号信号。

        Parameters
        ----------
        bits : np.ndarray
            输入比特流 (bool 数组)

        Returns
        -------
        Signal
            包含复符号的 Signal 对象
        """
        bits = bits.astype(bool)

        if self.n_pol == 1:
            symbols = self._map_bits_to_symbols(bits)
            data = symbols[np.newaxis, :]
        else:
            if len(bits) % 2 != 0:
                raise ValueError("For DP, number of bits must be even.")
            # 光通信通常交织或分块分配偏振比特
            symbols_x = self._map_bits_to_symbols(bits[0::2])
            symbols_y = self._map_bits_to_symbols(bits[1::2])
            data = np.stack([symbols_x, symbols_y], axis=0)

        # Signal Context 包含熵等 PS 参数
        ctx = SignalContext(
            sample_rate=1.0,
            center_freq=0.0,
            n_pol=self.n_pol,
            sps=1,
            bits_per_symbol=self.bits_per_symbol,
            meta={"modulation": self.modulation, "entropy": -np.sum(self.ps_probs * np.log2(self.ps_probs + 1e-12))},
        )
        ctx.sym_rate = ctx.sample_rate / ctx.sps

        return Signal(data=data.astype(np.complex64), ctx=ctx)
