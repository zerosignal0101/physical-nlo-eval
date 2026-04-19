"""光通信信号映射模块及星座图核心。"""

import re

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


def _gray_to_binary(n: np.ndarray) -> np.ndarray:
    """将整数数组从格雷码索引转换为二进制整数索引。"""
    mask = n >> 1
    res = n.copy()
    for _ in range(12):
        mask = res >> 1
        if not np.any(mask):
            break
        res = res ^ mask
    return res


class Constellation:
    """星座图核心类，处理LUT生成、归一化、差分编码和概率整形。"""

    def __init__(
        self,
        modulation: str,
        power_norm: bool = True,
        prob_shaping: dict[str, float] | None = None,
    ) -> None:
        self.modulation = modulation.upper()
        self.power_norm = power_norm
        self.prob_shaping = prob_shaping

        self.n_pol: int
        self.m_ary: int
        self.bits_per_symbol: int
        self.ps_probs: np.ndarray
        self.lut: np.ndarray

        self._parse()
        self._build()

    def _parse(self) -> None:
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

    def _build(self) -> None:
        """生成支持格雷映射的星座图查找表 (LUT)。"""
        if self.modulation.endswith("OOK"):
            constellation = np.array([0.0, 1.0], dtype=np.complex128)
        elif self.modulation.endswith("BPSK"):
            constellation = np.array([-1.0, 1.0], dtype=np.complex128)
        else:
            sqrt_m = int(np.sqrt(self.m_ary))
            idx = np.arange(sqrt_m)
            bin_val = _gray_to_binary(idx)
            pam_levels = 2 * bin_val - (sqrt_m - 1)
            i_grid, q_grid = np.meshgrid(pam_levels, pam_levels, indexing="ij")
            constellation = (i_grid + 1j * q_grid).flatten()

        # 概率整形分布
        if self.prob_shaping and "nu" in self.prob_shaping:
            nu = self.prob_shaping["nu"]
            probs = np.exp(-nu * (np.abs(constellation) ** 2))
            self.ps_probs = probs / np.sum(probs)
        else:
            self.ps_probs = np.ones(self.m_ary) / self.m_ary

        # 功率归一化
        if self.power_norm:
            avg_power = np.sum(self.ps_probs * (np.abs(constellation) ** 2))
            constellation /= np.sqrt(avg_power)

        self.lut = constellation


class Mapper(BaseBlock):
    """比特到符号的映射模块，支持导频插入和差分编码。

    Parameters
    ----------
    modulation : str
        调制格式。
    power_norm : bool, optional
        是否进行功率归一化，默认为 True。
    prob_shaping : dict[str, float] | None, optional
        概率整形参数。例如 {"nu": 0.1} (麦克斯韦-玻尔兹曼分布因子)。
    pilot_info : dict, optional
        导频信息配置，例如：
        {
            "symbols": np.array([1+1j, 1+1j]), # 导频符号序列
            "period": 100                      # 每 100 个净荷符号插入一次导频
        }
    diff_coding : bool, optional
        是否启用象限差分编码，默认为 False。
    """

    def __init__(
        self,
        modulation: str = "DP-16QAM",
        power_norm: bool = True,
        prob_shaping: dict[str, float] | None = None,
        pilot_info: dict | None = None,
        diff_coding: bool = False,
    ) -> None:
        super().__init__(name=f"Mapper({modulation})")
        # 组合 Constellation 类
        self.constellation = Constellation(modulation, power_norm, prob_shaping)
        self.pilot_info = pilot_info
        self.diff_coding = diff_coding

    def _diff_encode(self, symbol_indices: np.ndarray) -> np.ndarray:
        """基于象限的差分编码 (适用于 QPSK/QAM)。"""
        if self.constellation.m_ary < 4:
            return symbol_indices  # OOK/BPSK 暂不处理象限差分

        k_per_dim = self.constellation.bits_per_symbol // 2
        # 提取象限比特 (MSB of I and MSB of Q in Gray mapped QAM)
        i_sign_bit = (symbol_indices >> (self.constellation.bits_per_symbol - 1)) & 1
        q_sign_bit = (symbol_indices >> (k_per_dim - 1)) & 1

        quad_bits = (i_sign_bit << 1) | q_sign_bit
        encoded_quad = np.zeros_like(quad_bits)

        # 差分累加器
        current_phase = 0
        for i in range(len(quad_bits)):
            current_phase = (current_phase + quad_bits[i]) % 4
            encoded_quad[i] = current_phase

        # 将差分后的象限比特重新放回 symbol_indices
        mask = ~((1 << (self.constellation.bits_per_symbol - 1)) | (1 << (k_per_dim - 1)))
        new_indices = (
            (symbol_indices & mask)
            | ((encoded_quad >> 1) << (self.constellation.bits_per_symbol - 1))
            | ((encoded_quad & 1) << (k_per_dim - 1))
        )
        return new_indices

    def _map_payload(self, bits: np.ndarray) -> np.ndarray:
        """将比特流映射为净荷符号。"""
        bps = self.constellation.bits_per_symbol
        if len(bits) % bps != 0:
            raise ValueError(f"Bits length must be a multiple of {bps}.")

        bits_reshaped = bits.reshape(-1, bps)
        powers_of_two = 1 << np.arange(bps - 1, -1, -1)
        indices = np.sum(bits_reshaped * powers_of_two, axis=1)

        if self.diff_coding:
            indices = self._diff_encode(indices)

        return self.constellation.lut[indices]

    def _insert_pilots(self, payload: np.ndarray) -> np.ndarray:
        """在净荷数据中周期性插入导频符号。"""
        if not self.pilot_info:
            return payload

        period = self.pilot_info["period"]
        p_syms = np.atleast_1d(self.pilot_info["symbols"])
        p_len = len(p_syms)

        # 计算分块数量
        n_payload = len(payload)
        n_blocks = int(np.ceil(n_payload / period))

        # 预分配总内存
        total_len = n_payload + n_blocks * p_len
        framed_signal = np.zeros(total_len, dtype=payload.dtype)

        # 拼接块 (Block: [Pilot, Payload_chunk])
        idx_framed = 0
        idx_payload = 0
        for _ in range(n_blocks):
            # 插入导频
            framed_signal[idx_framed : idx_framed + p_len] = p_syms
            idx_framed += p_len

            # 插入净荷
            chunk_size = min(period, n_payload - idx_payload)
            framed_signal[idx_framed : idx_framed + chunk_size] = payload[idx_payload : idx_payload + chunk_size]
            idx_framed += chunk_size
            idx_payload += chunk_size

        return framed_signal

    @property
    def bits_per_symbol(self) -> int:
        """每个符号的比特数。"""
        return self.constellation.bits_per_symbol

    @property
    def n_pol(self) -> int:
        """偏振数 (1=单偏振, 2=双偏振)。"""
        return self.constellation.n_pol

    def forward(self, bits: np.ndarray) -> Signal:
        """将比特流映射为复符号信号，支持导频插入。"""
        bits = bits.astype(bool)
        cst = self.constellation

        if cst.n_pol == 1:
            symbols = self._map_payload(bits)
            framed_symbols = self._insert_pilots(symbols)
            data = framed_symbols[np.newaxis, :]
        else:
            symbols_x = self._map_payload(bits[0::2])
            symbols_y = self._map_payload(bits[1::2])
            framed_x = self._insert_pilots(symbols_x)
            framed_y = self._insert_pilots(symbols_y)
            data = np.stack([framed_x, framed_y], axis=0)

        ctx = SignalContext(
            sample_rate=1.0,
            center_freq=0.0,
            n_pol=cst.n_pol,
            sps=1,
            bits_per_symbol=cst.bits_per_symbol,
            meta={
                "modulation": cst.modulation,
                "pilot_info": self.pilot_info,
                "entropy": -np.sum(cst.ps_probs * np.log2(cst.ps_probs + 1e-12)),
            },
        )
        ctx.sym_rate = ctx.sample_rate / ctx.sps

        return Signal(data=data.astype(np.complex64), ctx=ctx)
