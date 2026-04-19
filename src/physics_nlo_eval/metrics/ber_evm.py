"""判决和 BER/EVM 计算所需的解映射模块。"""

import re

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal

# 复用 Mapper 中的格雷逆转换
from physics_nlo_eval.tx.mapper import _gray_to_binary


class Demapper(BaseBlock):
    """符号到比特的解映射模块。

    将接收到的符号判决到最近的理想星座点，解映射并恢复原始比特流。

    Parameters
    ----------
    modulation : str
        调制格式。
    power_norm : bool, optional
        星座图是否进行了功率归一化。
    diff_coding : bool, optional
        是否启用象限差分解码。
    prob_shaping : dict[str, float] | None, optional
        概率整形参数，须与 Mapper 保持一致。
    """

    def __init__(
        self,
        modulation: str = "DP-16QAM",
        power_norm: bool = True,
        diff_coding: bool = False,
        prob_shaping: dict[str, float] | None = None,
    ) -> None:
        super().__init__(name=f"Demapper({modulation})")
        self.modulation = modulation.upper()
        self.power_norm = power_norm
        self.diff_coding = diff_coding
        self.prob_shaping = prob_shaping

        self.n_pol: int
        self.m_ary: int
        self.bits_per_symbol: int
        self.lut: np.ndarray

        self._parse_modulation()
        self._generate_constellation_and_lut()

    def _parse_modulation(self) -> None:
        """解析调制格式 (逻辑与 Mapper 相同)"""
        match = re.match(r"^(DP|SP)-(OOK|BPSK|QPSK|(\d+)QAM)$", self.modulation)
        if not match:
            raise ValueError(f"Invalid modulation string: {self.modulation}.")

        pol_type, mod_type, qam_m = match.groups()
        self.n_pol = 2 if pol_type == "DP" else 1
        self.m_ary = 2 if mod_type in ["OOK", "BPSK"] else (4 if mod_type == "QPSK" else int(qam_m))
        self.bits_per_symbol = int(np.log2(self.m_ary))

    def _generate_constellation_and_lut(self) -> None:
        """生成接收端参考星座图。"""
        # (生成逻辑同 Mapper，为避免重复代码，在实际框架中这部分应提取为基类或 utils)
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

        if self.prob_shaping and "nu" in self.prob_shaping:
            nu = self.prob_shaping["nu"]
            probs = np.exp(-nu * (np.abs(constellation) ** 2))
            probs /= np.sum(probs)
        else:
            probs = np.ones(self.m_ary) / self.m_ary

        if self.power_norm:
            constellation /= np.sqrt(np.sum(probs * (np.abs(constellation) ** 2)))

        self.lut = constellation

    def _diff_decode(self, symbol_indices: np.ndarray) -> np.ndarray:
        """基于象限的差分解码。"""
        if self.m_ary < 4:
            return symbol_indices

        k_per_dim = self.bits_per_symbol // 2
        i_sign_bit = (symbol_indices >> (self.bits_per_symbol - 1)) & 1
        q_sign_bit = (symbol_indices >> (k_per_dim - 1)) & 1

        quad_bits = (i_sign_bit << 1) | q_sign_bit
        decoded_quad = np.zeros_like(quad_bits)

        # 差分减法
        decoded_quad[0] = quad_bits[0]  # 第一位参考
        decoded_quad[1:] = (quad_bits[1:] - quad_bits[:-1]) % 4

        mask = ~((1 << (self.bits_per_symbol - 1)) | (1 << (k_per_dim - 1)))
        new_indices = (
            (symbol_indices & mask)
            | ((decoded_quad >> 1) << (self.bits_per_symbol - 1))
            | ((decoded_quad & 1) << (k_per_dim - 1))
        )
        return new_indices

    def _demap_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """极速解映射：欧式距离判决 + 向量化位运算解包。"""
        # 广播计算欧式距离 (N_sym, M_ary)
        # 如果 symbols 非常大，可使用分块 (chunking) 避免 OOM，但通常足够
        distances = np.abs(symbols[:, np.newaxis] - self.lut[np.newaxis, :])
        decision_indices = np.argmin(distances, axis=1)

        if self.diff_coding:
            decision_indices = self._diff_decode(decision_indices)

        # 向量化整数到比特重组 (比原版字符串拼接快数百倍)
        powers_of_two = 1 << np.arange(self.bits_per_symbol - 1, -1, -1)
        bits_reshaped = (decision_indices[:, np.newaxis] & powers_of_two) > 0

        return bits_reshaped.flatten()

    def forward(self, symbols: Signal) -> np.ndarray:
        """将复符号信号解映射为比特流。

        Parameters
        ----------
        symbols : Signal
            接收到的复符号 Signal 对象

        Returns
        -------
        np.ndarray
            恢复的比特流 (bool 数组)
        """
        if symbols.data.ndim == 3:  # OFDM 等多载波架构支持
            flat_symbols = symbols.data.reshape(symbols.ctx.n_pol, -1)
        elif symbols.data.ndim == 2:
            flat_symbols = symbols.data
        else:
            raise ValueError("Unsupported symbol data dimension.")

        if self.n_pol == 1:
            return self._demap_symbols_to_bits(flat_symbols[0])
        else:
            bits_x = self._demap_symbols_to_bits(flat_symbols[0])
            bits_y = self._demap_symbols_to_bits(flat_symbols[1])
            # 与 Mapper 分发方式对应 (交织)
            final_bits = np.empty(len(bits_x) + len(bits_y), dtype=bool)
            final_bits[0::2] = bits_x
            final_bits[1::2] = bits_y
            return final_bits


class BERCalculator:
    """比特误码率 (BER) 和误差矢量幅度 (EVM) 计算器。"""

    def __init__(self, ignore_headers: int = 0) -> None:
        """
        初始化 BERCalculator。

        Parameters
        ----------
        ignore_headers : int, optional
            忽略前 n 个比特/符号，常用于跳过训练序列或帧头。
        """
        self.ignore_headers = ignore_headers

    def calculate_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
        """
        计算比特误码率 (BER)。

        Parameters
        ----------
        tx_bits : np.ndarray
            发送的原始比特流。
        rx_bits : np.ndarray
            接收并判决后的比特流。

        Returns
        -------
        float
            计算得到的 BER 值。

        Raises
        ------
        ValueError
            如果发送和接收比特流的长度不匹配。
        """
        if tx_bits.shape != rx_bits.shape:
            raise ValueError(
                f"TX bits shape {tx_bits.shape} does not match RX bits shape {rx_bits.shape}. "
                "Ensure correct length after demapping and potential truncation."
            )

        # Apply ignore_headers
        start_idx = self.ignore_headers
        if start_idx >= len(tx_bits):
            return 0.0  # No bits to compare after ignoring headers

        tx_bits_cropped = tx_bits[start_idx:]
        rx_bits_cropped = rx_bits[start_idx:]

        num_errors = np.sum(tx_bits_cropped != rx_bits_cropped)
        total_bits = len(tx_bits_cropped)

        if total_bits == 0:
            return 0.0  # Avoid division by zero
        return num_errors / total_bits

    def calculate_evm(self, tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
        """
        计算误差矢量幅度 (EVM)。

        Parameters
        ----------
        tx_symbols : np.ndarray
            发送的理想星座点符号。
        rx_symbols : np.ndarray
            接收到的实际符号 (判决前)。

        Returns
        -------
        float
            计算得到的 EVM 值 (百分比)。

        Raises
        ------
        ValueError
            如果发送和接收符号流的长度不匹配。
        """
        if tx_symbols.shape != rx_symbols.shape:
            raise ValueError(f"TX symbols shape {tx_symbols.shape} does not match RX symbols shape {rx_symbols.shape}.")

        # Apply ignore_headers (for symbols)
        # Assuming ignore_headers is in terms of symbols here.
        start_idx = self.ignore_headers
        if start_idx >= tx_symbols.shape[1]:  # If flat_symbols are (N_pol, N_symbols)
            return 0.0

        # If data is (n_pol, N_symbols) or (n_pol, N_subcarriers, N_symbols)
        if tx_symbols.ndim == 3:  # OFDM
            tx_symbols_cropped = tx_symbols[:, :, start_idx:]
            rx_symbols_cropped = rx_symbols[:, :, start_idx:]
        else:  # Single-carrier
            tx_symbols_cropped = tx_symbols[:, start_idx:]
            rx_symbols_cropped = rx_symbols[:, start_idx:]

        # Ensure tx_symbols are constellation points (normalized)
        # EVM = sqrt(P_error / P_signal) * 100%
        # P_error = sum(|rx_symbol - tx_symbol|**2)
        # P_signal = sum(|tx_symbol|**2)

        error_vectors = rx_symbols_cropped - tx_symbols_cropped

        p_error = np.mean(np.abs(error_vectors) ** 2)
        p_signal = np.mean(np.abs(tx_symbols_cropped) ** 2)

        if p_signal <= 1e-12:  # Avoid division by zero
            return np.inf

        evm = np.sqrt(p_error / p_signal) * 100  # Convert to percentage

        return evm
