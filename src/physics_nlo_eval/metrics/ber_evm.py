"""判决和 BER/EVM 计算所需的解映射模块。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.tx.mapper import Constellation


class Demapper(BaseBlock):
    """符号到比特的解映射模块，支持导频剔除和差分解码。"""

    def __init__(
        self,
        modulation: str = "DP-16QAM",
        power_norm: bool = True,
        prob_shaping: dict[str, float] | None = None,
        pilot_info: dict | None = None,
        diff_coding: bool = False,
    ) -> None:
        super().__init__(name=f"Demapper({modulation})")
        # 共享 Constellation 类，消除重复代码
        self.constellation = Constellation(modulation, power_norm, prob_shaping)
        self.pilot_info = pilot_info
        self.diff_coding = diff_coding

    @property
    def bits_per_symbol(self) -> int:
        """每个符号的比特数。"""
        return self.constellation.bits_per_symbol

    @property
    def n_pol(self) -> int:
        """偏振数 (1=单偏振, 2=双偏振)。"""
        return self.constellation.n_pol

    def _extract_payload(self, framed_signal: np.ndarray, pilot_info: dict) -> np.ndarray:
        """从接收信号中剔除导频，提取纯净的净荷符号。"""
        period = pilot_info["period"]
        p_len = len(np.atleast_1d(pilot_info["symbols"]))

        block_size = p_len + period
        n_total = len(framed_signal)
        n_blocks = int(np.ceil(n_total / block_size))

        payload_chunks = []
        for i in range(n_blocks):
            start_idx = i * block_size
            # 跳过前导 p_len 个导频符号
            payload_start = start_idx + p_len
            payload_end = min(start_idx + block_size, n_total)

            if payload_start < payload_end:
                payload_chunks.append(framed_signal[payload_start:payload_end])

        return np.concatenate(payload_chunks)

    def _diff_decode(self, symbol_indices: np.ndarray) -> np.ndarray:
        """基于象限的差分解码。"""
        if self.constellation.m_ary < 4:
            return symbol_indices

        k_per_dim = self.constellation.bits_per_symbol // 2
        i_sign_bit = (symbol_indices >> (self.constellation.bits_per_symbol - 1)) & 1
        q_sign_bit = (symbol_indices >> (k_per_dim - 1)) & 1

        quad_bits = (i_sign_bit << 1) | q_sign_bit
        decoded_quad = np.zeros_like(quad_bits)

        # 差分减法
        decoded_quad[0] = quad_bits[0]  # 第一位参考
        decoded_quad[1:] = (quad_bits[1:] - quad_bits[:-1]) % 4

        mask = ~((1 << (self.constellation.bits_per_symbol - 1)) | (1 << (k_per_dim - 1)))
        new_indices = (
            (symbol_indices & mask)
            | ((decoded_quad >> 1) << (self.constellation.bits_per_symbol - 1))
            | ((decoded_quad & 1) << (k_per_dim - 1))
        )
        return new_indices

    def _demap_payload(self, payload_symbols: np.ndarray) -> np.ndarray:
        """极速解映射：判决 + 向量化位运算解包。"""
        cst = self.constellation

        # 将输入物理信号功率拉回到理想星座图的归一化功率水平
        avg_power = np.mean(np.abs(payload_symbols) ** 2)
        if avg_power > 0:
            payload_symbols = payload_symbols / np.sqrt(avg_power)

        # 欧氏距离判决
        distances = np.abs(payload_symbols[:, np.newaxis] - cst.lut[np.newaxis, :])
        decision_indices = np.argmin(distances, axis=1)

        if self.diff_coding:
            decision_indices = self._diff_decode(decision_indices)

        # 向量化整数到比特重组
        powers_of_two = 1 << np.arange(cst.bits_per_symbol - 1, -1, -1)
        bits_reshaped = (decision_indices[:, np.newaxis] & powers_of_two) > 0

        return bits_reshaped.flatten()

    def forward(self, symbols: Signal) -> np.ndarray:
        """将复符号信号解映射为比特流，支持导频剔除。"""
        # 优先使用 Signal 流自带的导频元信息（如果存在）
        active_pilot_info = symbols.ctx.meta.get("pilot_info", self.pilot_info)

        flat_symbols = symbols.data.reshape(symbols.ctx.n_pol, -1) if symbols.data.ndim == 3 else symbols.data

        all_bits = []
        for pol_idx in range(self.constellation.n_pol):
            pol_symbols = flat_symbols[pol_idx]

            # 1. 剔除导频
            if active_pilot_info:
                pol_symbols = self._extract_payload(pol_symbols, active_pilot_info)

            # 2. 解映射
            bits = self._demap_payload(pol_symbols)
            all_bits.append(bits)

        if self.constellation.n_pol == 1:
            return all_bits[0]
        else:
            # 恢复双偏振的交织逻辑
            final_bits = np.empty(len(all_bits[0]) + len(all_bits[1]), dtype=bool)
            final_bits[0::2] = all_bits[0]
            final_bits[1::2] = all_bits[1]
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

    def calculate_evm(
        self,
        rx_symbols: np.ndarray,
        constellation_symbols: np.ndarray,
        constellation_indices: np.ndarray | None = None,
    ) -> float:
        """计算误差矢量幅度 (EVM)。

        EVM 衡量接收信号相对于理想星座点的偏差。

        Parameters
        ----------
        rx_symbols : np.ndarray
            接收到的实际符号，形状为 (n_pol, n_symbols) 或 (n_pol, n_subcarriers, n_symbols)。
        constellation_symbols : np.ndarray
            归一化后的理想星座点数组，形状为 (n_constellation_points,)。
        constellation_indices : np.ndarray | None
            每个接收符号对应的理想星座点索引，用于从 constellation_symbols 中查找理想点。

        Returns
        -------
        float
            计算得到的 EVM 值 (百分比)。

        Raises
        ------
        ValueError
            如果 rx_symbols 和 constellation_indices 长度不匹配。
        """
        # Reshape rx_symbols to (n_pol, -1) for easier processing
        if rx_symbols.ndim == 3:
            # OFDM: (n_pol, n_subcarriers, n_symbols) -> (n_pol, n_subcarriers * n_symbols)
            rx_flat = rx_symbols.reshape(rx_symbols.shape[0], -1)
        elif rx_symbols.ndim == 2:
            rx_flat = rx_symbols.reshape(rx_symbols.shape[0], -1)
        else:
            raise ValueError(f"rx_symbols must be 2D or 3D, got {rx_symbols.ndim}D")

        n_pol, n_total_symbols = rx_flat.shape

        # Apply ignore_headers
        start_idx = self.ignore_headers
        if start_idx >= n_total_symbols:
            return 0.0

        rx_cropped = rx_flat[:, start_idx:]

        # Normalize rx_symbols to unit power to match normalized constellation
        rx_power = np.mean(np.abs(rx_cropped) ** 2, axis=1, keepdims=True)
        rx_power[rx_power == 0] = 1.0  # Avoid division by zero
        rx_cropped_normalized = rx_cropped / np.sqrt(rx_power)

        if constellation_indices is not None:
            # Use the provided indices to get ideal constellation points
            indices_cropped = constellation_indices[start_idx:].flatten()
            n_symbols_expected = rx_cropped.shape[1]

            if len(indices_cropped) == n_symbols_expected:
                # indices length matches n_symbols: same index for all pols at each time slot
                ideal_symbols = constellation_symbols[indices_cropped]  # (n_symbols,)
                # Expand to (n_pol, n_symbols)
                ideal_cropped = np.repeat(ideal_symbols[np.newaxis, :], n_pol, axis=0)
            elif len(indices_cropped) == n_symbols_expected * n_pol:
                # indices are interleaved: [pol0_t0, pol1_t0, pol0_t1, pol1_t1, ...]
                if n_pol == 2:
                    ideal_pol0 = constellation_symbols[indices_cropped[0::2]]
                    ideal_pol1 = constellation_symbols[indices_cropped[1::2]]
                    ideal_cropped = np.vstack([ideal_pol0, ideal_pol1])
                else:
                    raise ValueError(f"Interleaved indices only supported for n_pol=2, got {n_pol}")
            else:
                raise ValueError(
                    f"constellation_indices length {len(indices_cropped)} does not match "
                    f"rx_symbols cropped length {n_symbols_expected} (n_pol={n_pol})"
                )
        else:
            # No indices provided - assume rx_symbols already aligned with constellation
            ideal_cropped = rx_cropped_normalized  # Fallback, will give zero EVM

        # EVM = sqrt(P_error / P_signal) * 100%
        # P_error = mean(|rx - ideal|^2)
        # P_signal = mean(|ideal|^2) - normalized constellation should have P_signal = 1

        error_vectors = rx_cropped_normalized - ideal_cropped
        p_error = np.mean(np.abs(error_vectors) ** 2)
        p_signal = np.mean(np.abs(ideal_cropped) ** 2)

        if p_signal <= 1e-12:
            return np.inf

        evm = np.sqrt(p_error / p_signal) * 100  # Convert to percentage
        return evm
