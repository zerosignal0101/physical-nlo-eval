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

    # ---- 辅助静态方法 ----

    @staticmethod
    def _extract_payload(framed_signal: np.ndarray, pilot_info: dict) -> np.ndarray:
        """从含导频的帧结构中剔除导频，提取净荷符号 (单偏振)。

        与 Demapper._extract_payload 逻辑完全一致。
        """
        period = pilot_info["period"]
        p_len = len(np.atleast_1d(pilot_info["symbols"]))

        block_size = p_len + period
        n_total = len(framed_signal)
        n_blocks = int(np.ceil(n_total / block_size))

        payload_chunks = []
        for i in range(n_blocks):
            start_idx = i * block_size
            payload_start = start_idx + p_len
            payload_end = min(start_idx + block_size, n_total)

            if payload_start < payload_end:
                payload_chunks.append(framed_signal[payload_start:payload_end])

        if payload_chunks:
            return np.concatenate(payload_chunks)
        return np.array([], dtype=framed_signal.dtype)

    @staticmethod
    def _diff_encode_indices(symbol_indices: np.ndarray, bits_per_symbol: int, m_ary: int) -> np.ndarray:
        """象限差分编码，与 Mapper._diff_encode 逻辑一致。

        仅编码象限比特 (I/Q 的 MSB)，低比特保持不变。
        m_ary < 4 (OOK/BPSK) 时不执行编码。
        """
        if m_ary < 4:
            return symbol_indices

        k_per_dim = bits_per_symbol // 2
        i_sign_bit = (symbol_indices >> (bits_per_symbol - 1)) & 1
        q_sign_bit = (symbol_indices >> (k_per_dim - 1)) & 1

        quad_bits = (i_sign_bit << 1) | q_sign_bit
        encoded_quad = np.zeros_like(quad_bits)

        current_phase = 0
        for i in range(len(quad_bits)):
            current_phase = (current_phase + quad_bits[i]) % 4
            encoded_quad[i] = current_phase

        mask = ~((1 << (bits_per_symbol - 1)) | (1 << (k_per_dim - 1)))
        return (
            (symbol_indices & mask)
            | ((encoded_quad >> 1) << (bits_per_symbol - 1))
            | ((encoded_quad & 1) << (k_per_dim - 1))
        )

    # ---- BER ----

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

    # ---- EVM ----

    def calculate_evm(
        self,
        rx_signal: Signal,
        constellation: Constellation,
        tx_bits: np.ndarray | None = None,
        diff_coding: bool = False,
        pilot_info: dict | None = None,
    ) -> float:
        """计算误差矢量幅度 (EVM)，与 Demapper 功能同步。

        预处理步骤与 Demapper.forward 保持一致：
        1. 导频剔除 (若存在 pilot_info)
        2. 降采样 (若 sps > 1)
        3. OFDM 展平 (3D → 2D)
        4. 功率归一化
        5. 从 tx_bits 重建理想星座点 (含差分解码逆操作)

        Parameters
        ----------
        rx_signal : Signal
            接收信号（经 DSP 后的符号同步信号）。
        constellation : Constellation
            星座图对象，提供归一化 LUT 和调制参数。
        tx_bits : np.ndarray | None
            发送比特流 (仅净荷，不含导频比特)。
        diff_coding : bool
            发送端是否启用了象限差分编码。
        pilot_info : dict | None
            导频配置。若为 None 则从 rx_signal.ctx.meta["pilot_info"] 获取。

        Returns
        -------
        float
            EVM 值 (百分比)。

        Raises
        ------
        ValueError
            如果 tx_bits 长度与净荷符号数不匹配。
        """
        # ---- 1. 降采样 ----
        sps = rx_signal.ctx.sps
        if sps > 1:
            rx_symbols = rx_signal.data[:, ::sps]
        else:
            rx_symbols = rx_signal.data

        # ---- 2. OFDM 展平 ----
        if rx_symbols.ndim == 3:
            rx_flat = rx_symbols.reshape(rx_symbols.shape[0], -1)
        else:
            rx_flat = rx_symbols

        # ---- 3. 导频剔除 (与 Demapper.forward 同步) ----
        active_pilot_info = pilot_info or rx_signal.ctx.meta.get("pilot_info")
        if active_pilot_info:
            payload_pols = []
            for pol in range(rx_flat.shape[0]):
                payload_pols.append(self._extract_payload(rx_flat[pol], active_pilot_info))
            rx_flat = np.stack(payload_pols, axis=0)

        n_total_syms = rx_flat.shape[1]

        if n_total_syms == 0:
            return 0.0

        # ---- 4. 功率归一化 ----
        rx_power = np.mean(np.abs(rx_flat) ** 2, axis=1, keepdims=True)
        rx_power[rx_power == 0] = 1.0
        rx_normalized = rx_flat / np.sqrt(rx_power)

        # ---- 5. 重建理想星座点 ----
        ideal_lut = constellation.lut
        bits_per_sym = constellation.bits_per_symbol
        n_pol_cst = constellation.n_pol

        if tx_bits is None:
            return 0.0

        expected_len = n_pol_cst * n_total_syms * bits_per_sym
        if len(tx_bits) < expected_len:
            raise ValueError(
                f"tx_bits length {len(tx_bits)} is too short. "
                f"Expected {expected_len} for n_pol={n_pol_cst}, n_syms={n_total_syms}, "
                f"bits_per_sym={bits_per_sym}."
            )

        # 按极化分离比特
        if n_pol_cst == 1:
            pol_bits = [tx_bits]
        else:
            pol_bits = [tx_bits[0::2], tx_bits[1::2]]

        # 计算 Gray 索引并可选差分编码 (每极化独立)
        powers_of_two = 1 << np.arange(bits_per_sym - 1, -1, -1)

        indices_per_pol = []
        for pol in range(n_pol_cst):
            bits_pol = pol_bits[pol][: n_total_syms * bits_per_sym]
            bits_reshaped = bits_pol.reshape(n_total_syms, bits_per_sym)
            idx = np.sum(bits_reshaped.astype(int) * powers_of_two[np.newaxis, :], axis=1)

            if diff_coding:
                idx = self._diff_encode_indices(idx, bits_per_sym, constellation.m_ary)

            indices_per_pol.append(idx)

        # 构建交织的 constellation_indices
        if n_pol_cst == 1:
            constellation_indices = indices_per_pol[0]
        else:
            indices_combined = np.empty(n_total_syms * n_pol_cst, dtype=int)
            indices_combined[0::2] = indices_per_pol[0]
            indices_combined[1::2] = indices_per_pol[1]
            constellation_indices = indices_combined

        # ---- 6. 底层 EVM ----
        return self._calculate_evm_arrays(rx_normalized, ideal_lut, constellation_indices)

    def _calculate_evm_arrays(
        self,
        rx_symbols: np.ndarray,
        constellation_symbols: np.ndarray,
        constellation_indices: np.ndarray | None = None,
    ) -> float:
        """底层 EVM 计算（数组接口，供内部或旧代码使用）。"""
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
            n_symbols_expected = rx_cropped.shape[1]

            # Determine index format and crop accordingly
            if len(constellation_indices) == n_total_symbols:
                # Same index for all pols at each symbol time (single-pol case)
                indices_cropped = constellation_indices[start_idx:].flatten()
            elif len(constellation_indices) == n_total_symbols * n_pol:
                # Interleaved: [pol0_t0, pol1_t0, pol0_t1, pol1_t1, ...]
                # Crop n_pol entries per symbol to keep interleaved alignment
                indices_cropped = constellation_indices[start_idx * n_pol:].flatten()
            else:
                raise ValueError(
                    f"constellation_indices length {len(constellation_indices)} does not match "
                    f"rx_symbols length {n_total_symbols} (n_pol={n_pol})"
                )

            if len(indices_cropped) == n_symbols_expected:
                # Same index for all pols at each time slot
                ideal_symbols = constellation_symbols[indices_cropped]  # (n_symbols,)
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
                    f"constellation_indices cropped length {len(indices_cropped)} does not match "
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
