"""判决和 BER/EVM 计算模块。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal


class Demapper(BaseBlock):
    """符号到比特的解映射模块。

    将接收到的 QAM 符号判决到最近的理想星座点，并转换为比特流。

    Parameters
    ----------
    modulation : str
        调制格式，例如 'DP-16QAM', 'SP-64QAM'。
    power_norm : bool, optional
        星座图是否进行了功率归一化。
    """

    _constellations: dict[str, np.ndarray] = {}

    def __init__(self, modulation: str = "DP-16QAM", power_norm: bool = True) -> None:
        """初始化 Demapper。"""
        super().__init__(name=f"Demapper({modulation})")
        self.modulation = modulation
        self.power_norm = power_norm
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
        if key in Demapper._constellations:
            return

        sqrt_m = int(np.sqrt(self.m_ary))
        if not sqrt_m * sqrt_m == self.m_ary:
            raise ValueError(f"Only square QAM constellations are supported for M={self.m_ary}.")

        levels = np.arange(-sqrt_m + 1, sqrt_m, 2)
        constellation = (levels[np.newaxis, :] + 1j * levels[:, np.newaxis]).flatten()

        if self.power_norm:
            constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))

        Demapper._constellations[key] = constellation

    def _demap_symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        将 QAM 符号判决并解映射为比特流。

        Parameters
        ----------
        symbols : np.ndarray
            输入 QAM 符号，形状为 (N_symbols,)。

        Returns
        -------
        np.ndarray
            解映射后的比特流，形状为 (N_bits,)。
        """
        constellation = Demapper._constellations[f"{self.m_ary}QAM_norm={self.power_norm}"]

        # 判决到最近的星座点
        # Calculate Euclidean distance from each received symbol to all constellation points
        distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :])

        # Find index of the nearest constellation point
        decision_indices = np.argmin(distances, axis=1)

        # Convert index back to bits (inverse of Gray mapping)
        # This needs a proper Gray code lookup table.
        # For simplicity, we convert index to binary string then flatten.
        # Assumes direct binary mapping for now.

        bit_strings: list[str] = []
        for idx in decision_indices:
            # Convert decimal index to binary string, padded with leading zeros
            bit_string = format(idx, f"0{self.bits_per_symbol}b")
            bit_strings.append(bit_string)

        # Flatten list of bit strings into a single numpy array of bits
        all_bits = np.array(list("".join(bit_strings)), dtype=int)

        return all_bits.astype(bool)

    def forward(self, symbols: Signal) -> np.ndarray:
        """
        对接收到的符号进行判决并解映射。

        Parameters
        ----------
        symbols : Signal
            接收到的符号信号。
            如果为单载波，数据形状为 (n_pol, N_symbols)。
            如果为 OFDM，数据形状为 (n_pol, N_subcarriers, N_symbols)。

        Returns
        -------
        np.ndarray
            判决后的比特流。
        """
        if symbols.data.ndim == 3:  # OFDM
            # Flatten OFDM symbols to (n_pol, N_total_symbols) for demapping
            flat_symbols = symbols.data.reshape(symbols.ctx.n_pol, -1)
        elif symbols.data.ndim == 2:  # Single-carrier
            flat_symbols = symbols.data
        else:
            raise ValueError("Unsupported symbol data dimension for demapper.")

        all_demapped_bits: list[np.ndarray] = []

        for pol_idx in range(self.n_pol):
            pol_symbols = flat_symbols[pol_idx, :]
            demapped_bits = self._demap_symbols_to_bits(pol_symbols)
            all_demapped_bits.append(demapped_bits)

        # Concatenate bits from all polarizations (X, then Y)
        # This assumes TX Mapper also concatenates X bits then Y bits
        final_bits = np.concatenate(all_demapped_bits)

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
