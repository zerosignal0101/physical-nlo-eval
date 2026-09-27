"""MIMO 偏振解复用和信道估计模块。"""

import warnings

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal


class MIMO_ChannelEstimation(BaseBlock):
    """MIMO 偏振解复用/信道估计模块。

    主要用于双偏振系统，通过训练序列 (TS) 或盲均衡算法
    估计和补偿偏振相关的信道损伤，包括偏振间串扰。

    LS/MMSE 从 ``mimo_training_tx`` / ``mimo_training_rx`` 元数据读取训练数据；
    对 OFDM 信号也兼容调制器原生的 ``ofdm_ts_pattern`` 与解调器提取的
    ``ofdm_ts_freq_symbols``。训练数组形状为 ``(n_pol, n_subcarrier, n_obs)``
    (OFDM) 或 ``(n_pol, n_obs)`` (单载波)。

    Parameters
    ----------
    method : str, optional
        信道估计方法，例如 'LS' (最小二乘), 'MMSE' (最小均方误差), 'CMA' (常模算法)。
    ts_interval : int, optional
        训练序列的间隔 (OFDM 符号数)。如果 method 为 'LS' 或 'MMSE' 且信号为 OFDM，则需要。
    equalizer_length : int, optional
        均衡器抽头长度，对于 CMA 等时域均衡器使用。
    cma_mu : float, optional
        CMA 算法的步长 (step size)。
    regularization : float, optional
        矩阵求逆的正则化系数，用于改善病态信道下的数值稳定性。
    """

    def __init__(
        self,
        method: str = "LS",
        ts_interval: int = 10,
        equalizer_length: int = 21,
        cma_mu: float = 1e-4,
        regularization: float = 1e-8,
    ) -> None:
        """初始化 MIMO_ChannelEstimation。"""
        method = method.upper()
        super().__init__(name=f"MIMO_CE_PD({method})")
        if method not in {"LS", "MMSE", "CMA"}:
            raise ValueError(f"Unsupported MIMO channel estimation method: {method}.")
        if ts_interval <= 0 or equalizer_length <= 0 or cma_mu <= 0 or regularization < 0:
            raise ValueError("MIMO estimator parameters must be positive.")
        self.method = method
        self.ts_interval = ts_interval
        self.equalizer_length = equalizer_length
        self.cma_mu = cma_mu
        self.regularization = regularization

    def forward(self, signal: Signal) -> Signal:
        """
        执行 MIMO 信道估计和偏振解复用。

        Parameters
        ----------
        signal : Signal
            输入的信号。
            如果是 OFDM 信号，数据形状为 (n_pol, N_subcarriers, N_symbols)。
            如果是单载波信号，数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            解复用后的信号。

        Raises
        ------
        ValueError
            偏振轴与上下文不一致，或偏振数既不是 1 也不是 2。
        """
        if signal.data.shape[0] != signal.ctx.n_pol:
            raise ValueError("Signal polarization axis does not match context.")
        if signal.ctx.n_pol == 1:
            # 单偏振无需解复用，返回副本以免调用方就地修改共享数据。
            return Signal(signal.data.copy(), signal.ctx.copy())
        if signal.ctx.n_pol != 2:
            raise ValueError("MIMO polarization equalizer currently supports exactly two polarizations.")
        if signal.ctx.is_ofdm:
            return self._ofdm_mimo_processing(signal)
        return self._sc_mimo_processing(signal)

    def _regularizer(self, signal: Signal) -> float:
        """返回本次求逆使用的正则化系数。"""
        if self.method == "MMSE":
            # MMSE 用噪声方差作为正则项；未提供时退回显式配置的正则化系数。
            return float(signal.ctx.meta.get("noise_variance", self.regularization or 1e-8))
        return self.regularization

    @staticmethod
    def _as_ofdm_training(value: object, n_sub: int) -> np.ndarray | None:
        """把 OFDM 训练数据规整为 (2, n_sub, n_obs) 形状。"""
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim == 2 and array.shape == (2, n_sub):
            return array[:, :, None]
        if array.ndim == 3 and array.shape[:2] == (2, n_sub):
            return array
        raise ValueError("OFDM MIMO training must have shape (2, n_sub) or (2, n_sub, n_obs).")

    def _ofdm_mimo_processing(self, signal: Signal) -> Signal:
        """OFDM 信号的 MIMO 处理 (逐子载波 2x2 信道估计和解复用)。

        对每个子载波用 LS/MMSE 求解完整的 2x2 信道矩阵：
        ``H = Y X^H (X X^H + reg I)^{-1}``，再取伪逆作为均衡器。
        与只做对角近似、忽略偏振串扰的做法不同，这里会估计并抵消
        非对角项。
        """
        if signal.data.ndim != 3:
            raise ValueError("OFDM MIMO input must have shape (2, n_sub, n_symbols).")
        _, n_sub, _ = signal.data.shape
        meta = signal.ctx.meta
        tx = self._as_ofdm_training(meta.get("mimo_training_tx", meta.get("ofdm_ts_pattern")), n_sub)
        rx = self._as_ofdm_training(meta.get("mimo_training_rx", meta.get("ofdm_ts_freq_symbols")), n_sub)
        if tx is None or rx is None:
            warnings.warn(
                "No complete OFDM MIMO training pair; signal was left unchanged.", RuntimeWarning, stacklevel=2
            )
            return Signal(signal.data.copy(), signal.ctx.copy())
        # 一侧只有单个观测时按另一侧重复，使两者观测数对齐。
        if tx.shape[2] == 1 and rx.shape[2] > 1:
            tx = np.repeat(tx, rx.shape[2], axis=2)
        if rx.shape[2] == 1 and tx.shape[2] > 1:
            rx = np.repeat(rx, tx.shape[2], axis=2)
        if tx.shape[2] != rx.shape[2]:
            raise ValueError("TX and RX MIMO training observation counts differ.")

        h_est = np.empty((2, 2, n_sub), dtype=np.complex128)
        equalizer = np.empty_like(h_est)
        reg = self._regularizer(signal)
        eye = np.eye(2)
        for k in range(n_sub):
            x = tx[:, k, :]
            y = rx[:, k, :]
            gram = x @ x.conj().T
            if np.linalg.matrix_rank(gram) < 2:
                # 训练观测不足以唯一确定 2x2 信道，退化为逐偏振对角估计。
                diagonal = np.ones(2, dtype=np.complex128)
                for pol in range(2):
                    denom = np.vdot(x[pol], x[pol]).real
                    if denom > 0:
                        diagonal[pol] = np.vdot(x[pol], y[pol]) / denom
                h_est[:, :, k] = np.diag(diagonal)
            else:
                h_est[:, :, k] = y @ x.conj().T @ np.linalg.inv(gram + reg * eye)
            equalizer[:, :, k] = np.linalg.pinv(h_est[:, :, k], rcond=max(reg, 1e-12))

        output = np.einsum("abk,bks->aks", equalizer, signal.data, optimize=True)
        ctx = signal.ctx.copy()
        ctx.meta.update(
            {
                "mimo_ofdm_demuxed": True,
                "ofdm_channel_estimated_H": h_est,
                "mimo_equalizer": equalizer,
                "mimo_method": self.method,
            }
        )
        return Signal(output.astype(signal.data.dtype, copy=False), ctx)

    def _sc_mimo_processing(self, signal: Signal) -> Signal:
        """单载波 (SC) 信号的 MIMO 处理。

        ``method='CMA'`` 走盲均衡 (:meth:`_cma`)；LS/MMSE 则用训练序列
        做一次 2x2 最小二乘信道估计并求逆。
        """
        if signal.data.ndim != 2:
            raise ValueError("Single-carrier MIMO input must have shape (2, samples).")
        if self.method == "CMA":
            return self._cma(signal)
        tx = signal.ctx.meta.get("mimo_training_tx")
        rx = signal.ctx.meta.get("mimo_training_rx")
        if tx is None:
            warnings.warn("No single-carrier MIMO training; signal was left unchanged.", RuntimeWarning, stacklevel=2)
            return Signal(signal.data.copy(), signal.ctx.copy())
        x = np.asarray(tx)
        y = np.asarray(rx) if rx is not None else signal.data[:, : x.shape[-1]]
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != 2 or y.shape != x.shape:
            raise ValueError("Single-carrier MIMO training arrays must have equal shape (2, n_obs).")
        reg = self._regularizer(signal)
        h_est = y @ x.conj().T @ np.linalg.inv(x @ x.conj().T + reg * np.eye(2))
        equalizer = np.linalg.pinv(h_est, rcond=max(reg, 1e-12))
        output = equalizer @ signal.data
        ctx = signal.ctx.copy()
        ctx.meta.update(
            {
                "mimo_sc_demuxed": True,
                "mimo_channel_estimated_H": h_est,
                "mimo_equalizer": equalizer,
                "mimo_method": self.method,
            }
        )
        return Signal(output.astype(signal.data.dtype, copy=False), ctx)

    def _cma(self, signal: Signal) -> Signal:
        """蝶形 FIR 常模自适应均衡。

        注意与旧实现的区别：这里是对 ``equalizer_length`` 个抽头做真正的
        FIR 滤波 (旧实现只用单个 2x2 抽头，``equalizer_length`` 形同虚设)，
        且模值参考 R 由信号的四阶矩推出，而非固定为 1。
        """
        length = self.equalizer_length
        center = length // 2
        taps = np.zeros((2, 2, length), dtype=np.complex128)
        # 初始化为对角延迟线，使均衡器起步时是一个带延迟的单位阵。
        taps[0, 0, center] = taps[1, 1, center] = 1
        padded = np.pad(signal.data, ((0, 0), (length - 1, 0)))
        output = np.empty_like(signal.data, dtype=np.complex128)
        p2 = np.mean(np.abs(signal.data) ** 2)
        # CMA 的模值参考 R = E[|x|^4] / E[|x|^2]。
        target = np.mean(np.abs(signal.data) ** 4) / max(p2, np.finfo(float).eps)
        for index in range(signal.data.shape[1]):
            vector = padded[:, index : index + length][:, ::-1]
            value = np.einsum("pql,ql->p", taps, vector)
            output[:, index] = value
            error = (np.abs(value) ** 2 - target) * value
            # 按输入能量归一化步长，避免深衰落处发散。
            norm = np.sum(np.abs(vector) ** 2) + 1e-12
            taps -= self.cma_mu * np.einsum("p,ql->pql", error, vector.conj()) / norm
        ctx = signal.ctx.copy()
        ctx.meta.update({"mimo_sc_demuxed": True, "mimo_equalizer_taps": taps, "mimo_method": "CMA"})
        return Signal(output.astype(signal.data.dtype, copy=False), ctx)
