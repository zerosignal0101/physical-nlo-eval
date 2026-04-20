"""载波相位恢复模块。"""

from typing import Any

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class PhaseNoiseCompensation(BaseBlock):
    """载波相位噪声补偿模块（单载波专用）。

    用于补偿残余的载波相位噪声，仅支持单载波信号。
    OFDM 场景下的相位噪声补偿由 CommonPhaseNoiseCompensate 模块处理。

    Parameters
    ----------
    method : str, optional
        相位噪声补偿方法，支持 'BPS' (盲相扫), 'V&V' (Viterbi-Viterbi), 'pilot_assisted' (导频辅助)。
    n_test_phases : int, optional
        BPS 方法中测试的相位数量，默认 32。
    block_size : int, optional
        分块处理时的符号块大小 (用于 BPS 和 V&V 的滑动平均或块平均)，默认 128。
    m_power : int, optional
        V&V 算法的 M 次方数，QPSK 和一般方形 QAM 的外部星环通常为 4。
    constellation : np.ndarray | None, optional
        理想星座点数组，提供给 BPS 算法以进行精确的欧氏距离判决。若未提供则使用简化代数约束。
    """

    def __init__(
        self,
        method: str = "BPS",
        n_test_phases: int = 32,
        block_size: int = 128,
        m_power: int = 4,
        constellation: np.ndarray | None = None,
    ) -> None:
        """初始化 PhaseNoiseCompensation。"""
        super().__init__(name=f"PhaseNoiseComp({method})")
        if method not in ["BPS", "V&V", "pilot_assisted"]:
            raise ValueError(f"Unsupported phase noise compensation method: {method}.")

        self.method = method
        self.n_test_phases = n_test_phases
        self.block_size = block_size
        self.m_power = m_power
        self.constellation = constellation

    def forward(self, signal: Signal) -> Signal:
        """执行载波相位噪声补偿。

        Parameters
        ----------
        signal : Signal
            输入的信号。数据形状为 (n_pol, N_samples) (单载波)。

        Returns
        -------
        Signal
            补偿相位噪声后的信号。
        """
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex.")

        if signal.ctx.is_ofdm:
            raise ValueError("OFDM phase noise compensation is handled by CommonPhaseNoiseCompensate. Use SC mode.")

        return self._sc_phase_noise_comp(signal)

    def _sc_phase_noise_comp(self, signal: Signal) -> Signal:
        """单载波 (SC) 信号的相位噪声补偿核心路由。"""
        n_pol, num_symbols = signal.data.shape
        compensated_data = np.empty_like(signal.data)

        for pol_idx in range(n_pol):
            pol_data = signal.data[pol_idx, :]

            if self.method == "BPS":
                compensated_data[pol_idx, :] = self._sc_bps(pol_data)
            elif self.method == "V&V":
                compensated_data[pol_idx, :] = self._sc_vv(pol_data)
            elif self.method == "pilot_assisted":
                pilot_info = signal.ctx.meta.get("pilot_info")
                if pilot_info is None:
                    raise ValueError("Pilot metadata 'pilot_info' not found in signal.ctx.meta.")
                compensated_data[pol_idx, :] = self._sc_pilot_assisted(pol_data, pilot_info)

        new_ctx = SignalContext(
            sample_rate=signal.ctx.sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={
                **(signal.ctx.meta or {}),
                "sc_phase_noise_compensated": True,
                "sc_phase_noise_method": self.method,
            },
        )
        return Signal(data=compensated_data, ctx=new_ctx)

    def _sc_bps(self, pol_data: np.ndarray) -> np.ndarray:
        """单载波 BPS (Blind Phase Search) 实现。"""
        compensated = np.empty_like(pol_data)
        # 测试相位通常取 [-pi/4, pi/4) 即能覆盖具备 pi/2 旋转对称性的 QAM/QPSK 星座
        test_phases = np.linspace(-np.pi / 4, np.pi / 4, self.n_test_phases, endpoint=False)
        test_phasors = np.exp(1j * test_phases)

        for i in range(0, len(pol_data), self.block_size):
            block = pol_data[i : i + self.block_size]
            # 广播计算旋转: (n_test_phases, block_size)
            rotated = block * test_phasors[:, np.newaxis]

            if self.constellation is not None:
                # 依据与理想星座图的欧氏距离作为代价函数
                diff = rotated[:, :, np.newaxis] - self.constellation[np.newaxis, np.newaxis, :]
                min_dist = np.min(np.abs(diff) ** 2, axis=2)
                cost = np.sum(min_dist, axis=1)
            else:
                # 简化的方形 QAM 盲当代价函数：(|Re| - |Im|)^2
                cost = np.sum((np.abs(rotated.real) - np.abs(rotated.imag)) ** 2, axis=1)

            best_idx = int(np.argmin(cost))
            compensated[i : i + self.block_size] = rotated[best_idx]

        return compensated

    def _sc_vv(self, pol_data: np.ndarray) -> np.ndarray:
        """单载波 Viterbi & Viterbi 实现。"""
        n_total = len(pol_data)
        n_blocks = n_total // self.block_size
        trunc_len = n_blocks * self.block_size

        phases = np.zeros(n_total)
        if n_blocks > 0:
            reshaped = pol_data[:trunc_len].reshape(n_blocks, self.block_size)
            # M-th Power, 参考 VV_ML_Common 实施幅度消除以减少码型噪声
            powered = reshaped**self.m_power
            den = np.abs(powered)
            den_safe = np.where(den == 0, 1.0, den)
            mean_pow = np.mean(powered / den_safe, axis=1)

            # 先解卷绕 M 倍相角，再除以 M
            block_phases = np.unwrap(np.angle(mean_pow)) / self.m_power
            phases[:trunc_len] = np.repeat(block_phases, self.block_size)

        if trunc_len < n_total:
            block_rem = pol_data[trunc_len:]
            powered_rem = block_rem**self.m_power
            mean_pow_rem = np.mean(powered_rem / np.where(np.abs(powered_rem) == 0, 1.0, np.abs(powered_rem)))
            rem_phase = np.angle(mean_pow_rem) / self.m_power
            # 将余下数据对齐到最后的相位卷绕层级
            if n_blocks > 0:
                diff = rem_phase - block_phases[-1]
                rem_phase -= np.round(diff / (np.pi / 2)) * (np.pi / 2)
            phases[trunc_len:] = rem_phase

        return pol_data * np.exp(-1j * phases)

    def _sc_pilot_assisted(self, pol_data: np.ndarray, pilot_info: dict[str, Any]) -> np.ndarray:
        """单载波基于导频序列的插值载波恢复。"""
        period = pilot_info["period"]
        p_syms = np.atleast_1d(pilot_info["symbols"])
        p_len = len(p_syms)
        block_len = period + p_len

        n_total = len(pol_data)
        n_blocks = int(np.ceil(n_total / block_len))

        pilot_phases = []
        pilot_centers = []

        # 提取导频块，计算平均相位差
        for i in range(n_blocks):
            start_idx = i * block_len
            end_idx = min(start_idx + p_len, n_total)
            if end_idx > start_idx:
                rx_p = pol_data[start_idx:end_idx]
                tx_p = p_syms[: end_idx - start_idx]
                # 最大似然相位评估
                phase = np.angle(np.mean(rx_p * tx_p.conj()))
                pilot_phases.append(phase)
                pilot_centers.append(start_idx + (end_idx - start_idx) / 2.0)

        if not pilot_phases:
            return pol_data.copy()

        # 相位解卷绕及线性插值到所有符号
        unwrapped_phases = np.unwrap(pilot_phases)
        all_indices = np.arange(n_total)
        interpolated_phases = np.interp(all_indices, pilot_centers, unwrapped_phases)

        return pol_data * np.exp(-1j * interpolated_phases)
