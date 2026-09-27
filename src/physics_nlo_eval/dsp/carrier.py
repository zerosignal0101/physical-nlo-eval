# src/physics_nlo_eval/dsp/carrier.py
"""载波相位恢复模块。"""

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.tx.mapper import Constellation


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
    constellation : np.ndarray | Constellation | None, optional
        理想星座点数组或 Constellation 对象，提供给 BPS 算法以进行精确的欧氏距离判决。
        若为 Constellation 实例，则使用其 .lut 属性。若未提供则使用简化代数约束。
    resolve_ambiguity : bool, optional
        仅 BPS 使用。是否沿块序列做连续性的 M 重相位解模糊，默认 False。
        两种取值的取舍见 `_sc_bps` 的说明，差异是实打实的，没有哪一方全面更好。
    """

    def __init__(
        self,
        method: str = "BPS",
        n_test_phases: int = 32,
        block_size: int = 128,
        m_power: int = 4,
        constellation: np.ndarray | Constellation | None = None,
        resolve_ambiguity: bool = False,
    ) -> None:
        """初始化 PhaseNoiseCompensation。"""
        super().__init__(name=f"PhaseNoiseComp({method})")
        if method not in ["BPS", "V&V", "pilot_assisted"]:
            raise ValueError(f"Unsupported phase noise compensation method: {method}.")

        self.method = method
        self.n_test_phases = n_test_phases
        self.block_size = block_size
        self.m_power = m_power
        self.resolve_ambiguity = resolve_ambiguity
        # 支持 Constellation 对象或原始数组
        if isinstance(constellation, Constellation):
            self._constellation_lut = constellation.lut
        else:
            self._constellation_lut = constellation

    def forward(self, signal: Signal) -> Signal:
        """执行载波相位噪声补偿。"""
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex.")

        if signal.ctx.is_ofdm:
            raise ValueError("OFDM phase noise compensation is handled by CommonPhaseNoiseCompensate. Use SC mode.")

        return self._sc_phase_noise_comp(signal)

    def _sc_phase_noise_comp(self, signal: Signal) -> Signal:
        """单载波 (SC) 信号的相位噪声补偿核心路由。"""
        n_pol, num_symbols = signal.data.shape
        compensated_data = np.empty_like(signal.data)

        pilot_info = signal.ctx.meta.get("pilot_info")

        for pol_idx in range(n_pol):
            pol_data = signal.data[pol_idx, :]

            if self.method == "BPS":
                compensated_data[pol_idx, :] = self._sc_bps(pol_data)
            elif self.method == "V&V":
                # 将 pilot_info 传递给 V&V 以实现模糊度与周跳纠正
                compensated_data[pol_idx, :] = self._sc_vv(pol_data, pilot_info)
            elif self.method == "pilot_assisted":
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
        """单载波 BPS (Blind Phase Search) 实现。

        搜索范围取星座的旋转对称周期 (方形 QAM 为 pi/2)，因此逐块的代价函数
        只能把相位确定到 pi/2 的整数倍。关键在于: 代价函数**本身就是** pi/2
        周期的，所以没有任何基于代价的判据能分辨这些等价分支 —— 分支序列只能
        靠先验去定，而不同的先验对应不同的取舍，没有哪一方全面更好。

        两种行为 (由 ``resolve_ambiguity`` 选择)，实测 (SP-QPSK, block=128):

        - ``False`` (默认): 每块独立取极值，等价于"取使残余最小的那个分支"。
          长程漂移一旦超过 +/-pi/4 就会锁到错误分支 (1 MHz 线宽连续维纳漂移
          BER 0.32，和不补偿一样)；但它是**无记忆**的，相位短时兜一圈又回来
          (升余弦 pi/2 脉冲) 时能自愈，实测 BER 1.1%。
        - ``True``: 每块取与上一块最接近的等价相位，等于用第一块锚定绝对相位
          后连续跟踪。连续维纳漂移 BER 0.00 (6 个种子全为 0)；代价是遇到真正
          的 pi/2 周跳时会把分支一路带错 (同一个脉冲场景退化到 22.7%)。

        两者都要求单块内的相位变化明显小于 pi/2，否则前者失锁、后者选错分支。
        绝对的 M 重模糊本身是盲估计无法分辨的 (需要导频或差分编码)，本实现不做
        处理 —— 需要周跳处理时请用 'V&V' + 导频或 'pilot_assisted'。
        """
        compensated = np.empty_like(pol_data)
        search_halfwidth = np.pi / 4
        test_phases = np.linspace(-search_halfwidth, search_halfwidth, self.n_test_phases, endpoint=False)
        test_phasors = np.exp(1j * test_phases)
        # 解模糊的步长就是搜索范围宽度 (即星座的 M 重旋转对称周期)
        ambiguity = 2 * search_halfwidth
        previous_phase: float | None = None

        for i in range(0, len(pol_data), self.block_size):
            block = pol_data[i : i + self.block_size]
            rotated = block * test_phasors[:, np.newaxis]

            if self._constellation_lut is not None:
                diff = rotated[:, :, np.newaxis] - self._constellation_lut[np.newaxis, np.newaxis, :]
                min_dist = np.min(np.abs(diff) ** 2, axis=2)
                cost = np.sum(min_dist, axis=1)
            else:
                cost = np.sum((np.abs(rotated.real) - np.abs(rotated.imag)) ** 2, axis=1)

            best_idx = int(np.argmin(cost))

            if not self.resolve_ambiguity:
                compensated[i : i + self.block_size] = rotated[best_idx]
                continue

            block_phase = float(test_phases[best_idx])
            if previous_phase is not None:
                block_phase += round((previous_phase - block_phase) / ambiguity) * ambiguity
            previous_phase = block_phase
            compensated[i : i + self.block_size] = block * np.exp(1j * block_phase)

        return compensated

    def _sc_vv(self, pol_data: np.ndarray, pilot_info: dict[str, Any] | None = None) -> np.ndarray:
        """单载波 Viterbi & Viterbi 实现。集成导频辅助的基准对齐与周跳纠正。"""
        n_total = len(pol_data)
        n_blocks = n_total // self.block_size
        trunc_len = n_blocks * self.block_size

        vv_phases = np.zeros(n_total)

        # 1. 基础 V&V 连续相位跟踪 (基于 M 次方)
        if n_blocks > 0:
            reshaped = pol_data[:trunc_len].reshape(n_blocks, self.block_size)
            powered = reshaped**self.m_power
            den = np.abs(powered)
            mean_pow = np.mean(powered / np.where(den == 0, 1.0, den), axis=1)

            # 提取相位、解卷绕并缩小 M 倍。
            # 注意: 此处不加固定偏置(如 +pi)，任何的基准模糊度将交由导频纠错处理
            block_phases = np.unwrap(np.angle(mean_pow)) / self.m_power
            vv_phases[:trunc_len] = np.repeat(block_phases, self.block_size)

        # 处理末尾剩余数据
        if trunc_len < n_total:
            block_rem = pol_data[trunc_len:]
            powered_rem = block_rem**self.m_power
            mean_pow_rem = np.mean(powered_rem / np.where(np.abs(powered_rem) == 0, 1.0, np.abs(powered_rem)))
            rem_phase = np.angle(mean_pow_rem) / self.m_power
            if n_blocks > 0:
                diff = rem_phase - block_phases[-1]
                ambiguity = 2 * np.pi / self.m_power
                rem_phase -= np.round(diff / ambiguity) * ambiguity
            vv_phases[trunc_len:] = rem_phase

        # 2. 导频辅助基准对齐 & 周跳纠正
        if pilot_info is not None:
            period = pilot_info["period"]
            p_syms = np.atleast_1d(pilot_info["symbols"])
            p_len = len(p_syms)
            block_len = period + p_len
            n_pilot_blocks = int(np.ceil(n_total / block_len))

            ambiguity_rad = 2 * np.pi / self.m_power  # M=4 时为 pi/2
            offsets_raw = []
            pilot_centers = []

            for i in range(n_pilot_blocks):
                start_idx = i * block_len
                end_idx = min(start_idx + p_len, n_total)
                if end_idx > start_idx:
                    rx_p = pol_data[start_idx:end_idx]
                    tx_p = p_syms[: end_idx - start_idx]

                    # 计算导频处的真实物理相位
                    true_phase = np.angle(np.mean(rx_p * tx_p.conj()))
                    # V&V 当前跟踪到的相位
                    vv_phase = np.mean(vv_phases[start_idx:end_idx])

                    # 计算需要修正的偏移量，并将其限制在 [-pi, pi) 之间
                    phase_diff = (true_phase - vv_phase + np.pi) % (2 * np.pi) - np.pi

                    offsets_raw.append(phase_diff)
                    pilot_centers.append(start_idx + (end_idx - start_idx) / 2.0)

            if offsets_raw:
                # 关键步骤：对 offsets_raw 以 ambiguity_rad 为步长进行解卷绕！
                # 这将保留真实发生的 pi/2 周跳跳变，同时滤除 V&V 噪声引起的非整数倍跳变
                unwrapped_offsets = offsets_raw.copy()
                for i in range(1, len(unwrapped_offsets)):
                    diff = unwrapped_offsets[i] - unwrapped_offsets[i - 1]
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                    step_correction = np.round(diff / ambiguity_rad) * ambiguity_rad
                    unwrapped_offsets[i] = unwrapped_offsets[i - 1] + step_correction

                # 使用零阶保持插值，将修正量应用到所有符号上
                if len(unwrapped_offsets) == 1:
                    full_correction = np.full(n_total, unwrapped_offsets[0])
                else:
                    f_interp = interp1d(
                        pilot_centers,
                        unwrapped_offsets,
                        kind="previous",
                        bounds_error=False,
                        fill_value=(unwrapped_offsets[0], unwrapped_offsets[-1]),
                    )
                    full_correction = f_interp(np.arange(n_total))

                vv_phases += full_correction

        return pol_data * np.exp(-1j * vv_phases)

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

        for i in range(n_blocks):
            start_idx = i * block_len
            end_idx = min(start_idx + p_len, n_total)
            if end_idx > start_idx:
                rx_p = pol_data[start_idx:end_idx]
                tx_p = p_syms[: end_idx - start_idx]
                phase = np.angle(np.mean(rx_p * tx_p.conj()))
                pilot_phases.append(phase)
                pilot_centers.append(start_idx + (end_idx - start_idx) / 2.0)

        if not pilot_phases:
            return pol_data.copy()

        unwrapped_phases = np.unwrap(pilot_phases)
        all_indices = np.arange(n_total)
        interpolated_phases = np.interp(all_indices, pilot_centers, unwrapped_phases)

        return pol_data * np.exp(-1j * interpolated_phases)
