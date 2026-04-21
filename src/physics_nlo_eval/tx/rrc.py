"""Root-Raised Cosine (RRC) 脉冲成形滤波器。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal


class RRCFilter(BaseBlock):
    """Root-Raised Cosine (RRC) 脉冲成形与匹配滤波器。

    该模块支持：
    1. 发送端上采样 + RRC 成形滤波。
    2. 接收端 RRC 匹配滤波。

    Parameters
    ----------
    sps : int
        目标每符号采样点数 (Samples per Symbol)。
    roll_off : float
        滚降系数 (0 ~ 1)，通常取 0.01 到 0.5。
    filter_span : int, optional
        滤波器的符号跨度（单侧长度），默认 10。总抽头数为 2 * filter_span * sps + 1。
    is_matched_filter : bool, optional
        是否作为匹配滤波器使用。
        - False (默认): 发送端成形。若输入 sps=1，则先进行零插值上采样。
        - True: 接收端匹配。仅滤波，不改变采样率。
    """

    def __init__(self, sps: int, roll_off: float, filter_span: int = 10, is_matched_filter: bool = False) -> None:
        name = "RRCMatchFilter" if is_matched_filter else "RRCShapeFilter"
        super().__init__(name=f"{name}(sps={sps}, alpha={roll_off})")

        self.sps = sps
        self.roll_off = roll_off
        self.filter_span = filter_span
        self.is_matched_filter = is_matched_filter

        # 预计算滤波器抽头 (Taps)
        self.taps = self._generate_rrc_taps()

    def _generate_rrc_taps(self) -> np.ndarray:
        """生成能量归一化的 RRC 脉冲响应。"""
        sps = self.sps
        alpha = self.roll_off
        span = self.filter_span

        # 时间轴：从 -span 到 +span，采样间隔为 1/sps
        t = np.arange(-span * sps, span * sps + 1) / sps
        taps = np.zeros_like(t)

        # 处理特殊点和普通点
        # RRC 理想公式：
        # h(t) = [sin(pi*t*(1-a)) + 4*a*t*cos(pi*t*(1+a))] / [pi*t*(1-(4*a*t)^2)]

        for i, ti in enumerate(t):
            if ti == 0.0:
                taps[i] = 1.0 - alpha + (4 * alpha / np.pi)
            elif alpha != 0 and abs(abs(4 * alpha * ti) - 1.0) < 1e-10:
                # 分母为 0 的特殊点 t = +/- 1/(4*alpha)
                taps[i] = (alpha / np.sqrt(2)) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            else:
                numerator = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
                denominator = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
                taps[i] = numerator / denominator

        # 能量归一化 (Sum of squares = 1)，确保滤波前后功率稳定
        return taps / np.sqrt(np.sum(taps**2))

    def _upsample(self, data: np.ndarray, factor: int) -> np.ndarray:
        """在符号间插入 factor-1 个零点。"""
        n_pol, n_samples = data.shape
        upsampled = np.zeros((n_pol, n_samples * factor), dtype=data.dtype)
        upsampled[:, ::factor] = data
        return upsampled

    def forward(self, signal: Signal) -> Signal:
        """
        执行 RRC 滤波逻辑。

        Parameters
        ----------
        signal : Signal
            输入信号。
            如果是 Tx 端 (is_matched_filter=False) 且 ctx.sps=1，则执行上采样。
        """
        data = signal.data
        ctx = signal.ctx

        # 1. 逻辑判断：是否需要上采样 (仅在 Tx 端且输入为 1 sps 时)
        if not self.is_matched_filter and ctx.sps == 1:
            data = self._upsample(data, self.sps)
            # 更新 Context 中的采样率
            ctx.sample_rate *= self.sps
            ctx.sps = self.sps
            ctx.sym_rate = ctx.sample_rate / ctx.sps

        # 2. 卷积滤波
        # 使用 np.convolve 处理每个偏振分量，'same' 保证信号长度不变且中心对齐
        processed_data = np.zeros_like(data)
        for p in range(data.shape[0]):
            processed_data[p] = np.convolve(data[p], self.taps, mode="same")

        # 3. 返回新信号对象（保持 Context 更新）
        return Signal(data=processed_data, ctx=ctx)
