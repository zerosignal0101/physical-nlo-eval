"""时域 DSP 模块：重采样、定时同步、粗频偏补偿。"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample_poly

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class Resampler(BaseBlock):
    """重采样模块。

    用于改变信号的采样率。内部使用多相 FIR 抗混叠滤波器进行高质量重采样。

    Parameters
    ----------
    target_sps : int
        目标每符号采样点数 (Samples per Symbol)。
    resampling_filter : str, optional
        重采样滤波器类型，目前支持 'fir' (使用 scipy resample_poly)。
    """

    def __init__(self, target_sps: int, resampling_filter: str = "fir") -> None:
        """初始化 Resampler。"""
        super().__init__(name=f"Resampler({target_sps}sps)")
        if target_sps <= 0:
            raise ValueError("Target sps must be positive.")
        if resampling_filter not in ["fir"]:
            raise ValueError(f"Unsupported resampling filter: {resampling_filter}. Currently only 'fir' is supported.")

        self.target_sps = target_sps
        self.resampling_filter = resampling_filter

    def forward(self, signal: Signal) -> Signal:
        """
        对信号进行重采样 (内置多相抗混叠滤波器)。

        Parameters
        ----------
        signal : Signal
            输入的信号。

        Returns
        -------
        Signal
            重采样后的信号。
        """
        current_sps = signal.ctx.sps
        if current_sps == self.target_sps:
            return signal  # No resampling needed

        # 使用 scipy.signal.resample_poly 处理任意有理数比例重采样，内置高阶多相 FIR 抗混叠滤波
        resampled_data = resample_poly(signal.data, self.target_sps, current_sps, axis=-1)

        new_sample_rate = signal.ctx.sym_rate * self.target_sps

        new_ctx = SignalContext(
            sample_rate=new_sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=signal.ctx.n_pol,
            sps=self.target_sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={**(signal.ctx.meta or {}), "resampled_sps": self.target_sps},
        )
        return Signal(data=resampled_data.astype(np.complex64), ctx=new_ctx)


class OFDMChannelEstimator(BaseBlock):
    """OFDM 信道估计与均衡模块。

    使用 Least Squares (LS) 算法估计每个子载波的频率响应，
    并对 payload 数据进行 Zero-Forcing (ZF) 均衡。

    Parameters
    ----------
    estimation_method : str, optional
        估计方法，目前支持 'LS' (Least Squares)。
    smooth_subcarriers : bool, optional
        是否对估计的信道响应在子载波维度进行平滑（减少噪声影响），默认 False。
    """

    def __init__(self, estimation_method: str = "LS", smooth_subcarriers: bool = False) -> None:
        super().__init__(name=f"ChannelEstimator({estimation_method})")
        self.estimation_method = estimation_method
        self.smooth_subcarriers = smooth_subcarriers

    def forward(self, signal: Signal) -> Signal:
        """执行信道估计与均衡。

        要求输入 Signal 的 meta 中包含:
        - 'ofdm_ts_freq_symbols': 接收端提取的 TS (n_pol, n_sub, n_frames)
        - 'ofdm_ts_pattern': 发射端原始 TS 模式 (n_pol, n_sub)
        - 'ofdm_num_data_symbols_per_frame': 每帧的数据符号数
        """
        if not signal.ctx.is_ofdm:
            raise ValueError("Signal is not OFDM.")

        meta = signal.ctx.meta or {}
        rx_ts = meta.get("ofdm_ts_freq_symbols")  # (n_pol, n_sub, n_frames)
        tx_ts = meta.get("ofdm_ts_pattern")  # (n_pol, n_sub)
        num_data_syms_per_frame = meta.get("ofdm_num_data_symbols_per_frame")

        if rx_ts is None or tx_ts is None:
            raise ValueError("TS patterns not found in signal metadata. Demodulator must extract TS first.")

        n_pol, n_sub, n_frames = rx_ts.shape
        # 数据当前的形状 (n_pol, n_sub, total_data_symbols)
        n_pol_data, n_sub_data, n_total_syms = signal.data.shape

        # 1. 信道估计 (LS: H = Y / X)
        # 将 tx_ts 广播到 rx_ts 的形状
        tx_ts_expanded = np.repeat(tx_ts[:, :, np.newaxis], n_frames, axis=2)

        # 计算每一帧、每一极化、每个子载波的信道响应 H
        # 避免除以 0
        eps = 1e-12
        h_est = rx_ts / (tx_ts_expanded + eps)

        # (可选) 在子载波上平滑以抑制噪声
        if self.smooth_subcarriers:
            from scipy.ndimage import uniform_filter1d

            h_est = uniform_filter1d(h_est, size=3, axis=1)

        # 2. 均衡 (ZF: X_hat = Y / H)
        equalized_data = np.zeros_like(signal.data)

        # 遍历每一帧进行均衡
        for f_idx in range(n_frames):
            start_idx = f_idx * num_data_syms_per_frame
            end_idx = min((f_idx + 1) * num_data_syms_per_frame, n_total_syms)

            if start_idx >= n_total_syms:
                break

            # 提取当前帧的信道估计结果 (n_pol, n_sub)
            h_frame = h_est[:, :, f_idx]

            # 对该帧内的所有数据符号进行均衡
            # 形状：(n_pol, n_sub, symbols_in_this_frame)
            frame_data = signal.data[:, :, start_idx:end_idx]

            # 执行均衡 (对每个符号应用对应的子载波增益)
            h_frame_expanded = h_frame[:, :, np.newaxis]
            equalized_data[:, :, start_idx:end_idx] = frame_data / (h_frame_expanded + eps)

        # Update Context
        new_meta = {
            **meta,
            "ofdm_channel_response": h_est,
            "ofdm_equalized": True,
        }
        new_ctx = signal.ctx.copy()
        new_ctx.meta = new_meta

        return Signal(data=equalized_data, ctx=new_ctx)


class TimingRecovery(BaseBlock):
    """基于 Oerder & Meyer 平方算法的前馈定时同步模块。

    该算法通过计算信号平方后的离散傅里叶变换(DFT)在符号速率处的分量，
    直接估计出全局或分段的定时相位偏移。没有反馈环路，无收敛时间。

    Parameters
    ----------
    sps_in : int
        输入信号的每符号采样点数。要求 >= 2，推荐 >= 4 以获得更高精度。
    block_size : int, optional
        进行相位估计的块大小（符号数）。如果设为 None，则对整个信号进行单一相位估计。
        如果信号存在采样频率失配（Sampling Frequency Offset），建议设置较小的块（如 1000）。
    """

    def __init__(self, sps_in: int = 4, block_size: int | None = None) -> None:
        super().__init__(name="OerderMeyerTimingRecovery")
        self.sps_in = sps_in
        self.block_size = block_size

    def _estimate_phase(self, data: np.ndarray) -> np.ndarray:
        """估计采样相位偏移 (U.I.)。"""
        n_samples = len(data)
        # 1. 信号非线性变换：平方（提取时钟音调）
        sq_data = np.abs(data) ** 2

        # 如果不分块，对整个信号计算
        if self.block_size is None:
            # 计算在符号速率频率处的 DFT
            # 频率对应于 1/sps_in (已归一化到采样率)
            t = np.arange(n_samples)
            clock_tone = np.sum(sq_data * np.exp(-1j * 2 * np.pi * t / self.sps_in))
            phase = -np.angle(clock_tone) / (2 * np.pi)
            return np.array([phase])
        else:
            # 分块估计相位以跟踪缓慢的定时漂移
            samples_per_block = self.block_size * self.sps_in
            num_blocks = n_samples // samples_per_block

            phases = []
            for i in range(num_blocks):
                block = sq_data[i * samples_per_block : (i + 1) * samples_per_block]
                t = np.arange(len(block))
                tone = np.sum(block * np.exp(-1j * 2 * np.pi * t / self.sps_in))
                phases.append(-np.angle(tone) / (2 * np.pi))

            # 解卷绕相位 (Unwrap) 避免突变
            phases = np.unwrap(np.array(phases) * 2 * np.pi) / (2 * np.pi)
            return phases

    def forward(self, signal: Signal) -> Signal:
        """
        对信号执行 Oerder-Meyer 前馈定时同步。

        Parameters
        ----------
        signal : Signal
            输入的信号。预期 `signal.ctx.sps` 为 `sps_in`。

        Returns
        -------
        Signal
            定时同步后的符号信号。`signal.ctx.sps` 将更新为 1。
        """
        if signal.ctx.sps != self.sps_in:
            raise ValueError(f"Expected sps {self.sps_in}, got {signal.ctx.sps}")

        n_pol, n_samples = signal.data.shape
        synced_list = []

        for p in range(n_pol):
            pol_data = signal.data[p]

            # 1. 估计相位
            phases = self._estimate_phase(pol_data)

            # 2. 构建重采样时刻序列
            # phase 是相对于符号起始位置的偏移 [0, 1)
            num_symbols = n_samples // self.sps_in

            if len(phases) == 1:
                # 全局单一相位
                # 理想采样点在：n * sps + phase * sps
                sample_instants = np.arange(num_symbols) * self.sps_in + phases[0] * self.sps_in
            else:
                # 分块插值相位
                # 将块相位映射到每个符号上
                xp = np.linspace(0, num_symbols, len(phases))
                interp_phase = np.interp(np.arange(num_symbols), xp, phases)
                sample_instants = np.arange(num_symbols) * self.sps_in + interp_phase * self.sps_in

            # 限制范围，防止插值越界
            sample_instants = np.clip(sample_instants, 0, n_samples - 1)

            # 3. 插值重采样 (使用三次样条插值)
            f_interp = interp1d(np.arange(n_samples), pol_data, kind="cubic", fill_value="extrapolate")
            synced_pol = f_interp(sample_instants)
            synced_list.append(synced_pol)

        # 保持长度一致
        min_len = min(len(s) for s in synced_list)
        output_data = np.array([s[:min_len] for s in synced_list])

        new_ctx = signal.ctx.copy()
        new_ctx.sps = 1
        new_ctx.sample_rate = signal.ctx.sym_rate
        new_ctx.meta["timing_method"] = "Oerder-Meyer"

        return Signal(data=output_data.astype(np.complex64), ctx=new_ctx)


class FOCoarse(BaseBlock):
    """粗频偏补偿模块。

    使用基于训练序列或 FFT 的方法进行粗频偏估计和补偿。
    此处实现一个基于 FFT 峰值的方法。

    Parameters
    ----------
    search_range_hz : float, optional
        频率搜索范围 (Hz)。
    fft_size_for_estimation : int, optional
        用于 FFT 估计的 FFT 大小。
    """

    def __init__(self, search_range_hz: float = 200e6, fft_size_for_estimation: int = 1024) -> None:
        """初始化 FOCoarse。"""
        super().__init__(name="FOCoarse")
        self.search_range_hz = search_range_hz
        self.fft_size_for_estimation = fft_size_for_estimation

    def forward(self, signal: Signal) -> Signal:
        """
        估计并补偿粗频偏。

        Parameters
        ----------
        signal : Signal
            输入的信号。数据形状为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            补偿粗频偏后的信号。
        """
        if signal.data.ndim != 2:
            raise ValueError("Input signal data must be 2D (n_pol, N_samples).")

        n_pol, num_samples = signal.data.shape
        sample_rate = signal.ctx.sample_rate

        # For simplicity, we assume an FFT-based estimation on one polarization.
        # More robust methods use squaring or training sequences.

        # Take a chunk of signal for estimation
        if num_samples < self.fft_size_for_estimation:
            fft_data = signal.data[:, :num_samples]
            fft_size = num_samples
        else:
            fft_data = signal.data[:, : self.fft_size_for_estimation]
            fft_size = self.fft_size_for_estimation

        estimated_freq_offset_hz_per_pol: list[float] = []

        for pol_idx in range(n_pol):
            # FFT of the squared signal (for BPSK/QPSK) or direct FFT (for presence of pilot/strong tone)
            # For general QAM, a frequency offset causes rotation in constellation.
            # A common method is to raise to the power of M (M-QAM) to remove data dependency.
            # Or use a pilot tone if available.

            # Simple direct FFT peak detection (assumes some residual carrier or pilot)
            spectrum = np.fft.fft(fft_data[pol_idx, :])
            freqs = np.fft.fftfreq(fft_size, d=1 / sample_rate)

            # Find peak in the spectrum (focus on positive frequencies or around expected offset)
            # For a pure baseband signal, this peak should be at 0.
            # If there's an offset, there will be a shift.

            # Limit search to search_range_hz
            freq_idx_range = np.where(
                np.logical_and(freqs >= -self.search_range_hz / 2, freqs <= self.search_range_hz / 2)
            )

            if freq_idx_range[0].size > 0:
                peak_idx = freq_idx_range[0][np.argmax(np.abs(spectrum[freq_idx_range]))]
                estimated_offset = freqs[peak_idx]
            else:
                estimated_offset = 0.0  # Default if no strong peak in range

            estimated_freq_offset_hz_per_pol.append(estimated_offset)

            # Compensate frequency offset for the entire signal
            t = np.arange(num_samples) / sample_rate
            compensation_phase = np.exp(-1j * 2 * np.pi * estimated_offset * t)
            signal.data[pol_idx, :] = signal.data[pol_idx, :] * compensation_phase

        # Assume average offset for now if multiple polarizations
        # More robust approach would be to compensate each polarization individually.
        avg_estimated_offset = np.mean(estimated_freq_offset_hz_per_pol)

        new_ctx = SignalContext(
            sample_rate=sample_rate,
            center_freq=signal.ctx.center_freq - avg_estimated_offset,  # Adjust center freq for estimation
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta={
                **(signal.ctx.meta or {}),
                "coarse_freq_offset_hz": avg_estimated_offset,
                "coarse_freq_offset_per_pol_hz": estimated_freq_offset_hz_per_pol,
            },
        )
        return Signal(data=signal.data, ctx=new_ctx)


class ChromaticDispersionComp(BaseBlock):
    """色度色散补偿 (CDC) 模块。

    在频域通过乘法器补偿光纤传输中产生的二阶色散 (GVD) 和三阶色散 (TOD)。
    该模块采用全信号 FFT 实现，适用于补偿累积色散。

    Parameters
    ----------
    fiber_length_km : float | None, optional
        需要补偿的光纤长度 (km)。如果为 None，则尝试从信号 metadata 中获取 "fiber_length_km"。
    beta2_ps2_per_km : float | None, optional
        光纤的二阶色散系数 (ps^2/km)。如果为 None，则尝试从信号 metadata 中获取。
    beta3_ps3_per_km : float | None, optional
        光纤的三阶色散系数 (ps^3/km)。如果为 None，则尝试从信号 metadata 中获取。
    """

    def __init__(
        self,
        fiber_length_km: float | None = None,
        beta2_ps2_per_km: float | None = None,
        beta3_ps3_per_km: float | None = None,
    ) -> None:
        """初始化 ChromaticDispersionComp。"""
        super().__init__(name="CDC")
        self.fiber_length_km = fiber_length_km
        self.beta2_ps2_per_km = beta2_ps2_per_km
        self.beta3_ps3_per_km = beta3_ps3_per_km

    def forward(self, signal: Signal) -> Signal:
        """执行频域色散补偿。

        Parameters
        ----------
        signal : Signal
            输入的复数基带信号，数据形状应为 (n_pol, N_samples)。

        Returns
        -------
        Signal
            补偿色散后的信号，上下文 metadata 中会记录补偿参数。

        Raises
        ------
        ValueError
            如果输入信号不是复数类型或维度不等于 2。
        """
        if not np.iscomplexobj(signal.data):
            raise ValueError("Input signal must be complex for CDC.")
        if signal.data.ndim != 2:
            raise ValueError("Input signal data must be 2D (n_pol, N_samples).")

        # 1. 参数提取与推断
        length_km = self.fiber_length_km
        if length_km is None:
            length_km = float(signal.ctx.meta.get("fiber_length_m", 0.0)) / 1000.0

        if length_km <= 0:
            # 若长度为0或未指定，直接返回原信号
            return signal

        beta2 = self.beta2_ps2_per_km
        if beta2 is None:
            beta2 = float(signal.ctx.meta.get("fiber_beta2_ps2_per_km", 0.0))

        beta3 = self.beta3_ps3_per_km
        if beta3 is None:
            beta3 = float(signal.ctx.meta.get("fiber_beta3_ps3_per_km", 0.0))

        if beta2 == 0.0 and beta3 == 0.0:
            raise ValueError("Zero beta, unable to compensate dispersion.")

        n_pol, num_samples = signal.data.shape
        sample_rate = signal.ctx.sample_rate

        # 2. 构建角频率网格 (rad/s)
        # freq: [-fs/2, ..., fs/2-1], omega = 2 * pi * freq
        #
        # 频率间隔必须与施加色散的一方保持一致。PyNLO 的
        # `Grid.create(n, v_center, bandwidth)` 把 bandwidth 解释为 n 个频点的
        # inclusive span，即频率间隔为 bandwidth/(n-1)，对应时间步长
        # (n-1)/(n*fs)；而 np.fft.fftfreq(n, 1/fs) 给出的间隔是 fs/n。两者相差
        # 一个 n/(n-1) 的因子。
        #
        # 色散相位正比于 omega^2，沿用 fftfreq 会让补偿相位带上约 2/(n-1) 的
        # 相对误差。n 大时无所谓（n=36000 时仅 0.58 rad），但 n 小时足以让补偿
        # 完全失效：n=3600 时在 576 GHz 处累积 5.8 rad，接近一整圈，BER 会从
        # 1e-4 量级恶化到 0.41。这里按 FiberChannel/PyNLO 的实际频率轴取值。
        freq_step_scale = num_samples / (num_samples - 1) if num_samples > 1 else 1.0
        freq = np.fft.fftfreq(num_samples, d=1.0 / sample_rate) * freq_step_scale
        omega = 2 * np.pi * freq

        # 3. 计算补偿相位
        # 物理公式: H_fiber = exp(-j * (beta2/2 * omega^2 + beta3/6 * omega^3) * L)
        # 补偿相位 (逆函数取共轭): phi = +(beta2/2 * omega^2 + beta3/6 * omega^3) * L
        # 量纲转换说明:
        # beta2: ps^2/km -> s^2/km 需要乘 1e-24
        # beta3: ps^3/km -> s^3/km 需要乘 1e-36
        # length_km 已是 km，无需额外转换
        phase = ((beta2 * 1e-24) * (omega**2) / 2 + (beta3 * 1e-36) * (omega**3) / 6) * length_km

        # 4. 频域滤波
        h_cdc = np.exp(1j * phase)
        compensated_data = np.empty_like(signal.data, dtype=np.complex128)

        for pol_idx in range(n_pol):
            # 时域 -> 频域
            sig_fft = np.fft.fft(signal.data[pol_idx, :])
            # 频域相乘
            sig_comp_fft = sig_fft * h_cdc
            # 频域 -> 时域
            compensated_data[pol_idx, :] = np.fft.ifft(sig_comp_fft)

        # 5. 更新上下文与元数据
        new_meta = (signal.ctx.meta or {}).copy()
        new_meta.update(
            {
                "cdc_compensated_length_km": length_km,
                "cdc_beta2_ps2_per_km": beta2,
                "cdc_beta3_ps3_per_km": beta3,
            }
        )

        new_ctx = SignalContext(
            sample_rate=sample_rate,
            center_freq=signal.ctx.center_freq,
            n_pol=n_pol,
            sps=signal.ctx.sps,
            sym_rate=signal.ctx.sym_rate,
            bits_per_symbol=signal.ctx.bits_per_symbol,
            is_ofdm=signal.ctx.is_ofdm,
            meta=new_meta,
        )

        return Signal(data=compensated_data, ctx=new_ctx)
