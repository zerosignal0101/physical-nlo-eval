"""实现 OFDM 调制（IFFT, CP 添加, 训练序列插入, 频域导频插入）。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class OFDMModulator(BaseBlock):
    """OFDM 调制模块。

    执行频域导频插入、IFFT、添加循环前缀 (CP) 和插入训练序列 (TS)。

    Parameters
    ----------
    n_subcarriers : int
        OFDM 总子载波数量（FFT 大小）。
    cp_len : int
        循环前缀的长度。
    ts_pattern : np.ndarray | None, optional
        用于训练序列的频域符号模式。形状应为 (n_pol, n_subcarriers)。
    num_data_symbols_per_frame : int, optional
        每帧包含的 OFDM 符号数（不含训练符号），默认 100。
    pilot_spacing : int, optional
        频域导频间隔。若为 0 则不插入导频。例如 8 表示每 8 个子载波插入一个导频。
    pilot_value : complex, optional
        导频子载波使用的复数符号，默认 1.0 + 0j。
    """

    def __init__(
        self,
        n_subcarriers: int,
        cp_len: int,
        ts_pattern: np.ndarray | None = None,
        num_data_symbols_per_frame: int = 100,
        pilot_spacing: int = 0,
        pilot_value: complex = 1.0 + 0j,
    ) -> None:
        """初始化 OFDMModulator。"""
        super().__init__(name="OFDMModulator")
        if not n_subcarriers > 0:
            raise ValueError("Number of subcarriers must be positive.")
        if pilot_spacing < 0:
            raise ValueError("Pilot spacing must be non-negative.")

        self.n_subcarriers = n_subcarriers
        self.cp_len = cp_len
        self.ts_pattern = ts_pattern
        self.num_data_symbols_per_frame = num_data_symbols_per_frame
        self.pilot_spacing = pilot_spacing
        self.pilot_value = pilot_value
        self.fft_size = n_subcarriers

        # 预计算导频和数据子载波的索引
        if self.pilot_spacing > 0:
            self.pilot_indices = np.arange(0, self.n_subcarriers, self.pilot_spacing)
            self.data_indices = np.setdiff1d(np.arange(self.n_subcarriers), self.pilot_indices)
        else:
            self.pilot_indices = np.array([], dtype=int)
            self.data_indices = np.arange(self.n_subcarriers)

        self.n_data_subcarriers = len(self.data_indices)

    def forward(self, symbols: Signal) -> Signal:
        """将 QAM 符号流转换为 OFDM 时域信号。

        Parameters
        ----------
        symbols : Signal
            输入 QAM 符号信号。数据形状为 (n_pol, N_total_qam_symbols)。

        Returns
        -------
        Signal
            输出 OFDM 时域信号。
        """
        if symbols.data.ndim != 2:
            raise ValueError("Input symbols data must be 2D (n_pol, N_symbols).")

        n_pol = symbols.ctx.n_pol
        total_input_symbols = symbols.data.shape[1]

        # 计算每帧需要的净荷符号数
        payload_symbols_per_frame = self.n_data_subcarriers * self.num_data_symbols_per_frame
        num_frames = total_input_symbols // payload_symbols_per_frame

        if num_frames == 0:
            raise ValueError(f"Not enough symbols for one OFDM frame. Need {payload_symbols_per_frame}.")

        ofdm_output_list: list[np.ndarray] = []
        ts_location_meta: list[tuple[int, int]] = []
        has_ts = self.ts_pattern is not None
        num_ts_symbols_per_frame = 1 if has_ts else 0

        # 准备导频参考信息供后续 DSP 使用
        # 形状为 (len(pilot_indices),)，在相位补偿时会被广播或按需复制
        expected_pilot_symbols = np.full(len(self.pilot_indices), self.pilot_value, dtype=np.complex64)

        for pol_idx in range(n_pol):
            pol_qam_data = symbols.data[pol_idx, :]
            pol_time_series: list[np.ndarray] = []

            for frame_idx in range(num_frames):
                # 1. 提取当前帧的净荷并填充到 OFDM 符号矩阵中
                start_s = frame_idx * payload_symbols_per_frame
                end_s = start_s + payload_symbols_per_frame

                # 形状: (n_data_subcarriers, num_data_symbols_per_frame)，F-order 保证每列是一个 OFDM 符号
                payload_grid = pol_qam_data[start_s:end_s].reshape(
                    self.n_data_subcarriers, self.num_data_symbols_per_frame, order="F"
                )

                # 2. 构建完整的频域矩阵 (n_subcarriers, num_data_symbols_per_frame)
                full_grid = np.zeros((self.n_subcarriers, self.num_data_symbols_per_frame), dtype=np.complex64)

                # 插入导频
                if self.pilot_spacing > 0:
                    full_grid[self.pilot_indices, :] = self.pilot_value

                # 插入净荷数据
                full_grid[self.data_indices, :] = payload_grid

                # 3. IFFT 变换到时域 (按列处理每个 OFDM 符号)
                time_domain_symbols = (
                    np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(full_grid, axes=0), axis=0), axes=0)
                    * self.n_subcarriers
                )

                # 4. 添加循环前缀 (CP)
                # 取最后 cp_len 行拼接到前面
                cp = time_domain_symbols[-self.cp_len :, :]
                ofdm_symbols_with_cp = np.vstack((cp, time_domain_symbols))

                # 5. 序列化当前帧的数据符号
                frame_time_data = ofdm_symbols_with_cp.flatten(order="F")  # 按列展开，保证时间连续

                # 6. 插入训练序列 (TS)
                if self.ts_pattern is not None:
                    ts_freq = self.ts_pattern[pol_idx, :]
                    ts_time = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ts_freq))) * self.n_subcarriers
                    ts_with_cp = np.concatenate((ts_time[-self.cp_len :], ts_time))

                    # 在帧头插入 TS
                    frame_time_data = np.concatenate([ts_with_cp, frame_time_data])

                    if pol_idx == 0:
                        ts_location_meta.append((frame_idx, 0))

                pol_time_series.append(frame_time_data)

            ofdm_output_list.append(np.concatenate(pol_time_series))

        # 更新上下文
        new_sps = self.fft_size + self.cp_len
        new_sample_rate = symbols.ctx.sym_rate * new_sps

        new_ctx = SignalContext(
            sample_rate=new_sample_rate,
            center_freq=0.0,
            n_pol=n_pol,
            sps=new_sps,
            sym_rate=symbols.ctx.sym_rate,
            bits_per_symbol=symbols.ctx.bits_per_symbol,
            is_ofdm=True,
            meta={
                **(symbols.ctx.meta or {}),
                "ofdm_n_subcarriers": self.n_subcarriers,
                "ofdm_cp_len": self.cp_len,
                "ofdm_data_indices": self.data_indices,
                "ofdm_pilot_subcarriers_indices": self.pilot_indices,
                "ofdm_expected_pilot_symbols": expected_pilot_symbols,
                "ofdm_pilot_spacing": self.pilot_spacing,
                "ofdm_ts_pattern": self.ts_pattern,
                "ofdm_ts_locations": ts_location_meta,
                "ofdm_num_data_symbols_per_frame": self.num_data_symbols_per_frame,
                "ofdm_has_ts": has_ts,
                "ofdm_num_ts_symbols_per_frame": num_ts_symbols_per_frame,
            },
        )

        return Signal(data=np.stack(ofdm_output_list, axis=0).astype(np.complex64), ctx=new_ctx)
