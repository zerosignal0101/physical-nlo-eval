"""实现 OFDM 调制（IFFT, CP 添加, 训练序列插入）。"""

import numpy as np

from physics_nlo_eval.core.base import BaseBlock
from physics_nlo_eval.core.signal import Signal, SignalContext


class OFDMModulator(BaseBlock):
    """OFDM 调制模块。

    执行 IFFT、添加循环前缀 (CP) 和插入训练序列 (TS)。

    Parameters
    ----------
    n_subcarriers : int
        OFDM 子载波数量。
    cp_len : int
        循环前缀的长度。
    ts_pattern : np.ndarray | None, optional
        用于训练序列的符号模式 (例如 Zadoff-Chu 序列)。
        如果为 None，则不插入训练序列。
        形状应为 (n_pol, n_subcarriers)。
    num_data_symbols_per_frame : int
        每帧的数据符号数 (不含训练符号)。
    """

    def __init__(
        self,
        n_subcarriers: int,
        cp_len: int,
        ts_pattern: np.ndarray | None = None,
        num_data_symbols_per_frame: int = 100,
    ) -> None:
        """初始化 OFDMModulator。"""
        super().__init__(name="OFDMModulator")
        if not n_subcarriers > 0:
            raise ValueError("Number of subcarriers must be positive.")
        if not cp_len >= 0:
            raise ValueError("CP length must be non-negative.")
        if not num_data_symbols_per_frame > 0:
            raise ValueError("Number of data symbols per frame must be positive.")

        self.n_subcarriers = n_subcarriers
        self.cp_len = cp_len
        self.ts_pattern = ts_pattern
        self.num_data_symbols_per_frame = num_data_symbols_per_frame
        self.fft_size = n_subcarriers  # OFDM 符号的时域长度 (不含 CP)

    def forward(self, symbols: Signal) -> Signal:
        """
        将 QAM 符号流转换为 OFDM 时域信号。

        Parameters
        ----------
        symbols : Signal
            输入 QAM 符号信号。数据形状为 (n_pol, N_total_qam_symbols)。
            每个符号已经通过 Mapper 进行了归一化。

        Returns
        -------
        Signal
            输出 OFDM 时域信号。数据形状为 (n_pol, N_ofdm_samples)。
            SignalContext 会更新 is_ofdm, sample_rate, sps 等。
        """
        if symbols.data.ndim != 2:
            raise ValueError("Input symbols data must be 2D (n_pol, N_symbols).")

        n_pol = symbols.ctx.n_pol
        total_qam_symbols = symbols.data.shape[1]

        # 确保每个偏振的符号数量足够填充帧
        symbols_per_frame = self.n_subcarriers * self.num_data_symbols_per_frame
        if total_qam_symbols < symbols_per_frame:
            raise ValueError(
                f"Not enough QAM symbols ({total_qam_symbols}) for one OFDM frame "
                f"({symbols_per_frame} required). Consider padding or increasing input data."
            )

        # 计算 OFDM 帧数量
        num_frames = total_qam_symbols // symbols_per_frame
        if total_qam_symbols % symbols_per_frame != 0:
            print(
                f"Warning: QAM symbols ({total_qam_symbols}) not perfectly divisible "
                f"by symbols per frame ({symbols_per_frame}). Some symbols will be truncated."
            )

        ofdm_output_list: list[np.ndarray] = []
        ts_location_meta: list[tuple[int, int]] = []  # (frame_idx, start_sample_in_frame)

        # 针对每个偏振进行 OFDM 调制
        for pol_idx in range(n_pol):
            pol_qam_symbols = symbols.data[pol_idx, :]
            pol_ofdm_frames: list[np.ndarray] = []

            for frame_idx in range(num_frames):
                frame_start_idx = frame_idx * symbols_per_frame
                frame_end_idx = frame_start_idx + symbols_per_frame

                # Reshape QAM symbols into subcarrier-symbol matrix
                # Shape: (n_subcarriers, num_data_symbols_per_frame)
                subcarrier_symbols = pol_qam_symbols[frame_start_idx:frame_end_idx].reshape(
                    self.n_subcarriers, self.num_data_symbols_per_frame
                )

                # Prepare for IFFT: (n_subcarriers, num_data_symbols_per_frame)
                # Apply IFFT column-wise (over subcarriers for each symbol)
                time_domain_symbols = (
                    np.fft.ifft(subcarrier_symbols, axis=0) * self.n_subcarriers
                )  # Scale by N for power preservation

                # Add CP to each OFDM symbol
                # Shape: (fft_size + cp_len, num_data_symbols_per_frame)
                ofdm_with_cp = np.vstack((time_domain_symbols[-self.cp_len :, :], time_domain_symbols))

                # Flatten frame (time domain)
                frame_time_domain = ofdm_with_cp.flatten()

                # Insert TS if provided
                if self.ts_pattern is not None:
                    # For simplicity, insert TS at the beginning of each frame (or after some symbols)
                    # Real systems have more complex pilot/TS insertion
                    ts_pol_pattern = self.ts_pattern[pol_idx, :]
                    ts_time_domain = np.fft.ifft(ts_pol_pattern) * self.n_subcarriers
                    ts_with_cp = np.hstack((ts_time_domain[-self.cp_len :], ts_time_domain))

                    # Decide insertion point. Here, prepend.
                    frame_time_domain = np.concatenate([ts_with_cp, frame_time_domain])
                    if pol_idx == 0:  # Only record TS location once per frame
                        # TS length (with CP) = self.fft_size + self.cp_len
                        # Start of TS within frame: 0
                        # End of TS within frame: self.fft_size + self.cp_len
                        ts_location_meta.append((frame_idx, 0))

                pol_ofdm_frames.append(frame_time_domain)

            ofdm_output_list.append(np.concatenate(pol_ofdm_frames))

        ofdm_data = np.stack(ofdm_output_list, axis=0)  # Final shape (n_pol, N_ofdm_samples)

        # Update SignalContext
        # OFDM 的采样率就是 IFFT 后的时域采样率
        # sps 在 OFDM 语境下通常指每子载波的符号数，但在这里为了保持与单载波一致，
        # 我们让它反映每个 OFDM 符号的时域采样点数 (fft_size + cp_len)
        # 此时的 sample_rate 对应于每个 OFDM 时域采样点
        new_sps = self.fft_size + self.cp_len
        new_sample_rate = symbols.ctx.sym_rate * new_sps  # Baseband electrical signal rate

        new_ctx = SignalContext(
            sample_rate=new_sample_rate,
            center_freq=0.0,  # 仍然是基带电信号
            n_pol=n_pol,
            sps=new_sps,  # Each OFDM symbol is effectively one "symbol" with this many samples
            sym_rate=symbols.ctx.sym_rate,  # The original QAM symbol rate per subcarrier
            bits_per_symbol=symbols.ctx.bits_per_symbol,  # QAM bits per symbol
            is_ofdm=True,
            meta={
                **(symbols.ctx.meta or {}),
                "ofdm_n_subcarriers": self.n_subcarriers,
                "ofdm_cp_len": self.cp_len,
                "ofdm_fft_size": self.fft_size,
                "ofdm_ts_pattern": self.ts_pattern,
                "ofdm_ts_locations": ts_location_meta,  # Store if any TS was inserted
                "ofdm_num_data_symbols_per_frame": self.num_data_symbols_per_frame,
            },
        )
        return Signal(data=ofdm_data.astype(np.complex64), ctx=new_ctx)
