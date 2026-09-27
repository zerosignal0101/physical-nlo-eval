"""实现 OFDM 调制（IFFT, CP/CS 添加, 训练序列插入, 频域导频插入）。"""

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
        if cp_len < 0:
            raise ValueError("CP length must be non-negative.")
        if num_data_symbols_per_frame <= 0:
            raise ValueError("Number of data symbols per frame must be positive.")
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

    # 封装 IFFT 和 CP 添加逻辑，便于 WindowedOFDMModulator 复用或修改
    def _process_single_ofdm_symbol(self, freq_domain_symbol: np.ndarray) -> np.ndarray:
        """
        对单个频域 OFDM 符号执行 IFFT 并添加 CP。

        Parameters
        ----------
        freq_domain_symbol : np.ndarray
            单个 OFDM 符号的频域数据 (n_subcarriers,)。

        Returns
        -------
        np.ndarray
            添加 CP 后的时域 OFDM 符号 (fft_size + cp_len,).
        """
        time_domain_symbol = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(freq_domain_symbol))) * self.n_subcarriers
        # cp_len == 0 时 [-0:] 会取到整个符号（Python 负数零切片陷阱），必须显式给空切片。
        cp = time_domain_symbol[-self.cp_len :] if self.cp_len else time_domain_symbol[:0]
        ofdm_symbol_with_cp = np.concatenate((cp, time_domain_symbol))
        return ofdm_symbol_with_cp

    def forward(self, symbols: Signal) -> Signal:
        """将 QAM 符号流转换为 OFDM 时域信号。"""
        if symbols.data.ndim != 2:
            raise ValueError("Input symbols data must be 2D (n_pol, N_symbols).")

        n_pol = symbols.ctx.n_pol
        total_input_symbols = symbols.data.shape[1]

        payload_symbols_per_frame = self.n_data_subcarriers * self.num_data_symbols_per_frame
        num_frames = total_input_symbols // payload_symbols_per_frame

        if num_frames == 0:
            raise ValueError(f"Not enough symbols for one OFDM frame. Need {payload_symbols_per_frame}.")

        ofdm_output_list: list[np.ndarray] = []
        ts_location_meta: list[tuple[int, int]] = []
        has_ts = self.ts_pattern is not None
        num_ts_symbols_per_frame = 1 if has_ts else 0

        # Prepare pilot reference info for downstream DSP
        expected_pilot_symbols = np.full(len(self.pilot_indices), self.pilot_value, dtype=np.complex64)

        for pol_idx in range(n_pol):
            pol_qam_data = symbols.data[pol_idx, :]
            pol_time_series: list[np.ndarray] = []

            for frame_idx in range(num_frames):
                start_s = frame_idx * payload_symbols_per_frame
                end_s = start_s + payload_symbols_per_frame

                payload_grid = pol_qam_data[start_s:end_s].reshape(
                    self.n_data_subcarriers, self.num_data_symbols_per_frame, order="F"
                )

                full_grid = np.zeros((self.n_subcarriers, self.num_data_symbols_per_frame), dtype=np.complex64)
                if self.pilot_spacing > 0:
                    full_grid[self.pilot_indices, :] = self.pilot_value
                full_grid[self.data_indices, :] = payload_grid

                frame_time_data_symbols: list[np.ndarray] = []
                for sym_col_idx in range(self.num_data_symbols_per_frame):
                    freq_sym = full_grid[:, sym_col_idx]
                    time_sym_with_cp = self._process_single_ofdm_symbol(freq_sym)
                    frame_time_data_symbols.append(time_sym_with_cp)

                # Concatenate data symbols for this frame
                frame_time_data = np.concatenate(frame_time_data_symbols)

                if self.ts_pattern is not None:
                    ts_freq = self.ts_pattern[pol_idx, :]
                    ts_time_with_cp = self._process_single_ofdm_symbol(ts_freq)
                    frame_time_data = np.concatenate([ts_time_with_cp, frame_time_data])

                    if pol_idx == 0:
                        ts_location_meta.append((frame_idx, 0))  # Store starting symbol index of TS within the frame

                pol_time_series.append(frame_time_data)

            ofdm_output_list.append(np.concatenate(pol_time_series))

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
                "ofdm_is_windowed": False,  # Add this flag
            },
        )

        return Signal(data=np.stack(ofdm_output_list, axis=0).astype(np.complex64), ctx=new_ctx)


class WindowedOFDMModulator(OFDMModulator):
    """带有时域升余弦加窗和重叠添加 (OLA) 的 OFDM 调制器 (W-OFDM)。

    Parameters
    ----------
    beta : float
        窗函数的滚降系数 (0 ~ 1)。决定了循环后缀 (CS) 的长度和窗的平滑程度。
        beta * n_subcarriers 决定了 CS 长度。
    """

    def __init__(
        self,
        n_subcarriers: int,
        cp_len: int,
        ts_pattern: np.ndarray | None = None,
        num_data_symbols_per_frame: int = 100,
        pilot_spacing: int = 0,
        pilot_value: complex = 1.0 + 0j,
        beta: float = 0.1,  # New parameter for windowing
    ) -> None:
        super().__init__(
            n_subcarriers,
            cp_len,
            ts_pattern,
            num_data_symbols_per_frame,
            pilot_spacing,
            pilot_value,
        )
        self.name = f"WindowedOFDMModulator(beta={beta})"
        if not 0 <= beta <= 1:
            raise ValueError("Beta (window roll-off factor) must be between 0 and 1.")
        self.beta = beta
        # Cyclic Suffix length (CS): typically beta * FFT_size
        self.cs_len = int(np.round(self.n_subcarriers * self.beta))
        # 重叠-相加 (OLA) 要求窗的上升沿不短于 CP，否则相邻符号的尾巴叠不满，
        # 窗不再满足 COLA。判据是 CS <= CP，而不是 CS < 符号总长。
        if self.cs_len > self.cp_len:
            raise ValueError("Window overlap (round(beta * n_subcarriers)) must not exceed CP length.")

        # Total length of an OFDM symbol for windowing (CP + FFT_size + CS)
        self.extended_symbol_len = self.cp_len + self.fft_size + self.cs_len
        # Effective output length of a single symbol after OLA
        self.effective_output_sym_len = self.fft_size + self.cp_len

        # Pre-compute the Raised Cosine window
        self._window = self._generate_raised_cosine_window()

    def _generate_raised_cosine_window(self) -> np.ndarray:
        """生成满足 COLA 的升余弦窗。

        重叠-相加 (OLA) 要求相邻窗在重叠区逐点相加恒为 1（constant overlap-add），
        否则 W-OFDM 的子载波正交性会被破坏，等效于引入符号间干扰。

        旧实现只把 CP 段和 CS 段各自做一条升余弦斜坡，两条斜坡之和在重叠区
        并不恒为 1；这里改用一对互补的半余弦：前半段取 ``sin²``、后半段取
        ``cos²``，由 ``sin²θ + cos²θ = 1`` 保证重叠区严格求和为 1。中间
        ``fft_size`` 段保持 1.0。
        """
        window = np.ones(self.extended_symbol_len, dtype=np.float64)
        if self.cs_len:
            # 相位取半格偏移 (+0.5)，避免窗口端点落在 0 而损失首尾样点能量。
            phase = np.pi * (np.arange(self.cs_len) + 0.5) / (2 * self.cs_len)
            window[: self.cs_len] = np.sin(phase) ** 2
            window[-self.cs_len :] = np.cos(phase) ** 2
        return window

    def forward(self, symbols: Signal) -> Signal:
        """将 QAM 符号流转换为加窗 OFDM 时域信号。"""
        if symbols.data.ndim != 2:
            raise ValueError("Input symbols data must be 2D (n_pol, N_symbols).")

        n_pol = symbols.ctx.n_pol
        total_input_symbols = symbols.data.shape[1]

        payload_symbols_per_frame = self.n_data_subcarriers * self.num_data_symbols_per_frame
        num_frames = total_input_symbols // payload_symbols_per_frame

        if num_frames == 0:
            raise ValueError(f"Not enough symbols for one OFDM frame. Need {payload_symbols_per_frame}.")

        ofdm_output_list: list[np.ndarray] = []
        ts_location_meta: list[tuple[int, int]] = []
        has_ts = self.ts_pattern is not None
        num_ts_symbols_per_frame = 1 if has_ts else 0

        expected_pilot_symbols = np.full(len(self.pilot_indices), self.pilot_value, dtype=np.complex64)

        for pol_idx in range(n_pol):
            pol_qam_data = symbols.data[pol_idx, :]
            current_qam_symbol_idx = 0
            current_output_time_series = np.array([], dtype=np.complex64)

            # Determine total number of effective OFDM symbols (including TS)
            total_effective_symbols = num_frames * (self.num_data_symbols_per_frame + num_ts_symbols_per_frame)

            for sym_global_idx in range(total_effective_symbols):
                # Check if current symbol is a TS
                is_ts_symbol = False
                if has_ts:
                    # Find which frame this symbol belongs to and its position within the frame
                    symbols_per_full_frame = self.num_data_symbols_per_frame + num_ts_symbols_per_frame
                    frame_idx = sym_global_idx // symbols_per_full_frame
                    sym_in_frame_idx = sym_global_idx % symbols_per_full_frame

                    if sym_in_frame_idx < num_ts_symbols_per_frame:  # Assuming TS is always at the beginning
                        is_ts_symbol = True
                        if pol_idx == 0:  # Only record for one polarization
                            if (frame_idx, sym_in_frame_idx) not in ts_location_meta:
                                ts_location_meta.append((frame_idx, sym_in_frame_idx))

                if is_ts_symbol:
                    # Use TS pattern
                    freq_sym = self.ts_pattern[pol_idx, :]
                else:
                    # Extract payload QAM symbols and fill grid
                    if current_qam_symbol_idx >= total_input_symbols:
                        # Should not happen if num_frames calculation is correct, but for safety
                        break

                    # Map QAM payload to full subcarrier grid
                    # current_qam_symbol_idx points to the first QAM symbol for the current OFDM symbol's data part
                    num_qam_for_this_ofdm = self.n_data_subcarriers
                    qam_payload = pol_qam_data[current_qam_symbol_idx : current_qam_symbol_idx + num_qam_for_this_ofdm]

                    if len(qam_payload) < num_qam_for_this_ofdm:
                        # Pad if not enough QAM symbols for the last OFDM symbol
                        qam_payload = np.pad(qam_payload, (0, num_qam_for_this_ofdm - len(qam_payload)), "constant")

                    freq_sym = np.zeros(self.n_subcarriers, dtype=np.complex64)
                    if self.pilot_spacing > 0:
                        freq_sym[self.pilot_indices] = self.pilot_value
                    freq_sym[self.data_indices] = qam_payload

                    current_qam_symbol_idx += num_qam_for_this_ofdm

                # 1. IFFT + CP (from base class)
                time_domain_symbol_no_cs = self._process_single_ofdm_symbol(freq_sym)  # Length: fft_size + cp_len

                # 2. Add Cyclic Suffix (CS)
                # CS is a copy of the beginning of the FFT part
                cs = time_domain_symbol_no_cs[
                    self.cp_len : self.cp_len + self.cs_len
                ]  # First `cs_len` samples of IFFT part
                extended_time_symbol = np.concatenate(
                    (time_domain_symbol_no_cs, cs)
                )  # Length: cp_len + fft_size + cs_len

                # 3. Apply window
                windowed_symbol = extended_time_symbol * self._window

                # 4. Overlap-and-Add (OLA)
                # The effective output symbol duration is fft_size + cp_len
                # The last 'cs_len' samples of current symbol overlap with the first 'cp_len' samples of next.
                # Here, we add the overlapping tail of the previous symbol to the head of the current symbol.

                # If this is the first symbol, just append the full windowed symbol.
                # cs_len == 0 时没有重叠区，后续符号直接首尾相接即可；否则下面的
                # [-0:] 切片会取到整段并造成形状错位。
                if sym_global_idx == 0 or self.cs_len == 0:
                    if sym_global_idx == 0:
                        current_output_time_series = windowed_symbol
                    else:
                        current_output_time_series = np.concatenate((current_output_time_series, windowed_symbol))
                else:
                    # Overlap: the last 'cs_len' samples of the previous *effective* symbol (before windowing)
                    # will be merged with the first 'cp_len' samples of the current windowed symbol.
                    # The previous symbol's actual output was `effective_output_sym_len`.
                    # The current symbol's OLA-ed part takes from `cs_len` from the previous and `cp_len` from current.

                    # The `current_output_time_series` stores the *already OLA-ed* sequence.
                    # The `windowed_symbol` is the *new* symbol.

                    # Overlap length for OLA:
                    overlap_len = self.cp_len  # For standard OLA, this is the length of the overlapping part.
                    # The windowed symbol has length extended_symbol_len.
                    # The part to add is the (extended_symbol_len - effective_output_sym_len)
                    # This part is 'cs_len'

                    # Overlap and add the `cs_len` tail of the previous effective symbol
                    # (which is already in current_output_time_series's end)
                    # with the `cp_len` head of the current symbol.

                    # current_output_time_series[-overlap_len:] += windowed_symbol[:overlap_len]
                    # current_output_time_series = np.concatenate(
                    #     (current_output_time_series[:-overlap_len], windowed_symbol))

                    # A more correct OLA implementation for windowed OFDM:
                    # Each windowed symbol is `extended_symbol_len` long.
                    # The *new* block to add is `effective_output_sym_len` long.
                    # The overlap is `cs_len` samples.

                    # `new_block_to_add` is `windowed_symbol[cp_len:]` effectively.
                    # It has length `fft_size + cs_len`.

                    # Overlap with the existing tail of `current_output_time_series`.
                    # The tail of `current_output_time_series` is `cs_len` long (from previous symbol's tail).

                    # For a truly windowed OFDM with OLA:
                    # `current_output_time_series` is built by appending `extended_symbol_len - overlap_len` samples
                    # and adding `overlap_len` samples to the overlap region.
                    # Overlap region is `cs_len`.

                    # The samples from `windowed_symbol[0:cs_len]` are added to
                    # `current_output_time_series[-cs_len:]`.
                    # Then `windowed_symbol[cs_len:]` is appended.

                    # The output sample rate should reflect `effective_output_sym_len`.
                    overlap_length_for_ola = (
                        self.cs_len
                    )  # The part of the previous symbol that overlaps with the current

                    current_output_time_series[-overlap_length_for_ola:] += windowed_symbol[0:overlap_length_for_ola]
                    current_output_time_series = np.concatenate(
                        (current_output_time_series, windowed_symbol[overlap_length_for_ola:])
                    )

            ofdm_output_list.append(current_output_time_series)

        new_sps = self.effective_output_sym_len  # Each effective symbol has this many samples
        new_sample_rate = symbols.ctx.sym_rate * new_sps

        new_ctx = SignalContext(
            sample_rate=new_sample_rate,
            center_freq=0.0,
            n_pol=n_pol,
            sps=new_sps,  # Updated SPS
            sym_rate=symbols.ctx.sym_rate,
            bits_per_symbol=symbols.ctx.bits_per_symbol,
            is_ofdm=True,
            meta={
                **(symbols.ctx.meta or {}),
                "ofdm_n_subcarriers": self.n_subcarriers,
                "ofdm_cp_len": self.cp_len,
                "ofdm_cs_len": self.cs_len,  # New
                "ofdm_beta": self.beta,  # New
                "ofdm_data_indices": self.data_indices,
                "ofdm_pilot_subcarriers_indices": self.pilot_indices,
                "ofdm_expected_pilot_symbols": expected_pilot_symbols,
                "ofdm_pilot_spacing": self.pilot_spacing,
                "ofdm_ts_pattern": self.ts_pattern,
                "ofdm_ts_locations": ts_location_meta,
                "ofdm_num_data_symbols_per_frame": self.num_data_symbols_per_frame,
                "ofdm_has_ts": has_ts,
                "ofdm_num_ts_symbols_per_frame": num_ts_symbols_per_frame,
                "ofdm_is_windowed": True,  # Flag this as windowed
            },
        )

        # 截断最后一个符号多出的 CS 尾巴，使输出长度对齐有效符号长度
        # 这样做在连续传输模拟中更方便，且不会损失信息（因为信息在前面的 FFT 窗里）
        # cs_len == 0 时 [: -0] 会截成空数组，必须退化为不截断。
        stacked = np.stack(ofdm_output_list, axis=0)
        final_data = stacked[:, : -self.cs_len] if self.cs_len else stacked

        return Signal(data=final_data.astype(np.complex64), ctx=new_ctx)
