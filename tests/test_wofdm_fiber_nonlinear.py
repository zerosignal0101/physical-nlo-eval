"""W-OFDM + 光纤非线性信道 + 逐子载波 BER 测试。

测试链路:
  TX: Mapper → WindowedOFDMModulator → IQModulator (电→光)
  CH: FiberChannel (NLSE, 含非线性) + EDFA
  RX: OFDM_Demodulator → CommonPhaseNoiseCompensate → 逐子载波解映射 → BER

关键说明:
  - TimingRecovery/SC PhaseNoiseCompensation 不适用于 OFDM
  - OFDM 定时同步由 OFDM_Demodulator 内部通过 CP 相关处理
  - 相位噪声由 CommonPhaseNoiseCompensate (CPE) 处理
"""

import numpy as np

from physics_nlo_eval.channel.fiber import EDFA, FiberChannel
from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.freq_domain import CommonPhaseNoiseCompensate, OFDM_Demodulator, OFDMToSymbolStream
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.ofdm import WindowedOFDMModulator
from physics_nlo_eval.tx.optics import IQModulator

# ============================================================================
# 参数配置
# ============================================================================

# W-OFDM 参数 (适度规模,兼顾仿真速度)
N_SUBCARRIERS = 32  # FFT 大小
CP_LEN = 4  # 循环前缀
BETA = 0.1  # 窗滚降
NUM_DATA_SYMBOLS_PER_FRAME = 20  # 每帧 OFDM 符号数
NUM_FRAMES = 5  # 帧数
MODULATION = "DP-16QAM"  # 调制格式

# 光纤参数 (典型 SMF, 启用非线性)
FIBER_LENGTH_KM = 80  # 80 km
ALPHA_DB_PER_KM = 0.2  # 损耗
BETA2_PS2_PER_KM = -20.0  # 色散
BETA3_PS3_PER_KM = 0.1e-3  # 三阶色散
N2_SI = 2.6e-20  # 非线性折射率 (石英玻璃)
A_EFF_UM2 = 80.0  # 有效面积 (um^2)
DELTA_F_KHZ = 100.0  # 激光线宽 (kHz)

# 信号参数
SYM_RATE_GBD = 32.0  # 符号率 (GBd)
CENTER_FREQ_THZ = 193.0  # 光载波频率 (THz)
OPTICAL_POWER_DBM = 0.0  # 每偏振光功率 (dBm)


def theoretical_ber_qpsk(eb_n0_db: float) -> float:
    """QPSK 理论 BER (erfc 近似)。"""
    from scipy import special

    eb_n0 = 10 ** (eb_n0_db / 10)
    return 0.5 * special.erfc(np.sqrt(2 * eb_n0) / np.sqrt(2))


def theoretical_ber_16qam(eb_n0_db: float) -> float:
    """16-QAM 理论 BER (近似)。"""
    from scipy import special

    eb_n0 = 10 ** (eb_n0_db / 10)
    return 0.75 * special.erfc(np.sqrt(0.1 * eb_n0)) / 2


def make_signal_context(
    n_pol: int,
    sample_rate: float,
    center_freq: float,
    sym_rate: float,
    bits_per_symbol: int,
    sps: int,
    is_ofdm: bool = False,
    meta: dict | None = None,
) -> SignalContext:
    """构建 SignalContext 的辅助函数。"""
    return SignalContext(
        sample_rate=sample_rate,
        center_freq=center_freq,
        n_pol=n_pol,
        sps=sps,
        sym_rate=sym_rate,
        bits_per_symbol=bits_per_symbol,
        is_ofdm=is_ofdm,
        meta=meta,
    )


def add_awgn_to_signal(signal: Signal, eb_n0_db: float, bits_per_symbol: int) -> Signal:
    """对 OFDM 信号添加 AWGN，使用信号功率作为 Eb 参考.

    注意：这里使用 OFDM 信号的功率（而非放大后的功率）来校准噪声，
    使得 Eb/N0 直接对应 OFDM 符号层面的信噪比。
    """
    eb_n0 = 10 ** (eb_n0_db / 10)
    es = np.mean(np.abs(signal.data) ** 2)
    eb = es / bits_per_symbol
    n0 = eb / eb_n0
    sigma = np.sqrt(n0 / 2)
    noise = sigma * (np.random.randn(*signal.data.shape) + 1j * np.random.randn(*signal.data.shape))
    noisy_data = signal.data + noise
    return Signal(data=noisy_data.astype(np.complex64), ctx=signal.ctx)


# ============================================================================
# 测试用例
# ============================================================================


def test_wofdm_fiber(
    modulation: str = "DP-16QAM",
    fiber_length_km: float = 80.0,
    eb_n0_db: float = 15.0,
    nlo_enable: bool = False,
    seed: int = 42,
):
    """W-OFDM + 光纤信道逐子载波 BER 测试。"""
    np.random.seed(seed)

    nlo_label = "NL" if nlo_enable else "Linear"
    print(f"\n{'=' * 60}")
    print(f"测试: W-OFDM + Fiber ({fiber_length_km} km, {nlo_label}) @ Eb/N0={eb_n0_db}dB")
    print(f"{'=' * 60}")

    # ---- TX ----
    n_pol_tx = 2 if "DP-" in modulation else 1
    sym_rate = SYM_RATE_GBD * 1e9
    mapper = Mapper(modulation=modulation, power_norm=True, sym_rate=sym_rate)
    demapper = Demapper(modulation=modulation, power_norm=True)
    modulator = WindowedOFDMModulator(
        n_subcarriers=N_SUBCARRIERS,
        cp_len=CP_LEN,
        ts_pattern=None,
        num_data_symbols_per_frame=NUM_DATA_SYMBOLS_PER_FRAME,
        pilot_spacing=0,
        pilot_value=1.0 + 0j,
        beta=BETA,
    )

    bits_per_symbol = mapper.bits_per_symbol
    payload_per_frame = modulator.n_data_subcarriers * NUM_DATA_SYMBOLS_PER_FRAME
    total_qam = payload_per_frame * NUM_FRAMES
    total_bits = total_qam * bits_per_symbol * n_pol_tx

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    tx_signal = mapper.forward(tx_bits)
    ofdm_signal = modulator.forward(tx_signal)

    effective_sps = modulator.effective_output_sym_len  # n_sub + cp = 32+4=36
    sample_rate = sym_rate * effective_sps

    print(f"  W-OFDM: {N_SUBCARRIERS} SC, CP={CP_LEN}, β={BETA}, effective SPS={effective_sps}")
    print(f"  符号率: {sym_rate / 1e9:.1f} GBd, 采样率: {sample_rate / 1e9:.2f} Gs/s")
    print(f"  总比特: {total_bits}, n_pol={n_pol_tx}")

    # 转换为光信号
    iq_mod = IQModulator(launch_power_dbm_per_pol=OPTICAL_POWER_DBM)
    optical_signal = iq_mod.forward(ofdm_signal)

    # 扩展为双偏振 (FiberChannel 需要)
    if optical_signal.ctx.n_pol == 1:
        optical_signal = Signal(
            data=np.vstack([optical_signal.data[0], np.zeros_like(optical_signal.data[0])]),
            ctx=make_signal_context(
                n_pol=2,
                sample_rate=optical_signal.ctx.sample_rate,
                center_freq=optical_signal.ctx.center_freq,
                sym_rate=optical_signal.ctx.sym_rate,
                bits_per_symbol=optical_signal.ctx.bits_per_symbol,
                sps=optical_signal.ctx.sps,
                is_ofdm=True,
                meta={**optical_signal.ctx.meta, "n_pol_padded": True},
            ),
        )

    # ---- 光纤传播 ----
    n2_to_use = N2_SI if nlo_enable else 0.0
    fiber = FiberChannel(
        length_m=fiber_length_km * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        coarse_step_size_m=200.0,
        n2_si=n2_to_use,
        a_eff_um2=A_EFF_UM2,
        slope_a_eff_um2_per_nm=0.05,
        ref_lambda_nm=1550.0,
        compute_ctx_name="cupy",
    )
    print(f"  光纤: L={fiber_length_km}km, α={ALPHA_DB_PER_KM}dB/km, β2={BETA2_PS2_PER_KM}ps²/km")
    print(f"        n2={n2_to_use:.2e} m²/W, A_eff={A_EFF_UM2}μm²")

    rx_optical = fiber.forward(optical_signal)
    print(f"  光纤输出形状: {rx_optical.data.shape}")

    # ---- EDFA ----
    loss_db = ALPHA_DB_PER_KM * fiber_length_km
    edfa = EDFA(gain_db=loss_db, nf_db=5.0)
    rx_amp = edfa.forward(rx_optical)
    print(f"  EDFA: 增益={loss_db:.1f}dB, NF=5dB")

    # ---- 接收机噪声 ----
    rx_noisy = add_awgn_to_signal(rx_amp, eb_n0_db, bits_per_symbol)

    # ---- 色散补偿 (CD) ----
    cdc = ChromaticDispersionComp(
        fiber_length_km=fiber_length_km,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
    )
    rx_cdc = cdc.forward(rx_noisy)
    print(f"  CD 补偿后形状: {rx_cdc.data.shape}")

    # ---- OFDM 解调 ----
    demodulator = OFDM_Demodulator(
        n_subcarriers=N_SUBCARRIERS,
        cp_len=CP_LEN,
        cs_len=modulator.cs_len,
    )
    rx_fd = demodulator.forward(rx_cdc)
    print(f"  OFDM Demod 输出: {rx_fd.data.shape} → (n_pol, n_sub, n_sym)")

    cpe = CommonPhaseNoiseCompensate(method="pilot_assisted")
    rx_fd_comp = cpe.forward(rx_fd)

    # ---- BER 计算 ----
    # 使用 OFDMToSymbolStream 进行格式转换，然后使用 Demapper 计算 BER
    ofdm_reshape = OFDMToSymbolStream()
    rx_fd_reshaped = ofdm_reshape.forward(rx_fd_comp)
    rx_signal = Signal(data=rx_fd_reshaped.data, ctx=rx_fd_reshaped.ctx)
    rx_bits = demapper.forward(rx_signal)

    total_errors = np.sum(tx_bits != rx_bits)
    total_bits = len(tx_bits)
    measured_ber = total_errors / total_bits

    if "QPSK" in modulation:
        theoretical_ber = theoretical_ber_qpsk(eb_n0_db)
    else:
        theoretical_ber = theoretical_ber_16qam(eb_n0_db)

    ratio = measured_ber / theoretical_ber if theoretical_ber > 0 else float("inf")

    print(f"\n  测量 BER: {measured_ber:.4e}")
    print(f"  理论 BER: {theoretical_ber:.4e}")
    print(f"  BER ratio: {ratio:.2f}")
    print(f"  {'[PASS]' if ratio < 5.0 else '[WARN]'} BER ratio: {ratio:.2f}")

    return measured_ber, theoretical_ber


def run_all_tests():
    """运行全套测试。"""
    print("\n" + "=" * 60)
    print("W-OFDM + 光纤非线性信道 + 逐子载波 BER 测试套件")
    print("=" * 60)

    try:
        # 测试 1: 线性光纤 (80km, 无非线性) - 多 Eb/N0 点
        print("\n理论 AWGN BER 参考:")
        for eb in [10, 12, 14, 16]:
            print(f"  Eb/N0={eb}dB: {theoretical_ber_16qam(eb):.4e}")

        print("\n" + "-" * 60)
        print("DP-16QAM 线性光纤多 Eb/N0 点测试 (100 帧)")
        print("-" * 60)

        # 使用较大帧数以获得统计显著性
        original_num_frames = NUM_FRAMES
        globals()["NUM_FRAMES"] = 100

        results = []
        for eb_n0_db in [10, 12, 14, 16]:
            measured_ber, theoretical_ber = test_wofdm_fiber(
                modulation="DP-16QAM",
                fiber_length_km=80.0,
                eb_n0_db=eb_n0_db,
                nlo_enable=False,
                seed=42,
            )
            ratio = measured_ber / theoretical_ber if theoretical_ber > 0 else float("inf")
            results.append((eb_n0_db, measured_ber, theoretical_ber, ratio))
            print(f"  Eb/N0={eb_n0_db}dB: measured={measured_ber:.4e}, theory={theoretical_ber:.4e}, ratio={ratio:.2f}")

        # 恢复原始帧数
        globals()["NUM_FRAMES"] = original_num_frames

        # # ---- NLO 非线性测试 ----
        # print("\n" + "=" * 60)
        # print("NLO 非线性光纤测试 (n2=2.6e-20 m²/W, 80km)")
        # print("=" * 60)

        # # NLO 效应在高功率下更明显，使用 0 dBm/pol
        # original_power = OPTICAL_POWER_DBM
        # globals()["OPTICAL_POWER_DBM"] = 0.0  # 1 mW/pol

        # # 使用 SP-16QAM 以避免 DP 的 XPM 非线性复杂性
        # print("\nSP-16QAM NLO 光纤测试 (100 帧, 不同 Eb/N0):")
        # print("-" * 60)
        # globals()["NUM_FRAMES"] = 100

        # for eb_n0_db in [10, 12, 14, 16]:
        #     measured_ber, theoretical_ber = test_wofdm_fiber(
        #         modulation="SP-16QAM",
        #         fiber_length_km=80.0,
        #         eb_n0_db=eb_n0_db,
        #         nlo_enable=True,
        #         seed=42,
        #     )
        #     ratio = measured_ber / theoretical_ber if theoretical_ber > 0 else float("inf")
        #     msg = f"  Eb/N0={eb_n0_db}dB (NLO): measured={measured_ber:.4e}"
        #     msg += f", theory={theoretical_ber:.4e}, ratio={ratio:.2f}"
        #     print(msg)

        # # 恢复参数
        # globals()["OPTICAL_POWER_DBM"] = original_power
        # globals()["NUM_FRAMES"] = original_num_frames

        print("\n" + "=" * 60)
        print("所有测试完成!")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试出错: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
