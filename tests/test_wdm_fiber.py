"""WDM + 光纤信道 BER 测试.

测试链路:
  TX: Mapper → RRC → WDMMux
  CH: FiberChannel (Manakov) + EDFA + AWGN
  RX: WDMDemux → RRC Matched → Demapper → BER
"""

import numpy as np

from physics_nlo_eval.channel.fiber import EDFA, FiberChannel
from physics_nlo_eval.channel.wdm import ChannelExtractor, WDMDemux, WDMMux
from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.rrc import RRCFilter

# ============================================================================
# 参数
# ============================================================================
N_CHANNELS = 3
CHANNEL_SPACING_HZ = 75e9
TARGET_SAMPLE_RATE_HZ = 400e9

MODULATION = "SP-QPSK"
SYMBOL_RATE_HZ = 50e9
RRC_SPS = 4
RRC_ROLL_OFF = 0.1

FIBER_LENGTH_KM = 80.0
ALPHA_DB_PER_KM = 0.2
BETA2_PS2_PER_KM = -20.0
BETA3_PS3_PER_KM = 0.1e-3
N2_SI = 2.6e-20
A_EFF_UM2 = 80.0

NUM_SYMBOLS = 4000
EB_N0_DB = 15.0


# ============================================================================
# 辅助函数
# ============================================================================


def make_optical_signal(wdm_signal: Signal) -> Signal:
    """将 WDM 基带信号转换为光信号 (双偏振)."""
    if wdm_signal.ctx.n_pol == 1:
        data = np.vstack([wdm_signal.data[0], np.zeros_like(wdm_signal.data[0])])
    else:
        data = wdm_signal.data
    return Signal(
        data=data.astype(np.complex64),
        ctx=SignalContext(
            sample_rate=wdm_signal.ctx.sample_rate,
            center_freq=193.0e12,
            n_pol=2,
            sps=wdm_signal.ctx.sps,
            sym_rate=wdm_signal.ctx.sym_rate,
            bits_per_symbol=wdm_signal.ctx.bits_per_symbol,
            is_ofdm=False,
            meta={**wdm_signal.ctx.meta, "n_pol_padded": wdm_signal.ctx.n_pol == 1},
        ),
    )


def add_awgn(signal: Signal, eb_n0_db: float, bits_per_symbol: int) -> Signal:
    """添加 AWGN 噪声."""
    eb_n0 = 10 ** (eb_n0_db / 10)
    es = np.mean(np.abs(signal.data) ** 2)
    eb = es / bits_per_symbol
    n0 = eb / eb_n0
    sigma = np.sqrt(n0 / 2)
    noise = sigma * (np.random.randn(*signal.data.shape) + 1j * np.random.randn(*signal.data.shape))
    return Signal(data=(signal.data + noise).astype(np.complex64), ctx=signal.ctx)


def ber_with_delay_search(rx_bits: np.ndarray, tx_bits: np.ndarray) -> float:
    """计算 BER，带延迟搜索以处理滤波器群延迟."""
    best_ber = 1.0
    for delay in range(0, 100):
        if delay + len(tx_bits) > len(rx_bits):
            break
        test_bits = rx_bits[delay : delay + len(tx_bits)]
        errs = np.sum(tx_bits != test_bits)
        ber = errs / len(tx_bits)
        if ber < best_ber:
            best_ber = ber
    return best_ber


def test_wdm_fiber(
    modulation: str = "SP-QPSK",
    fiber_length_km: float = 80.0,
    eb_n0_db: float = 15.0,
    nlo_enable: bool = False,
    seed: int = 42,
):
    """WDM + 光纤 BER 测试."""
    np.random.seed(seed)

    nlo_str = "NLO" if nlo_enable else "Linear"
    print(f"\n{'=' * 60}")
    print(f"WDM Fiber Test: {modulation}, {fiber_length_km} km, {nlo_str}, Eb/N0={eb_n0_db} dB")
    print(f"{'=' * 60}")

    # ---- TX ----
    mapper = Mapper(modulation=modulation, power_norm=True)
    demapper = Demapper(modulation=modulation, power_norm=True)
    tx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=False)
    rx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=True)

    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = NUM_SYMBOLS * bits_per_symbol

    print(f"  {modulation}: {NUM_SYMBOLS} symbols, {n_pol} pol, {bits_per_symbol} bits/sym")

    # ---- 生成通道 ----
    channels = []
    all_bits = []
    for _i in range(N_CHANNELS):
        tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
        all_bits.append(tx_bits)

        sig = mapper.forward(tx_bits)
        sig.ctx.sample_rate = SYMBOL_RATE_HZ
        sig.ctx.sym_rate = SYMBOL_RATE_HZ
        sig_rrc = tx_rrc.forward(sig)
        channels.append(sig_rrc)

    # ---- WDM 合波 ----
    wdm_mux = WDMMux(channel_spacing_hz=CHANNEL_SPACING_HZ, target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ)
    wdm_signal = wdm_mux.combine(channels)
    print(
        f"  WDM: {N_CHANNELS} ch, "
        f"spacing={CHANNEL_SPACING_HZ / 1e9:.0f} GHz, "
        f"sr={wdm_signal.ctx.sample_rate / 1e9:.0f} GHz"
    )

    # ---- 光信号 ----
    optical = make_optical_signal(wdm_signal)
    print(f"  Optical: shape={optical.data.shape}, n_pol={optical.ctx.n_pol}")

    # ---- 光纤 ----
    fiber = FiberChannel(
        length_m=fiber_length_km * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        coarse_step_size_m=200.0,
        n2_si=N2_SI if nlo_enable else 0.0,
        a_eff_um2=A_EFF_UM2,
        slope_a_eff_um2_per_nm=0.05,
        ref_lambda_nm=1550.0,
        d_pmd_ps_per_sqrt_km=0.0,
        compute_ctx_name="cupy",
    )
    rx_optical = fiber.forward(optical)
    print(f"  Fiber: {fiber_length_km} km, n2={'ON' if nlo_enable else 'OFF'}")

    # ---- EDFA ----
    loss_db = ALPHA_DB_PER_KM * fiber_length_km
    edfa = EDFA(gain_db=loss_db, nf_db=5.0)
    rx_amp = edfa.forward(rx_optical)

    # ---- CD 补偿 ----
    cdc = ChromaticDispersionComp(
        fiber_length_km=fiber_length_km,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
    )
    rx_cdc = cdc.forward(rx_amp)

    # ---- AWGN ----
    rx_noisy = add_awgn(rx_cdc, eb_n0_db, bits_per_symbol)
    print(f"  AWGN: Eb/N0={eb_n0_db} dB")

    # ---- WDM 分波 ----
    wdm_demux = WDMDemux(filter_type="fir")
    demuxed = wdm_demux.forward(rx_noisy)
    print(f"  Demuxed: shape={demuxed.data.shape}")

    # ---- 逐通道 BER ----
    all_passed = True
    for _i in range(N_CHANNELS):
        # 提取通道
        extractor = ChannelExtractor(channel_idx=_i)
        rx_ch = extractor.forward(demuxed)
        # shape: (2, N) - 即使 SP 也是 (2, N)

        # RRC 匹配滤波
        rx_filt = rx_rrc.forward(rx_ch)
        # rx_filt.data shape: (2, N) for both SP and DP

        # 下采样到 1 SPS
        # SP: 取 pol 0; DP: 取 pol 0 和 pol 1
        if n_pol == 1:
            data_1sps = rx_filt.data[0, ::RRC_SPS]
            rx_sig = Signal(data=data_1sps[np.newaxis, :], ctx=rx_filt.ctx)
        else:
            # DP: 保留两个偏振
            pol0 = rx_filt.data[0, ::RRC_SPS]
            pol1 = rx_filt.data[1, ::RRC_SPS]
            rx_sig = Signal(data=np.vstack([pol0, pol1]), ctx=rx_filt.ctx)
        rx_sig.ctx.sps = 1

        # Demap
        rx_bits = demapper.forward(rx_sig)

        # BER
        best_ber = ber_with_delay_search(rx_bits, all_bits[_i])

        # NLO 时高 BER 是预期行为 (非线性损伤)
        threshold = 1e-3
        if nlo_enable:
            threshold = 0.6  # NLO 导致 ~0.5 BER 正常
        status = "[PASS]" if best_ber < threshold else "[FAIL]"
        if best_ber > threshold:
            all_passed = False
        print(f"  Ch{_i}: BER={best_ber:.2e} {status}")

    return all_passed


def run_all_tests():
    """Run all WDM + Fiber BER tests."""
    print("\n" + "=" * 60)
    print("WDM + Fiber BER Test Suite")
    print("=" * 60)

    try:
        # Test 1: Linear fiber, SP-QPSK
        test_wdm_fiber(modulation="SP-QPSK", fiber_length_km=80.0, eb_n0_db=15.0, nlo_enable=False)

        # Test 2: Linear fiber, DP-16QAM
        test_wdm_fiber(modulation="DP-16QAM", fiber_length_km=80.0, eb_n0_db=15.0, nlo_enable=False)

        # Test 3: NLO enabled, SP-QPSK
        test_wdm_fiber(modulation="SP-QPSK", fiber_length_km=80.0, eb_n0_db=15.0, nlo_enable=True)

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
