"""WDM + 光纤信道 BER 测试 (不含 EDFA)."""

import numpy as np

from physics_nlo_eval.channel.fiber import FiberChannel
from physics_nlo_eval.channel.wdm import ChannelExtractor, ChannelSpec, WDMDemux, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp, TimingRecovery
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter

# ============================================================================
# 参数
# ============================================================================
WDM_PLAN = WDMPlan(
    reference_freq_hz=193.1e12,
    channels=[
        ChannelSpec(channel_id=0, freq_offset_hz=-80e9),
        ChannelSpec(channel_id=1, freq_offset_hz=0e9),
        ChannelSpec(channel_id=2, freq_offset_hz=80e9),
    ],
)
TARGET_SAMPLE_RATE_HZ = 400e9

MODULATION = "DP-QPSK"
SYMBOL_RATE_HZ = 50e9
RRC_SPS = 4
RRC_ROLL_OFF = 0.1

LAUNCH_POWER_DBM_PER_POL = 0.0

FIBER_LENGTH_KM = 80.0
ALPHA_DB_PER_KM = 0.2
BETA2_PS2_PER_KM = -20.0
BETA3_PS3_PER_KM = 0.1e-3
N2_SI = 0.0
A_EFF_UM2 = 80.0

NUM_SYMBOLS = 4000
EB_N0_DB = 15.0


# ============================================================================
# 辅助函数
# ============================================================================


def add_awgn_relative(signal: Signal, es_n0_db: float, bits_per_symbol: int) -> Signal:
    """基于信号当前功率添加相对 AWGN 噪声."""
    sig_power_per_pol = np.mean(np.abs(signal.data) ** 2, axis=-1, keepdims=True)
    es_n0 = 10 ** (es_n0_db / 10) * bits_per_symbol
    noise_power_per_pol = sig_power_per_pol / es_n0
    sigma = np.sqrt(noise_power_per_pol / 2)
    noise = sigma * (np.random.randn(*signal.data.shape) + 1j * np.random.randn(*signal.data.shape))
    return Signal(data=(signal.data + noise).astype(np.complex128), ctx=signal.ctx)


def ber_with_delay_search(rx_bits: np.ndarray, tx_bits: np.ndarray) -> float:
    """计算 BER，带延迟搜索."""
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
    modulation: str = "DP-QPSK",
    fiber_length_km: float = 80.0,
    eb_n0_db: float = 15.0,
    nlo_enable: bool = False,
    seed: int = 42,
):
    """WDM + 光纤 BER 测试 (无 EDFA)."""
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"WDM Fiber Test: {modulation}, {fiber_length_km} km, Eb/N0={eb_n0_db} dB")
    print(f"Launch Power: {LAUNCH_POWER_DBM_PER_POL} dBm/Pol")
    print(f"{'=' * 60}")

    # ---- TX 组件 ----
    mapper = Mapper(modulation=modulation, power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    demapper = Demapper(modulation=modulation, power_norm=True)
    tx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=False)
    rx_rrc = RRCFilter(sps=RRC_SPS, roll_off=RRC_ROLL_OFF, is_matched_filter=True)
    iq_mod = IQModulator(launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL)

    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = NUM_SYMBOLS * bits_per_symbol * n_pol

    print(f"  {modulation}: {NUM_SYMBOLS} symbols, {n_pol} pol, {bits_per_symbol} bits/sym")

    # ---- 生成通道 ----
    channels = []
    all_bits = []
    for ch_spec in WDM_PLAN.channels:
        tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
        all_bits.append(tx_bits)

        sig = mapper.forward(tx_bits)
        sig_rrc = tx_rrc.forward(sig)
        sig_optical = iq_mod.forward(sig_rrc)

        avg_pwr_w = np.mean(np.abs(sig_optical.data) ** 2)
        print(f"  Ch{ch_spec.channel_id}: launch power {avg_pwr_w * 1e3:.4f} mW")

        channels.append(sig_optical)

    # ---- WDM 合波 ----
    wdm_mux = WDMMux(wdm_plan=WDM_PLAN, target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ)
    wdm_signal = wdm_mux.combine(channels)
    print(f"  WDM: {len(WDM_PLAN.channels)} ch, sr={wdm_signal.ctx.sample_rate / 1e9:.0f} GHz")

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
        compute_ctx_name="numpy",
    )
    rx_optical = fiber.forward(wdm_signal)

    fiber_loss = ALPHA_DB_PER_KM * fiber_length_km
    rx_power = np.mean(np.abs(rx_optical.data) ** 2) * 1e3
    expected_power = 10 ** (LAUNCH_POWER_DBM_PER_POL / 10) * 10 ** (-fiber_loss / 10)
    print(
        f"  Fiber: {fiber_length_km} km, loss={fiber_loss:.1f} dB, "
        f"rx power={rx_power:.4f} mW (expected ~{expected_power * 1e3:.4f} mW)"
    )

    # ---- CD 补偿 ----
    cdc = ChromaticDispersionComp(
        fiber_length_km=fiber_length_km,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
    )
    rx_cdc = cdc.forward(rx_optical)

    # ---- AWGN ----
    rx_noisy = add_awgn_relative(rx_cdc, eb_n0_db, bits_per_symbol)
    print(f"  AWGN: Eb/N0={eb_n0_db} dB, Es/N0~{eb_n0_db + 10 * np.log10(bits_per_symbol):.1f} dB")

    # ---- WDM 分波 ----
    wdm_demux = WDMDemux(filter_type="fir")
    demuxed = wdm_demux.forward(rx_noisy)
    print(f"  Demux: shape={demuxed.data.shape}")

    # ---- 逐通道 BER ----
    all_passed = True
    timing_recovery = TimingRecovery(sps_in=demuxed.ctx.sps)
    for i, ch_spec in enumerate(WDM_PLAN.channels):
        extractor = ChannelExtractor(channel_idx=i)
        rx_ch = extractor.forward(demuxed)
        rx_filt = rx_rrc.forward(rx_ch)
        rx_sync = timing_recovery.forward(rx_filt)
        rx_bits = demapper.forward(rx_sync)

        best_ber = ber_with_delay_search(rx_bits, all_bits[i])

        threshold = 1e-3
        if nlo_enable:
            threshold = 0.3
        status = "[PASS]" if best_ber < threshold else "[FAIL]"
        if best_ber > threshold:
            all_passed = False
        print(f"  Ch{i} ({ch_spec.freq_offset_hz / 1e9} GHz): BER={best_ber:.2e} {status}")

    return all_passed


def run_all_tests():
    """Run all WDM + Fiber BER tests."""
    print("\n" + "=" * 60)
    print("WDM + Fiber BER Test Suite (No EDFA)")
    print("=" * 60)

    try:
        test_wdm_fiber(modulation="DP-QPSK", fiber_length_km=80.0, eb_n0_db=15.0, nlo_enable=False)
        test_wdm_fiber(modulation="DP-QPSK", fiber_length_km=80.0, eb_n0_db=15.0, nlo_enable=True)
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
