"""
QoT 评估器对比测试 - PyNLO 仿真 vs ISRS-GN 解析模型

比较基于 PyNLO 物理层仿真的 BER 评估与 ISRS-GN 解析模型的 SNR 估算。
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "D:/C-Codes/Optical-Networks/network-nlo-eval/src")

from physics_nlo_eval.channel.fiber import FiberChannel
from physics_nlo_eval.channel.wdm import ChannelExtractor, ChannelSpec, WDMDemux, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.dsp.carrier import PhaseNoiseCompensation
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp, TimingRecovery
from physics_nlo_eval.metrics.ber_evm import BERCalculator, Demapper
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ISRS-GN imports - kept for future reference
# from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath as GNOpticalPath
# from network_nlo_eval.network.elements import FiberSpanConfig as GNFiberSpanConfig, EDFAConfig as GNEDFAConfig
# from network_nlo_eval.core.spectrum import SpectrumGrid
# from network_nlo_eval.core.constants import C_LIGHT


@dataclass
class FiberSpanConfig:
    """单跨段光纤的配置参数。"""

    length_km: float = 100.0
    alpha_db_per_km: float = 0.2
    beta2_ps2_per_km: float = 17.0
    beta3_ps3_per_km: float = 0.06
    n2_si: float = 2.6e-20
    a_eff_um2: float = 80.0
    slope_a_eff_um2_per_nm: float = 0.05
    ref_lambda_nm: float = 1550.0
    d_pmd_ps_per_sqrt_km: float = 0.0
    coarse_step_size_m: float = 1000.0


@dataclass
class EDFAConfig:
    """EDFA 放大器配置。"""

    gain_db: float = 20.0
    nf_db: float = 5.0


@dataclass
class MultiSpanOpticalPath:
    """定义一条光路，包含多个跨段及其对应的配置。"""

    fiber_configs: list[FiberSpanConfig] = field(default_factory=list)
    edfa_configs: list[EDFAConfig] = field(default_factory=list)
    roadm_loss_db: float = 12.0
    total_length_km: float = 0.0

    def __post_init__(self):
        """初始化后计算总光纤长度。"""
        if self.total_length_km == 0.0:
            self.total_length_km = sum(f.length_km for f in self.fiber_configs)


# Parameters
REFERENCE_FREQ_HZ = 193.1e12
SYMBOL_RATE_HZ = 50e9
TARGET_SAMPLE_RATE_HZ = 400e9
MODULATION = "DP-QPSK"
SPS = 4
RRC_ROLL_OFF = 0.1

# WDM Plan - 3 channels
WDM_PLAN = WDMPlan(
    reference_freq_hz=REFERENCE_FREQ_HZ,
    channels=[
        ChannelSpec(channel_id=0, freq_offset_hz=-80e9),
        ChannelSpec(channel_id=1, freq_offset_hz=0e9),
        ChannelSpec(channel_id=2, freq_offset_hz=80e9),
    ],
)

# Fiber parameters (same for both models)
FIBER_LENGTH_KM = 80.0
ALPHA_DB_PER_KM = 0.2
BETA2_PS2_PER_KM = -20.0
BETA3_PS3_PER_KM = 0.1e-3
N2_SI = 2.6e-20
A_EFF_UM2 = 80.0
NUM_SPANS = 1


def ber_with_delay_search(rx_bits: np.ndarray, tx_bits: np.ndarray) -> float:
    """计算 BER，带延迟搜索。"""
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


class PyNLOQOTBenchmark:
    """基于 PyNLO 的 QoT 基准测试。"""

    def __init__(
        self,
        wdm_plan: WDMPlan,
        symbol_rate_hz: float = 50e9,
        target_sample_rate_hz: float = 400e9,
        modulation: str = "DP-QPSK",
        sps: int = 4,
        rrc_roll_off: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.wdm_plan = wdm_plan
        self.symbol_rate_hz = symbol_rate_hz
        self.target_sample_rate_hz = target_sample_rate_hz
        self.modulation = modulation
        self.sps = sps
        self.rrc_roll_off = rrc_roll_off
        self.seed = seed

        self.mapper = Mapper(modulation=modulation, power_norm=True, sym_rate=symbol_rate_hz)
        self.demapper = Demapper(modulation=modulation, power_norm=True)
        self.tx_rrc = RRCFilter(sps=sps, roll_off=rrc_roll_off, is_matched_filter=False)
        self.rx_rrc = RRCFilter(sps=sps, roll_off=rrc_roll_off, is_matched_filter=True)

    def create_wdm_signal(
        self,
        launch_power_dbm_per_pol: float,
        num_symbols: int = 4000,
    ) -> tuple[Signal, list[np.ndarray]]:
        """创建 WDM 信号和原始比特。"""
        np.random.seed(self.seed)
        channels = []
        all_bits = []

        for _ch_spec in self.wdm_plan.channels:
            bits_per_symbol = self.mapper.bits_per_symbol
            n_pol = self.mapper.n_pol
            total_bits = num_symbols * bits_per_symbol * n_pol

            tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
            all_bits.append(tx_bits)

            sig = self.mapper.forward(tx_bits)
            sig_rrc = self.tx_rrc.forward(sig)
            iq_mod = IQModulator(launch_power_dbm_per_pol=launch_power_dbm_per_pol)
            sig_optical = iq_mod.forward(sig_rrc)

            channels.append(sig_optical)

        mux = WDMMux(wdm_plan=self.wdm_plan, target_sample_rate_hz=self.target_sample_rate_hz)
        wdm_signal = mux.combine(channels)

        return wdm_signal, all_bits

    def run_through_fiber(
        self,
        wdm_signal: Signal,
        fiber_length_km: float,
        alpha_db_per_km: float = 0.2,
        beta2_ps2_per_km: float = -20.0,
        beta3_ps3_per_km: float = 0.1e-3,
        n2_si: float = 2.6e-20,
        enable_raman: bool = False,
        raman_r3=None,
    ) -> Signal:
        """通过光纤传播。"""
        fiber = FiberChannel(
            length_m=fiber_length_km * 1000,
            alpha_db_per_km=alpha_db_per_km,
            beta2_ps2_per_km=beta2_ps2_per_km,
            beta3_ps3_per_km=beta3_ps3_per_km,
            coarse_step_size_m=5000.0,
            n2_si=n2_si,
            a_eff_um2=80.0,
            slope_a_eff_um2_per_nm=0.05,
            ref_lambda_nm=1550.0,
            d_pmd_ps_per_sqrt_km=0.0,
            raman_r3=raman_r3,
            compute_ctx_name="numpy",
        )
        return fiber.forward(wdm_signal)

    def evaluate_ber(self, rx_wdm: Signal, tx_bits: list[np.ndarray], fiber_length_km: float) -> dict:
        """评估 BER 和 EVM（均先经 BPS 相位补偿）。"""
        cdc = ChromaticDispersionComp(
            fiber_length_km=fiber_length_km,
            beta2_ps2_per_km=-20.0,
            beta3_ps3_per_km=0.1e-3,
        )
        rx_cdc = cdc.forward(rx_wdm)
        demux = WDMDemux(filter_type="fir")
        demuxed = demux.forward(rx_cdc)

        results = {}
        timing_recovery = TimingRecovery(sps_in=demuxed.ctx.sps)
        ideal_constellation = self.demapper.constellation.lut

        for i, _ch_spec in enumerate(self.wdm_plan.channels):
            extractor = ChannelExtractor(channel_idx=i)
            rx_ch = extractor.forward(demuxed)
            rx_filt = self.rx_rrc.forward(rx_ch)
            rx_sync = timing_recovery.forward(rx_filt)

            # BPS 相位补偿（消除 Gordon-Mollenauer 效应导致的固定旋转）
            bps = PhaseNoiseCompensation(
                method="BPS",
                n_test_phases=256,
                block_size=256,
                constellation=self.demapper.constellation,
            )
            rx_comp = bps.forward(rx_sync)

            # BER（经 BPS 补偿后）
            rx_bits = self.demapper.forward(rx_comp)
            ber = ber_with_delay_search(rx_bits, tx_bits[i])

            # EVM（使用新 API：Signal + Constellation + tx_bits）
            ber_calc = BERCalculator(ignore_headers=0)
            evm = ber_calc.calculate_evm(rx_comp, self.demapper.constellation, tx_bits=tx_bits[i])

            # 诊断：错误符号分析
            self._diagnose_errors(rx_comp, rx_bits, tx_bits[i], ideal_constellation, i)

            results[i] = {"ber": ber, "evm": evm, "rx_comp": rx_comp}

        return results

    def _diagnose_errors(
        self, rx_comp: Signal, rx_bits: np.ndarray, tx_bits: np.ndarray, ideal_constellation: np.ndarray, ch_idx: int
    ) -> None:
        """诊断错误符号，分析错误原因。"""
        err_mask = rx_bits != tx_bits
        err_indices = np.where(err_mask)[0]
        if len(err_indices) == 0:
            return

        print(f"\n  === Channel {ch_idx} Error Analysis ({len(err_indices)} errors) ===")

        # 提取符号级数据
        sps = rx_comp.ctx.sps
        rx_syms = rx_comp.data[:, ::sps]  # (2, n_sym)
        n_sym = rx_syms.shape[1]

        # 从 tx_bits 推导每符号的理想索引
        bits_per_sym = 2
        tx_bits_i = tx_bits[: n_sym * bits_per_sym]
        pol0_bits = tx_bits_i[0::2].reshape(n_sym, 2)
        pol1_bits = tx_bits_i[1::2].reshape(n_sym, 2)
        tx_idx_pol0 = pol0_bits[:, 0] * 2 + pol0_bits[:, 1]
        tx_idx_pol1 = pol1_bits[:, 0] * 2 + pol1_bits[:, 1]

        # 逐极化分析
        for pol in range(2):
            pol_err_mask = err_mask[pol::2]
            pol_err_syms = np.where(pol_err_mask)[0]
            if len(pol_err_syms) == 0:
                continue

            tx_idx = tx_idx_pol0 if pol == 0 else tx_idx_pol1
            ideal_pts = ideal_constellation[tx_idx]

            print(f"\n  Pol {pol}: {len(pol_err_syms)} symbol errors")
            print(f"  {'Sym':>6} {'TX idx':>6} {'Ideal':>12} {'RX':>12} {'Error':>10}")
            for sym_idx in pol_err_syms[:10]:  # 只打印前10个
                ideal = ideal_pts[sym_idx]
                rx_pt = rx_syms[pol, sym_idx]
                err_vec = rx_pt - ideal
                dist = np.abs(err_vec)
                print(  # noqa: E501
                    f"  {sym_idx:>6} {tx_idx[sym_idx]:>6} ({ideal.real:+.3f}{ideal.imag:+.3f}j) "
                    f"({rx_pt.real:+.3f}{rx_pt.imag:+.3f}j) {dist:.4f}"
                )

            # 分析错误分布：是否集中在某个符号值
            err_symbol_values, err_counts = np.unique(tx_idx[pol_err_syms], return_counts=True)
            print("  Error distribution by TX symbol value:")
            for sv, cnt in zip(err_symbol_values, err_counts, strict=True):
                qpsk_labels = ["00→(1+j)", "01→(-1+j)", "10→(-1-j)", "11→(1-j)"]
                print(f"    idx={sv} ({qpsk_labels[sv]}): {cnt} errors")


def test_pynlo_qot():
    """运行 PyNLO QoT 基准测试。"""
    print("=" * 70)
    print("PyNLO QoT Benchmark Test")
    print("=" * 70)

    np.random.seed(42)
    benchmark = PyNLOQOTBenchmark(
        wdm_plan=WDM_PLAN,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        target_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
        modulation=MODULATION,
        sps=SPS,
        rrc_roll_off=RRC_ROLL_OFF,
        seed=42,
    )

    # Create signal
    launch_power_dbm = 0.0
    wdm_signal, tx_bits = benchmark.create_wdm_signal(launch_power_dbm_per_pol=launch_power_dbm)
    print(f"\nSignal: {len(WDM_PLAN.channels)} channels, launch power = {launch_power_dbm} dBm/pol")

    fiber_length_km = 80.0
    n2_si = 0.0  # Linear only first

    # Test 1: Linear (NLO disabled)
    print(f"\n--- Test 1: Linear (NLO disabled), {fiber_length_km} km ---")
    rx_wdm = benchmark.run_through_fiber(wdm_signal, fiber_length_km, n2_si=0.0)
    ber_results = benchmark.evaluate_ber(rx_wdm, tx_bits, fiber_length_km)
    ber_strs = [f"{ber_results[i]['ber']:.2e}" for i in range(len(WDM_PLAN.channels))]
    evm_strs = [f"{ber_results[i]['evm']:.2f}%" for i in range(len(WDM_PLAN.channels))]
    print(f"BER results: [{', '.join(ber_strs)}]")
    print(f"EVM  results: [{', '.join(evm_strs)}]")

    # Test 2: Nonlinear (NLO enabled)
    print(f"\n--- Test 2: Nonlinear (NLO enabled), {fiber_length_km} km ---")
    rx_wdm_nl = benchmark.run_through_fiber(wdm_signal, fiber_length_km, n2_si=2.6e-20)
    ber_results_nl = benchmark.evaluate_ber(rx_wdm_nl, tx_bits, fiber_length_km)
    ber_strs = [f"{ber_results_nl[i]['ber']:.2e}" for i in range(len(WDM_PLAN.channels))]
    evm_strs = [f"{ber_results_nl[i]['evm']:.2f}%" for i in range(len(WDM_PLAN.channels))]
    print(f"BER results: [{', '.join(ber_strs)}]")
    print(f"EVM  results: [{', '.join(evm_strs)}]")

    # Test 3: Nonlinear + Raman
    print(f"\n--- Test 3: Nonlinear + Raman, {fiber_length_km} km ---")
    from pynlo.utility.chi3 import raman_from_dat

    n_samples = wdm_signal.num_samples
    dt = 1 / TARGET_SAMPLE_RATE_HZ
    r3 = raman_from_dat(
        n_samples,
        dt,
        filepath="D:/C-Codes/Optical-Networks/PyNLO/pynlo/utility/RamanGain.dat",
        raman_frac=0.18,
        analytic=True,
    )[1]
    rx_wdm_raman = benchmark.run_through_fiber(
        wdm_signal, fiber_length_km, n2_si=2.6e-20, enable_raman=True, raman_r3=r3
    )
    ber_results_raman = benchmark.evaluate_ber(rx_wdm_raman, tx_bits, fiber_length_km)
    ber_strs = [f"{ber_results_raman[i]['ber']:.2e}" for i in range(len(WDM_PLAN.channels))]
    evm_strs = [f"{ber_results_raman[i]['evm']:.2f}%" for i in range(len(WDM_PLAN.channels))]
    print(f"BER results: [{', '.join(ber_strs)}]")
    print(f"EVM  results: [{', '.join(evm_strs)}]")

    return (
        (ber_results, rx_wdm),
        (ber_results_nl, rx_wdm_nl),
        (ber_results_raman, rx_wdm_raman),
        benchmark,
        tx_bits,
        fiber_length_km,
    )


def plot_constellation(rx_symbols, ideal_constellation, title, filename, n_pol=2):
    """绘制双偏振星座图，带理想点参考。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    pol_labels = ["Pol 0", "Pol 1"]
    colors = ["blue", "orange"]

    for pol in range(n_pol):
        ax = axes[pol]
        rx_pol = rx_symbols[pol, :]

        # 理想星座点
        ax.scatter(
            ideal_constellation.real,
            ideal_constellation.imag,
            c="red",
            marker="x",
            s=80,
            linewidths=1.5,
            label="Ideal",
            zorder=5,
        )

        # 接收符号（取前 2000 个避免过于密集）
        n_show = min(len(rx_pol), 2000)
        ax.scatter(
            rx_pol.real[:n_show],
            rx_pol.imag[:n_show],
            c=colors[pol],
            alpha=0.3,
            s=8,
            label=f"RX (n={n_show})",
        )

        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title(f"{pol_labels[pol]} - EVM")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_constellation_comparison(results_by_case, benchmark, tx_bits, fiber_length_km):
    """为每种 case 绘制星座图：原始 vs BPS 补偿后。"""
    import matplotlib

    matplotlib.use("Agg")

    case_names = {
        "linear": "Linear (NLO disabled)",
        "nonlinear": "Nonlinear (NLO enabled)",
        "raman": "Nonlinear + Raman",
    }

    ideal_constellation = benchmark.demapper.constellation.lut
    ber_calc = BERCalculator(ignore_headers=0)

    for case_key, case_name in case_names.items():
        results = results_by_case[case_key]
        rx_wdm = results_by_case[f"{case_key}_rx"]

        for ch_idx in range(len(benchmark.wdm_plan.channels)):
            evm = results[ch_idx]["evm"]

            # RAW: 重新执行 DSP chain（CDC → Demux → Extractor → RRC → Timing）
            # 提取原始符号（未经过 BPS 补偿）
            cdc = ChromaticDispersionComp(
                fiber_length_km=fiber_length_km,
                beta2_ps2_per_km=-20.0,
                beta3_ps3_per_km=0.1e-3,
            )
            rx_cdc = cdc.forward(rx_wdm)
            demux = WDMDemux(filter_type="fir")
            demuxed = demux.forward(rx_cdc)
            extractor = ChannelExtractor(channel_idx=ch_idx)
            rx_ch = extractor.forward(demuxed)
            rx_filt = benchmark.rx_rrc.forward(rx_ch)
            timing_recovery = TimingRecovery(sps_in=demuxed.ctx.sps)
            rx_sync = timing_recovery.forward(rx_filt)

            sps = rx_sync.ctx.sps
            rx_sym_raw = rx_sync.data[:, ::sps].copy()
            rx_power = np.mean(np.abs(rx_sym_raw) ** 2, axis=1, keepdims=True)
            rx_power[rx_power == 0] = 1.0
            rx_sym_raw = rx_sym_raw / np.sqrt(rx_power)

            # RAW 绘图
            title_orig = f"{case_name} - Ch {ch_idx} RAW (EVM={evm:.2f}%)"
            fname_orig = f"constellation_{case_key}_ch{ch_idx}_raw.png"
            plot_constellation(rx_sym_raw, ideal_constellation, title_orig, fname_orig)

            # BPS 补偿：直接使用 evaluate_ber 已存储的 rx_comp
            rx_comp = results[ch_idx]["rx_comp"]
            rx_sym_comp = rx_comp.data[:, :: rx_comp.ctx.sps].copy()
            rx_power_comp = np.mean(np.abs(rx_sym_comp) ** 2, axis=1, keepdims=True)
            rx_power_comp[rx_power_comp == 0] = 1.0
            rx_sym_comp = rx_sym_comp / np.sqrt(rx_power_comp)

            # 估算 BPS 估计的相位旋转
            phase_rot_pol0 = np.angle(np.mean(rx_sym_raw[0] / (rx_sym_comp[0] + 1e-12))) * 180 / np.pi
            phase_rot_pol1 = np.angle(np.mean(rx_sym_raw[1] / (rx_sym_comp[1] + 1e-12))) * 180 / np.pi

            # EVM（单行调用，使用 Signal-based API）
            evm_comp = ber_calc.calculate_evm(rx_comp, benchmark.demapper.constellation, tx_bits=tx_bits[ch_idx])

            title_comp = (
                f"{case_name} - Ch {ch_idx} BPS (EVM={evm_comp:.2f}%, rot={phase_rot_pol0:.1f}°/{phase_rot_pol1:.1f}°)"
            )
            fname_comp = f"constellation_{case_key}_ch{ch_idx}_bps.png"
            plot_constellation(rx_sym_comp, ideal_constellation, title_comp, fname_comp)


def main():
    """主函数。"""
    print("PyNLO QoT Comparison Test")
    print("=" * 70)

    # Run PyNLO benchmark
    (ber_lin, rx_wdm_lin), (ber_nl, rx_wdm_nl), (ber_raman, rx_wdm_raman), benchmark, tx_bits, fiber_length_km = (
        test_pynlo_qot()
    )

    print("\n" + "=" * 70)
    print("Summary: PyNLO BER/EVM Comparison")
    print("=" * 70)
    header = (
        f"{'Channel':>10} {'Linear BER':>12} {'Nonlin BER':>12} {'+Raman BER':>12} "
        f"| {'Linear EVM':>10} {'Nonlin EVM':>10} {'+Raman EVM':>10}"
    )
    print(header)
    print("-" * 90)
    for i in range(len(WDM_PLAN.channels)):
        row = (
            f"  Ch{i:>3} {ber_lin[i]['ber']:>12.2e} {ber_nl[i]['ber']:>12.2e} "
            f"{ber_raman[i]['ber']:>12.2e} | {ber_lin[i]['evm']:>10.2f}% "
            f"{ber_nl[i]['evm']:>10.2f}% {ber_raman[i]['evm']:>10.2f}%"
        )
        print(row)

    # Calculate SNR from BER (approximate)
    print("\nNote: For QPSK at Eb/N0=10dB, theoretical BER ~ 3.9e-6")
    print("      For reference: BER 1e-3 corresponds to ~ 9.6 dB Eb/N0")

    # Plot constellation diagrams
    print("\n--- Generating constellation plots ---")
    results_by_case = {
        "linear": ber_lin,
        "linear_rx": rx_wdm_lin,
        "nonlinear": ber_nl,
        "nonlinear_rx": rx_wdm_nl,
        "raman": ber_raman,
        "raman_rx": rx_wdm_raman,
    }
    plot_constellation_comparison(results_by_case, benchmark, tx_bits, fiber_length_km)


if __name__ == "__main__":
    main()
