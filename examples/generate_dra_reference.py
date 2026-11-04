#!/usr/bin/env python3
"""生成分布式拉曼放大 (DRA) 的独立波形参考数据集。

用法::

    python examples/generate_dra_reference.py
    python examples/generate_dra_reference.py --output-dir examples/dra_reference
    python examples/generate_dra_reference.py --depleted            # 追加耗尽数据集
    python examples/generate_dra_reference.py --depleted --skip-base

收敛数据（62.5 m）生成到独立目录::

    python examples/generate_dra_reference.py \
        --output-dir examples/dra_reference_converged --coarse-step-m 62.5
    python examples/generate_dra_reference.py \
        --output-dir examples/dra_reference_converged --coarse-step-m 62.5 \
        --depleted --skip-base

默认输出 ``dra_reference.npz``（输入/输出复波形）和 ``dra_reference.json``
（参数、逐信道功率/NLI/SNR、EVM 等元数据）。``--depleted`` 另行输出
``dra_reference_depleted.{npz,json}``：3/5 信道耗尽参考、1 信道 SPM 探针，
以及 3 信道前向泵的发射功率扫描。数据集刻意保持小规模：3/5 信道、
50 GBd DP-QPSK、100 km 单跨，便于完整复现和逐位核对。

全 WDM（c57/cl152、75 GHz、64 GBd）::

    # 只跑步长探针，写 step_convergence.json
    python examples/generate_dra_reference.py --wideband --wideband-probe-only

    # 复用探针并生成主数据（每个计划单独跑，避免超过命令超时）
    python examples/generate_dra_reference.py --wideband --reuse-probe \
        --wideband-plans c57 --wideband-step-map c57=0.48828125
    python examples/generate_dra_reference.py --wideband --reuse-probe \
        --wideband-plans cl152 --wideband-symbols-map cl152=128 \
        --wideband-step-map cl152=0.48828125

全 WDM 的步长收敛远比 3/5/7ch 苛刻（见 ``docs/dra-waveform.md`` §12）；
默认的历史 500 m 路径保持不变。
"""

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from physics_nlo_eval.channel import (
    ChannelExtractor,
    ChannelSpec,
    DistributedRamanFiber,
    PumpSpec,
    WDMDemux,
    WDMMux,
    WDMPlan,
)
from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.dsp.time_domain import ChromaticDispersionComp
from physics_nlo_eval.metrics.ber_evm import BERCalculator
from physics_nlo_eval.tx.mapper import Constellation, Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter

MODULATION = "DP-QPSK"
SYMBOL_RATE_HZ = 50e9
SPS = 4
ROLL_OFF = 0.1
#: NLI 分析带宽与解析 GN 模型对齐：取符号率（50 GHz），而不是含滚降的 55 GHz。
NLI_ANALYSIS_BANDWIDTH_HZ = SYMBOL_RATE_HZ
LAUNCH_POWER_DBM_PER_POL = 0.0

FIBER_LENGTH_KM = 100.0
ALPHA_DB_PER_KM = 0.2
BETA2_PS2_PER_KM = -20.0
BETA3_PS3_PER_KM = 0.1e-3
N2_SI = 2.6e-20
A_EFF_UM2 = 80.0
COARSE_STEP_M = 500.0

PUMP_WAVELENGTH_NM = 1450.0
PUMP_ATTENUATION_DB_PER_KM = 0.25

DATASETS = {
    "3ch": {"count": 3, "offsets_hz": np.array([-100e9, 0.0, 100e9]), "sample_rate_hz": 500e9},
    "5ch": {
        "count": 5,
        "offsets_hz": np.array([-200e9, -100e9, 0.0, 100e9, 200e9]),
        "sample_rate_hz": 700e9,
    },
    "7ch": {
        "count": 7,
        "offsets_hz": np.array([-300e9, -200e9, -100e9, 0.0, 100e9, 200e9, 300e9]),
        "sample_rate_hz": 900e9,
    },
}
#: 单信道 SPM 探针（不属于默认数据集）。
ONE_CHANNEL_DATASET = {"count": 1, "offsets_hz": np.array([0.0]), "sample_rate_hz": 100e9}

CONFIGS: dict[str, list[PumpSpec]] = {
    "nopump": [],
    "fwd_0p10w": [PumpSpec(PUMP_WAVELENGTH_NM, 0.10, "forward", PUMP_ATTENUATION_DB_PER_KM)],
    "fwd_0p20w": [PumpSpec(PUMP_WAVELENGTH_NM, 0.20, "forward", PUMP_ATTENUATION_DB_PER_KM)],
    "bwd_0p10w": [PumpSpec(PUMP_WAVELENGTH_NM, 0.10, "backward", PUMP_ATTENUATION_DB_PER_KM)],
    "bwd_0p20w": [PumpSpec(PUMP_WAVELENGTH_NM, 0.20, "backward", PUMP_ATTENUATION_DB_PER_KM)],
}

#: 1 信道探针只需无泵与 0.2 W 前后向，用于隔离 SPM。
ONE_CHANNEL_CONFIGS = ("nopump", "fwd_0p20w", "bwd_0p20w")
#: 发射功率扫描（dBm/偏振）：0.5/1/2 mW。
POWER_SWEEP_DBM_PER_POL = (-3.0103, 0.0, 3.0103)
POWER_SWEEP_TAG = "3ch_power_sweep"

#: 历史数据使用的粗步长；新生成的收敛数据应显著小于它。
HISTORICAL_STEP_M = 500.0
#: 步长收敛判据：相邻两档步长的中心信道 SNR 差应小于该值。
STEP_CONVERGENCE_TOL_DB = 0.01
#: 走离经验判据：每步的群速度走离应 ≲0.15 symbol。
WALK_OFF_TARGET_SYMBOLS = 0.15

# ---------------------------------------------------------------------------
# 全 WDM 参考（c57 / cl152）：75 GHz 栅格、64 GBd、C 或 C+L
# ---------------------------------------------------------------------------

#: 与 network-nlo-eval 的 ``SMF_100KM`` 近似对齐的光纤参数。
WIDEBAND_FIBER_LENGTH_KM = 100.0
#: 1550 nm 处的多项式衰减值。第一版对所有信道用平坦近似；C+L 边缘的
#: 实际值约为 0.162–0.178 dB/km，差异在文档中列为已知限制。
WIDEBAND_ALPHA_DB_PER_KM = 0.162
#: D=17 ps/nm/km @1550 nm 对应的 beta2。
WIDEBAND_BETA2_PS2_PER_KM = -21.7
#: 在 64 GBd 信道内 beta3 可忽略（与历史数据同为 1e-4 ps^3/km 量级）。
WIDEBAND_BETA3_PS3_PER_KM = 1e-4
WIDEBAND_N2_SI = 2.6e-20
WIDEBAND_A_EFF_UM2 = 80.0
WIDEBAND_PUMP_ATTENUATION_DB_PER_KM = 0.25

WIDEBAND_NUM_SYMBOLS = 256
WIDEBAND_SEED = 20261104
WIDEBAND_SECOND_SEED = 20261105
#: 步长探针只需短记录；64 symbols 足以稳定中心信道 NLI 的步长趋势。
WIDEBAND_PROBE_NUM_SYMBOLS = 64
#: 从历史 500 m 一路加密到 0.98 m；全 WDM 的 NLI 在可负担步长内仍未收敛，
#: 探针会继续自动减半到 ``WIDEBAND_MIN_PROBE_STEP_M``。
WIDEBAND_PROBE_STEPS_M = (
    500.0,
    250.0,
    125.0,
    62.5,
    31.25,
    15.625,
    7.8125,
    3.90625,
    1.953125,
    0.9765625,
)
#: 自动加密的最小步长；再小则单次传播超过可接受时间，明确记录未收敛。
WIDEBAND_MIN_PROBE_STEP_M = 0.48828125

#: 六泵后向配置（C+L）。功率为单泵入纤功率 (W)。
SIX_PUMP_WAVELENGTHS_NM = (1340.0, 1360.0, 1427.0, 1445.0, 1476.0, 1505.0)
SIX_PUMP_POWERS_W = (0.63, 0.63, 0.165, 0.131, 0.125, 0.075)


@dataclass(frozen=True)
class WidebandPlan:
    """一个全 WDM 参考数据集的频率与波形规划。

    该类型刻意不按算例名分支：``c57`` / ``cl152`` 只是两个实例。采样率
    必须是单信道采样率（``symbol_rate_hz * sps``）的整数倍，这样
    :class:`WDMMux` 的 ``resample_poly`` 比值才是整数，避免重采样伪影。

    Parameters
    ----------
    name : str
        数据集标签（同时用作数组键前缀）。
    n_channels : int
        信道数。
    channel_spacing_hz : float
        信道间隔 (Hz)。
    center_frequency_hz : float
        WDM 复用信号的绝对中心频率 (Hz)。
    symbol_rate_hz : float
        单信道符号率 (Hz)。
    roll_off : float
        RRC 滚降系数。
    sps : int
        每符号采样数（单信道基带）。
    sample_rate_hz : float
        合波后的采样率 (Hz)。
    launch_power_dbm_per_pol : float, optional
        每信道每偏振入纤功率 (dBm)。默认 0 dBm。
    num_symbols : int, optional
        每信道符号数。
    """

    name: str
    n_channels: int
    channel_spacing_hz: float
    center_frequency_hz: float
    symbol_rate_hz: float
    roll_off: float
    sps: int
    sample_rate_hz: float
    launch_power_dbm_per_pol: float = 0.0
    num_symbols: int = WIDEBAND_NUM_SYMBOLS

    def __post_init__(self) -> None:
        """校验栅格覆盖、信道不重叠，以及整数重采样比。"""
        if not self.name:
            raise ValueError("plan name must be non-empty.")
        if self.n_channels < 1:
            raise ValueError("n_channels must be positive.")
        if self.channel_spacing_hz <= 0 or self.center_frequency_hz <= 0:
            raise ValueError("channel spacing and center frequency must be positive.")
        if self.symbol_rate_hz <= 0 or self.sps < 1:
            raise ValueError("symbol rate and sps must be positive.")
        if not 0.0 <= self.roll_off < 1.0:
            raise ValueError("roll_off must be in [0, 1).")
        if self.num_symbols < 1:
            raise ValueError("num_symbols must be positive.")

        occupied = self.symbol_rate_hz * (1.0 + self.roll_off)
        if self.channel_spacing_hz < occupied:
            raise ValueError(
                f"{self.name}: channel spacing {self.channel_spacing_hz:.6g} Hz is smaller than "
                f"the occupied bandwidth {occupied:.6g} Hz; channels overlap."
            )
        span = (self.n_channels - 1) * self.channel_spacing_hz + occupied
        if self.sample_rate_hz < span:
            raise ValueError(
                f"{self.name}: sample rate {self.sample_rate_hz:.6g} Hz is smaller than the "
                f"muxed span {span:.6g} Hz; the WDM signal would alias."
            )

        channel_rate = self.symbol_rate_hz * self.sps
        ratio = self.sample_rate_hz / channel_rate
        if not np.isclose(ratio, round(ratio), rtol=1e-9, atol=0.0):
            raise ValueError(
                f"{self.name}: sample rate must be an integer multiple of the channel "
                f"sample rate ({channel_rate:.6g} Hz); got ratio {ratio:.6g}."
            )

    @property
    def offsets_hz(self) -> np.ndarray:
        """各信道相对 ``center_frequency_hz`` 的频偏 (Hz)。"""
        return (np.arange(self.n_channels) - (self.n_channels - 1) / 2.0) * self.channel_spacing_hz

    @property
    def occupied_bandwidth_hz(self) -> float:
        """单信道占用带宽 ``Rs * (1 + roll_off)`` (Hz)。"""
        return self.symbol_rate_hz * (1.0 + self.roll_off)

    @property
    def channel_sample_rate_hz(self) -> float:
        """单信道基带采样率 (Hz)。"""
        return self.symbol_rate_hz * self.sps

    @property
    def resample_ratio(self) -> int:
        """``WDMMux`` 的整数上采样比（目标采样率 / 单信道采样率）。"""
        return int(round(self.sample_rate_hz / self.channel_sample_rate_hz))

    @property
    def samples_per_channel(self) -> int:
        """单信道成形后的样本数。"""
        return self.num_symbols * self.sps

    def with_num_symbols(self, num_symbols: int) -> "WidebandPlan":
        """返回同参数、不同符号数的副本。"""
        return replace(self, num_symbols=num_symbols)

    def to_metadata(self) -> dict:
        """可 JSON 序列化的计划描述。"""
        offsets = self.offsets_hz
        return {
            "name": self.name,
            "n_channels": self.n_channels,
            "channel_spacing_hz": self.channel_spacing_hz,
            "center_frequency_hz": self.center_frequency_hz,
            "symbol_rate_hz": self.symbol_rate_hz,
            "roll_off": self.roll_off,
            "sps": self.sps,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_sample_rate_hz": self.channel_sample_rate_hz,
            "resample_ratio": self.resample_ratio,
            "launch_power_dbm_per_pol": self.launch_power_dbm_per_pol,
            "num_symbols": self.num_symbols,
            "samples_per_channel": self.samples_per_channel,
            "occupied_bandwidth_hz": self.occupied_bandwidth_hz,
            "first_channel_offset_hz": float(offsets[0]),
            "last_channel_offset_hz": float(offsets[-1]),
        }


#: 全 WDM 计划。中心频率与 network-nlo-eval 的 ``c_57ch`` / ``cl_152ch``
#: 网格中心一致；采样率是 64 GBd × 4 sps 的 18 / 45 倍。
WIDEBAND_PLANS: dict[str, WidebandPlan] = {
    "c57": WidebandPlan(
        name="c57",
        n_channels=57,
        channel_spacing_hz=75e9,
        center_frequency_hz=193.75e12,
        symbol_rate_hz=64e9,
        roll_off=0.1,
        sps=SPS,
        sample_rate_hz=4.608e12,
        launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL,
        num_symbols=WIDEBAND_NUM_SYMBOLS,
    ),
    "cl152": WidebandPlan(
        name="cl152",
        n_channels=152,
        channel_spacing_hz=75e9,
        center_frequency_hz=190.15e12,
        symbol_rate_hz=64e9,
        roll_off=0.1,
        sps=SPS,
        sample_rate_hz=11.52e12,
        launch_power_dbm_per_pol=LAUNCH_POWER_DBM_PER_POL,
        num_symbols=WIDEBAND_NUM_SYMBOLS,
    ),
}


def _single_pump(power_w: float, direction: str) -> PumpSpec:
    """1450 nm 单泵，参数与历史 3/5/7ch 数据一致。"""
    return PumpSpec(
        PUMP_WAVELENGTH_NM,
        power_w,
        direction,
        WIDEBAND_PUMP_ATTENUATION_DB_PER_KM,
    )


#: 全 WDM 的泵浦配置。``c57`` 与历史数据同口径；``cl152`` 用六泵后向。
WIDEBAND_CONFIGS: dict[str, dict[str, list[PumpSpec]]] = {
    "c57": {
        "nopump": [],
        "fwd_0p20w": [_single_pump(0.20, "forward")],
        "bwd_0p20w": [_single_pump(0.20, "backward")],
    },
    "cl152": {
        "nopump": [],
        "bwd_sixpumps": [
            PumpSpec(wavelength_nm, power_w, "backward", WIDEBAND_PUMP_ATTENUATION_DB_PER_KM)
            for wavelength_nm, power_w in zip(
                SIX_PUMP_WAVELENGTHS_NM, SIX_PUMP_POWERS_W, strict=True
            )
        ],
        # 可选趋势对照：与 c57 的前向单泵同功率。
        "fwd_0p20w": [_single_pump(0.20, "forward")],
    },
}


def _sweep_tag(power_dbm_per_pol: float) -> str:
    """把发射功率 (dBm/偏振) 映射成稳定的数据集标签，如 ``3ch_0p5mw``。"""
    milliwatt = 10 ** (power_dbm_per_pol / 10)
    return f"3ch_{milliwatt:.1f}mw".replace(".", "p")


def build_wdm_dataset(
    tag: str,
    spec: dict,
    num_symbols: int,
    seed: int,
    launch_power_dbm_per_pol: float = LAUNCH_POWER_DBM_PER_POL,
) -> tuple[Signal, np.ndarray, Constellation]:
    """生成一个 WDM 输入波形、逐通道比特和星座。"""
    rng = np.random.default_rng(seed)
    mapper = Mapper(modulation=MODULATION, power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    rrc = RRCFilter(sps=SPS, roll_off=ROLL_OFF)
    iq_mod = IQModulator(launch_power_dbm_per_pol=launch_power_dbm_per_pol)
    constellation = Constellation(MODULATION, power_norm=True)

    bits_per_channel = num_symbols * mapper.bits_per_symbol * mapper.n_pol
    bits = rng.integers(0, 2, size=(spec["count"], bits_per_channel), dtype=np.int8).astype(bool)

    channel_signals = []
    for idx in range(spec["count"]):
        symbols = mapper.forward(bits[idx])
        shaped = rrc.forward(symbols)
        optical = iq_mod.forward(shaped)
        channel_signals.append(optical)

    plan = WDMPlan(
        reference_freq_hz=193.1e12,
        channels=[
            ChannelSpec(
                channel_id=idx,
                freq_offset_hz=float(spec["offsets_hz"][idx]),
                power_dbm_per_pol=launch_power_dbm_per_pol,
            )
            for idx in range(spec["count"])
        ],
    )
    mux = WDMMux(plan, target_sample_rate_hz=spec["sample_rate_hz"])
    wdm_input = mux.combine(channel_signals)
    wdm_input.ctx.meta["dra_reference_tag"] = tag
    return wdm_input, bits, constellation


def band_nli_metrics(
    nonlinear_data: np.ndarray,
    linear_data: np.ndarray,
    sample_rate_hz: float,
    offset_hz: float,
    bandwidth_hz: float,
) -> tuple[float, float, float, float, float]:
    """计算一个信道带内的信号功率、NLI 和公共复增益。

    先对每个信道用单一复标量 ``a`` 拟合 ``NL ≈ a * LIN``（最小二乘），
    再取残差作为 NLI。这样剔除的是相干接收机本来就会补偿的公共相位/
    幅度，保留的是信道内失真。同时返回未拟合的原始差值功率以便审计。
    """
    n_samples = nonlinear_data.shape[-1]
    spectrum_nl = np.fft.fft(nonlinear_data, axis=-1)
    spectrum_lin = np.fft.fft(linear_data, axis=-1)
    freqs = np.fft.fftfreq(n_samples, d=1.0 / sample_rate_hz)
    mask = np.abs(freqs - offset_hz) <= bandwidth_hz / 2.0

    nl_band = spectrum_nl[:, mask]
    lin_band = spectrum_lin[:, mask]
    signal_power = float(np.sum(np.abs(nl_band) ** 2) / (n_samples**2))
    raw_nli_power = float(np.sum(np.abs(nl_band - lin_band) ** 2) / (n_samples**2))
    denominator = np.sum(np.abs(lin_band) ** 2)
    if denominator == 0:
        return signal_power, raw_nli_power, raw_nli_power, 0.0, 0.0
    common_gain = np.sum(np.conj(lin_band) * nl_band) / denominator
    residual = nl_band - common_gain * lin_band
    nli_power = float(np.sum(np.abs(residual) ** 2) / (n_samples**2))
    return (
        signal_power,
        nli_power,
        raw_nli_power,
        float(20 * np.log10(np.abs(common_gain))) if np.abs(common_gain) > 0 else 0.0,
        float(np.angle(common_gain, deg=True)),
    )


def center_channel_evm(
    nonlinear_output: Signal,
    linear_output: Signal,
    center_idx: int,
    tx_bits: np.ndarray,
    constellation: Constellation,
) -> tuple[float, float, float, float]:
    """对中心信道执行 demux → 公共相位对齐 → CDC → 匹配滤波 → EVM。"""
    # 理想矩形滤波避免短 FIR 在 27.5 GHz 截止处的边带失真。
    stacked_nl = WDMDemux(filter_type="ideal_rectangular").forward(nonlinear_output)
    stacked_lin = WDMDemux(filter_type="ideal_rectangular").forward(linear_output)
    channel_nl = ChannelExtractor(center_idx).forward(stacked_nl)
    channel_lin = ChannelExtractor(center_idx).forward(stacked_lin)

    # 单复标量对齐，剔除接收机本来会补偿的公共相位/幅度。
    denominator = np.vdot(channel_lin.data.ravel(), channel_lin.data.ravel())
    common_gain = np.vdot(channel_lin.data.ravel(), channel_nl.data.ravel()) / denominator
    if common_gain == 0:
        common_gain = 1.0
    aligned_data = channel_nl.data / common_gain
    channel_nl = Signal(data=aligned_data, ctx=channel_nl.ctx)

    channel_nl = ChromaticDispersionComp(
        fiber_length_km=FIBER_LENGTH_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
    ).forward(channel_nl)
    channel_nl = RRCFilter(sps=SPS, roll_off=ROLL_OFF, is_matched_filter=True).forward(channel_nl)
    evm_percent = BERCalculator(ignore_headers=30).calculate_evm(channel_nl, constellation, tx_bits)
    snr_db = float(20 * np.log10(100.0 / evm_percent)) if evm_percent > 0 else float("inf")
    return (
        float(evm_percent),
        snr_db,
        float(20 * np.log10(np.abs(common_gain))),
        float(np.angle(common_gain, deg=True)),
    )


def make_fiber(
    pumps: list[PumpSpec],
    *,
    nonlinear: bool,
    pump_model: str = "undepleted",
    coarse_step_m: float = COARSE_STEP_M,
) -> DistributedRamanFiber:
    """构造 DRA 光纤；``nonlinear=False`` 时关闭 Kerr 作为线性参考。"""
    return DistributedRamanFiber(
        length_m=FIBER_LENGTH_KM * 1e3,
        alpha_db_per_km=ALPHA_DB_PER_KM,
        beta2_ps2_per_km=BETA2_PS2_PER_KM,
        beta3_ps3_per_km=BETA3_PS3_PER_KM,
        pumps=pumps,
        coarse_step_size_m=coarse_step_m,
        n2_si=N2_SI if nonlinear else 0.0,
        a_eff_um2=A_EFF_UM2,
        pump_model=pump_model,
        add_ase=False,
    )


def walkoff_per_step_symbols(max_offset_hz: float, coarse_step_m: float) -> float:
    """返回每一步 SSFM 的群速度走离，单位为 symbol。

    ``Δτ = 2π · Δf · |β2| · dz``，再除以符号周期 ``1/Rs``。``Δf`` 取最外
    信道相对中心信道的频偏。
    """
    beta2_s2_per_m = abs(BETA2_PS2_PER_KM) * 1e-24 / 1e3
    return float(2 * np.pi * max_offset_hz * beta2_s2_per_m * coarse_step_m * SYMBOL_RATE_HZ)


def recommended_step_m(max_offset_hz: float) -> float | None:
    """按每步走离 0.15 symbol 折算的建议步长 (m)；单信道无走离时返回 None。"""
    walkoff = walkoff_per_step_symbols(max_offset_hz, 1.0)
    return float(WALK_OFF_TARGET_SYMBOLS / walkoff) if walkoff > 0 else None


def _step_criterion_metadata(coarse_step_m: float) -> dict:
    """走离判据及其对各数据集的建议步长。"""
    per_dataset = {}
    for tag, spec in DATASETS.items():
        max_offset = float(np.max(np.abs(spec["offsets_hz"])))
        per_dataset[tag] = {
            "max_offset_from_center_hz": max_offset,
            "walkoff_per_step_symbols": walkoff_per_step_symbols(max_offset, coarse_step_m),
            "recommended_step_m_for_target": recommended_step_m(max_offset),
        }
    return {
        "walkoff_formula": "2*pi*delta_f*abs(beta2)*dz*symbol_rate",
        "target_walkoff_symbols": WALK_OFF_TARGET_SYMBOLS,
        "coarse_step_m": float(coarse_step_m),
        "datasets": per_dataset,
    }


def center_channel_snr_db(
    wdm_input: Signal,
    spec: dict,
    coarse_step_m: float,
    pump_model: str = "undepleted",
) -> float:
    """无泵配置下中心信道的 SNR_NLI，用于步长收敛审计。"""
    nonlinear_output = make_fiber(
        [], nonlinear=True, pump_model=pump_model, coarse_step_m=coarse_step_m
    ).forward(wdm_input)
    linear_output = make_fiber(
        [], nonlinear=False, pump_model=pump_model, coarse_step_m=coarse_step_m
    ).forward(wdm_input)
    center_idx = spec["count"] // 2
    power, nli_power, *_ = band_nli_metrics(
        nonlinear_output.data,
        linear_output.data,
        spec["sample_rate_hz"],
        float(spec["offsets_hz"][center_idx]),
        NLI_ANALYSIS_BANDWIDTH_HZ,
    )
    return float(10 * np.log10(power / nli_power)) if nli_power > 0 else float("inf")


def step_convergence_metadata(
    wdm_input: Signal,
    spec: dict,
    coarse_step_m: float,
    pump_model: str = "undepleted",
) -> dict:
    """计算数据集的步长收敛元数据。

    以无泵中心信道 SNR 为准：``last_increment_db`` 是粗步长与半步长的差，
    ``step_artifact_500m_db`` 是历史 500 m 相对当前步长的差（负值表示
    500 m 高估了 NLI）。
    """
    snr_coarse = center_channel_snr_db(wdm_input, spec, coarse_step_m, pump_model)
    snr_half = center_channel_snr_db(wdm_input, spec, coarse_step_m / 2.0, pump_model)
    snr_500 = center_channel_snr_db(wdm_input, spec, HISTORICAL_STEP_M, pump_model)
    max_offset = float(np.max(np.abs(spec["offsets_hz"])))
    walkoff = walkoff_per_step_symbols(max_offset, coarse_step_m)
    return {
        "coarse_step_m": float(coarse_step_m),
        "half_step_m": float(coarse_step_m / 2.0),
        "center_snr_db": snr_coarse,
        "last_increment_db": snr_coarse - snr_half,
        "tolerance_db": STEP_CONVERGENCE_TOL_DB,
        "step_converged": bool(abs(snr_coarse - snr_half) < STEP_CONVERGENCE_TOL_DB),
        "step_artifact_500m_db": snr_500 - snr_coarse,
        "max_offset_from_center_hz": max_offset,
        "walkoff_per_step_symbols": walkoff,
        "walkoff_target_symbols": WALK_OFF_TARGET_SYMBOLS,
        "recommended_step_m_for_target": recommended_step_m(max_offset),
        "metric": "nopump center-channel snr_nli_db",
    }


def build_wideband_dataset(
    plan: WidebandPlan,
    seed: int,
) -> tuple[Signal, np.ndarray, Constellation]:
    """生成一个全 WDM 输入波形、逐信道比特和星座。

    与历史 :func:`build_wdm_dataset` 的唯一区别是全部参数来自
    :class:`WidebandPlan`，不使用模块级的 50 GBd / 100 GHz 常量。
    """
    rng = np.random.default_rng(seed)
    mapper = Mapper(modulation=MODULATION, power_norm=True, sym_rate=plan.symbol_rate_hz)
    rrc = RRCFilter(sps=plan.sps, roll_off=plan.roll_off)
    iq_mod = IQModulator(launch_power_dbm_per_pol=plan.launch_power_dbm_per_pol)
    constellation = Constellation(MODULATION, power_norm=True)

    bits_per_channel = plan.num_symbols * mapper.bits_per_symbol * mapper.n_pol
    bits = rng.integers(
        0, 2, size=(plan.n_channels, bits_per_channel), dtype=np.int8
    ).astype(bool)

    channel_signals = []
    for idx in range(plan.n_channels):
        symbols = mapper.forward(bits[idx])
        shaped = rrc.forward(symbols)
        channel_signals.append(iq_mod.forward(shaped))

    wdm_plan = WDMPlan(
        reference_freq_hz=plan.center_frequency_hz,
        channels=[
            ChannelSpec(
                channel_id=idx,
                freq_offset_hz=float(plan.offsets_hz[idx]),
                power_dbm_per_pol=plan.launch_power_dbm_per_pol,
            )
            for idx in range(plan.n_channels)
        ],
    )
    mux = WDMMux(wdm_plan, target_sample_rate_hz=plan.sample_rate_hz)
    wdm_input = mux.combine(channel_signals)
    wdm_input.ctx.meta["dra_reference_tag"] = plan.name
    return wdm_input, bits, constellation


def make_wideband_fiber(
    pumps: list[PumpSpec],
    *,
    nonlinear: bool,
    pump_model: str = "depleted",
    coarse_step_m: float,
) -> DistributedRamanFiber:
    """构造全 WDM 参考用的 DRA 光纤（平坦 alpha，1550 nm beta2）。"""
    return DistributedRamanFiber(
        length_m=WIDEBAND_FIBER_LENGTH_KM * 1e3,
        alpha_db_per_km=WIDEBAND_ALPHA_DB_PER_KM,
        beta2_ps2_per_km=WIDEBAND_BETA2_PS2_PER_KM,
        beta3_ps3_per_km=WIDEBAND_BETA3_PS3_PER_KM,
        pumps=pumps,
        coarse_step_size_m=coarse_step_m,
        n2_si=WIDEBAND_N2_SI if nonlinear else 0.0,
        a_eff_um2=WIDEBAND_A_EFF_UM2,
        pump_model=pump_model,
        add_ase=False,
    )


def wideband_walkoff_per_step_symbols(plan: WidebandPlan, step_m: float) -> float:
    """每一步 SSFM 的群速度走离（symbol）。

    ``Δτ = 2π · Δf_max · |β2| · dz``，再除以符号周期 ``1/Rs``；``Δf_max``
    取最外信道相对中心的频偏。
    """
    max_offset_hz = float(np.max(np.abs(plan.offsets_hz)))
    beta2_s2_per_m = abs(WIDEBAND_BETA2_PS2_PER_KM) * 1e-24 / 1e3
    return float(2 * np.pi * max_offset_hz * beta2_s2_per_m * step_m * plan.symbol_rate_hz)


def wideband_recommended_step_m(plan: WidebandPlan) -> float | None:
    """按每步走离 0.15 symbol 折算的建议步长 (m)。"""
    walkoff = wideband_walkoff_per_step_symbols(plan, 1.0)
    return float(WALK_OFF_TARGET_SYMBOLS / walkoff) if walkoff > 0 else None


def wideband_center_channel_metrics(
    plan: WidebandPlan,
    wdm_input: Signal,
    step_m: float,
    pumps: list[PumpSpec],
    pump_model: str = "depleted",
) -> dict:
    """在给定步长下传播，返回中心信道的功率/NLI/SNR。"""
    nonlinear = make_wideband_fiber(
        pumps, nonlinear=True, pump_model=pump_model, coarse_step_m=step_m
    ).forward(wdm_input)
    linear = make_wideband_fiber(
        pumps, nonlinear=False, pump_model=pump_model, coarse_step_m=step_m
    ).forward(wdm_input)
    center_idx = plan.n_channels // 2
    power, nli_power, raw_nli_power, common_gain_db, common_phase_deg = band_nli_metrics(
        nonlinear.data,
        linear.data,
        plan.sample_rate_hz,
        float(plan.offsets_hz[center_idx]),
        plan.symbol_rate_hz,
    )
    snr_nli_db = float(10 * np.log10(power / nli_power)) if nli_power > 0 else None
    return {
        "center_channel_index": center_idx,
        "center_freq_offset_hz": float(plan.offsets_hz[center_idx]),
        "signal_power_w": power,
        "nli_power_w": nli_power,
        "nli_power_raw_w": raw_nli_power,
        "snr_nli_db": snr_nli_db,
        "common_gain_db": common_gain_db,
        "common_phase_deg": common_phase_deg,
    }


def wideband_step_convergence(
    plan: WidebandPlan,
    steps_m: tuple[float, ...] = WIDEBAND_PROBE_STEPS_M,
    *,
    num_symbols: int = WIDEBAND_PROBE_NUM_SYMBOLS,
    seed: int = WIDEBAND_SEED,
    pump_model: str = "depleted",
    tolerance_db: float = STEP_CONVERGENCE_TOL_DB,
    extend: bool = True,
) -> dict:
    """用短记录测无泵中心信道的步长收敛性。

    从 ``steps_m`` 开始，逐档减半重复最后一步，直到相邻两档的 SNR 增量
    ``< tolerance_db`` 或触及 ``WIDEBAND_MIN_PROBE_STEP_M``。返回值记录
    每档的 SNR、走离和耗时；未收敛时 ``converged_step_m`` 为 ``None``。
    """
    steps = [float(step) for step in steps_m]
    if not steps:
        raise ValueError("steps_m must not be empty.")
    if any(step <= 0 for step in steps):
        raise ValueError("step sizes must be positive.")

    probe_plan = plan.with_num_symbols(num_symbols)
    probe_input, _, _ = build_wideband_dataset(probe_plan, seed)
    pumps = WIDEBAND_CONFIGS.get(plan.name, {}).get("nopump", [])

    def evaluate(step_m: float) -> dict:
        started = time.perf_counter()
        metrics = wideband_center_channel_metrics(
            probe_plan, probe_input, step_m, pumps, pump_model
        )
        elapsed = time.perf_counter() - started
        return {
            "step_m": float(step_m),
            "n_steps": int(np.ceil(WIDEBAND_FIBER_LENGTH_KM * 1e3 / step_m)),
            "walkoff_per_step_symbols": wideband_walkoff_per_step_symbols(probe_plan, step_m),
            "elapsed_s": float(elapsed),
            **metrics,
        }

    rows = [evaluate(step) for step in steps]
    max_extensions = 2
    while extend and len(rows) >= 2:
        increment = rows[-1]["snr_nli_db"] - rows[-2]["snr_nli_db"]
        if abs(increment) < tolerance_db:
            break
        next_step = rows[-1]["step_m"] / 2.0
        if next_step < WIDEBAND_MIN_PROBE_STEP_M or len(rows) >= len(steps) + max_extensions:
            break
        rows.append(evaluate(next_step))

    last_increment = (
        float(rows[-1]["snr_nli_db"] - rows[-2]["snr_nli_db"]) if len(rows) >= 2 else float("nan")
    )
    step_converged = bool(abs(last_increment) < tolerance_db)
    snr_500 = next((row["snr_nli_db"] for row in rows if row["step_m"] == HISTORICAL_STEP_M), None)
    if snr_500 is None:
        snr_500 = evaluate(HISTORICAL_STEP_M)["snr_nli_db"]
    max_offset = float(np.max(np.abs(plan.offsets_hz)))

    return {
        "plan": plan.to_metadata(),
        "probe_num_symbols": int(num_symbols),
        "probe_seed": int(seed),
        "pump_model": pump_model,
        "config": "nopump",
        "num_samples": int(probe_input.data.shape[-1]),
        "steps": rows,
        "last_increment_db": last_increment,
        "tolerance_db": float(tolerance_db),
        "step_converged": step_converged,
        "converged_step_m": float(rows[-1]["step_m"]) if step_converged else None,
        "step_artifact_500m_db": float(snr_500 - rows[-1]["snr_nli_db"]),
        "max_offset_from_center_hz": max_offset,
        "walkoff_per_step_symbols": wideband_walkoff_per_step_symbols(
            plan, rows[-1]["step_m"]
        ),
        "walkoff_target_symbols": WALK_OFF_TARGET_SYMBOLS,
        "walkoff_formula": "2*pi*delta_f_max*abs(beta2)*dz*symbol_rate",
        "recommended_step_m_for_target": wideband_recommended_step_m(plan),
        "metric": "nopump center-channel snr_nli_db",
        "elapsed_s": float(sum(row["elapsed_s"] for row in rows)),
    }


def _wideband_pump_metadata(plan: WidebandPlan) -> dict:
    """把该计划的泵浦配置整理成可序列化字典。"""
    return {
        "attenuation_db_per_km": WIDEBAND_PUMP_ATTENUATION_DB_PER_KM,
        "configs": {
            config_name: [
                {
                    "wavelength_nm": pump.wavelength_nm,
                    "power_w": pump.power_w,
                    "direction": pump.direction,
                    "attenuation_db_per_km": pump.attenuation_db_per_km,
                }
                for pump in pumps
            ]
            for config_name, pumps in WIDEBAND_CONFIGS[plan.name].items()
        },
    }


def _wideband_notes(plan: WidebandPlan, seed: int) -> list[str]:
    """全 WDM 数据集的限制与口径说明。"""
    return [
        "NLI analysis band is the symbol rate (64 GHz), not the 70.4 GHz occupied bandwidth.",
        "Fibre alpha is flat at the 1550 nm polynomial value (0.162 dB/km); the SMF_100KM "
        "polynomial gives ~0.162-0.178 dB/km across C+L, so L-band edge loss is underestimated.",
        "beta2 is flat -21.7 ps^2/km at 1550 nm; beta3=1e-4 ps^3/km is negligible over 64 GBd.",
        "Depleted pump model uses the in-band total power at the plan centre frequency; it does "
        "not model pump-pump Raman transfer or pump ASE.",
        "Channel-to-channel ISRS is not modelled (raman_r3=None).",
        f"Each waveform is one {seed} realisation; statistical uncertainty is documented in "
        "docs/dra-waveform.md and needs a second seed for a formal estimate.",
        "step_artifact_500m_db = SNR(500 m) - SNR(generation step); negative means 500 m "
        "overestimates NLI.",
    ]


def generate_wideband_plan(
    plan: WidebandPlan,
    output_dir: Path,
    *,
    seed: int,
    step_m: float,
    probe: dict,
    pump_model: str = "depleted",
    file_stem: str | None = None,
    check_undepleted: bool = True,
) -> dict:
    """生成并保存一个全 WDM 计划的数据集，返回其元数据。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    wdm_input, bits, _ = build_wideband_dataset(plan, seed)
    tag = plan.name
    arrays: dict[str, np.ndarray] = {
        f"input_{tag}": wdm_input.data.astype(np.complex64),
        f"bits_{tag}": bits,
        f"offsets_{tag}": plan.offsets_hz,
    }

    configs = WIDEBAND_CONFIGS[plan.name]
    raw_metrics: dict[str, list[dict]] = {}
    nonlinear_outputs: dict[str, np.ndarray] = {}
    output_powers: dict[str, float] = {}
    for config_name, pumps in configs.items():
        nonlinear = make_wideband_fiber(
            pumps, nonlinear=True, pump_model=pump_model, coarse_step_m=step_m
        ).forward(wdm_input)
        linear = make_wideband_fiber(
            pumps, nonlinear=False, pump_model=pump_model, coarse_step_m=step_m
        ).forward(wdm_input)
        arrays[f"nl_{tag}_{config_name}"] = nonlinear.data.astype(np.complex64)
        arrays[f"lin_{tag}_{config_name}"] = linear.data.astype(np.complex64)
        nonlinear_outputs[config_name] = nonlinear.data
        output_powers[config_name] = float(np.mean(np.abs(nonlinear.data) ** 2))
        raw_metrics[config_name] = [
            {
                "channel_id": idx,
                "freq_offset_hz": float(offset),
                "signal_power_w": power,
                "nli_power_w": nli_power,
                "nli_power_raw_w": raw_nli_power,
                "snr_nli_db": (
                    float(10 * np.log10(power / nli_power)) if nli_power > 0 else None
                ),
                "snr_nli_raw_db": (
                    float(10 * np.log10(power / raw_nli_power)) if raw_nli_power > 0 else None
                ),
                "common_gain_db": gain_db,
                "common_phase_deg": phase_deg,
            }
            for idx, offset in enumerate(plan.offsets_hz)
            for power, nli_power, raw_nli_power, gain_db, phase_deg in [
                band_nli_metrics(
                    nonlinear.data,
                    linear.data,
                    plan.sample_rate_hz,
                    float(offset),
                    plan.symbol_rate_hz,
                )
            ]
        ]

    nopump_powers = [row["signal_power_w"] for row in raw_metrics["nopump"]]
    config_documents: dict[str, dict] = {}
    for config_name, rows in raw_metrics.items():
        per_channel = []
        for row in rows:
            reference_power = nopump_powers[row["channel_id"]]
            per_channel.append(
                {
                    **row,
                    "on_off_gain_db": (
                        float(10 * np.log10(row["signal_power_w"] / reference_power))
                        if reference_power > 0
                        else None
                    ),
                }
            )
        config_documents[config_name] = {
            "waveform_nonlinear_key": f"nl_{tag}_{config_name}",
            "waveform_linear_key": f"lin_{tag}_{config_name}",
            "pumps": [
                {
                    "wavelength_nm": pump.wavelength_nm,
                    "power_w": pump.power_w,
                    "direction": pump.direction,
                    "attenuation_db_per_km": pump.attenuation_db_per_km,
                }
                for pump in configs[config_name]
            ],
            "output_power_per_pol_w": output_powers[config_name],
            "dra_ase_model": "none",
            "dra_ase_power_per_pol_w": 0.0,
            "per_channel": per_channel,
        }

    nopump_undepleted_bit_exact: bool | None = None
    if check_undepleted and pump_model == "depleted" and "nopump" in nonlinear_outputs:
        undepleted = make_wideband_fiber(
            [], nonlinear=True, pump_model="undepleted", coarse_step_m=step_m
        ).forward(wdm_input)
        nopump_undepleted_bit_exact = bool(
            np.array_equal(undepleted.data, nonlinear_outputs["nopump"])
        )

    stem = file_stem if file_stem is not None else tag
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **arrays)
    document = {
        "schema_version": 1,
        "dataset_family": "dra_wideband",
        "model": {
            "name": "DistributedRamanFiber",
            "propagator": "in-house symmetric NumPy SSFM (never PyNLO path)",
            "pump_model": pump_model,
            "pump_model_detail": (
                "Depleted pumps are integrated with RK4 along z at 25 m steps; the signal is "
                "represented by its in-band total power at the channel-plan centre frequency. "
                "No pump-pump Raman transfer and no pump ASE."
            ),
            "raman_spectrum": "Gaussian, peak 13.2 THz, sigma 2.12 THz",
            "signal_effects": "loss + CD + Kerr retained; channel ISRS not modelled",
            "ase": "not included in this reference dataset (dra_ase_model='none')",
        },
        "plan": plan.to_metadata(),
        "fiber": {
            "length_km": WIDEBAND_FIBER_LENGTH_KM,
            "alpha_db_per_km": WIDEBAND_ALPHA_DB_PER_KM,
            "alpha_model": "flat; 1550 nm value of the SMF_100KM polynomial",
            "beta2_ps2_per_km": WIDEBAND_BETA2_PS2_PER_KM,
            "beta3_ps3_per_km": WIDEBAND_BETA3_PS3_PER_KM,
            "n2_si": WIDEBAND_N2_SI,
            "a_eff_um2": WIDEBAND_A_EFF_UM2,
            "coarse_step_m": float(step_m),
        },
        "signal": {
            "modulation": MODULATION,
            "symbol_rate_hz": plan.symbol_rate_hz,
            "sps": plan.sps,
            "roll_off": plan.roll_off,
            "launch_power_dbm_per_pol": plan.launch_power_dbm_per_pol,
            "num_symbols": plan.num_symbols,
            "seed": int(seed),
        },
        "pump": _wideband_pump_metadata(plan),
        "nli_analysis_bandwidth_hz": plan.symbol_rate_hz,
        "nli_method": (
            "Per channel, fit one complex scalar a to NL ≈ a*LIN in the 64 GHz symbol-rate band, "
            "then take the band power of the residual NL-a*LIN as NLI. This removes the common "
            "phase/gain that a coherent receiver would equalize. nli_power_raw_w (no fit) is "
            "also recorded. out_linear uses the same pumps/loss/CD but n2=0."
        ),
        "coarse_step_m": float(step_m),
        "step_converged": bool(probe["step_converged"]),
        "step_artifact_500m_db": float(probe["step_artifact_500m_db"]),
        "step_convergence": probe,
        "nopump_undepleted_bit_exact": nopump_undepleted_bit_exact,
        "num_samples": int(wdm_input.data.shape[-1]),
        "samples_per_channel": plan.samples_per_channel,
        "notes": _wideband_notes(plan, seed),
        "configs": config_documents,
    }
    document["generation"] = {
        "elapsed_s": float(time.perf_counter() - started),
        "npz_size_bytes": int(npz_path.stat().st_size),
        "json_path": str(output_dir / f"{stem}.json"),
        "npz_path": str(npz_path),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return document


def generate_wideband(
    output_dir: Path,
    *,
    plan_names: tuple[str, ...] = tuple(WIDEBAND_PLANS),
    num_symbols: int = WIDEBAND_NUM_SYMBOLS,
    num_symbols_by_plan: dict[str, int] | None = None,
    seeds: tuple[int, ...] = (WIDEBAND_SEED,),
    step_m: float | None = None,
    step_overrides: dict[str, float] | None = None,
    pump_model: str = "depleted",
    probe_only: bool = False,
    probe_num_symbols: int = WIDEBAND_PROBE_NUM_SYMBOLS,
    probe_steps_m: tuple[float, ...] = WIDEBAND_PROBE_STEPS_M,
    reuse_probe: bool = False,
) -> dict:
    """生成 c57/cl152 全 WDM 参考数据（或只跑步长探针）。

    ``step_m`` / ``step_overrides`` / ``probe`` 的优先级：逐计划覆盖 >
    全局步长 > 探针收敛步长 > 探针最后一档。``reuse_probe=True`` 时读取已
    有的 ``step_convergence.json``，避免重复跑昂贵的探针。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = output_dir / "step_convergence.json"
    cached_probes: dict[str, dict] = {}
    if reuse_probe and probe_path.exists():
        cached_probes = json.loads(probe_path.read_text(encoding="utf-8")).get("plans", {})
    probes: dict[str, dict] = dict(cached_probes) if reuse_probe else {}
    documents: dict[str, dict] = {}
    for name in plan_names:
        if name not in WIDEBAND_PLANS:
            raise KeyError(f"unknown wideband plan {name!r}; available: {sorted(WIDEBAND_PLANS)}")
        base_plan = WIDEBAND_PLANS[name]
        probe = cached_probes.get(name)
        if probe is None:
            probe = wideband_step_convergence(
                base_plan,
                probe_steps_m,
                num_symbols=probe_num_symbols,
                seed=seeds[0],
                pump_model=pump_model,
            )
        probes[name] = probe
        if probe_only:
            continue
        override = None if step_overrides is None else step_overrides.get(name)
        if override is not None:
            chosen_step = float(override)
        elif step_m is not None:
            chosen_step = float(step_m)
        else:
            chosen_step = float(probe["converged_step_m"] or probe["steps"][-1]["step_m"])
        symbols = (
            num_symbols_by_plan.get(name, num_symbols) if num_symbols_by_plan else num_symbols
        )
        plan = base_plan.with_num_symbols(symbols)
        for seed in seeds:
            stem = plan.name if seed == WIDEBAND_SEED else f"{plan.name}_seed{seed}"
            documents[stem] = generate_wideband_plan(
                plan,
                output_dir,
                seed=seed,
                step_m=chosen_step,
                probe=probe,
                pump_model=pump_model,
                file_stem=stem,
                # 152ch 的 nopump 额外传播会把单配置推到 ~17 min；该等价性
                # 已由短记录单测覆盖，主数据生成时只对 <=57ch 做逐位复核。
                check_undepleted=plan.n_channels <= 57,
            )

    # 复用探针时只读不写，避免两个计划并发生成时互相覆盖。
    if not reuse_probe or probe_only:
        probe_path.write_text(
            json.dumps(
                {
                    "generated_by": "examples/generate_dra_reference.py --wideband",
                    "probe_num_symbols": int(probe_num_symbols),
                    "probe_seed": int(seeds[0]),
                    "pump_model": pump_model,
                    "tolerance_db": STEP_CONVERGENCE_TOL_DB,
                    "plans": probes,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {probe_path}")
    return {"probes": probes, "documents": documents}


def _populate_dataset(
    arrays: dict[str, np.ndarray],
    metadata: dict,
    tag: str,
    spec: dict,
    num_symbols: int,
    seed: int,
    configs: dict[str, list[PumpSpec]],
    pump_model: str,
    launch_power_dbm_per_pol: float,
    purpose: str | None = None,
    coarse_step_m: float = COARSE_STEP_M,
    record_step_metadata: bool = False,
) -> None:
    """生成并记录一个数据集（输入波形 + 各配置的非线性/线性输出）。

    ``purpose`` 为 None 时保持默认数据集的 JSON 结构不变；耗尽数据集会额外
    写入 ``pump_model`` / ``purpose`` / ``launch_power_dbm_per_pol``。
    ``record_step_metadata=True`` 时写入步长收敛字段（用于收敛数据集）。
    """
    wdm_input, bits, constellation = build_wdm_dataset(
        tag, spec, num_symbols, seed, launch_power_dbm_per_pol
    )
    arrays[f"input_{tag}"] = wdm_input.data.astype(np.complex64)
    arrays[f"bits_{tag}"] = bits
    arrays[f"offsets_{tag}"] = spec["offsets_hz"]
    center_idx = spec["count"] // 2
    channel_bandwidth_hz = SYMBOL_RATE_HZ * (1 + ROLL_OFF)

    dataset_meta: dict = {
        "input_key": f"input_{tag}",
        "bits_key": f"bits_{tag}",
        "offsets_key": f"offsets_{tag}",
        "n_channels": spec["count"],
        "center_channel_index": center_idx,
        "sample_rate_hz": float(spec["sample_rate_hz"]),
        "num_samples": int(wdm_input.data.shape[-1]),
        "channel_bandwidth_hz": channel_bandwidth_hz,
        "nli_analysis_bandwidth_hz": NLI_ANALYSIS_BANDWIDTH_HZ,
        "nli_definition": (
            "In-band additive NLI after one complex common-gain projection NL-a*LIN; "
            "50 GHz ideal rectangular analysis band; SNR_NLI=signal/NLI; no ASE."
        ),
        "configs": {},
    }
    if purpose is not None:
        dataset_meta["pump_model"] = pump_model
        dataset_meta["purpose"] = purpose
        dataset_meta["launch_power_dbm_per_pol"] = launch_power_dbm_per_pol
    if record_step_metadata:
        dataset_meta["pump_model"] = pump_model
        dataset_meta["coarse_step_m"] = float(coarse_step_m)
        dataset_meta["step_convergence"] = step_convergence_metadata(
            wdm_input, spec, coarse_step_m, pump_model
        )
        dataset_meta["step_converged"] = dataset_meta["step_convergence"]["step_converged"]
        dataset_meta["step_artifact_500m_db"] = dataset_meta["step_convergence"][
            "step_artifact_500m_db"
        ]

    no_pump_channel_powers: list[float] | None = None
    for config_name, pumps in configs.items():
        nonlinear_output = make_fiber(
            pumps, nonlinear=True, pump_model=pump_model, coarse_step_m=coarse_step_m
        ).forward(wdm_input)
        linear_output = make_fiber(
            pumps, nonlinear=False, pump_model=pump_model, coarse_step_m=coarse_step_m
        ).forward(wdm_input)

        arrays[f"nl_{tag}_{config_name}"] = nonlinear_output.data.astype(np.complex64)
        arrays[f"lin_{tag}_{config_name}"] = linear_output.data.astype(np.complex64)

        channel_powers = []
        nli_powers = []
        raw_nli_powers = []
        common_gains_db = []
        common_phases_deg = []
        for offset in spec["offsets_hz"]:
            power, nli_power, raw_nli_power, common_gain_db, common_phase_deg = band_nli_metrics(
                nonlinear_output.data,
                linear_output.data,
                spec["sample_rate_hz"],
                float(offset),
                NLI_ANALYSIS_BANDWIDTH_HZ,
            )
            channel_powers.append(power)
            nli_powers.append(nli_power)
            raw_nli_powers.append(raw_nli_power)
            common_gains_db.append(common_gain_db)
            common_phases_deg.append(common_phase_deg)
        if no_pump_channel_powers is None:
            no_pump_channel_powers = list(channel_powers)

        per_channel = []
        for idx, (offset, power, nli_power) in enumerate(
            zip(spec["offsets_hz"], channel_powers, nli_powers, strict=True)
        ):
            snr_nli_db = float(10 * np.log10(power / nli_power)) if nli_power > 0 else None
            snr_nli_raw_db = (
                float(10 * np.log10(power / raw_nli_powers[idx])) if raw_nli_powers[idx] > 0 else None
            )
            no_pump_power = no_pump_channel_powers[idx]
            on_off_db = float(10 * np.log10(power / no_pump_power)) if no_pump_power > 0 else None
            per_channel.append(
                {
                    "channel_id": idx,
                    "freq_offset_hz": float(offset),
                    "signal_power_w": power,
                    "nli_power_w": nli_power,
                    "nli_power_raw_w": raw_nli_powers[idx],
                    "snr_nli_db": snr_nli_db,
                    "snr_nli_raw_db": snr_nli_raw_db,
                    "common_gain_db": common_gains_db[idx],
                    "common_phase_deg": common_phases_deg[idx],
                    "on_off_gain_db": on_off_db,
                }
            )

        evm_percent, snr_evm_db, evm_common_gain_db, evm_common_phase_deg = center_channel_evm(
            nonlinear_output, linear_output, center_idx, bits[center_idx], constellation
        )
        output_power_per_pol_w = float(np.mean(np.abs(nonlinear_output.data) ** 2))
        dataset_meta["configs"][config_name] = {
            "waveform_nonlinear_key": f"nl_{tag}_{config_name}",
            "waveform_linear_key": f"lin_{tag}_{config_name}",
            "pumps": [
                {
                    "wavelength_nm": pump.wavelength_nm,
                    "power_w": pump.power_w,
                    "direction": pump.direction,
                    "attenuation_db_per_km": pump.attenuation_db_per_km,
                }
                for pump in pumps
            ],
            "output_power_per_pol_w": output_power_per_pol_w,
            "dra_ase_model": "none",
            "dra_ase_power_per_pol_w": 0.0,
            "center_channel_evm_percent": evm_percent,
            "center_channel_snr_evm_db": snr_evm_db,
            "center_channel_common_gain_db": evm_common_gain_db,
            "center_channel_common_phase_deg": evm_common_phase_deg,
            "per_channel": per_channel,
        }

    metadata["datasets"][tag] = dataset_meta


def generate(
    output_dir: Path, num_symbols: int, seed: int, coarse_step_m: float = COARSE_STEP_M
) -> None:
    """生成并保存默认（未耗尽）参考数据集。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    record_step_metadata = coarse_step_m != COARSE_STEP_M
    arrays: dict[str, np.ndarray] = {}
    metadata: dict = {
        "schema_version": 1,
        "model": {
            "name": "DistributedRamanFiber",
            "propagator": "in-house symmetric NumPy SSFM (never PyNLO path)",
            "pump_model": "undepleted analytical pump attenuation",
            "raman_spectrum": "Gaussian, peak 13.2 THz, sigma 2.12 THz",
            "signal_effects": "loss + CD + Kerr retained; channel ISRS not modelled",
            "ase": "not included in this reference dataset (dra_ase_model='none')",
        },
        "fiber": {
            "length_km": FIBER_LENGTH_KM,
            "alpha_db_per_km": ALPHA_DB_PER_KM,
            "beta2_ps2_per_km": BETA2_PS2_PER_KM,
            "beta3_ps3_per_km": BETA3_PS3_PER_KM,
            "n2_si": N2_SI,
            "a_eff_um2": A_EFF_UM2,
            "coarse_step_m": float(coarse_step_m),
        },
        "signal": {
            "modulation": MODULATION,
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "sps": SPS,
            "roll_off": ROLL_OFF,
            "launch_power_dbm_per_pol": LAUNCH_POWER_DBM_PER_POL,
            "num_symbols": num_symbols,
            "seed": seed,
        },
        "pump": {
            "wavelength_nm": PUMP_WAVELENGTH_NM,
            "attenuation_db_per_km": PUMP_ATTENUATION_DB_PER_KM,
            "directions": ["forward", "backward"],
            "powers_w": [0.10, 0.20],
        },
        "nli_method": (
            "Per channel, fit one complex scalar a to NL ≈ a*LIN in the channel band, then take "
            "the band power of the residual NL-a*LIN as NLI. This removes the common phase/gain "
            "that a coherent receiver would equalize. nli_power_raw_w (no fit) is also recorded. "
            "out_linear uses the same pumps/loss/CD but n2=0."
        ),
        "notes": [
            "DRA ASE is optional in the model but disabled for this dataset; SNR values are NLI-limited.",
            "Pumps are undepleted and uncoupled; Gaussian Raman spectrum with 13.2 THz peak, 2.12 THz sigma.",
            "Channel-to-channel ISRS is not modelled (raman_r3=None) in this dataset.",
        ],
        "datasets": {},
    }
    if record_step_metadata:
        metadata["coarse_step_m"] = float(coarse_step_m)
        metadata["pump_model"] = "undepleted"
        metadata["step_criterion"] = _step_criterion_metadata(coarse_step_m)
        metadata["notes"].append(
            f"Step-converged dataset generated at {coarse_step_m} m instead of the historical "
            f"{HISTORICAL_STEP_M} m; per-dataset step_artifact_500m_db records the residual "
            "difference of the historical step."
        )

    for tag, spec in DATASETS.items():
        _populate_dataset(
            arrays,
            metadata,
            tag,
            spec,
            num_symbols,
            seed,
            CONFIGS,
            "undepleted",
            LAUNCH_POWER_DBM_PER_POL,
            None,
            coarse_step_m,
            record_step_metadata,
        )

    np.savez_compressed(output_dir / "dra_reference.npz", **arrays)
    (output_dir / "dra_reference.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def generate_depleted(
    output_dir: Path, num_symbols: int, seed: int, coarse_step_m: float = COARSE_STEP_M
) -> None:
    """生成泵浦耗尽数据集：3/5/7 信道、1 信道 SPM 探针、发射功率扫描。

    7ch 只在非默认步长的收敛数据中加入；历史 500 m 耗尽数据集保持 3/5ch，
    以免改变已提交的默认输出。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    record_step_metadata = coarse_step_m != COARSE_STEP_M
    # 7ch 的 500 m 数据未收敛，因此只在收敛数据集（非 500 m）中生成。
    depleted_tags = ("3ch", "5ch", "7ch") if record_step_metadata else ("3ch", "5ch")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict = {
        "schema_version": 1,
        "model": {
            "name": "DistributedRamanFiber",
            "propagator": "in-house symmetric NumPy SSFM (never PyNLO path)",
            "pump_model": "depleted coupled signal/pump power ODE",
            "pump_model_detail": (
                "Pump powers are state variables integrated with RK4 along z at 25 m steps. "
                "The signal is represented by its in-band total power (both polarisations) at "
                "the channel-plan centre frequency. Depleted pump profiles then drive the "
                "per-frequency Raman gain inside the SSFM. Backward pumps close the boundary "
                "value problem by fixed-point iteration."
            ),
            "raman_spectrum": "Gaussian, peak 13.2 THz, sigma 2.12 THz",
            "signal_effects": "loss + CD + Kerr retained; channel ISRS not modelled",
            "ase": "not included in this reference dataset (dra_ase_model='none')",
        },
        "fiber": {
            "length_km": FIBER_LENGTH_KM,
            "alpha_db_per_km": ALPHA_DB_PER_KM,
            "beta2_ps2_per_km": BETA2_PS2_PER_KM,
            "beta3_ps3_per_km": BETA3_PS3_PER_KM,
            "n2_si": N2_SI,
            "a_eff_um2": A_EFF_UM2,
            "coarse_step_m": float(coarse_step_m),
        },
        "signal": {
            "modulation": MODULATION,
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "sps": SPS,
            "roll_off": ROLL_OFF,
            "launch_power_dbm_per_pol": LAUNCH_POWER_DBM_PER_POL,
            "num_symbols": num_symbols,
            "seed": seed,
        },
        "pump": {
            "wavelength_nm": PUMP_WAVELENGTH_NM,
            "attenuation_db_per_km": PUMP_ATTENUATION_DB_PER_KM,
            "directions": ["forward", "backward"],
            "powers_w": [0.10, 0.20],
        },
        "nli_method": (
            "Per channel, fit one complex scalar a to NL ≈ a*LIN in the channel band, then take "
            "the band power of the residual NL-a*LIN as NLI. This removes the common phase/gain "
            "that a coherent receiver would equalize. nli_power_raw_w (no fit) is also recorded. "
            "out_linear uses the same pumps/loss/CD but n2=0."
        ),
        "notes": [
            "Same grid and NLI extraction as dra_reference.json; only the pump model differs.",
            "Pumps are depleted by the signal but have no pump-pump Raman transfer.",
            "Channel-to-channel ISRS is not modelled (raman_r3=None).",
            "The 1ch probe isolates SPM; the 3ch power sweep tests NLI ∝ P^3.",
        ],
        "datasets": {},
    }
    if record_step_metadata:
        metadata["coarse_step_m"] = float(coarse_step_m)
        metadata["pump_model"] = "depleted"
        metadata["step_criterion"] = _step_criterion_metadata(coarse_step_m)
        metadata["notes"].append(
            f"Step-converged dataset generated at {coarse_step_m} m instead of the historical "
            f"{HISTORICAL_STEP_M} m; per-dataset step_artifact_500m_db records the residual "
            "difference of the historical step."
        )
        metadata["notes"].append(
            "The converged depleted dataset includes 7ch; the historical 500 m depleted "
            "dataset stays at 3/5ch because its 7ch result is not step-converged."
        )

    for tag in depleted_tags:
        spec = DATASETS[tag]
        _populate_dataset(
            arrays,
            metadata,
            tag,
            spec,
            num_symbols,
            seed,
            CONFIGS,
            "depleted",
            LAUNCH_POWER_DBM_PER_POL,
            (
                f"{spec['count']}-channel depleted-pump counterpart of dra_reference; "
                "compare on_off_gain_db and snr_nli_db against the undepleted dataset."
            ),
            coarse_step_m,
            record_step_metadata,
        )

    _populate_dataset(
        arrays,
        metadata,
        "1ch",
        ONE_CHANNEL_DATASET,
        num_symbols,
        seed,
        {name: CONFIGS[name] for name in ONE_CHANNEL_CONFIGS},
        "depleted",
        LAUNCH_POWER_DBM_PER_POL,
        "Single-channel SPM probe: one channel, no XPM, used to separate SPM from XPM under pumping.",
        coarse_step_m,
        record_step_metadata,
    )

    for power_dbm in POWER_SWEEP_DBM_PER_POL:
        tag = _sweep_tag(power_dbm)
        _populate_dataset(
            arrays,
            metadata,
            tag,
            DATASETS["3ch"],
            num_symbols,
            seed,
            {"nopump": CONFIGS["nopump"], "fwd_0p20w": CONFIGS["fwd_0p20w"]},
            "depleted",
            power_dbm,
            (
                "Launch-power sweep at 3ch with a 0.2 W forward pump; "
                f"launch {10 ** (power_dbm / 10):.2f} mW per pol. Used to fit NLI ∝ P^3."
            ),
            coarse_step_m,
            record_step_metadata,
        )

    metadata["power_sweep"] = {
        "dataset_tags": [_sweep_tag(power_dbm) for power_dbm in POWER_SWEEP_DBM_PER_POL],
        "launch_power_mw_per_pol": [float(10 ** (power_dbm / 10)) for power_dbm in POWER_SWEEP_DBM_PER_POL],
        "config": "fwd_0p20w",
        "model": "depleted",
        "purpose": "Fit the log-log slope of nli_power_w vs launch power; GN predicts 3.",
    }

    np.savez_compressed(output_dir / "dra_reference_depleted.npz", **arrays)
    (output_dir / "dra_reference_depleted.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _parse_named_values(items: list[str] | None, value_type):
    """把 ``NAME=VALUE`` 列表解析成字典；用于逐计划覆盖步长/符号数。"""
    if not items:
        return None
    parsed = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        try:
            parsed[name] = value_type(value)
        except ValueError as exc:
            raise ValueError(f"invalid value in {item!r}: {exc}") from exc
    return parsed


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dra_reference",
        help="输出目录 (默认: examples/dra_reference)",
    )
    parser.add_argument("--symbols", type=int, default=256, help="每信道符号数")
    parser.add_argument("--seed", type=int, default=20261104, help="比特/波形生成种子")
    parser.add_argument(
        "--coarse-step-m",
        type=float,
        default=COARSE_STEP_M,
        help=(
            "SSFM 粗步长 (m)。默认 500 m 与历史数据逐位一致；"
            "收敛数据建议 62.5 m（7ch 可用 31.25 m）"
        ),
    )
    parser.add_argument(
        "--depleted",
        action="store_true",
        help="同时生成 dra_reference_depleted.{npz,json}（3/5ch、1ch、发射功率扫描）",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="跳过默认的 dra_reference.{npz,json}（配合 --depleted 使用）",
    )
    parser.add_argument(
        "--wideband",
        action="store_true",
        help="生成 c57/cl152 全 WDM 参考（默认目录 examples/dra_reference_wide）",
    )
    parser.add_argument(
        "--wideband-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dra_reference_wide",
        help="全 WDM 输出目录 (默认: examples/dra_reference_wide)",
    )
    parser.add_argument(
        "--wideband-plans",
        nargs="+",
        choices=sorted(WIDEBAND_PLANS),
        default=None,
        help="要生成的全 WDM 计划 (默认: c57 cl152)",
    )
    parser.add_argument(
        "--wideband-step-m",
        type=float,
        default=None,
        help="覆盖所有全 WDM 计划的 SSFM 步长；省略时用 64-symbol 探针自动确定",
    )
    parser.add_argument(
        "--wideband-step-map",
        nargs="+",
        default=None,
        metavar="NAME=STEP",
        help="逐计划覆盖 SSFM 步长，如 c57=0.9765625 cl152=0.48828125",
    )
    parser.add_argument(
        "--wideband-symbols-map",
        nargs="+",
        default=None,
        metavar="NAME=SYMBOLS",
        help="逐计划覆盖符号数，如 cl152=128",
    )
    parser.add_argument(
        "--reuse-probe",
        action="store_true",
        help="复用输出目录已有的 step_convergence.json，不重跑探针",
    )
    parser.add_argument(
        "--wideband-probe-only",
        action="store_true",
        help="只跑步长探针并写 step_convergence.json，不生成主数据集",
    )
    parser.add_argument(
        "--wideband-probe-symbols",
        type=int,
        default=WIDEBAND_PROBE_NUM_SYMBOLS,
        help="步长探针的符号数 (默认 64)",
    )
    parser.add_argument(
        "--wideband-probe-steps",
        type=float,
        nargs="+",
        default=list(WIDEBAND_PROBE_STEPS_M),
        help="步长探针的初始步长序列 (m)",
    )
    parser.add_argument(
        "--wideband-second-seed",
        action="store_true",
        help=f"同时生成第二个 seed ({WIDEBAND_SECOND_SEED}) 的数据集",
    )
    args = parser.parse_args()
    if args.coarse_step_m <= 0:
        parser.error("--coarse-step-m must be positive")
    if args.wideband:
        if args.wideband_step_m is not None and args.wideband_step_m <= 0:
            parser.error("--wideband-step-m must be positive")
        if args.wideband_probe_symbols < 1:
            parser.error("--wideband-probe-symbols must be positive")
        if any(step <= 0 for step in args.wideband_probe_steps):
            parser.error("--wideband-probe-steps must be positive")
        try:
            step_overrides = _parse_named_values(args.wideband_step_map, float)
            num_symbols_by_plan = _parse_named_values(args.wideband_symbols_map, int)
        except ValueError as exc:
            parser.error(str(exc))
        seeds = (
            (args.seed, WIDEBAND_SECOND_SEED)
            if args.wideband_second_seed
            else (args.seed,)
        )
        generate_wideband(
            args.wideband_dir,
            plan_names=tuple(args.wideband_plans or WIDEBAND_PLANS),
            num_symbols=args.symbols,
            num_symbols_by_plan=num_symbols_by_plan,
            seeds=seeds,
            step_m=args.wideband_step_m,
            step_overrides=step_overrides,
            probe_only=args.wideband_probe_only,
            probe_num_symbols=args.wideband_probe_symbols,
            probe_steps_m=tuple(args.wideband_probe_steps),
            reuse_probe=args.reuse_probe,
        )
        return
    if not args.skip_base:
        generate(args.output_dir, args.symbols, args.seed, args.coarse_step_m)
        print(f"Wrote {args.output_dir / 'dra_reference.npz'} and dra_reference.json")
    if args.depleted:
        generate_depleted(args.output_dir, args.symbols, args.seed, args.coarse_step_m)
        print(
            f"Wrote {args.output_dir / 'dra_reference_depleted.npz'} "
            "and dra_reference_depleted.json"
        )


if __name__ == "__main__":
    main()
