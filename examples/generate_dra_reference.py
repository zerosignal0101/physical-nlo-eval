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
"""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()
    if args.coarse_step_m <= 0:
        parser.error("--coarse-step-m must be positive")
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
