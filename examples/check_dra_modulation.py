#!/usr/bin/env python3
"""量化 GN 的 Gaussian 假设与 DP-QPSK 波形之间的调制口径差。

全 WDM 参考波形用恒模 DP-QPSK，而解析 GN 模型假设每偏振独立复高斯
符号。本脚本复用 :mod:`generate_dra_reference` 的 ``WidebandPlan``、
``WIDEBAND_CONFIGS``、``make_wideband_fiber`` 与 ``band_nli_metrics``，
只在符号生成处切换调制：

* ``qpsk``：与 ``build_wideband_dataset`` 完全相同的比特→符号→RRC→IQ；
* ``gaussian``：每偏振独立复高斯 ``(randn + 1j*randn)/sqrt(2)``，按实际
  样本功率归一化到单位每偏振功率，再走同一 RRC/IQ 路径。

高斯与 QPSK 的 RRC 参数（sps、roll_off）和发射功率完全一致，因此两者的
唯一差异是符号分布。``qpsk`` 路径会与 ``build_wideband_dataset`` 的输入
波形逐位比对；不等则直接报错。

用法::

    # 一个组合：单计划、单种子、单调制（便于 nohup 并行）
    PYTHONPATH=src python examples/check_dra_modulation.py \
        --plan cl152 --modulations gaussian --symbols 128 \
        --seeds 20261104 --step-m 0.244140625 \
        --sample-rate-factor 2 --output /tmp/cl152_gauss_seed20261104.json

    # 合并同一计划/采样率的多个单组合 JSON（配对 QPSK 与高斯并算 NLI 比）
    PYTHONPATH=src python examples/check_dra_modulation.py \
        --merge /tmp/cl152_*.json --output \
        examples/dra_reference_wide/diagnostics/cl152_modulation_2x_128sym.json

结果只写小 JSON，不修改 ``examples/dra_reference_wide/`` 的主数据。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import generate_dra_reference as generator  # noqa: E402

from physics_nlo_eval.core.signal import Signal, SignalContext  # noqa: E402
from physics_nlo_eval.tx.mapper import Mapper  # noqa: E402
from physics_nlo_eval.tx.optics import IQModulator  # noqa: E402
from physics_nlo_eval.tx.rrc import RRCFilter  # noqa: E402

MODULATIONS = ("qpsk", "gaussian")
DEFAULT_PLAN = "cl152"
DEFAULT_SYMBOLS = 128
DEFAULT_STEP_M = 0.244140625
DEFAULT_SAMPLE_RATE_FACTOR = 1
DEFAULT_CONFIG = "nopump"
DIAGNOSTICS_DIR = _EXAMPLES_DIR / "dra_reference_wide" / "diagnostics"
#: NLI 比的标准差/极差来自独立种子的样本统计；单种子时 std 为 None。
REQUIRED_ROW_FIELDS = (
    "plan",
    "config",
    "pump_model",
    "modulation",
    "sample_rate_factor",
    "resample_ratio",
    "num_symbols",
    "seed",
    "step_m",
    "num_samples",
    "signal_power_w",
    "nli_power_w",
    "snr_nli_db",
    "elapsed_s",
)


def make_plan(plan_name: str, sample_rate_factor: int, num_symbols: int) -> generator.WidebandPlan:
    """构造采样率放大 ``sample_rate_factor`` 倍的计划。

    2× 是 cl152 alias-free 的最低要求；单信道采样率必须是目标采样率的
    整数分频，因此重采样比也必须保持整数。
    """
    if plan_name not in generator.WIDEBAND_PLANS:
        raise KeyError(f"unknown plan {plan_name!r}; available: {sorted(generator.WIDEBAND_PLANS)}")
    if sample_rate_factor < 1:
        raise ValueError("sample_rate_factor must be >= 1.")
    base = generator.WIDEBAND_PLANS[plan_name]
    plan = dataclasses.replace(
        base,
        sample_rate_hz=base.sample_rate_hz * sample_rate_factor,
    ).with_num_symbols(num_symbols)
    ratio = plan.sample_rate_hz / plan.channel_sample_rate_hz
    if not float(ratio).is_integer():
        raise ValueError(
            f"{plan_name}: sample rate {plan.sample_rate_hz:.6g} Hz is not an integer "
            f"multiple of the channel sample rate {plan.channel_sample_rate_hz:.6g} Hz "
            f"(ratio {ratio:.6g})."
        )
    if plan.resample_ratio != int(round(ratio)):
        raise ValueError(f"{plan_name}: resample_ratio {plan.resample_ratio} disagrees with ratio {ratio:.6g}.")
    return plan


def gaussian_symbols(rng: np.random.Generator, n_pol: int, num_symbols: int) -> np.ndarray:
    """每偏振独立复高斯符号，按实际样本功率归一化到单位平均功率。

    ``(randn + 1j*randn)/sqrt(2)`` 的期望功率为 1；这里再做一次有限样本
    归一化，使每偏振平均功率与 ``power_norm=True`` 的 QPSK 星座一致。
    返回 float64 复数组；进入信号链时再转成与 ``Mapper.forward`` 相同的
    ``complex64``。
    """
    symbols = (rng.standard_normal((n_pol, num_symbols)) + 1j * rng.standard_normal((n_pol, num_symbols))) / np.sqrt(
        2.0
    )
    sample_power = np.mean(np.abs(symbols) ** 2, axis=1, keepdims=True)
    return symbols / np.sqrt(sample_power)


def gaussian_signal(plan: generator.WidebandPlan, symbols: np.ndarray) -> Signal:
    """把高斯符号包装成与 ``Mapper.forward`` 同构的 1 sps Signal。"""
    ctx = SignalContext(
        sample_rate=plan.symbol_rate_hz,
        center_freq=0.0,
        n_pol=int(symbols.shape[0]),
        sps=1,
        sym_rate=plan.symbol_rate_hz,
        bits_per_symbol=1,
        meta={"modulation": "GAUSSIAN"},
    )
    # 与 Mapper.forward 一样以 complex64 进入 RRC/IQ，保证两条路径的
    # 数值精度口径一致。
    return Signal(data=symbols.astype(np.complex64), ctx=ctx)


def build_symbol_streams(
    plan: generator.WidebandPlan,
    seed: int,
    modulation: str,
) -> tuple[list[Signal], np.ndarray | None]:
    """生成逐信道 1 sps 符号流；返回 ``(symbol_streams, bits)``。

    ``qpsk`` 分支与 :func:`generate_dra_reference.build_wideband_dataset`
    的 RNG 消费、比特形状、Mapper 调用顺序完全一致；``gaussian`` 分支
    只替换符号来源。``bits`` 对高斯无意义，返回 ``None``。
    """
    if modulation not in MODULATIONS:
        raise ValueError(f"modulation must be one of {MODULATIONS}, got {modulation!r}.")

    rng = np.random.default_rng(seed)
    bits: np.ndarray | None = None
    symbol_streams: list[Signal] = []
    if modulation == "qpsk":
        mapper = Mapper(
            modulation=generator.MODULATION,
            power_norm=True,
            sym_rate=plan.symbol_rate_hz,
        )
        bits_per_channel = plan.num_symbols * mapper.bits_per_symbol * mapper.n_pol
        bits = rng.integers(0, 2, size=(plan.n_channels, bits_per_channel), dtype=np.int8).astype(bool)
        for idx in range(plan.n_channels):
            symbol_streams.append(mapper.forward(bits[idx]))
    else:
        for _ in range(plan.n_channels):
            symbol_streams.append(gaussian_signal(plan, gaussian_symbols(rng, 2, plan.num_symbols)))
    return symbol_streams, bits


def build_modulated_channels(
    plan: generator.WidebandPlan,
    seed: int,
    modulation: str,
) -> tuple[list[Signal], np.ndarray | None]:
    """生成一个计划的逐信道光信号；返回 ``(channel_signals, bits)``。

    符号来源由 :func:`build_symbol_streams` 决定，之后统一走与参考生成器
    相同的 RRC 成形和 IQ 调制。
    """
    symbol_streams, bits = build_symbol_streams(plan, seed, modulation)
    rrc = RRCFilter(sps=plan.sps, roll_off=plan.roll_off)
    iq_mod = IQModulator(launch_power_dbm_per_pol=plan.launch_power_dbm_per_pol)
    channel_signals = [iq_mod.forward(rrc.forward(signal)) for signal in symbol_streams]
    return channel_signals, bits


def build_modulated_dataset(
    plan: generator.WidebandPlan,
    seed: int,
    modulation: str,
) -> tuple[Signal, np.ndarray | None, list[Signal]]:
    """生成一个全 WDM 输入波形；返回 ``(wdm_input, bits, channel_signals)``。

    合波代码与 :func:`generate_dra_reference.build_wideband_dataset` 相同，
    保证 QPSK 路径可以逐位复现参考输入。
    """
    channel_signals, bits = build_modulated_channels(plan, seed, modulation)
    wdm_plan = generator.WDMPlan(
        reference_freq_hz=plan.center_frequency_hz,
        channels=[
            generator.ChannelSpec(
                channel_id=idx,
                freq_offset_hz=float(plan.offsets_hz[idx]),
                power_dbm_per_pol=plan.launch_power_dbm_per_pol,
            )
            for idx in range(plan.n_channels)
        ],
    )
    mux = generator.WDMMux(wdm_plan, target_sample_rate_hz=plan.sample_rate_hz)
    wdm_input = mux.combine(channel_signals)
    wdm_input.ctx.meta["dra_reference_tag"] = plan.name
    wdm_input.ctx.meta["modulation_variant"] = modulation
    return wdm_input, bits, channel_signals


def qpsk_self_check(
    plan: generator.WidebandPlan,
    seed: int,
    wdm_input: Signal,
) -> dict:
    """把自建 QPSK 输入与 ``build_wideband_dataset`` 逐位比对。"""
    reference, _, _ = generator.build_wideband_dataset(plan, seed)
    if wdm_input.data.shape != reference.data.shape:
        raise SystemExit(f"QPSK self-check failed: shape mismatch {wdm_input.data.shape} vs {reference.data.shape}.")
    max_abs_diff = float(np.max(np.abs(wdm_input.data - reference.data)))
    bit_exact = bool(np.array_equal(wdm_input.data, reference.data))
    if not bit_exact:
        raise SystemExit(
            "QPSK self-check failed: input waveform differs from "
            f"build_wideband_dataset (max |diff| = {max_abs_diff:.3e})."
        )
    return {
        "bit_exact": bit_exact,
        "max_abs_diff": max_abs_diff,
        "num_samples": int(wdm_input.data.shape[-1]),
    }


def run_combination(
    plan: generator.WidebandPlan,
    *,
    plan_name: str,
    seed: int,
    modulation: str,
    config_name: str,
    pumps: list,
    step_m: float,
    sample_rate_factor: int,
    pump_model: str,
) -> tuple[dict, dict]:
    """传播一个 (modulation, seed) 组合并返回 ``(row, self_check)``。

    每次调用做两次传播：一次非线性、一次 n2=0 的线性参考。中心信道
    NLI 复用生成器的 :func:`wideband_center_channel_metrics`。
    """
    wdm_input, _, channel_signals = build_modulated_dataset(plan, seed, modulation)

    self_check: dict = {"bit_exact": None, "max_abs_diff": None}
    if modulation == "qpsk":
        self_check = qpsk_self_check(plan, seed, wdm_input)

    # 逐信道光功率（合波前）用于验证两条调制路径的发射功率一致。
    channel_power_w = [float(np.mean(np.abs(signal.data) ** 2)) for signal in channel_signals]

    started = time.perf_counter()
    metrics = generator.wideband_center_channel_metrics(plan, wdm_input, step_m, pumps, pump_model)
    elapsed = time.perf_counter() - started

    row = {
        "plan": plan_name,
        "config": config_name,
        "pump_model": pump_model,
        "modulation": modulation,
        "sample_rate_factor": int(sample_rate_factor),
        "resample_ratio": plan.resample_ratio,
        "num_symbols": plan.num_symbols,
        "seed": int(seed),
        "step_m": float(step_m),
        "num_samples": int(wdm_input.data.shape[-1]),
        "channel_power_mean_w": float(np.mean(channel_power_w)),
        "channel_power_rel_spread": (
            float(np.ptp(channel_power_w) / np.mean(channel_power_w)) if channel_power_w else None
        ),
        **metrics,
        "elapsed_s": float(elapsed),
        "qpsk_self_check": self_check,
    }
    print(
        f"{plan_name} {config_name} {modulation} sym={plan.num_symbols} "
        f"x{sample_rate_factor} seed={seed} dz={step_m:g}: "
        f"snr={row['snr_nli_db']:.4f} dB nli={row['nli_power_w']:.6e} W "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return row, self_check


def _pair_key(row: dict) -> tuple:
    """QPSK/高斯配对键：除 modulation 外的所有组合参数。"""
    return (
        row["plan"],
        row["config"],
        row["pump_model"],
        int(row["sample_rate_factor"]),
        int(row["resample_ratio"]),
        int(row["num_symbols"]),
        int(row["seed"]),
        float(row["step_m"]),
    )


def _stats(values: list[float]) -> dict:
    """mean、样本标准差 (ddof=1) 与 min/max。"""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def build_document(
    rows: list[dict],
    *,
    plan: generator.WidebandPlan,
    config_name: str,
    pump_model: str,
    sample_rate_factor: int,
    num_symbols: int,
) -> dict:
    """把行结果整理成 JSON 文档，并配对 QPSK/高斯计算 NLI 比。"""
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        by_key.setdefault(_pair_key(row), {})[row["modulation"]] = row

    comparisons = []
    for key, pair in sorted(by_key.items()):
        if "qpsk" not in pair or "gaussian" not in pair:
            continue
        qpsk = pair["qpsk"]
        gaussian = pair["gaussian"]
        nli_ratio_db = (
            float(10 * np.log10(gaussian["nli_power_w"] / qpsk["nli_power_w"])) if qpsk["nli_power_w"] > 0 else None
        )
        signal_ratio_db = (
            float(10 * np.log10(gaussian["signal_power_w"] / qpsk["signal_power_w"]))
            if qpsk["signal_power_w"] > 0
            else None
        )
        comparisons.append(
            {
                "plan": key[0],
                "config": key[1],
                "pump_model": key[2],
                "sample_rate_factor": key[3],
                "resample_ratio": key[4],
                "num_symbols": key[5],
                "seed": key[6],
                "step_m": key[7],
                "qpsk_snr_nli_db": qpsk["snr_nli_db"],
                "gaussian_snr_nli_db": gaussian["snr_nli_db"],
                "delta_snr_db": (
                    float(gaussian["snr_nli_db"] - qpsk["snr_nli_db"])
                    if qpsk["snr_nli_db"] is not None and gaussian["snr_nli_db"] is not None
                    else None
                ),
                "nli_ratio_db": nli_ratio_db,
                "signal_power_ratio_db": signal_ratio_db,
                "qpsk_nli_power_w": qpsk["nli_power_w"],
                "gaussian_nli_power_w": gaussian["nli_power_w"],
            }
        )

    modulation_summary = {}
    for modulation in sorted({row["modulation"] for row in rows}):
        subset = [row for row in rows if row["modulation"] == modulation]
        modulation_summary[modulation] = {
            "snr_nli_db": _stats([float(row["snr_nli_db"]) for row in subset]),
            "nli_power_w": _stats([float(row["nli_power_w"]) for row in subset]),
            "signal_power_w": _stats([float(row["signal_power_w"]) for row in subset]),
        }

    nli_ratios = [c["nli_ratio_db"] for c in comparisons if c["nli_ratio_db"] is not None]
    step_values = sorted({float(row["step_m"]) for row in rows})
    self_checks = [row["qpsk_self_check"] for row in rows if row["modulation"] == "qpsk"]
    return {
        "schema_version": 1,
        "generated_by": "examples/check_dra_modulation.py",
        "plan": plan.to_metadata(),
        "config": config_name,
        "pump_model": pump_model,
        "sample_rate_factor": int(sample_rate_factor),
        "sample_rate_hz": float(plan.sample_rate_hz),
        "resample_ratio": plan.resample_ratio,
        "num_symbols": int(num_symbols),
        "step_m": step_values[0] if len(step_values) == 1 else None,
        "step_m_values": step_values,
        "nli_analysis_bandwidth_hz": float(plan.symbol_rate_hz),
        "modulations": sorted({row["modulation"] for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "qpsk_self_check": {
            "all_bit_exact": all(bool(check["bit_exact"]) for check in self_checks) if self_checks else None,
            "max_abs_diff": max(
                (check["max_abs_diff"] for check in self_checks if check["max_abs_diff"] is not None),
                default=None,
            ),
        },
        "modulation_summary": modulation_summary,
        "nli_ratio_db_summary": _stats(nli_ratios),
        "comparisons": comparisons,
        "rows": sorted(
            rows,
            key=lambda row: (
                float(row["step_m"]),
                int(row["seed"]),
                row["modulation"],
            ),
        ),
        "notes": [
            "Both modulations use the same RRC (sps, roll_off) and IQ launch power; "
            "only the symbol distribution differs.",
            "Gaussian symbols are per-polarisation complex normal, power-normalised on the "
            "finite record to unit per-pol power before RRC/IQ.",
            "nli_ratio_db = 10*log10(NLI_gaussian / NLI_qpsk); delta_snr_db = SNR_gaussian - "
            "SNR_qpsk. A positive nli_ratio_db means the Gaussian control produces more NLI.",
            "This is a nopump comparison unless config is a pumped one; pumped modulation "
            "spacing may differ and must not be extrapolated.",
            "Single-realisation NLI has 1-2 dB scatter; quote multi-seed means, not one seed.",
        ],
    }


def write_document(path: Path, document: dict) -> None:
    """写小 JSON 文件并确保父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def merge_documents(paths: list[Path], output: Path) -> dict:
    """合并多个单组合文档，重新配对并写合并结果。"""
    rows: list[dict] = []
    first: dict | None = None
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        if first is None:
            first = document
        rows.extend(document["rows"])
    if first is None:
        raise SystemExit("no input documents to merge.")
    plan = generator.WidebandPlan(
        **{key: value for key, value in first["plan"].items() if key in generator.WidebandPlan.__dataclass_fields__}
    )
    document = build_document(
        rows,
        plan=plan,
        config_name=first["config"],
        pump_model=first["pump_model"],
        sample_rate_factor=int(first["sample_rate_factor"]),
        num_symbols=int(first["num_symbols"]),
    )
    write_document(output, document)
    summary = document["nli_ratio_db_summary"]
    print(
        f"nli_ratio_db mean={summary['mean']} dB std={summary['std']} n={summary['n']}",
        flush=True,
    )
    return document


def _default_output(args: argparse.Namespace, plan_name: str) -> Path:
    """单组合的默认输出文件名（计划/采样率/步长/符号/种子/调制）。"""
    stem = f"{plan_name}_x{args.sample_rate_factor}_{args.step_m:g}m_{args.symbols}sym"
    if len(args.seeds) == 1:
        stem += f"_seed{args.seeds[0]}"
    if len(args.modulations) == 1:
        stem += f"_{args.modulations[0]}"
    return DIAGNOSTICS_DIR / f"dra_modulation_{stem}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行。"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plan",
        choices=sorted(generator.WIDEBAND_PLANS),
        default=DEFAULT_PLAN,
        help=f"全 WDM 计划 (默认 {DEFAULT_PLAN})",
    )
    parser.add_argument(
        "--modulations",
        nargs="+",
        choices=MODULATIONS,
        default=list(MODULATIONS),
        help="要传播的调制 (默认 qpsk gaussian)",
    )
    parser.add_argument("--symbols", type=int, default=DEFAULT_SYMBOLS, help="每信道符号数")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[generator.WIDEBAND_SEED],
        help="随机种子列表",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=DEFAULT_STEP_M,
        help="SSFM 步长 (m)",
    )
    parser.add_argument(
        "--sample-rate-factor",
        type=int,
        default=DEFAULT_SAMPLE_RATE_FACTOR,
        help="相对计划采样率的倍数；cl152 alias-free 至少 2",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[DEFAULT_CONFIG],
        help=f"泵浦配置名 (默认 {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--pump-model",
        choices=("depleted", "undepleted"),
        default="depleted",
        help="泵浦模型 (默认 depleted，与参考数据一致)",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出 JSON 路径")
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        metavar="JSON",
        help="只合并已有单组合 JSON，不重新传播",
    )
    args = parser.parse_args(argv)
    if args.symbols < 1:
        parser.error("--symbols must be positive")
    if args.step_m <= 0:
        parser.error("--step-m must be positive")
    if args.sample_rate_factor < 1:
        parser.error("--sample-rate-factor must be >= 1")
    if args.merge is None:
        available = set(generator.WIDEBAND_CONFIGS[args.plan])
        unknown = sorted(set(args.configs) - available)
        if unknown:
            parser.error(f"unknown config(s) {unknown} for {args.plan}; available: {sorted(available)}")
    return args


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    if args.merge is not None:
        output = args.output
        if output is None:
            parser_error = "merge requires --output"
            raise SystemExit(parser_error)
        merge_documents(args.merge, output)
        return 0

    plan = make_plan(args.plan, args.sample_rate_factor, args.symbols)
    output = args.output if args.output is not None else _default_output(args, args.plan)
    rows: list[dict] = []
    for config_name in args.configs:
        pumps = generator.WIDEBAND_CONFIGS[args.plan][config_name]
        for seed in args.seeds:
            for modulation in args.modulations:
                row, _ = run_combination(
                    plan,
                    plan_name=args.plan,
                    seed=seed,
                    modulation=modulation,
                    config_name=config_name,
                    pumps=pumps,
                    step_m=args.step_m,
                    sample_rate_factor=args.sample_rate_factor,
                    pump_model=args.pump_model,
                )
                rows.append(row)

    document = build_document(
        rows,
        plan=plan,
        config_name=args.configs[0] if len(args.configs) == 1 else "mixed",
        pump_model=args.pump_model,
        sample_rate_factor=args.sample_rate_factor,
        num_symbols=args.symbols,
    )
    write_document(output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
