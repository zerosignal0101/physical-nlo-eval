#!/usr/bin/env python3
"""cl152 nopump 的 alias-free（2x 采样）种子集合统计。

现有全 WDM 工艺诊断在 ``cl152`` 上只跑过单次实现：2x 采样、64 symbols、
0.244140625 m 得到中心信道 SNR 35.635 dB，而 1x 为 32.473 dB。单次
实现的 2-3 dB 散布使这个绝对值不能直接作为校准上限。本脚本对固定
``(plan, config, symbols, step)`` 扫描多个随机种子，给出 alias-free
中心信道 SNR/NLI 的 mean、样本标准差与 min/max。

每个种子的传播互相独立，可以用 ``--seed-single`` 一个进程跑一个种子，
再用 ``--combine`` 合并；也可以在单进程内用 ``--workers`` 并行。
每个种子单独写 ``<output stem>_seed<seed>.json``，合并结果写
``--output``（默认 ``examples/dra_reference_wide/diagnostics/
cl152_nopump_2x_ensemble.json``）。

用法::

    # 4 个种子各起一个进程（推荐，避免单个命令超时）
    for seed in 20261104 20261105 20261106 20261107; do
        PYTHONPATH=src nohup python examples/check_dra_cl152_ensemble.py \
            --seed-single "$seed" --symbols 64 \
            > "/tmp/cl152_ensemble_${seed}.log" 2>&1 &
    done

    # 全部完成后合并（不重新传播）
    PYTHONPATH=src python examples/check_dra_cl152_ensemble.py --combine

    # 单进程内并行 4 个种子
    PYTHONPATH=src python examples/check_dra_cl152_ensemble.py --workers 4

结果只写小 JSON，不修改 ``examples/dra_reference_wide/`` 的主数据。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import generate_dra_reference as generator

DEFAULT_PLAN = "cl152"
DEFAULT_CONFIG = "nopump"
DEFAULT_SEEDS = (20261104, 20261105, 20261106, 20261107)
DEFAULT_SYMBOLS = 64
DEFAULT_STEP_M = 0.244140625
DEFAULT_SAMPLE_RATE_FACTOR = 2
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "dra_reference_wide" / "diagnostics" / "cl152_nopump_2x_ensemble.json"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="随机种子列表 (默认 20261104 20261105 20261106 20261107)",
    )
    parser.add_argument("--symbols", type=int, default=DEFAULT_SYMBOLS, help="符号数 (默认 64)")
    parser.add_argument("--step-m", type=float, default=DEFAULT_STEP_M, help="SSFM 步长 (m, 默认 0.244140625)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "合并结果 JSON 路径；每个种子写入同目录下的 "
            "<stem>_seed<seed>.json (默认 examples/dra_reference_wide/"
            "diagnostics/cl152_nopump_2x_ensemble.json)"
        ),
    )
    parser.add_argument(
        "--seed-single",
        type=int,
        default=None,
        help="只跑一个种子并写单种子 JSON，用于 nohup 多进程并行",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="单进程内并行的进程数 (>1 时用 ProcessPoolExecutor)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="不传播，读取已有的 _seed<seed>.json 并写合并结果",
    )
    parser.add_argument("--pump-model", default="depleted")
    return parser.parse_args(argv)


def _ensemble_plan(symbols: int) -> tuple[generator.WidebandPlan, generator.WidebandPlan]:
    """构造 2x 采样的 cl152 计划，并校验上采样比是整数。"""
    base_plan = generator.WIDEBAND_PLANS[DEFAULT_PLAN]
    plan = dataclasses.replace(
        base_plan,
        sample_rate_hz=base_plan.sample_rate_hz * DEFAULT_SAMPLE_RATE_FACTOR,
    ).with_num_symbols(symbols)
    ratio = plan.sample_rate_hz / plan.channel_sample_rate_hz
    if not float(ratio).is_integer():
        raise SystemExit(
            f"{plan.name}: 2x sample rate {plan.sample_rate_hz:.6g} Hz is not an integer "
            f"multiple of the channel sample rate {plan.channel_sample_rate_hz:.6g} Hz "
            f"(ratio {ratio:.6g})."
        )
    if plan.resample_ratio != int(round(ratio)):
        raise SystemExit(f"{plan.name}: resample_ratio {plan.resample_ratio} disagrees with ratio {ratio:.6g}.")
    return base_plan, plan


def _seed_path(output: Path, seed: int) -> Path:
    """返回单种子 JSON 路径（与合并输出同目录）。"""
    return output.with_name(f"{output.stem}_seed{seed}{output.suffix}")


def _resolve_output(args: argparse.Namespace) -> Path:
    """确定合并输出路径；单种子结果写到同目录的 ``_seed<seed>.json``。"""
    return args.output if args.output is not None else DEFAULT_OUTPUT


def _run_one(task: tuple[int, int, float, str]) -> dict:
    """传播一个种子并返回一行指标。该函数必须可被 ``ProcessPoolExecutor`` 序列化。"""
    seed, symbols, step_m, pump_model = task
    _, plan = _ensemble_plan(symbols)
    pumps = generator.WIDEBAND_CONFIGS[DEFAULT_PLAN][DEFAULT_CONFIG]
    wdm_input, _, _ = generator.build_wideband_dataset(plan, seed)
    started = time.perf_counter()
    metrics = generator.wideband_center_channel_metrics(plan, wdm_input, step_m, pumps, pump_model)
    row = {
        "plan": DEFAULT_PLAN,
        "config": DEFAULT_CONFIG,
        "pump_model": pump_model,
        "sample_rate_factor": DEFAULT_SAMPLE_RATE_FACTOR,
        "resample_ratio": plan.resample_ratio,
        "num_symbols": symbols,
        "seed": seed,
        "step_m": float(step_m),
        "num_samples": int(wdm_input.data.shape[-1]),
        "elapsed_s": float(time.perf_counter() - started),
        **metrics,
    }
    print(
        f"{DEFAULT_PLAN} {DEFAULT_CONFIG} sym={symbols} x{DEFAULT_SAMPLE_RATE_FACTOR} "
        f"seed={seed} dz={step_m:g}: snr={row['snr_nli_db']:.4f} dB "
        f"nli={row['nli_power_w']:.6e} W ({row['elapsed_s']:.1f}s)",
        flush=True,
    )
    return row


def _stats(values: list[float]) -> dict:
    """mean、样本标准差 (ddof=1) 与 min/max。"""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else None,
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def _summarize(rows: list[dict]) -> dict:
    """汇总 ``snr_nli_db`` 与 ``nli_power_w`` 的均值/散布。"""
    snr = [float(row["snr_nli_db"]) for row in rows if row.get("snr_nli_db") is not None]
    nli = [float(row["nli_power_w"]) for row in rows if row.get("nli_power_w") is not None]
    return {
        "count": len(rows),
        "seeds": sorted(int(row["seed"]) for row in rows),
        "snr_nli_db": _stats(snr),
        "nli_power_w": _stats(nli),
        "snr_nli_db_range": (float(max(snr) - min(snr)) if len(snr) > 1 else None),
    }


def _document(rows: list[dict], args: argparse.Namespace) -> dict:
    """构造合并/单种子 JSON 文档。"""
    _, plan = _ensemble_plan(args.symbols)
    return {
        "generated_by": "examples/check_dra_cl152_ensemble.py",
        "plan": DEFAULT_PLAN,
        "config": DEFAULT_CONFIG,
        "pump_model": args.pump_model,
        "sample_rate_factor": DEFAULT_SAMPLE_RATE_FACTOR,
        "sample_rate_hz": float(plan.sample_rate_hz),
        "resample_ratio": plan.resample_ratio,
        "num_symbols": args.symbols,
        "step_m": float(args.step_m),
        "note": (
            "Alias-free (2x) cl152 nopump center-channel ensemble. Full-WDM waveform "
            "NLI has a 2-3 dB single-realization scatter; quote the mean over seeds and "
            "the sample std, not a single realization. 4 seeds give a rough mean but a "
            "sub-dB confidence interval needs more seeds. Each row is an independent "
            "random symbol record propagated at the same step."
        ),
        "summary": _summarize(rows),
        "rows": sorted(rows, key=lambda row: int(row["seed"])),
    }


def _write_json(path: Path, document: dict) -> None:
    """写小 JSON 文件并确保父目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _run_seeds(
    tasks: list[tuple[int, int, float, str]],
    workers: int,
    seed_outputs: dict[int, Path],
    args: argparse.Namespace,
) -> list[dict]:
    """按顺序或进程池执行所有种子；每完成一个立即落盘单种子 JSON。"""
    rows: list[dict] = []
    if workers <= 1:
        for task in tasks:
            row = _run_one(task)
            rows.append(row)
            _write_json(seed_outputs[int(task[0])], _document([row], args))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_one, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                row = future.result()
                rows.append(row)
                _write_json(seed_outputs[int(task[0])], _document([row], args))
    return rows


def _print_summary(summary: dict) -> None:
    """打印 SNR/NLI 的 mean/std/min/max；单样本时 std 为 ``n/a``。"""
    for key, unit in (("snr_nli_db", "dB"), ("nli_power_w", "W")):
        stats = summary[key]
        std = "n/a" if stats["std"] is None else f"{stats['std']:.4g}"
        print(
            f"{key} mean={stats['mean']:.6g} {unit} std={std} "
            f"min={stats['min']:.6g} max={stats['max']:.6g} (n={stats['n']})"
        )


def _combine(args: argparse.Namespace, output: Path) -> int:
    """读取单种子 JSON 并写合并结果，不重新传播。"""
    rows: list[dict] = []
    missing: list[Path] = []
    for seed in args.seeds:
        path = _seed_path(output, seed)
        if not path.exists():
            missing.append(path)
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(document["rows"])
    if missing:
        raise SystemExit("missing per-seed result files: " + ", ".join(str(path) for path in missing))
    combined = _document(rows, args)
    _write_json(output, combined)
    _print_summary(combined["summary"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    args = _parse_args(argv)
    if args.combine and args.seed_single is not None:
        raise SystemExit("--combine and --seed-single are mutually exclusive.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    output = _resolve_output(args)
    if args.combine:
        return _combine(args, output)

    seeds = [args.seed_single] if args.seed_single is not None else list(args.seeds)
    tasks = [(seed, args.symbols, float(args.step_m), args.pump_model) for seed in seeds]
    seed_outputs = {seed: _seed_path(output, seed) for seed in seeds}
    rows = _run_seeds(tasks, args.workers, seed_outputs, args)
    if args.seed_single is not None:
        return 0

    combined = _document(rows, args)
    _write_json(output, combined)
    _print_summary(combined["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
