#!/usr/bin/env python3
"""全 WDM 波形参考的采样率（混叠）与记录长度诊断。

`generate_dra_reference.py --wideband-probe-only` 只扫描步长。本脚本补两类
诊断，它们决定全 WDM 的绝对 NLI 能否作为基准：

* ``--mode oversample``：固定步长/符号数，改变合波采样率因子 1/2/3。
  1× 采样率下三次非线性产物会折叠回中心信道；2× 与 3× 的差小于
  0.01 dB 时说明 2× 已足够。
* ``--mode record``：固定步长/采样率，改变符号数或 seed，估计 NLI 的
  统计散布。全 WDM 的中心信道 NLI 在 64–512 symbols 之间没有单调
  收敛趋势，单次实现有 ~1–2 dB 散布。

用法::

    # 混叠：cl152 nopump，0.488/0.244 m，1×/2×/3×
    PYTHONPATH=src python examples/check_dra_wideband_convergence.py \
        --plan cl152 --mode oversample --config nopump \
        --steps 0.48828125 0.244140625 --factors 1 2 3 --symbols 64

    # 记录长度：c57 nopump，0.488 m，64/128/256/512 symbols
    PYTHONPATH=src python examples/check_dra_wideband_convergence.py \
        --plan c57 --mode record --config nopump --steps 0.48828125 \
        --symbols 64 128 256 512 --seeds 20261104 20261105

结果写 JSON；不修改 ``examples/dra_reference_wide/`` 的主数据。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import generate_dra_reference as generator

#: 采样率因子必须让 ``WDMMux`` 的 resample_poly 比值保持整数。
SUPPORTED_SAMPLE_RATE_FACTORS = (1, 2, 3)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", choices=sorted(generator.WIDEBAND_PLANS), required=True)
    parser.add_argument("--mode", choices=("oversample", "record"), required=True)
    parser.add_argument("--config", default="nopump", help="泵浦配置名 (默认 nopump)")
    parser.add_argument("--steps", type=float, nargs="+", required=True, help="SSFM 步长 (m)")
    parser.add_argument("--symbols", type=int, nargs="+", required=True, help="符号数")
    parser.add_argument(
        "--factors",
        type=int,
        nargs="+",
        default=list(SUPPORTED_SAMPLE_RATE_FACTORS),
        help="采样率因子 (仅 oversample 模式)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[generator.WIDEBAND_SEED],
        help="随机种子 (oversample 与 record 模式)",
    )
    parser.add_argument("--pump-model", default="depleted")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSON (默认 examples/dra_reference_wide/convergence_diagnostics.json)",
    )
    return parser.parse_args(argv)


def _rows(args: argparse.Namespace) -> list[dict]:
    """按模式生成 (symbols, factor, seed, step) 组合并逐点传播。"""
    base_plan = generator.WIDEBAND_PLANS[args.plan]
    if args.config not in generator.WIDEBAND_CONFIGS[args.plan]:
        raise SystemExit(
            f"unknown config {args.config!r}; available: "
            f"{sorted(generator.WIDEBAND_CONFIGS[args.plan])}"
        )
    pumps = generator.WIDEBAND_CONFIGS[args.plan][args.config]

    if args.mode == "oversample":
        combinations = [
            (symbols, factor, seed, step)
            for step in args.steps
            for symbols in args.symbols
            for factor in args.factors
            for seed in args.seeds
        ]
    else:
        combinations = [
            (symbols, 1, seed, step)
            for step in args.steps
            for symbols in args.symbols
            for seed in args.seeds
        ]

    rows = []
    for symbols, factor, seed, step in combinations:
        if factor not in SUPPORTED_SAMPLE_RATE_FACTORS:
            raise SystemExit(
                f"sample-rate factor {factor} is not supported; "
                f"use {SUPPORTED_SAMPLE_RATE_FACTORS}"
            )
        plan = dataclasses.replace(
            base_plan, sample_rate_hz=base_plan.sample_rate_hz * factor
        ).with_num_symbols(symbols)
        wdm_input, _, _ = generator.build_wideband_dataset(plan, seed)
        started = time.perf_counter()
        metrics = generator.wideband_center_channel_metrics(
            plan, wdm_input, step, pumps, args.pump_model
        )
        row = {
            "plan": args.plan,
            "config": args.config,
            "pump_model": args.pump_model,
            "num_symbols": symbols,
            "sample_rate_factor": factor,
            "seed": seed,
            "step_m": float(step),
            "num_samples": int(wdm_input.data.shape[-1]),
            "elapsed_s": float(time.perf_counter() - started),
            **metrics,
        }
        rows.append(row)
        print(
            f"{args.plan} {args.config} sym={symbols} x{factor} seed={seed} "
            f"dz={step:g}: snr={row['snr_nli_db']:.4f} dB ({row['elapsed_s']:.1f}s)",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    args = _parse_args(argv)
    rows = _rows(args)
    output = args.output or (
        Path(__file__).resolve().parent
        / "dra_reference_wide"
        / "convergence_diagnostics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_by": "examples/check_dra_wideband_convergence.py",
        "mode": args.mode,
        "plan": args.plan,
        "config": args.config,
        "pump_model": args.pump_model,
        "note": (
            "Oversample mode checks aliasing: 1x and 2x can differ by ~0.4-1.4 dB; "
            "2x and 3x should agree within 0.01 dB. Record mode checks statistical "
            "scatter: full-WDM center NLI has ~1-2 dB seed/record-length spread."
        ),
        "rows": rows,
    }
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
