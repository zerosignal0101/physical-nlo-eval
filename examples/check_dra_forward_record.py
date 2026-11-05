#!/usr/bin/env python3
"""c57 前向泵记录长度诊断：频谱 NLI vs 接收机 EVM。

§12.2.2 发现 c57 ``fwd_0p20w`` 的中心信道频谱 SNR 随记录长度持续
下降（64→512 symbols，26.78→14.15 dB，512 仍未饱和），而无泵没有
单调趋势。本脚本在同一批波形上同时测两个量：

* ``snr_nli_db``：频带内 NLI 残差（``band_nli_metrics``），即现有
  校准口径；
* ``center_channel_snr_evm_db``：接收机 EVM（demux → 公共增益对齐 →
  CDC → RRC → EVM），即实际可测量的接收质量。

若 EVM SNR 随记录长度稳定、而频谱 NLI 发散，说明发散来自频域
估计器/短周期记录；若两者同步发散，则说明周期性记录的传播本身在
随记录长度积累 NLI。

用法::

    # 完整诊断（256/512 很慢，建议 nohup 后台运行）
    PYTHONPATH=src nohup python examples/check_dra_forward_record.py \
        > /tmp/c57_forward_record.log 2>&1 &

    # 快速冒烟
    PYTHONPATH=src python examples/check_dra_forward_record.py --symbols 64

    # 2× 合波采样率（factor 必须是整数，且重采样比仍为整数）
    PYTHONPATH=src python examples/check_dra_forward_record.py \
        --symbols 512 --sample-rate-factor 2

结果写 ``examples/dra_reference_wide/diagnostics/c57_forward_record_evm.json``；
不修改主参考数据。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import generate_dra_reference as generator

DEFAULT_SYMBOLS = (64, 128, 256, 512)
DEFAULT_CONFIGS = ("nopump", "fwd_0p20w")
DEFAULT_STEP_M = 0.48828125
DEFAULT_SEED = 20261104
DEFAULT_PLAN = "c57"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", default=DEFAULT_PLAN, choices=sorted(generator.WIDEBAND_PLANS))
    parser.add_argument("--symbols", type=int, nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--step-m", type=float, default=DEFAULT_STEP_M)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pump-model", default="depleted")
    parser.add_argument(
        "--sample-rate-factor",
        type=int,
        default=1,
        help=(
            "合波采样率相对计划默认值的整数倍 (默认 1)。用于混叠诊断；"
            "必须保持重采样比为整数。"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "输出 JSON (默认 "
            "examples/dra_reference_wide/diagnostics/c57_forward_record_evm.json)"
        ),
    )
    return parser.parse_args(argv)


def _plan_for_symbols(base_plan, symbols: int, factor: int):
    """构造指定符号数与采样率倍数的计划，并校验整数重采样比。"""
    plan = base_plan.with_num_symbols(symbols)
    if factor != 1:
        plan = replace(plan, sample_rate_hz=base_plan.sample_rate_hz * factor)
    ratio = plan.sample_rate_hz / plan.channel_sample_rate_hz
    if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=0.0):
        raise SystemExit(
            f"sample-rate-factor {factor} gives a non-integer resample ratio "
            f"{ratio:.12g} for plan {plan.name}"
        )
    return plan


def _rows(args: argparse.Namespace) -> list[dict]:
    """对每个记录长度复用同一输入，逐配置传播并同时测频谱与 EVM。"""
    if args.sample_rate_factor < 1:
        raise SystemExit("sample-rate-factor must be a positive integer")
    base_plan = generator.WIDEBAND_PLANS[args.plan]
    available = generator.WIDEBAND_CONFIGS[args.plan]
    unknown = [name for name in args.configs if name not in available]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; available: {sorted(available)}")

    rows = []
    for symbols in args.symbols:
        plan = _plan_for_symbols(base_plan, symbols, args.sample_rate_factor)
        wdm_input, bits, constellation = generator.build_wideband_dataset(plan, args.seed)
        for config_name in args.configs:
            pumps = available[config_name]
            started = time.perf_counter()
            metrics = generator.wideband_center_channel_metrics(
                plan,
                wdm_input,
                args.step_m,
                pumps,
                args.pump_model,
                bits=bits,
                constellation=constellation,
            )
            row = {
                "plan": args.plan,
                "config": config_name,
                "pump_model": args.pump_model,
                "num_symbols": symbols,
                "seed": args.seed,
                "step_m": float(args.step_m),
                "num_samples": int(wdm_input.data.shape[-1]),
                "elapsed_s": float(time.perf_counter() - started),
                **metrics,
            }
            rows.append(row)
            print(
                f"{args.plan} {config_name} sym={symbols} seed={args.seed} "
                f"factor={args.sample_rate_factor} dz={args.step_m:g}: "
                f"snr_nli={row['snr_nli_db']:.4f} dB "
                f"snr_evm={row['center_channel_snr_evm_db']:.4f} dB "
                f"({row['elapsed_s']:.1f}s)",
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
        / "diagnostics"
        / "c57_forward_record_evm.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_by": "examples/check_dra_forward_record.py",
        "plan": args.plan,
        "configs": args.configs,
        "pump_model": args.pump_model,
        "step_m": float(args.step_m),
        "seed": int(args.seed),
        "sample_rate_factor": int(args.sample_rate_factor),
        "symbols": list(args.symbols),
        "evm_fiber_parameters": {
            "length_km": generator.WIDEBAND_FIBER_LENGTH_KM,
            "beta2_ps2_per_km": generator.WIDEBAND_BETA2_PS2_PER_KM,
            "beta3_ps3_per_km": generator.WIDEBAND_BETA3_PS3_PER_KM,
            "sps": generator.WIDEBAND_PLANS[args.plan].sps,
            "roll_off": generator.WIDEBAND_PLANS[args.plan].roll_off,
            "cdc_freq_scale": "n_full/(n_full-1), matching the full-band propagator",
        },
        "note": (
            "snr_nli_db is the in-band residual after one complex common-gain "
            "projection; center_channel_snr_evm_db is the receiver EVM SNR after "
            "demux -> common-gain alignment -> CDC -> matched RRC. Comparing their "
            "record-length trends separates a spectral-estimator/periodic-record "
            "artifact from a real propagation effect."
        ),
        "rows": rows,
    }
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
