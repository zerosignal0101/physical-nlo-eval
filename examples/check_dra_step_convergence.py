#!/usr/bin/env python3
"""检查 DRA 参考波形的 SSFM 步长收敛性。

用参考数据集里的 ``input_*`` 波形，在 500/250/125/62.5/31.25 m 五档步长下
重跑无泵与 7ch 泵浦配置，输出中心/均值/最小 SNR、历史 500 m 的伪影
（``artifact_500m_db``）以及最后两档步长的增量。若最后两档的中心信道
SNR 差 >= ``--tolerance``（默认 0.01 dB），脚本以非零状态退出。

用法::

    python examples/check_dra_step_convergence.py
    python examples/check_dra_step_convergence.py \
        --npz examples/dra_reference_converged/dra_reference.npz \
        --output /tmp/dra_step_convergence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 复用生成器的光纤构造与 NLI 定义，避免两套口径漂移。
_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from generate_dra_reference import (  # noqa: E402
    CONFIGS,
    FIBER_LENGTH_KM,
    NLI_ANALYSIS_BANDWIDTH_HZ,
    band_nli_metrics,
    make_fiber,
    walkoff_per_step_symbols,
)

from physics_nlo_eval.core.signal import Signal, SignalContext  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = REPO_ROOT / "examples" / "dra_reference" / "dra_reference.npz"
DEFAULT_OUTPUT = REPO_ROOT / "examples" / "dra_reference_converged" / "step_convergence.json"
DEFAULT_STEPS_M = (500.0, 250.0, 125.0, 62.5, 31.25)
DEFAULT_TOLERANCE_DB = 0.01
#: (数据集 tag, 配置名)；覆盖 3/5/7ch 无泵与 7ch 前/后向 0.2 W。
DEFAULT_PROBES = (
    ("3ch", "nopump"),
    ("5ch", "nopump"),
    ("7ch", "nopump"),
    ("7ch", "fwd_0p20w"),
    ("7ch", "bwd_0p20w"),
)


def load_reference(npz_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """读取参考 JSON 元数据与 npz 中的 ``input_*`` 波形。"""
    json_path = npz_path.with_suffix(".json")
    if not npz_path.exists():
        raise FileNotFoundError(f"missing reference waveforms: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"missing reference metadata: {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as arrays:
        inputs = {
            key: np.asarray(arrays[key]) for key in arrays.files if key.startswith("input_")
        }
    return metadata, inputs


def _channel_metrics(
    nonlinear_data: np.ndarray,
    linear_data: np.ndarray,
    sample_rate_hz: float,
    offsets_hz: np.ndarray,
) -> list[dict]:
    """逐信道计算 NLI SNR（与参考 JSON 同一套单复标量拟合）。"""
    channels = []
    for index, offset in enumerate(offsets_hz):
        power, nli_power, raw_nli_power, _, _ = band_nli_metrics(
            nonlinear_data,
            linear_data,
            sample_rate_hz,
            float(offset),
            NLI_ANALYSIS_BANDWIDTH_HZ,
        )
        snr_db = float(10 * np.log10(power / nli_power)) if nli_power > 0 else None
        channels.append(
            {
                "channel": index,
                "freq_offset_hz": float(offset),
                "signal_power_w": power,
                "nli_power_w": nli_power,
                "nli_power_raw_w": raw_nli_power,
                "snr_nli_db": snr_db,
            }
        )
    return channels


def probe_config(
    config_name: str,
    input_data: np.ndarray,
    dataset_meta: dict,
    offsets_hz: np.ndarray,
    steps_m: tuple[float, ...] = DEFAULT_STEPS_M,
) -> dict:
    """对同一输入按不同步长重跑一个配置，返回逐信道 SNR 序列。"""
    sample_rate_hz = float(dataset_meta["sample_rate_hz"])
    signal_context = SignalContext(
        sample_rate=sample_rate_hz,
        center_freq=193.1e12,
        n_pol=int(input_data.shape[0]),
        sps=4,
        sym_rate=sample_rate_hz / 4.0,
    )
    signal = Signal(data=input_data.astype(np.complex128), ctx=signal_context)
    offsets = np.asarray(offsets_hz, dtype=np.float64)
    pumps = CONFIGS[config_name]

    series = []
    for step_m in steps_m:
        nonlinear = make_fiber(pumps, nonlinear=True, coarse_step_m=step_m).forward(signal)
        linear = make_fiber(pumps, nonlinear=False, coarse_step_m=step_m).forward(signal)
        channels = _channel_metrics(nonlinear.data, linear.data, sample_rate_hz, offsets)
        snr = np.asarray([row["snr_nli_db"] for row in channels], dtype=np.float64)
        series.append(
            {
                "step_m": float(step_m),
                "n_steps": int(np.ceil(FIBER_LENGTH_KM * 1e3 / step_m)),
                "center_snr_db": float(snr[len(snr) // 2]),
                "mean_snr_db": float(snr.mean()),
                "min_snr_db": float(snr.min()),
                "center_nli_w": float(channels[len(channels) // 2]["nli_power_w"]),
                "walkoff_per_step_symbols": walkoff_per_step_symbols(
                    float(np.max(np.abs(offsets))), step_m
                ),
                "per_channel": channels,
            }
        )

    result = {
        "config": config_name,
        "steps": series,
        "last_increment_db": float(
            series[-1]["center_snr_db"] - series[-2]["center_snr_db"]
        ),
    }
    if series[0]["step_m"] == 500.0:
        result["artifact_500m_db"] = float(
            series[0]["center_snr_db"] - series[-1]["center_snr_db"]
        )
    return result


def run_probes(
    npz_path: Path = DEFAULT_NPZ,
    probes: tuple[tuple[str, str], ...] = DEFAULT_PROBES,
    steps_m: tuple[float, ...] = DEFAULT_STEPS_M,
) -> dict:
    """运行全部探针并返回可序列化的报告。"""
    if len(steps_m) < 2:
        raise ValueError("at least two step sizes are required")
    if any(step <= 0 for step in steps_m):
        raise ValueError("step sizes must be positive")

    metadata, inputs = load_reference(npz_path)
    report: dict = {
        "generated_by": "examples/check_dra_step_convergence.py",
        "npz": str(npz_path),
        "steps_m": [float(step) for step in steps_m],
        "probes": {},
    }
    for tag, config_name in probes:
        dataset = metadata["datasets"][tag]
        input_key = dataset["input_key"]
        if input_key not in inputs:
            raise KeyError(f"{npz_path} has no {input_key!r}")
        with np.load(npz_path) as arrays:
            offsets = np.asarray(arrays[dataset["offsets_key"]])
        report["probes"].setdefault(tag, {})[config_name] = probe_config(
            config_name, inputs[input_key], dataset, offsets, steps_m
        )

    increments = [
        (tag, config_name, probe["last_increment_db"])
        for tag, configs in report["probes"].items()
        for config_name, probe in configs.items()
    ]
    report["summary"] = {
        "max_abs_last_increment_db": max(abs(value) for _, _, value in increments),
        "last_increments_db": {
            f"{tag}/{config_name}": value for tag, config_name, value in increments
        },
    }
    report["summary"]["converged"] = (
        report["summary"]["max_abs_last_increment_db"] < DEFAULT_TOLERANCE_DB
    )
    return report


def failed_increments(report: dict, tolerance_db: float) -> list[tuple[str, str, float]]:
    """返回最后一档步长增量超过容差的 (tag, config, increment)。"""
    failures = []
    for tag, configs in report["probes"].items():
        for config_name, probe in configs.items():
            increment = float(probe["last_increment_db"])
            if abs(increment) >= tolerance_db:
                failures.append((tag, config_name, increment))
    if "summary" in report:
        report["summary"]["converged"] = not failures
        report["summary"]["tolerance_db"] = float(tolerance_db)
    return failures


def main(argv: list[str] | None = None) -> int:
    """命令行入口；未收敛时返回 1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="参考数据 npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_DB,
        help="最后两档步长的中心 SNR 容差 (dB)",
    )
    args = parser.parse_args(argv)

    report = run_probes(args.npz)
    failures = failed_increments(report, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = report["summary"]
    print(f"steps: {report['steps_m']}")
    for tag, configs in report["probes"].items():
        for config_name, probe in configs.items():
            seq = " -> ".join(
                f"{row['center_snr_db']:.4f}" for row in probe["steps"]
            )
            print(
                f"{tag}/{config_name}: {seq} dB "
                f"(last increment {probe['last_increment_db']:+.4f} dB, "
                f"500m artifact {probe.get('artifact_500m_db', float('nan')):+.4f} dB)"
            )
    print(f"wrote {args.output}")

    if failures:
        print(
            f"FAIL: last-step increment exceeds {args.tolerance} dB in "
            f"{len(failures)} probe(s): {failures}",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: max |last increment| = {summary['max_abs_last_increment_db']:.4f} dB "
        f"< {args.tolerance} dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
