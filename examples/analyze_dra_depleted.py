#!/usr/bin/env python3
"""分析泵浦耗尽参考数据集。

用法::

    python examples/analyze_dra_depleted.py
    python examples/analyze_dra_depleted.py --json analysis.json

读取 ``examples/dra_reference/dra_reference_depleted.json``（以及可选的
``dra_reference.json``），输出三类诊断：

1. 单信道 SPM 探针：无泵 vs 前向 0.2 W 的 NLI SNR 变化；
2. 3ch 前向 0.2 W 发射功率扫描的 ``NLI ∝ P^k`` 拟合指数；
3. 耗尽 vs 未耗尽参考的开关增益与 SNR 差异。

脚本只使用物理侧数据，不依赖 ``network-nlo-eval`` 解析模型；解析残差对照
需要用该仓的 ``scripts/dra_waveform_compare.py``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REFERENCE_DIR = Path(__file__).resolve().parent / "dra_reference"
DEPLETED_JSON = REFERENCE_DIR / "dra_reference_depleted.json"
BASE_JSON = REFERENCE_DIR / "dra_reference.json"


def _center_channel(metadata: dict, tag: str, config: str) -> dict:
    """返回数据集中某个配置的中心信道指标。"""
    dataset = metadata["datasets"][tag]
    return dataset["configs"][config]["per_channel"][dataset["center_channel_index"]]


def _snr_nli_db(metadata: dict, tag: str, config: str) -> float:
    """中心信道 NLI SNR (dB)。"""
    return float(_center_channel(metadata, tag, config)["snr_nli_db"])


def _p3_fit(metadata: dict) -> dict:
    """对发射功率扫描做 log(NLI) vs log(P) 最小二乘拟合。"""
    sweep = metadata["power_sweep"]
    config = sweep["config"]
    launch_mw = []
    nli_w = []
    for tag in sweep["dataset_tags"]:
        launch_mw.append(float(metadata["datasets"][tag]["launch_power_dbm_per_pol"]))
        nli_w.append(float(_center_channel(metadata, tag, config)["nli_power_w"]))
    log_power = np.log10(10 ** (np.array(launch_mw) / 10.0))
    log_nli = np.log10(np.array(nli_w))
    slope, intercept = np.polyfit(log_power, log_nli, 1)
    return {
        "config": config,
        "launch_power_mw_per_pol": list(10 ** (np.array(launch_mw) / 10.0)),
        "nli_power_w": nli_w,
        "slope": float(slope),
        "intercept": float(intercept),
        "segments": [float(value) for value in np.diff(log_nli) / np.diff(log_power)],
    }


def main() -> None:
    """打印诊断并把结果写成 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depleted-json", type=Path, default=DEPLETED_JSON)
    parser.add_argument("--base-json", type=Path, default=BASE_JSON)
    parser.add_argument("--json", type=Path, default=None, help="可选的输出 JSON 路径")
    args = parser.parse_args()

    depleted = json.loads(args.depleted_json.read_text(encoding="utf-8"))
    report: dict = {
        "depleted": str(args.depleted_json),
        "pump_model": depleted["model"]["pump_model"],
    }

    # 1. 单信道 SPM 探针。
    probe = {
        "snr_nopump_db": _snr_nli_db(depleted, "1ch", "nopump"),
        "snr_forward_0p20w_db": _snr_nli_db(depleted, "1ch", "fwd_0p20w"),
        "snr_backward_0p20w_db": _snr_nli_db(depleted, "1ch", "bwd_0p20w"),
    }
    probe["forward_enhancement_db"] = probe["snr_nopump_db"] - probe["snr_forward_0p20w_db"]
    probe["backward_enhancement_db"] = probe["snr_nopump_db"] - probe["snr_backward_0p20w_db"]
    report["one_channel_probe"] = probe

    # 2. 功率扫描的 P^k 拟合；无泵对照组给出提取方法的自检。
    report["p3_fit_forward_0p20w"] = _p3_fit(depleted)

    # 3. 耗尽 vs 未耗尽（仅当默认数据集存在）。
    if args.base_json.exists():
        base = json.loads(args.base_json.read_text(encoding="utf-8"))
        comparison = {}
        for tag in ("3ch", "5ch"):
            if tag not in base["datasets"] or tag not in depleted["datasets"]:
                continue
            for config in depleted["datasets"][tag]["configs"]:
                if config not in base["datasets"][tag]["configs"]:
                    continue
                comparison[f"{tag}/{config}"] = {
                    "on_off_gain_db_undepleted": _center_channel(base, tag, config)["on_off_gain_db"],
                    "on_off_gain_db_depleted": _center_channel(depleted, tag, config)["on_off_gain_db"],
                    "snr_nli_db_undepleted": _snr_nli_db(base, tag, config),
                    "snr_nli_db_depleted": _snr_nli_db(depleted, tag, config),
                }
        report["depleted_vs_undepleted"] = comparison

    # 控制台摘要。
    print(f"pump model: {report['pump_model']}")
    print("\n== 1ch SPM probe ==")
    print(
        f"  SNR_NLI nopump {probe['snr_nopump_db']:.3f} dB, "
        f"fwd 0.2 W {probe['snr_forward_0p20w_db']:.3f} dB, "
        f"bwd 0.2 W {probe['snr_backward_0p20w_db']:.3f} dB"
    )
    print(
        f"  forward enhancement {probe['forward_enhancement_db']:.3f} dB, "
        f"backward {probe['backward_enhancement_db']:.3f} dB"
    )

    fit = report["p3_fit_forward_0p20w"]
    print("\n== NLI scaling (3ch, fwd 0.2 W) ==")
    for power_mw, nli_w in zip(fit["launch_power_mw_per_pol"], fit["nli_power_w"], strict=True):
        print(f"  launch {power_mw:.2f} mW/pol: NLI {nli_w:.4e} W")
    slopes = ", ".join(f"{value:.3f}" for value in fit["segments"])
    print(f"  local slopes [{slopes}], fit {fit['slope']:.3f}")

    if "depleted_vs_undepleted" in report:
        print("\n== depleted vs undepleted ==")
        for key, row in report["depleted_vs_undepleted"].items():
            print(
                f"  {key:20s} on/off {row['on_off_gain_db_undepleted']:6.3f} -> "
                f"{row['on_off_gain_db_depleted']:6.3f} dB, SNR_NLI "
                f"{row['snr_nli_db_undepleted']:6.3f} -> {row['snr_nli_db_depleted']:6.3f} dB"
            )

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
