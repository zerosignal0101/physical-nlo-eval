#!/usr/bin/env python3
"""用 network-nlo-eval 的 hybrid_raman_paper 只读对照全 WDM 波形参考。

脚本不修改 network-nlo-eval，只把它的 ``src`` 加入 ``sys.path`` 后调用
``SpectrumGrid`` / ``create_model``。对照口径：

* 与波形相同的平坦 alpha、beta2（``dispersion_slope_s=0``）、n2、A_eff；
* ``symbol_rate_hz=64e9``，``channel_bandwidth_source="symbol_rate"``，
  即 NLI 分析带宽 = 符号率（与波形元数据一致）；
* 只比较 nopump 的中心信道；解析侧 ``pump_depletion=True``（无泵时无影响）。

用法::

    python examples/check_dra_wideband_analytic.py
    python examples/check_dra_wideband_analytic.py \
        --network-repo /path/to/mm-integrate \
        --output /tmp/wide_analytic.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_NETWORK_REPO = Path("/home/zsig/Documents/gits/wt/mm-integrate")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dra_reference_wide"


def _import_network(repo: Path):
    """把 network-nlo-eval 的 src 加入路径并导入所需符号。"""
    src = repo / "src"
    if not (src / "network_nlo_eval").is_dir():
        raise SystemExit(
            f"{repo} does not look like network-nlo-eval (missing src/network_nlo_eval)"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    return SpectrumGrid, MultiSpanOpticalPath, create_model, EDFAConfig, FiberSpanConfig


C_LIGHT_M_S = 2.99792458e8


def _beta2_to_dispersion(beta2_ps2_per_km: float, wavelength_nm: float = 1550.0) -> float:
    """把 β2 (ps²/km) 换算为 D (ps/(nm·km))。"""
    wavelength_m = float(wavelength_nm) * 1e-9
    beta2_s2_per_m = float(beta2_ps2_per_km) * 1e-24 / 1e3
    return float(-(2.0 * np.pi * C_LIGHT_M_S / wavelength_m**2) * beta2_s2_per_m * 1e6)


def _beta3_to_slope(
    beta3_ps3_per_km: float, dispersion_ps_nm_km: float, wavelength_nm: float = 1550.0
) -> float:
    """把 β3 (ps³/km) 换算成解析侧色散斜率 S (ps/(nm²·km))。

    与 network-nlo-eval 的 ``_calc_beta3_from_s_d`` 互逆，确保解析侧
    β3 与波形参考一致（波形用的是 1e-4 ps³/km，等价于很小的 β3）。
    """
    wavelength_m = float(wavelength_nm) * 1e-9
    beta3_s3_per_m = float(beta3_ps3_per_km) * 1e-36 / 1e3
    numerator = (
        beta3_s3_per_m * (2.0 * np.pi * C_LIGHT_M_S) ** 2 / wavelength_m**2
        - 2.0 * wavelength_m * float(dispersion_ps_nm_km) * 1e-6
    )
    return float(numerator / (wavelength_m**2 * 1e3))


def _fiber_from_metadata(metadata: dict, FiberSpanConfig):
    """按波形元数据构造解析侧光纤（平坦 alpha/beta2，β3 匹配）。"""
    fiber = metadata["fiber"]
    wavelength_nm = float(fiber.get("reference_wavelength_nm", 1550.0))
    if "beta2_ps2_per_km" in fiber:
        dispersion = _beta2_to_dispersion(float(fiber["beta2_ps2_per_km"]), wavelength_nm)
    else:
        dispersion = 17.0
    if "beta3_ps3_per_km" in fiber:
        slope = _beta3_to_slope(float(fiber["beta3_ps3_per_km"]), dispersion, wavelength_nm)
    else:
        slope = 0.06
    return FiberSpanConfig(
        length_km=float(fiber["length_km"]),
        attenuation_db_km_ref=float(fiber["alpha_db_per_km"]),
        use_attenuation_polynomial=False,
        dispersion_parameter_d=dispersion,
        dispersion_slope_s=slope,
        nonlinear_index_n2=float(fiber["n2_si"]),
        effective_area_um2_ref=float(fiber["a_eff_um2"]),
    )


def _pump_list(metadata: dict, config_name: str, RamanPumpConfig):
    """把参考 JSON 的泵浦描述转成解析侧配置。"""
    pumps = []
    for pump in metadata["pump"]["configs"][config_name]:
        pumps.append(
            RamanPumpConfig(
                wavelength_nm=float(pump["wavelength_nm"]),
                power_w=float(pump["power_w"]),
                direction=pump["direction"],
                attenuation_db_per_km=float(pump["attenuation_db_per_km"]),
            )
        )
    return pumps


def _run_analytic(
    metadata: dict,
    *,
    SpectrumGrid,
    MultiSpanOpticalPath,
    create_model,
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
    config_name: str = "nopump",
) -> dict:
    """返回中心信道的解析 SNR_NLI 与 NLI 功率。"""
    plan = metadata["plan"]
    n_channels = int(plan["n_channels"])
    grid = SpectrumGrid(
        n_channels,
        float(plan["channel_spacing_hz"]),
        float(plan["center_frequency_hz"]),
    )
    fiber = _fiber_from_metadata(metadata, FiberSpanConfig)
    pumps = _pump_list(metadata, config_name, RamanPumpConfig)
    options = {"pump_depletion": True, "raman_ase": False}
    if pumps:
        options["raman_pumps"] = pumps
    model = create_model(
        "hybrid_raman_paper",
        grid,
        fiber,
        symbol_rate_hz=float(plan["symbol_rate_hz"]),
        channel_bandwidth_source="symbol_rate",
        **options,
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=5.5)])
    launch_power_w = float(10.0 ** (float(plan["launch_power_dbm_per_pol"]) / 10.0) * 1e-3)
    result = model.evaluate_path(path, np.ones(n_channels) * launch_power_w)
    nli = np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)
    center = n_channels // 2
    signal = float(np.asarray(result.signal_power_w)[center])
    return {
        "model": "hybrid_raman_paper",
        "config": config_name,
        "pump_depletion": True,
        "channel_bandwidth_source": "symbol_rate",
        "center_channel_index": center,
        "signal_power_w": signal,
        "nli_power_w": float(nli[center]),
        "snr_nli_db": float(10.0 * np.log10(signal / float(nli[center]))),
    }


def compare_plan(
    metadata_path: Path,
    *,
    SpectrumGrid,
    MultiSpanOpticalPath,
    create_model,
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
) -> dict:
    """对一个计划 JSON 做 nopump 中心信道对照。"""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    center = int(metadata["plan"]["n_channels"]) // 2
    waveform = metadata["configs"]["nopump"]["per_channel"][center]
    analytic = _run_analytic(
        metadata,
        SpectrumGrid=SpectrumGrid,
        MultiSpanOpticalPath=MultiSpanOpticalPath,
        create_model=create_model,
        EDFAConfig=EDFAConfig,
        FiberSpanConfig=FiberSpanConfig,
        RamanPumpConfig=RamanPumpConfig,
    )
    residual = float(analytic["snr_nli_db"] - waveform["snr_nli_db"])
    return {
        "plan": metadata["plan"]["name"],
        "n_channels": metadata["plan"]["n_channels"],
        "waveform_snr_nli_db": float(waveform["snr_nli_db"]),
        "waveform_nli_w": float(waveform["nli_power_w"]),
        "coarse_step_m": metadata["coarse_step_m"],
        "step_converged": metadata["step_converged"],
        "analytic": analytic,
        "residual_db": residual,
        "residual_definition": "analytic_snr_nli_db - waveform_snr_nli_db; positive means analytic is more optimistic",
        "note": (
            "Analytic side uses the same flat alpha=0.162 dB/km, beta2 and beta3-derived "
            "dispersion slope, n2 and A_eff as the waveform; only the nopump case is compared."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="全 WDM 数据目录")
    parser.add_argument(
        "--network-repo",
        type=Path,
        default=DEFAULT_NETWORK_REPO,
        help="network-nlo-eval 工作树（只读调用）",
    )
    parser.add_argument("--output", type=Path, default=None, help="输出 JSON")
    parser.add_argument(
        "--plans",
        nargs="+",
        default=["c57", "cl152"],
        help="要对照的计划文件名（不含扩展名）",
    )
    args = parser.parse_args(argv)

    (
        SpectrumGrid,
        MultiSpanOpticalPath,
        create_model,
        EDFAConfig,
        FiberSpanConfig,
    ) = _import_network(args.network_repo)
    from network_nlo_eval.network.elements import RamanPumpConfig

    rows = []
    for name in args.plans:
        metadata_path = args.data_dir / f"{name}.json"
        if not metadata_path.exists():
            raise SystemExit(f"missing {metadata_path}; run generate_dra_reference.py --wideband first")
        rows.append(
            compare_plan(
                metadata_path,
                SpectrumGrid=SpectrumGrid,
                MultiSpanOpticalPath=MultiSpanOpticalPath,
                create_model=create_model,
                EDFAConfig=EDFAConfig,
                FiberSpanConfig=FiberSpanConfig,
                RamanPumpConfig=RamanPumpConfig,
            )
        )

    output = args.output or (args.data_dir / "analytic_compare.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "generated_by": "examples/check_dra_wideband_analytic.py (read-only call to network-nlo-eval)",
        "network_repo": str(args.network_repo),
        "plans": {row["plan"]: row for row in rows},
    }
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'plan':>8} {'n':>4} {'step_m':>10} {'wave':>9} {'analytic':>9} {'residual':>9}")
    for row in rows:
        print(
            f"{row['plan']:>8} {row['n_channels']:>4} {row['coarse_step_m']:>10.6g} "
            f"{row['waveform_snr_nli_db']:>9.3f} {row['analytic']['snr_nli_db']:>9.3f} "
            f"{row['residual_db']:>+9.3f}"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
