#!/usr/bin/env python3
"""物理波形与解析 Raman ODE 的“剖面差异”诊断（只读）。

对比两侧在同一泵浦配置下的：

* 泵浦总功率剖面 ``P_p(z)``（对各自发射端归一化）与残余功率；
* 中心信道 / 带边信道的增益剖面 ``G(z)`` 与开关增益；
* 带内平均耦合 vs 中心频率耦合（解析侧逐信道、物理侧单标量）；
* 解析侧 5 参数剖面拟合的残差。

物理侧直接调用 :meth:`DistributedRamanFiber._solve_depleted_profiles`，
输入功率取参考数据 ``input_<plan>`` 波形的带内总功率，因此与
``generate_dra_reference.py`` 生成的参考逐点同口径。解析侧只读调用
``network-nlo-eval`` 的 ``hybrid_raman_paper`` 求解器，用实例方法包装记录
正向/反向积分得到的泵浦与信号剖面（不修改 network 仓）。

用法::

    PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
        examples/diagnose_dra_profile.py \
        --cases cl152:bwd_sixpumps cl152:fwd_0p20w c57:bwd_0p20w c57:fwd_0p20w \
        --output examples/dra_reference_wide/diagnostics/profile_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "examples" / "dra_reference_wide"
DEFAULT_NETWORK_REPO = Path("/home/zsig/Documents/gits/wt/mm-integrate")

#: 剖面对比用的归一化 z 采样点数（含两端）。
N_PROFILE_POINTS = 21
#: 解析侧每跨积分步数，与 ``hybrid_raman_paper`` 的 SPP 网格一致（32）。
ANALYTIC_N_STEPS = 32


def _import_network(repo: Path):
    """把 network-nlo-eval 加入 ``sys.path`` 并导入所需符号。"""
    src = repo / "src"
    if not (src / "network_nlo_eval").is_dir():
        raise SystemExit(f"{repo} does not look like network-nlo-eval")
    for path in (str(repo / "scripts"), str(src)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from network_nlo_eval.core.spectrum import SpectrumGrid  # noqa: PLC0415
    from network_nlo_eval.models.base import create_model  # noqa: PLC0415
    from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig  # noqa: PLC0415

    return SpectrumGrid, create_model, FiberSpanConfig, RamanPumpConfig


def _beta2_to_dispersion(beta2_ps2_per_km: float, wavelength_nm: float = 1550.0) -> float:
    """把 β2 (ps²/km) 换算为 D (ps/(nm·km))，与参考脚本一致。"""
    c_light = 2.99792458e8
    wavelength_m = float(wavelength_nm) * 1e-9
    beta2_s2_per_m = float(beta2_ps2_per_km) * 1e-24 / 1e3
    return float(-(2.0 * np.pi * c_light / wavelength_m**2) * beta2_s2_per_m * 1e6)


def _beta3_to_slope(
    beta3_ps3_per_km: float, dispersion_ps_nm_km: float, wavelength_nm: float = 1550.0
) -> float:
    """把 β3 (ps³/km) 换算为色散斜率 S (ps/(nm²·km))，与参考脚本一致。"""
    c_light = 2.99792458e8
    wavelength_m = float(wavelength_nm) * 1e-9
    beta3_s3_per_m = float(beta3_ps3_per_km) * 1e-36 / 1e3
    numerator = (
        beta3_s3_per_m * (2.0 * np.pi * c_light) ** 2 / wavelength_m**2
        - 2.0 * wavelength_m * float(dispersion_ps_nm_km) * 1e-6
    )
    return float(numerator / (wavelength_m**2 * 1e3))


def _fiber_config(metadata: dict, FiberSpanConfig):
    """按参考元数据构造解析侧光纤（平坦 alpha、β2 与 β3 反解斜率）。"""
    fiber = metadata["fiber"]
    wavelength_nm = float(fiber.get("reference_wavelength_nm", 1550.0))
    dispersion = _beta2_to_dispersion(float(fiber["beta2_ps2_per_km"]), wavelength_nm)
    slope = _beta3_to_slope(float(fiber["beta3_ps3_per_km"]), dispersion, wavelength_nm)
    return FiberSpanConfig(
        length_km=float(fiber["length_km"]),
        attenuation_db_km_ref=float(fiber["alpha_db_per_km"]),
        use_attenuation_polynomial=False,
        dispersion_parameter_d=dispersion,
        dispersion_slope_s=slope,
        nonlinear_index_n2=float(fiber["n2_si"]),
        effective_area_um2_ref=float(fiber["a_eff_um2"]),
    )


def _pump_specs(metadata: dict, config: str):
    """参考 JSON 的泵浦 -> ``PumpSpec`` 与解析侧配置。"""
    from physics_nlo_eval.channel import PumpSpec  # noqa: PLC0415

    entries = metadata["pump"]["configs"][config]
    specs = [
        PumpSpec(
            wavelength_nm=float(entry["wavelength_nm"]),
            power_w=float(entry["power_w"]),
            direction=str(entry["direction"]),
            attenuation_db_per_km=float(entry["attenuation_db_per_km"]),
        )
        for entry in entries
    ]
    return specs, entries


def _physics_profiles(metadata: dict, config: str, waveform: np.ndarray, coupling: str, extra: dict):
    """物理耗尽模型：返回归一化 z 网格上的泵浦/信号剖面与能量账。"""
    from physics_nlo_eval.channel import DistributedRamanFiber  # noqa: PLC0415

    specs, _ = _pump_specs(metadata, config)
    fiber = DistributedRamanFiber(
        length_m=float(metadata["fiber"]["length_km"]) * 1e3,
        alpha_db_per_km=float(metadata["fiber"]["alpha_db_per_km"]),
        beta2_ps2_per_km=float(metadata["fiber"]["beta2_ps2_per_km"]),
        beta3_ps3_per_km=float(metadata["fiber"]["beta3_ps3_per_km"]),
        pumps=specs,
        coarse_step_size_m=float(extra.get("step_m") or metadata["coarse_step_m"]),
        n2_si=0.0,
        a_eff_um2=float(metadata["fiber"]["a_eff_um2"]),
        pump_model="depleted",
        depleted_max_iterations=int(extra.get("iterations", 8)),
        depletion_coupling=coupling,
    )
    input_power_w = float(np.sum(np.mean(np.abs(waveform) ** 2, axis=1)))
    band_freqs_hz = None
    band_weights = None
    if coupling == "band_average":
        n_samples = waveform.shape[-1]
        sample_rate = float(metadata["plan"]["sample_rate_hz"])
        freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
        band_freqs_hz = float(metadata["plan"]["center_frequency_hz"]) + np.fft.fftfreq(
            n_samples, d=1.0 / sample_rate
        ) * freq_scale
        spectrum = np.mean(np.abs(np.fft.fft(waveform, axis=-1)) ** 2, axis=0)
        total = float(np.sum(spectrum))
        band_weights = spectrum / total if total > 0.0 else None
    z, profile, ps, diagnostics = fiber._solve_depleted_profiles(
        input_power_w,
        float(metadata["plan"]["center_frequency_hz"]),
        band_freqs_hz=band_freqs_hz,
        band_weights=band_weights,
    )
    # 让 on_off_gain() 能读到本次求解的剖面（forward() 之外不会自动写入）。
    fiber._depleted_pump_z = z
    fiber._depleted_pump_power = profile
    return {
        "fiber": fiber,
        "z_m": z,
        "pump_total_w": profile.sum(axis=0),
        "pump_w": profile,
        "signal_w": ps,
        "diagnostics": diagnostics,
        "pump_final_w": np.array(
            [
                float(profile[i, -1] if pump.direction == "forward" else profile[i, 0])
                for i, pump in enumerate(specs)
            ]
        ),
    }


def _analytic_profiles(
    metadata: dict,
    config: str,
    factor: int,
    SpectrumGrid,
    create_model,
    FiberSpanConfig,
    *,
    disable_pump_pump: bool = False,
):
    """解析 ODE：返回泵浦/逐信道信号剖面与收敛信息。"""
    plan = metadata["plan"]
    grid = SpectrumGrid(
        int(plan["n_channels"]),
        float(plan["channel_spacing_hz"]),
        float(plan["center_frequency_hz"]),
    )
    fiber_cfg = _fiber_config(metadata, FiberSpanConfig)
    _, entries = _pump_specs(metadata, config)
    from network_nlo_eval.network.elements import RamanPumpConfig  # noqa: PLC0415

    pumps = [
        RamanPumpConfig(
            wavelength_nm=float(entry["wavelength_nm"]),
            power_w=float(entry["power_w"]),
            direction=str(entry["direction"]),
            attenuation_db_per_km=float(entry["attenuation_db_per_km"]),
        )
        for entry in entries
    ]
    model = create_model(
        "hybrid_raman_paper",
        grid,
        fiber_cfg,
        symbol_rate_hz=float(plan["symbol_rate_hz"]),
        channel_bandwidth_source="symbol_rate",
        raman_pumps=pumps,
        pump_depletion=True,
        signal_polarizations=int(factor),
        raman_ase=False,
        spp_steps=ANALYTIC_N_STEPS,
    )
    kernel = model.kernel_for(fiber_cfg)
    solver = kernel._solver_for(float(metadata["fiber"]["length_km"]) * 1e3)
    if disable_pump_pump:
        # 消融：只关掉泵浦-泵浦拉曼转移，保留泵浦-信号耦合。
        pump_slots = np.flatnonzero(solver._is_pump)
        solver._coupling[np.ix_(pump_slots, pump_slots)] = 0.0

    recorded: dict[str, tuple] = {}
    original_forward = solver._integrate_forward
    original_backward = solver._integrate_backward

    def record_forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        recorded["forward"] = out
        return out

    def record_backward(*args, **kwargs):
        out = original_backward(*args, **kwargs)
        recorded["backward"] = out
        return out

    solver._integrate_forward = record_forward
    solver._integrate_backward = record_backward
    try:
        launch_w = float(1e-3 * 10 ** (float(plan["launch_power_dbm_per_pol"]) / 10.0))
        power_in = np.full(int(plan["n_channels"]), launch_w)
        result = solver.solve(power_in)
    finally:
        solver._integrate_forward = original_forward
        solver._integrate_backward = original_backward

    _, _, power_grid, _, samples = recorded["forward"]
    step = solver._z_m[1] - solver._z_m[0]
    sample_z = np.empty(samples.shape[0])
    sample_z[0::2] = solver._z_m[:-1] + 0.5 * step
    sample_z[1::2] = solver._z_m[1:]

    n_pumps = solver._n_pump
    pump_profile = np.zeros((n_pumps, sample_z.size))
    fwd_pump_columns = np.flatnonzero(~solver._fwd_is_signal)
    for column, pump_index in zip(fwd_pump_columns, solver._fwd_pump_original[fwd_pump_columns], strict=True):
        pump_profile[pump_index] = np.interp(sample_z, solver._z_m, power_grid[:, column])
    if solver._bw_slots.size:
        # 反向泵网格是 n_steps+1 点，插值到与信号样本相同的 z 上。
        backward_grid_s = recorded["backward"][2][::-1]
        backward_z = solver._z_m
        for row, pump_index in zip(backward_grid_s.T, solver._bw_original, strict=True):
            pump_profile[pump_index] = np.interp(sample_z, backward_z, row)

    signal_columns = np.flatnonzero(solver._fwd_is_signal)
    signal = np.zeros((int(plan["n_channels"]), samples.shape[0]))
    signal[solver._fwd_signal_original[signal_columns]] = samples[:, signal_columns].T
    # ``signal_columns`` 是 ``_fwd_slots`` 内的位置，频率要经 ``_fwd_slots`` 索引。
    signal_freqs = solver._f_hz[solver._fwd_slots[signal_columns]]

    return {
        "solver": solver,
        "z_m": sample_z,
        "pump_total_w": pump_profile.sum(axis=0),
        "signal_w": signal,
        "freqs_hz": signal_freqs,
        "launch_w": launch_w,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "power_out_w": np.asarray(result.power_out_w),
    }


def _interp(z_query: np.ndarray, z: np.ndarray, values: np.ndarray) -> np.ndarray:
    """把 ``values(z)`` 线性插值到 ``z_query``。"""
    return np.interp(z_query, z, values)


def _linear_forward(metadata: dict, waveform: np.ndarray, fiber, profiles: np.ndarray) -> np.ndarray:
    """线性传播的频域精确解（n2=0）。

    对称分步里的色散/损耗/拉曼增益都是频域对角算符，彼此对易，因此
    ``exp(A dz/2) exp(B dz) exp(A dz/2) = exp((A+B) dz)``，可以一次乘完。
    泵浦积分用与 SSFM 相同的步中点求和，结果与逐步步进一致（实测相对
    误差 ~4e-8，即 complex64 舍入）。
    """
    n_samples = waveform.shape[-1]
    sample_rate = float(metadata["plan"]["sample_rate_hz"])
    n_steps = max(1, int(np.ceil(fiber.length_m / fiber.coarse_step_size_m)))
    dz = fiber.length_m / n_steps
    midpoints = (np.arange(n_steps) + 0.5) * dz
    pump_integral = np.array(
        [float(np.sum(np.interp(midpoints, profiles["z_m"], row))) * dz for row in profiles["pump_w"]]
    )
    freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
    omega = 2 * np.pi * np.fft.fftfreq(n_samples, d=1.0 / sample_rate) * freq_scale
    beta = fiber.beta2_si * omega**2 / 2 + fiber.beta3_si * omega**3 / 6
    alpha = np.asarray(fiber.alpha_np_inv_m)
    optical_freq = float(metadata["plan"]["center_frequency_hz"]) + omega / (2 * np.pi)
    from physics_nlo_eval.channel.dra import raman_gain_spectrum  # noqa: PLC0415

    rate = np.zeros(n_samples)
    for pump, length_w_m in zip(fiber.pumps, pump_integral, strict=True):
        rate += (
            fiber.raman_peak_gain_m_per_w
            * raman_gain_spectrum(
                fiber._pump_frequency_hz(pump) - optical_freq,
                peak_shift_hz=fiber.raman_peak_shift_hz,
                fwhm_hz=fiber.raman_fwhm_hz,
            )
            * length_w_m
            / fiber.a_eff_si
        )
    transfer = np.exp((-1j * beta - alpha / 2) * fiber.length_m + 0.5 * rate)
    return np.fft.ifft(np.fft.fft(waveform, axis=-1) * transfer[None, :], axis=-1)


def _waveform_snr(metadata: dict, config: str, waveform: np.ndarray, coupling: str, physics: dict) -> dict:
    """跑一次非线性 SSFM；线性参考用频域精确解。"""
    import generate_dra_reference as generator  # noqa: PLC0415

    from physics_nlo_eval.core.signal import Signal, SignalContext  # noqa: PLC0415

    plan = metadata["plan"]
    ctx = SignalContext(
        sample_rate=float(plan["sample_rate_hz"]),
        center_freq=float(plan["center_frequency_hz"]),
        n_pol=2,
        sps=int(plan["sps"]),
        sym_rate=float(plan["symbol_rate_hz"]),
    )
    signal = Signal(waveform.astype(np.complex64), ctx)
    # 复用剖面求解用的同一个 fiber 实例：步长/泵浦口径与 profile 完全一致。
    fiber = physics["fiber"]
    fiber.n2_si = float(metadata["fiber"]["n2_si"])
    nonlinear = fiber.forward(signal)
    linear = _linear_forward(metadata, waveform, fiber, physics)
    center = int(plan["n_channels"]) // 2
    offsets = np.linspace(
        float(plan["first_channel_offset_hz"]),
        float(plan["last_channel_offset_hz"]),
        int(plan["n_channels"]),
    )
    power, nli, raw_nli, common_gain_db, _ = generator.band_nli_metrics(
        nonlinear.data,
        linear,
        float(plan["sample_rate_hz"]),
        float(offsets[center]),
        float(plan["symbol_rate_hz"]),
    )
    return {
        "center_channel_index": center,
        "signal_power_w": float(power),
        "nli_power_w": float(nli),
        "nli_power_raw_w": float(raw_nli),
        "common_gain_db": float(common_gain_db),
        "snr_nli_db": float(10 * np.log10(power / nli)) if nli > 0 else None,
        "linear_model": "exact_frequency_domain",
    }


def _analytic_snr(metadata: dict, config: str, factor: int, SpectrumGrid, create_model, FiberSpanConfig) -> dict:
    """解析侧中心信道 SNR_NLI（transmitter-plane GN），与参考对照口径一致。"""
    from network_nlo_eval.models.base import MultiSpanOpticalPath  # noqa: PLC0415
    from network_nlo_eval.network.elements import EDFAConfig, RamanPumpConfig  # noqa: PLC0415

    plan = metadata["plan"]
    grid = SpectrumGrid(
        int(plan["n_channels"]),
        float(plan["channel_spacing_hz"]),
        float(plan["center_frequency_hz"]),
    )
    fiber_cfg = _fiber_config(metadata, FiberSpanConfig)
    _, entries = _pump_specs(metadata, config)
    pumps = [
        RamanPumpConfig(
            wavelength_nm=float(entry["wavelength_nm"]),
            power_w=float(entry["power_w"]),
            direction=str(entry["direction"]),
            attenuation_db_per_km=float(entry["attenuation_db_per_km"]),
        )
        for entry in entries
    ]
    model = create_model(
        "hybrid_raman_paper",
        grid,
        fiber_cfg,
        symbol_rate_hz=float(plan["symbol_rate_hz"]),
        channel_bandwidth_source="symbol_rate",
        raman_pumps=pumps,
        pump_depletion=True,
        signal_polarizations=int(factor),
        raman_ase=False,
    )
    path = MultiSpanOpticalPath([fiber_cfg], [EDFAConfig(noise_figure_db=5.5)])
    launch = float(1e-3 * 10 ** (float(plan["launch_power_dbm_per_pol"]) / 10.0))
    result = model.evaluate_path(path, np.full(int(plan["n_channels"]), launch))
    center = int(plan["n_channels"]) // 2
    signal = float(np.asarray(result.signal_power_w)[center])
    nli = float(np.asarray(result.spm_noise_w)[center] + np.asarray(result.xpm_noise_w)[center])
    return {
        "signal_power_w": signal,
        "nli_power_w": nli,
        "snr_nli_db": float(10 * np.log10(signal / nli)) if nli > 0 else None,
    }


def _relative_difference(query_z, reference, candidate):
    """两条归一化剖面的相对差统计（max / rms）。"""
    scale = max(float(np.max(np.abs(reference))), 1e-30)
    diff = candidate - reference
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff**2))),
        "max_relative": float(np.max(np.abs(diff)) / scale),
        "rms_relative": float(np.sqrt(np.mean(diff**2)) / scale),
    }


def _case(plan: str, config: str, data_dir: Path, coupling: str, extra: dict, network) -> dict:
    """跑一个 plan/config 的剖面诊断。"""
    SpectrumGrid, create_model, FiberSpanConfig, _ = network
    metadata = json.loads((data_dir / f"{plan}.json").read_text(encoding="utf-8"))
    arrays = np.load(data_dir / f"{plan}.npz")
    waveform = arrays[f"input_{plan}"].astype(np.complex128)
    input_power_w = float(np.sum(np.mean(np.abs(waveform) ** 2, axis=1)))

    physics = _physics_profiles(metadata, config, waveform, coupling, extra)
    analytic = {}
    for factor in (1, 2):
        analytic[factor] = _analytic_profiles(
            metadata, config, factor, SpectrumGrid, create_model, FiberSpanConfig
        )
    ablation = _analytic_profiles(
        metadata,
        config,
        1,
        SpectrumGrid,
        create_model,
        FiberSpanConfig,
        disable_pump_pump=True,
    )

    # 剖面对比从解析侧第一个样本点（z = h/2）开始，避免把 z=0 与 z=h/2 当成同一点。
    analytic_step = float(metadata["fiber"]["length_km"]) * 1e3 / ANALYTIC_N_STEPS
    z_query = np.linspace(
        0.5 * analytic_step, float(metadata["fiber"]["length_km"]) * 1e3, N_PROFILE_POINTS
    )
    launch_end = -1 if physics["fiber"].pumps[0].direction == "backward" else 0
    # 泵浦剖面按发射端功率归一化（两侧都用标称功率，避免解析侧首样本不在 z=0）。
    launch_pump_w = float(sum(pump.power_w for pump in physics["fiber"].pumps))
    physics_pump_norm = physics["pump_total_w"] / launch_pump_w
    physics_signal_norm = physics["signal_w"] / physics["signal_w"][0]
    alpha_length_db = float(metadata["fiber"]["alpha_db_per_km"]) * float(
        metadata["fiber"]["length_km"]
    )
    freq_query = np.array(
        [
            float(metadata["plan"]["center_frequency_hz"]),
            float(metadata["plan"]["center_frequency_hz"])
            - 0.5 * float(metadata["plan"]["channel_spacing_hz"]) * (int(metadata["plan"]["n_channels"]) - 1),
            float(metadata["plan"]["center_frequency_hz"])
            + 0.5 * float(metadata["plan"]["channel_spacing_hz"]) * (int(metadata["plan"]["n_channels"]) - 1),
        ]
    )
    physics_on_off_db = 10 * np.log10(physics["fiber"].on_off_gain(freq_query))

    # 带内功率加权的耦合 vs 中心频率耦合：这是 "center" 近似的直接误差。
    from physics_nlo_eval.channel.dra import raman_gain_spectrum  # noqa: PLC0415

    n_samples = waveform.shape[-1]
    sample_rate = float(metadata["plan"]["sample_rate_hz"])
    freq_scale = n_samples / (n_samples - 1) if n_samples > 1 else 1.0
    band_freqs = float(metadata["plan"]["center_frequency_hz"]) + np.fft.fftfreq(
        n_samples, d=1.0 / sample_rate
    ) * freq_scale
    band_psd = np.mean(np.abs(np.fft.fft(waveform, axis=-1)) ** 2, axis=0)
    band_weights = band_psd / float(np.sum(band_psd))
    gamma_center = physics["fiber"]._pump_coupling_rates(float(metadata["plan"]["center_frequency_hz"]))
    gamma_band = np.zeros_like(gamma_center)
    for index, pump in enumerate(physics["fiber"].pumps):
        spectrum = raman_gain_spectrum(
            physics["fiber"]._pump_frequency_hz(pump) - band_freqs,
            peak_shift_hz=physics["fiber"].raman_peak_shift_hz,
            fwhm_hz=physics["fiber"].raman_fwhm_hz,
        )
        gamma_band[index] = float(
            np.sum(band_weights * spectrum) * physics["fiber"].raman_peak_gain_m_per_w / physics["fiber"].a_eff_si
        )
    coupling_section = {
        "definition": "band average = signal PSD weighted average of g_R(f_p - f)/A_eff over the waveform band",
        "pumps": [
            {
                "wavelength_nm": float(pump.wavelength_nm),
                "power_w": float(pump.power_w),
                "direction": str(pump.direction),
                "gamma_center_per_w_m": float(gamma_center[index]),
                "gamma_band_average_per_w_m": float(gamma_band[index]),
                "ratio_db": float(
                    10 * np.log10(max(gamma_band[index], 1e-300) / max(gamma_center[index], 1e-300))
                ),
            }
            for index, pump in enumerate(physics["fiber"].pumps)
        ],
        "power_weighted_ratio_db": float(
            10
            * np.log10(
                max(float(gamma_band @ np.array([p.power_w for p in physics["fiber"].pumps])), 1e-300)
                / max(float(gamma_center @ np.array([p.power_w for p in physics["fiber"].pumps])), 1e-300)
            )
        ),
    }

    record = {
        "plan": plan,
        "config": config,
        "depletion_coupling": coupling,
        "n_channels": int(metadata["plan"]["n_channels"]),
        "input_power_w": input_power_w,
        "center_frequency_hz": float(metadata["plan"]["center_frequency_hz"]),
        "depleted_max_iterations": int(extra.get("iterations", 8)),
        "coarse_step_m": float(extra.get("step_m") or metadata["coarse_step_m"]),
        "coupling": coupling_section,
        "physics": {
            "depletion_converged": bool(physics["diagnostics"]["converged"]),
            "depletion_iterations": int(physics["diagnostics"]["iterations"]),
            "pump_residual_w": physics["pump_final_w"].tolist(),
            "pump_launch_w": [float(pump.power_w) for pump in physics["fiber"].pumps],
            "pump_transferred_w": [
                float(p["transferred_to_signal_w"]) for p in physics["diagnostics"]["pumps"]
            ],
            "pump_passive_loss_w": [
                float(p["passive_loss_w"]) for p in physics["diagnostics"]["pumps"]
            ],
            "signal_out_w": float(physics["signal_w"][-1]),
            "on_off_gain_db": {
                "center": float(physics_on_off_db[0]),
                "low": float(physics_on_off_db[1]),
                "high": float(physics_on_off_db[2]),
            },
        },
        "analytic": {},
        "profiles": {
            "z_m": z_query.tolist(),
            "pump_physics": _interp(z_query, physics["z_m"], physics_pump_norm).tolist(),
            "signal_physics_center": _interp(z_query, physics["z_m"], physics_signal_norm).tolist(),
        },
        "profile_difference": {},
    }

    for factor, data in analytic.items():
        solver = data["solver"]
        freqs = data["freqs_hz"]  # 降序
        center_target = float(metadata["plan"]["center_frequency_hz"])
        center_index = int(np.argmin(np.abs(freqs - center_target)))
        launch = data["launch_w"]
        signal_norm = {
            "center": data["signal_w"][center_index] / launch,
            "low": data["signal_w"][-1] / launch,
            "high": data["signal_w"][0] / launch,
        }
        pump_norm = data["pump_total_w"] / launch_pump_w
        record["analytic"][str(factor)] = {
            "depletion_converged": data["converged"],
            "depletion_iterations": data["iterations"],
            "pump_residual_w": float(data["pump_total_w"][0 if launch_end == -1 else -1]),
            "signal_out_per_pol_w": float(data["power_out_w"][center_index]),
            "center_channel_index": center_index,
            "center_frequency_hz": float(freqs[center_index]),
            "on_off_gain_db": {
                key: float(10 * np.log10(values[-1]) + alpha_length_db)
                for key, values in signal_norm.items()
            },
        }
        record["profiles"][f"pump_analytic_factor{factor}"] = _interp(
            z_query, data["z_m"], pump_norm
        ).tolist()
        record["profiles"][f"signal_analytic_factor{factor}_center"] = _interp(
            z_query, data["z_m"], signal_norm["center"]
        ).tolist()
        record["profile_difference"][f"pump_factor{factor}"] = _relative_difference(
            z_query,
            _interp(z_query, physics["z_m"], physics_pump_norm),
            _interp(z_query, data["z_m"], pump_norm),
        )
        record["profile_difference"][f"signal_center_factor{factor}"] = _relative_difference(
            z_query,
            _interp(z_query, physics["z_m"], physics_signal_norm),
            _interp(z_query, data["z_m"], signal_norm["center"]),
        )

    # 消融：解析侧关掉泵浦-泵浦转移后，中心信道剖面离物理侧多远。
    ablate_freqs = ablation["freqs_hz"]
    ablate_center = int(np.argmin(np.abs(ablate_freqs - float(metadata["plan"]["center_frequency_hz"]))))
    ablate_norm = ablation["signal_w"][ablate_center] / ablation["launch_w"]
    record["ablation"] = {
        "analytic_factor1_without_pump_pump": {
            "pump_residual_w": float(ablation["pump_total_w"][0 if launch_end == -1 else -1]),
            "on_off_gain_center_db": float(10 * np.log10(ablate_norm[-1]) + alpha_length_db),
            "signal_center_difference": _relative_difference(
                z_query,
                _interp(z_query, physics["z_m"], physics_signal_norm),
                _interp(z_query, ablation["z_m"], ablate_norm),
            ),
        },
        "profiles": {
            "signal_analytic_factor1_without_pump_pump_center": _interp(
                z_query, ablation["z_m"], ablate_norm
            ).tolist(),
        },
    }

    if extra.get("snr"):
        reference_snr = float(
            metadata["configs"][config]["per_channel"][int(metadata["plan"]["n_channels"]) // 2][
                "snr_nli_db"
            ]
        )
        waveform_snr = _waveform_snr(metadata, config, waveform, coupling, physics)
        analytic_snr = {
            str(factor): _analytic_snr(
                metadata, config, factor, SpectrumGrid, create_model, FiberSpanConfig
            )
            for factor in (1, 2)
        }
        record["snr"] = {
            "reference_waveform_snr_nli_db": reference_snr,
            "waveform_snr_nli_db": waveform_snr["snr_nli_db"],
            "waveform_shift_vs_reference_db": float(waveform_snr["snr_nli_db"] - reference_snr),
            "analytic": analytic_snr,
            "residual_db": {
                factor: float(row["snr_nli_db"] - waveform_snr["snr_nli_db"])
                for factor, row in analytic_snr.items()
            },
        }

    return record


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--network-repo", type=Path, default=DEFAULT_NETWORK_REPO)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["cl152:bwd_sixpumps", "cl152:fwd_0p20w", "c57:bwd_0p20w", "c57:fwd_0p20w"],
    )
    parser.add_argument("--coupling", default="center", help="物理侧 depletion_coupling")
    parser.add_argument("--iterations", type=int, default=8, help="物理耗尽模型不动点迭代上限")
    parser.add_argument("--step-m", type=float, default=None, help="覆盖物理侧 SSFM 步长 (m)")
    parser.add_argument("--snr", action="store_true", help="额外跑一次 SSFM，给出中心信道 SNR")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    network = _import_network(args.network_repo)
    cases = []
    for tag in args.cases:
        plan, config = tag.split(":", 1)
        row = _case(
            plan,
            config,
            args.data_dir,
            args.coupling,
            {"iterations": args.iterations, "snr": args.snr, "step_m": args.step_m},
            network,
        )
        cases.append(row)
        physics = row["physics"]
        print(
            f"{tag:22s} phys pump residual {np.sum(physics['pump_residual_w']):.6g} W "
            f"signal_out {physics['signal_out_w']:.6g} W conv={physics['depletion_converged']}"
        )
        for factor in ("1", "2"):
            ana = row["analytic"][factor]
            print(
                f"    factor={factor} conv={ana['depletion_converged']} "
                f"pump_residual {ana['pump_residual_w']:.6g} W "
                f"signal_out {ana['signal_out_per_pol_w']:.6g} W"
            )
            print(
                f"      on-off dB phys {row['physics']['on_off_gain_db']} "
                f"ana {ana['on_off_gain_db']}"
            )
        if "snr" in row:
            snr = row["snr"]
            print(
                f"    SNR ref {snr['reference_waveform_snr_nli_db']:.3f} "
                f"wave {snr['waveform_snr_nli_db']:.3f} "
                f"(shift {snr['waveform_shift_vs_reference_db']:+.3f}) "
                f"residual f1 {snr['residual_db']['1']:+.3f} f2 {snr['residual_db']['2']:+.3f}"
            )

    document = {
        "generated_by": "examples/diagnose_dra_profile.py (read-only call to network-nlo-eval)",
        "network_repo": str(args.network_repo),
        "note": "Profiles are normalised at the pump launch end; analytic signal profiles are per-polarisation.",
        "cases": {f"{row['plan']}:{row['config']}": row for row in cases},
    }
    output = args.output or (args.data_dir / "diagnostics" / "profile_diagnosis.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
