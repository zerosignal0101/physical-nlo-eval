"""分布式拉曼放大 (DRA) 波形模型的回归测试。

覆盖：

* 无泵 DRA 与现有 NumPy ``FiberChannel`` 逐位一致；
* 泵浦功率单调性、前向/后向剖面差异；
* 未耗尽开关增益与 lumped ``RamanPump`` 的频率形状一致；
* 可选 ASE 与 EDFA ASE 分离记录；
* 已提交的小规模 3/5 信道参考数据集 schema 与 SNR 数据点；
* 泵浦耗尽模型：与旧数据逐位兼容、能量/光子账、信道数依赖、
  后向泵低信号功率极限。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c

from physics_nlo_eval.channel import (
    DEPLETION_MODELS,
    DistributedRamanFiber,
    FiberChannel,
    PumpSpec,
    RamanPump,
    raman_gain_spectrum,
    signal_spectrum_bins,
)
from physics_nlo_eval.core.signal import Signal, SignalContext

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "examples" / "dra_reference"
CONVERGED_DIR = REPO_ROOT / "examples" / "dra_reference_converged"
WIDEBAND_DIR = REPO_ROOT / "examples" / "dra_reference_wide"
GENERATOR_PATH = REPO_ROOT / "examples" / "generate_dra_reference.py"
STEP_CHECKER_PATH = REPO_ROOT / "examples" / "check_dra_step_convergence.py"
CONVERGENCE_CHECKER_PATH = REPO_ROOT / "examples" / "check_dra_wideband_convergence.py"
EXAMPLES_DIR = REPO_ROOT / "examples"


def _signal(
    n_samples: int = 512,
    *,
    sample_rate_hz: float = 100e9,
    center_freq_hz: float = 193.1e12,
    n_pol: int = 2,
    seed: int = 0,
    power_w: float = 1e-3,
) -> Signal:
    """构造双偏振带限随机光信号，每偏振功率 ``power_w``。"""
    rng = np.random.default_rng(seed)
    data = (rng.normal(size=(n_pol, n_samples)) + 1j * rng.normal(size=(n_pol, n_samples))).astype(np.complex128)
    data *= np.sqrt(power_w / np.mean(np.abs(data) ** 2, axis=1, keepdims=True))
    ctx = SignalContext(
        sample_rate=sample_rate_hz,
        center_freq=center_freq_hz,
        n_pol=n_pol,
        sps=4,
        sym_rate=sample_rate_hz / 4,
    )
    return Signal(data, ctx)


def _fiber_kwargs(**overrides: float) -> dict:
    """典型 100 km SMF 参数，可按测试覆盖。"""
    kwargs = {
        "length_m": 100e3,
        "alpha_db_per_km": 0.2,
        "beta2_ps2_per_km": -20.0,
        "beta3_ps3_per_km": 0.1e-3,
        "coarse_step_size_m": 500.0,
        "n2_si": 2.6e-20,
        "a_eff_um2": 80.0,
        "d_pmd_ps_per_sqrt_km": 0.0,
    }
    kwargs.update(overrides)
    return kwargs


def _output_power_w(signal: Signal) -> float:
    """返回双偏振平均功率 (W)。"""
    return float(np.mean(np.abs(signal.data) ** 2))


def test_no_pump_is_bit_exact_with_numpy_fiber_channel() -> None:
    """无泵 DRA 必须与 FiberChannel 的 NumPy 路径逐位一致。"""
    signal = _signal()
    common = _fiber_kwargs(length_m=20e3)

    reference_fiber = FiberChannel(compute_ctx_name="numpy", **common)
    reference_fiber.compute_ctx = None  # 强制 NumPy SSFM
    reference = reference_fiber.forward(signal)

    dra = DistributedRamanFiber(pumps=None, **common)
    output = dra.forward(signal)

    assert dra.compute_ctx is None
    assert output.ctx.meta["fiber_solver"] == "numpy-ssfm-dra"
    assert output.ctx.meta["dra_n_pumps"] == 0
    assert np.array_equal(output.data, reference.data)


def test_zero_power_pump_is_also_bit_exact() -> None:
    """0 W 泵浦的增益因子严格为 1，不应改变任何比特。"""
    signal = _signal()
    common = _fiber_kwargs(length_m=20e3)

    reference_fiber = FiberChannel(compute_ctx_name="numpy", **common)
    reference_fiber.compute_ctx = None
    reference = reference_fiber.forward(signal)

    output = DistributedRamanFiber(pumps=PumpSpec(1450.0, 0.0, "forward"), **common).forward(signal)
    assert np.array_equal(output.data, reference.data)


def test_pump_power_monotonic_and_direction_profiles() -> None:
    """信号输出功率随泵浦功率单调增加；前向/后向泵剖面不同但积分增益相同。"""
    signal = _signal(n_samples=1024)
    common = _fiber_kwargs(length_m=20e3, n2_si=0.0, beta2_ps2_per_km=0.0, beta3_ps3_per_km=0.0)
    no_pump = _output_power_w(DistributedRamanFiber(pumps=None, **common).forward(signal))

    powers = {}
    for direction in ("forward", "backward"):
        values = []
        for pump_power in (0.10, 0.20):
            fiber = DistributedRamanFiber(
                pumps=PumpSpec(1450.0, pump_power, direction),
                **common,
            )
            values.append(_output_power_w(fiber.forward(signal)))
        powers[direction] = values
        assert values[0] > no_pump
        assert values[1] > values[0]

    # 未耗尽近似下，总开关增益只取决于泵浦积分功率，前后向相同。
    forward_gain = DistributedRamanFiber(pumps=PumpSpec(1450.0, 0.2, "forward"), **common).on_off_gain(
        193.1e12
    )
    backward_gain = DistributedRamanFiber(pumps=PumpSpec(1450.0, 0.2, "backward"), **common).on_off_gain(
        193.1e12
    )
    assert np.isclose(forward_gain, backward_gain, rtol=1e-12)

    # 局部功率剖面必须不同：前向泵在输入端最强，后向泵在输出端最强。
    z = np.linspace(0.0, 20e3, 11)
    forward = DistributedRamanFiber(pumps=PumpSpec(1450.0, 0.2, "forward"), **common)
    backward = DistributedRamanFiber(pumps=PumpSpec(1450.0, 0.2, "backward"), **common)
    forward_profile = forward.pump_power_w(z)[0]
    backward_profile = backward.pump_power_w(z)[0]
    assert forward_profile[0] > forward_profile[-1]
    assert backward_profile[-1] > backward_profile[0]
    assert not np.allclose(forward_profile, backward_profile)


def test_on_off_gain_matches_lumped_raman_pump_shape() -> None:
    """未耗尽 DRA 开关增益的频率形状与 lumped RamanPump 一致。"""
    frequencies = 193.1e12 + np.linspace(-200e9, 200e9, 41)
    for direction in ("forward", "backward"):
        pump = PumpSpec(1450.0, 0.2, direction)
        dra = DistributedRamanFiber(pumps=pump, **_fiber_kwargs(n2_si=0.0))
        lumped = RamanPump(pump, 100.0, add_noise=False, max_gain_db=100.0)

        dra_gain_db = 10 * np.log10(dra.on_off_gain(frequencies))
        lumped_gain_db = 10 * np.log10(lumped._gain(frequencies))
        assert np.allclose(dra_gain_db, lumped_gain_db, atol=1e-9)
        assert dra_gain_db.max() > 10.0


def test_metadata_records_dra_configuration() -> None:
    """输出元数据应记录泵浦、模型、开关增益和 ISRS 处理方式。"""
    signal = _signal()
    common = _fiber_kwargs(length_m=20e3)
    dra = DistributedRamanFiber(
        pumps=[PumpSpec(1450.0, 0.1, "forward"), PumpSpec(1450.0, 0.05, "backward")],
        **common,
    )
    output = dra.forward(signal)
    meta = output.ctx.meta
    assert meta["fiber_solver"] == "numpy-ssfm-dra"
    assert meta["dra_model"] == "undepleted"
    assert meta["dra_n_pumps"] == 2
    assert len(meta["dra_pumps"]) == 2
    assert meta["dra_channel_isrs"] == "not_modelled"
    assert meta["dra_on_off_gain_db_max"] > meta["dra_on_off_gain_db_min"]


def test_raman_spectrum_peak_and_negative_shift() -> None:
    """Gaussian 拉曼谱在峰值处为 1，负频移侧趋于零。"""
    assert np.isclose(raman_gain_spectrum(13.2e12), 1.0)
    assert raman_gain_spectrum(-13.2e12) < 1e-6
    assert raman_gain_spectrum(0.0) < 1e-6


def test_optional_ase_is_separate_and_reproducible() -> None:
    """可选 ASE 与 EDFA ASE 分开记录，且相同 seed 可复现。"""
    signal = _signal(n_samples=1024)
    common = _fiber_kwargs(length_m=20e3, n2_si=0.0)

    without_ase = DistributedRamanFiber(pumps=PumpSpec(1450.0, 0.2, "forward"), add_ase=False, **common).forward(signal)
    assert without_ase.ctx.meta["dra_ase_model"] == "none"
    assert without_ase.ctx.meta["dra_ase_power_per_pol_w"] == 0.0

    first = DistributedRamanFiber(
        pumps=PumpSpec(1450.0, 0.2, "forward"), add_ase=True, seed=7, **common
    ).forward(signal)
    second = DistributedRamanFiber(
        pumps=PumpSpec(1450.0, 0.2, "forward"), add_ase=True, seed=7, **common
    ).forward(signal)
    third = DistributedRamanFiber(
        pumps=PumpSpec(1450.0, 0.2, "forward"), add_ase=True, seed=8, **common
    ).forward(signal)

    assert first.ctx.meta["dra_ase_model"] == "undepleted-raman-ase"
    assert first.ctx.meta["dra_ase_power_per_pol_w"] > 0
    assert "edfa_ase_power_per_pol_w" not in first.ctx.meta
    assert np.array_equal(first.data, second.data)
    assert not np.array_equal(first.data, third.data)
    assert _output_power_w(first) > _output_power_w(without_ase)


def test_invalid_models_raise() -> None:
    """不支持的泵浦模型和非法谱参数必须显式报错。"""
    common = _fiber_kwargs()
    with pytest.raises(ValueError, match="pump_model"):
        DistributedRamanFiber(pumps=None, pump_model="coupled", **common)
    with pytest.raises(ValueError, match="FWHM"):
        DistributedRamanFiber(pumps=None, raman_fwhm_hz=0.0, **common)
    with pytest.raises(ValueError, match="spontaneous"):
        DistributedRamanFiber(pumps=None, ase_spontaneous_factor=-1.0, **common)
    with pytest.raises(ValueError, match="depletion_coupling"):
        DistributedRamanFiber(
            pumps=PumpSpec(1450.0, 0.2, "forward"),
            depletion_coupling="channelwise",
            **common,
        )


def test_reference_dataset_schema_and_snr() -> None:
    """已提交的 3/5/7 信道参考数据应包含可用的功率/NLI/SNR 数据点。"""
    json_path = REFERENCE_DIR / "dra_reference.json"
    npz_path = REFERENCE_DIR / "dra_reference.npz"
    assert json_path.exists(), f"missing reference metadata: {json_path}"
    assert npz_path.exists(), f"missing reference waveforms: {npz_path}"

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as arrays:
        assert metadata["schema_version"] == 1
        assert set(metadata["datasets"]) == {"3ch", "5ch", "7ch"}

        for dataset in metadata["datasets"].values():
            input_key = dataset["input_key"]
            assert input_key in arrays
            assert arrays[input_key].ndim == 2
            assert arrays[input_key].shape[0] == 2
            assert arrays[input_key].shape[1] == dataset["num_samples"]

            assert set(dataset["configs"]) == {
                "nopump",
                "fwd_0p10w",
                "fwd_0p20w",
                "bwd_0p10w",
                "bwd_0p20w",
            }
            for config in dataset["configs"].values():
                nl = arrays[config["waveform_nonlinear_key"]]
                lin = arrays[config["waveform_linear_key"]]
                assert np.all(np.isfinite(nl))
                assert np.all(np.isfinite(lin))
                assert nl.shape == lin.shape == arrays[input_key].shape
                assert config["dra_ase_model"] == "none"
                assert len(config["per_channel"]) == dataset["n_channels"]

        three_channel = metadata["datasets"]["3ch"]
        no_pump = three_channel["configs"]["nopump"]
        forward_010 = three_channel["configs"]["fwd_0p10w"]
        forward_020 = three_channel["configs"]["fwd_0p20w"]
        backward_020 = three_channel["configs"]["bwd_0p20w"]

        # 至少一个 3 信道波形 SNR 数据点（NLI 残差，已剔除公共相位）。
        for config in (no_pump, backward_020):
            assert config["center_channel_snr_evm_db"] > 20.0
            assert config["per_channel"][0]["snr_nli_db"] > 20.0

        # 泵浦越强，开关增益越大。
        gain_010 = forward_010["per_channel"][0]["on_off_gain_db"]
        gain_020 = forward_020["per_channel"][0]["on_off_gain_db"]
        assert gain_020 > gain_010 > 0.0
        assert np.isclose(gain_010, 5.5, atol=0.2)
        assert np.isclose(gain_020, 11.0, atol=0.3)


def _load_generator():
    """把参考数据生成脚本作为模块加载，复用其波形构造与光纤参数。"""
    spec = importlib.util.spec_from_file_location("_dra_reference_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wideband_convergence_diagnostic_oversample(monkeypatch) -> None:
    """采样率诊断脚本应在小 plan 上返回每个 factor 的有限 SNR。"""
    import argparse
    import importlib.util
    import sys

    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    spec = importlib.util.spec_from_file_location(
        "_dra_wideband_convergence_diagnostic", CONVERGENCE_CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = module.generator
    plan = generator.WidebandPlan(
        name="probe3",
        n_channels=3,
        channel_spacing_hz=75e9,
        center_frequency_hz=193.75e12,
        symbol_rate_hz=64e9,
        roll_off=0.1,
        sps=4,
        sample_rate_hz=4.608e12,
        num_symbols=64,
    )
    monkeypatch.setitem(generator.WIDEBAND_PLANS, "probe3", plan)
    monkeypatch.setitem(generator.WIDEBAND_CONFIGS, "probe3", {"nopump": []})
    args = argparse.Namespace(
        plan="probe3",
        mode="oversample",
        config="nopump",
        steps=[500.0],
        symbols=[64],
        factors=[1, 2],
        seeds=[1],
        pump_model="depleted",
        output=None,
    )
    rows = module._rows(args)
    assert len(rows) == 2
    assert {row["sample_rate_factor"] for row in rows} == {1, 2}
    for row in rows:
        assert np.isfinite(row["snr_nli_db"])
        assert row["snr_nli_db"] > 0
        assert row["num_symbols"] == 64


def test_undepleted_path_matches_committed_waveform_bit_exactly() -> None:
    """显式 ``pump_model="undepleted"`` 必须逐位复现已提交的参考波形。

    这条测试锁住"新模型不得改动旧数值路径"的约定：波形由固定种子生成、
    SSFM 完全确定性，因此重算的数组应与 ``dra_reference.npz`` 逐位相等。
    """
    metadata = json.loads((REFERENCE_DIR / "dra_reference.json").read_text(encoding="utf-8"))
    generator = _load_generator()
    tag = "3ch"
    dataset = metadata["datasets"][tag]
    wdm_input, _, _ = generator.build_wdm_dataset(
        tag,
        generator.DATASETS[tag],
        int(metadata["signal"]["num_symbols"]),
        int(metadata["signal"]["seed"]),
    )

    output = generator.make_fiber(generator.CONFIGS["fwd_0p20w"], nonlinear=True).forward(wdm_input)
    assert output.ctx.meta["dra_model"] == "undepleted"
    with np.load(REFERENCE_DIR / "dra_reference.npz") as arrays:
        assert np.array_equal(output.data, arrays[f"nl_{tag}_fwd_0p20w"])


def test_depleted_energy_and_photon_balance() -> None:
    """耗尽模型的泵浦能量账必须闭合，且信号增益满足光子数守恒。"""
    signal = _signal(n_samples=2048, power_w=1e-3)
    pumps = PumpSpec(1450.0, 0.2, "forward")
    common = _fiber_kwargs()

    fiber = DistributedRamanFiber(pumps=pumps, pump_model="depleted", **common)
    output = fiber.forward(signal)
    balance = output.ctx.meta["dra_pump_energy_balance"]
    assert output.ctx.meta["dra_model"] == "depleted"
    assert balance["converged"] is True

    pump = balance["pumps"][0]
    # 泵浦初值 = 残余 + 光纤损耗 + 转移给信号的功率。
    transferred = pump["passive_loss_w"] + pump["transferred_to_signal_w"]
    assert np.isclose(
        pump["launch_power_w"] - pump["residual_power_w"], transferred, rtol=1e-4
    )

    # 信号获得的拉曼功率与泵浦转移功率按光子数守恒换算。
    photon_ratio = signal.ctx.center_freq / (c / (1450.0 * 1e-9))
    assert np.isclose(
        balance["signal_raman_gain_w"],
        pump["transferred_to_signal_w"] * photon_ratio,
        rtol=1e-9,
    )

    # 波形侧交叉校验：SSFM 的开关增益必须与耗尽剖面反算的开关增益一致。
    no_pump = DistributedRamanFiber(pumps=None, **common).forward(signal)
    measured_db = 10 * np.log10(_output_power_w(output) / _output_power_w(no_pump))
    modeled_db = 10 * np.log10(fiber.on_off_gain(signal.ctx.center_freq))
    assert np.isclose(measured_db, modeled_db, atol=0.05)


def test_depleted_on_off_gain_falls_with_channel_count() -> None:
    """前向 0.2 W 下信道数越多，泵浦耗尽越强，开关增益越低。"""
    common = _fiber_kwargs()
    gains_db = []
    for n_channels in (1, 3, 5):
        fiber = DistributedRamanFiber(
            pumps=PumpSpec(1450.0, 0.2, "forward"), pump_model="depleted", **common
        )
        # 每信道每偏振 1 mW，与参考数据集的 0 dBm/pol 一致。
        fiber.forward(_signal(power_w=n_channels * 1e-3))
        gains_db.append(float(10 * np.log10(fiber.on_off_gain(193.1e12))))

    assert gains_db[0] > gains_db[1] > gains_db[2]
    # 单信道到 5 信道应有零点几 dB 的可见退化，但不应崩塌。
    assert 0.3 < gains_db[0] - gains_db[2] < 2.0


def test_depleted_backward_pump_matches_undepleted_at_low_power() -> None:
    """信号功率极低时，后向泵耗尽模型的开关增益退化为未耗尽结果。"""
    common = _fiber_kwargs()
    pumps = PumpSpec(1450.0, 0.2, "backward")

    depleted = DistributedRamanFiber(pumps=pumps, pump_model="depleted", **common)
    depleted.forward(_signal(power_w=1e-7))
    undepleted = DistributedRamanFiber(pumps=pumps, pump_model="undepleted", **common)

    depleted_db = 10 * np.log10(depleted.on_off_gain(193.1e12))
    undepleted_db = 10 * np.log10(undepleted.on_off_gain(193.1e12))
    assert np.isclose(depleted_db, undepleted_db, atol=1e-3)
    # 后向泵在低信号功率下应已收敛到不动点。
    assert depleted._depletion_diagnostics is not None
    assert depleted._depletion_diagnostics["converged"] is True


def test_depleted_profile_requires_forward() -> None:
    """耗尽剖面依赖信号，``forward()`` 之前调用必须显式报错。"""
    fiber = DistributedRamanFiber(
        pumps=PumpSpec(1450.0, 0.2, "forward"), pump_model="depleted", **_fiber_kwargs()
    )
    with pytest.raises(RuntimeError, match="forward"):
        fiber.pump_power_w(0.0)
    with pytest.raises(RuntimeError, match="forward"):
        fiber.on_off_gain(193.1e12)


def test_depletion_coupling_center_reproduces_committed_depleted_waveform() -> None:
    """默认 ``depletion_coupling="center"`` 必须逐位复现已提交的耗尽波形。

    锁定"新增的带内平均选项不得改动旧数值路径"这一约定。3ch/0.2 W 前向泵
    数据集在上游只跑 200 步，重算成本很低。
    """
    metadata = json.loads(
        (REFERENCE_DIR / "dra_reference_depleted.json").read_text(encoding="utf-8")
    )
    generator = _load_generator()
    tag = "3ch"
    wdm_input, _, _ = generator.build_wdm_dataset(
        tag,
        generator.DATASETS[tag],
        int(metadata["signal"]["num_symbols"]),
        int(metadata["signal"]["seed"]),
    )
    output = generator.make_fiber(
        generator.CONFIGS["fwd_0p20w"], nonlinear=True, pump_model="depleted"
    ).forward(wdm_input)
    assert output.ctx.meta["dra_depletion_coupling"] == "center"
    with np.load(REFERENCE_DIR / "dra_reference_depleted.npz") as arrays:
        assert np.array_equal(output.data, arrays[f"nl_{tag}_fwd_0p20w"])


def test_depletion_coupling_default_equals_explicit_center() -> None:
    """缺省参数与显式 ``"center"`` 必须给出逐位相同的波形和泵浦剖面。"""
    signal = _signal(n_samples=512, power_w=5e-3)
    common = _fiber_kwargs(pumps=PumpSpec(1450.0, 0.2, "backward"), pump_model="depleted")

    default = DistributedRamanFiber(**common).forward(signal)
    explicit = DistributedRamanFiber(depletion_coupling="center", **common).forward(signal)
    assert np.array_equal(default.data, explicit.data)
    assert (
        default.ctx.meta["dra_pump_energy_balance"]["pumps"][0]["transferred_to_signal_w"]
        == explicit.ctx.meta["dra_pump_energy_balance"]["pumps"][0]["transferred_to_signal_w"]
    )
    assert default.ctx.meta["dra_depletion_coupling"] == "center"
    assert explicit.ctx.meta["dra_depletion_coupling"] == "center"


def test_band_average_matches_center_for_narrowband_signal() -> None:
    """窄带信号（100 GHz）带内拉曼谱几乎不变，两种口径必须几乎一致。"""
    signal = _signal(n_samples=1024, sample_rate_hz=100e9, power_w=5e-2, seed=5)
    residuals = {}
    for coupling in ("center", "band_average"):
        fiber = DistributedRamanFiber(
            pumps=PumpSpec(1450.0, 0.2, "forward"),
            pump_model="depleted",
            depletion_coupling=coupling,
            **_fiber_kwargs(coarse_step_size_m=2000.0),
        )
        fiber.forward(signal)
        residuals[coupling] = float(fiber.pump_power_w(fiber.length_m)[0])
    assert np.isclose(
        residuals["band_average"], residuals["center"], rtol=2e-3, atol=0.0
    )


def test_band_average_changes_wideband_pump_residual_for_both_directions() -> None:
    """宽带信号下带内平均耦合低于中心耦合，泵浦耗尽更弱、残余功率更高。

    1450 nm 泵浦在中心频率正好落在 13.2 THz 拉曼峰上，而 4.6 THz 带宽的
    两侧都在峰外，因此 ``γ_band < γ_center``；前向和后向泵都应表现为更少的
    耗尽（即更高的残余功率）。符号与幅度都要可解释，而不是数值噪声。
    """
    signal = _signal(n_samples=2048, sample_rate_hz=4.608e12, power_w=5e-2, seed=7)
    for direction in ("forward", "backward"):
        residuals = {}
        for coupling in ("center", "band_average"):
            fiber = DistributedRamanFiber(
                pumps=PumpSpec(1450.0, 0.2, direction),
                pump_model="depleted",
                depletion_coupling=coupling,
                **_fiber_kwargs(coarse_step_size_m=2000.0),
            )
            fiber.forward(signal)
            residuals[coupling] = float(
                fiber.pump_power_w(fiber.length_m if direction == "forward" else 0.0)[0]
            )
        # 少耗尽 => 残余更高。窄带时该差异 <0.1%，宽带时应远大于此。
        assert residuals["band_average"] > residuals["center"]
        assert residuals["band_average"] / residuals["center"] > 1.05


def test_depletion_model_defaults_to_scalar_and_is_bit_exact() -> None:
    """缺省 ``depletion_model="scalar"`` 必须与显式 scalar 逐位一致。"""
    signal = _signal(n_samples=1024, power_w=5e-3, seed=11)
    common = _fiber_kwargs(
        pumps=PumpSpec(1450.0, 0.2, "backward"), pump_model="depleted"
    )
    default = DistributedRamanFiber(**common)
    explicit = DistributedRamanFiber(depletion_model="scalar", **common)
    assert default.depletion_model == "scalar"
    assert default.spectral_bins == 256
    assert "scalar" in DEPLETION_MODELS and "spectral" in DEPLETION_MODELS

    default_out = default.forward(signal)
    explicit_out = explicit.forward(signal)
    assert np.array_equal(default_out.data, explicit_out.data)
    assert np.array_equal(default._depleted_pump_power, explicit._depleted_pump_power)
    assert np.array_equal(default._depleted_signal_power, explicit._depleted_signal_power)
    assert default._depletion_diagnostics["spectral_bins"] is None
    assert default_out.ctx.meta["dra_depletion_model"] == "scalar"
    assert default_out.ctx.meta["dra_spectral_bins"] is None


def test_spectral_narrowband_matches_scalar() -> None:
    """100 GHz 窄带信号下，逐频箱口径应退化到标量口径（<0.1%）。"""
    signal = _signal(n_samples=1024, sample_rate_hz=100e9, power_w=5e-2, seed=5)
    residuals = {}
    outputs = {}
    for model in ("scalar", "spectral"):
        fiber = DistributedRamanFiber(
            pumps=PumpSpec(1450.0, 0.2, "forward"),
            pump_model="depleted",
            depletion_model=model,
            **_fiber_kwargs(coarse_step_size_m=2000.0),
        )
        output = fiber.forward(signal)
        residuals[model] = float(fiber.pump_power_w(fiber.length_m)[0])
        outputs[model] = _output_power_w(output)
    assert np.isclose(residuals["spectral"], residuals["scalar"], rtol=1e-3)
    assert np.isclose(outputs["spectral"], outputs["scalar"], rtol=1e-3)


def test_spectral_bins_normalise_power_and_drop_empty_bins() -> None:
    """频箱权重和为 1，总功率可还原；空信号退化为单箱。"""
    signal = _signal(n_samples=1024, sample_rate_hz=100e9, seed=3)
    freqs, weights = signal_spectrum_bins(
        signal.data, signal.ctx.sample_rate, signal.ctx.center_freq, 64
    )
    assert freqs.shape == weights.shape
    assert 1 <= freqs.size <= 64
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(freqs > 0.0)
    assert np.all(weights > 0.0)

    freqs0, weights0 = signal_spectrum_bins(
        np.zeros_like(signal.data), signal.ctx.sample_rate, signal.ctx.center_freq, 64
    )
    assert freqs0.size == 1 and weights0[0] == 1.0


def test_spectral_energy_and_photon_balance() -> None:
    """谱域耗尽：逐箱功率、逐泵能量账与光子数账都必须闭合。"""
    signal = _signal(n_samples=2048, sample_rate_hz=100e9, power_w=5e-3, seed=7)
    fiber = DistributedRamanFiber(
        pumps=PumpSpec(1450.0, 0.2, "forward"),
        pump_model="depleted",
        depletion_model="spectral",
        spectral_bins=128,
        **_fiber_kwargs(coarse_step_size_m=2000.0),
    )
    input_power_w = float(np.sum(np.mean(np.abs(signal.data) ** 2, axis=1)))
    freqs, weights = signal_spectrum_bins(
        signal.data, signal.ctx.sample_rate, signal.ctx.center_freq, 128
    )
    z, profile, ps, diagnostics = fiber._solve_depleted_profiles(
        input_power_w,
        signal.ctx.center_freq,
        bin_freqs_hz=freqs,
        bin_weights=weights,
    )

    assert diagnostics["depletion_model"] == "spectral"
    assert diagnostics["spectral_bins"] == freqs.size
    assert diagnostics["converged"] is True
    assert ps.shape == (freqs.size, z.size)
    assert np.all(ps >= 0.0)
    assert np.all(profile >= 0.0)
    # 总输入功率按权重分配，逐箱初值求和 = input_power_w。
    assert np.isclose(np.sum(ps[:, 0]), input_power_w, rtol=1e-12)
    assert np.isclose(diagnostics["signal_out_w"], float(np.sum(ps[:, -1])), rtol=1e-12)

    for pump in diagnostics["pumps"]:
        balance = (
            pump["launch_power_w"]
            - pump["residual_power_w"]
            - pump["passive_loss_w"]
            - pump["transferred_to_signal_w"]
            + pump["pump_pump_net_w"]
        )
        assert abs(balance) < 1e-6

    # 逐箱光子数守恒：Σ_b ΔP_b/f_b = Σ_p transfer_p/f_p。
    assert np.isclose(
        diagnostics["signal_photon_gain_per_s"],
        diagnostics["pump_photon_transfer_per_s"],
        rtol=1e-9,
    )
    assert diagnostics["signal_raman_gain_w"] > 0.0

    # ODE 的总信号输出必须与 SSFM 实际输出同口径（差别只来自损耗/色散细节）。
    output = fiber.forward(signal)
    ssfm_total = float(np.sum(np.mean(np.abs(output.data) ** 2, axis=1)))
    assert np.isclose(diagnostics["signal_out_w"], ssfm_total, rtol=2e-2)


def test_spectral_is_required_for_pump_pump() -> None:
    """泵浦-泵浦转移必须显式启用谱域耗尽，不能静默用标量驱动。"""
    with pytest.raises(ValueError, match="spectral"):
        DistributedRamanFiber(
            pumps=[PumpSpec(1360.0, 0.2, "forward"), PumpSpec(1450.0, 0.2, "forward")],
            pump_model="depleted",
            pump_pump_raman=True,
            **_fiber_kwargs(),
        )
    with pytest.raises(ValueError, match="spectral_bins"):
        DistributedRamanFiber(
            pumps=PumpSpec(1450.0, 0.2, "forward"),
            pump_model="depleted",
            depletion_model="spectral",
            spectral_bins=0,
            **_fiber_kwargs(),
        )


def test_spectral_pump_pump_has_no_unphysical_amplification() -> None:
    """六泵后向 + 谱域 + pp：总输出必须低于泵浦+信号总注入，能量账闭合。"""
    generator = _load_generator()
    pumps = generator.WIDEBAND_CONFIGS["cl152"]["bwd_sixpumps"]
    signal = _signal(n_samples=512, sample_rate_hz=100e9, center_freq_hz=190.15e12)
    fiber = generator.make_wideband_fiber(
        pumps,
        nonlinear=True,
        pump_model="depleted",
        coarse_step_m=100.0,
        depletion_model="spectral",
        pump_pump_raman=True,
    )
    fiber.depleted_step_m = 100.0
    fiber.depleted_max_iterations = 128
    output = fiber.forward(signal)
    balance = output.ctx.meta["dra_pump_energy_balance"]

    assert balance["converged"] is True
    assert balance["depletion_model"] == "spectral"
    assert balance["spectral_bins"] is not None
    assert balance["signal_out_w"] < sum(pump.power_w for pump in pumps)
    assert np.all(fiber._depleted_pump_power >= 0.0)
    for pump in balance["pumps"]:
        residual = (
            pump["launch_power_w"]
            - pump["residual_power_w"]
            - pump["passive_loss_w"]
            - pump["transferred_to_signal_w"]
            + pump["pump_pump_net_w"]
        )
        assert abs(residual) < 1e-4 * max(pump.power_w for pump in pumps)


def _multichannel_signal(
    n_channels: int = 5,
    *,
    spacing_hz: float = 100e9,
    power_per_channel_w: float = 1e-3,
    n_samples: int = 512,
    sample_rate_hz: float = 1.6e12,
    center_freq_hz: float = 193.1e12,
    seed: int = 0,
) -> Signal:
    """构造 ``n_channels`` 个等间隔单音（每信道每偏振 ``power_per_channel_w``）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate_hz
    offsets = (np.arange(n_channels) - (n_channels - 1) / 2.0) * spacing_hz
    data = np.zeros((2, n_samples), dtype=np.complex128)
    amplitude = np.sqrt(power_per_channel_w)
    for offset in offsets:
        phase = rng.uniform(0.0, 2 * np.pi, size=(2, 1))
        data += amplitude * np.exp(1j * (2 * np.pi * offset * t + phase))
    ctx = SignalContext(
        sample_rate=sample_rate_hz,
        center_freq=center_freq_hz,
        n_pol=2,
        sps=4,
        sym_rate=sample_rate_hz / 4,
    )
    return Signal(data.astype(np.complex64), ctx)


def test_spectral_multichannel_mixed_pumps_energy_balance() -> None:
    """3–8 信道、前向+后向泵：频箱功率、能量账与光子账都闭合。"""
    signal = _multichannel_signal(n_channels=5)
    pumps = [
        PumpSpec(1450.0, 0.2, "forward"),
        PumpSpec(1460.0, 0.1, "backward"),
    ]
    fiber = DistributedRamanFiber(
        pumps=pumps,
        pump_model="depleted",
        depletion_model="spectral",
        spectral_bins=64,
        depleted_step_m=100.0,
        **_fiber_kwargs(length_m=100e3, coarse_step_size_m=2000.0),
    )
    input_power_w = float(np.sum(np.mean(np.abs(signal.data) ** 2, axis=1)))
    freqs, weights = signal_spectrum_bins(
        signal.data, signal.ctx.sample_rate, signal.ctx.center_freq, 64
    )
    # 5 个单音占据主要频箱；complex64 的数值泄漏会带出少量邻近箱。
    assert 5 <= freqs.size <= 64
    dominant = np.argsort(weights)[-5:]
    assert np.isclose(float(np.sum(weights[dominant])), 1.0, atol=1e-6)
    z, profile, ps, diagnostics = fiber._solve_depleted_profiles(
        input_power_w,
        signal.ctx.center_freq,
        bin_freqs_hz=freqs,
        bin_weights=weights,
    )

    assert diagnostics["converged"] is True
    assert ps.shape == (freqs.size, z.size)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.isclose(np.sum(ps[:, 0]), input_power_w, rtol=1e-12)
    assert np.all(ps >= 0.0)
    assert np.all(profile >= 0.0)
    assert diagnostics["signal_raman_gain_w"] > 0.0
    for pump in diagnostics["pumps"]:
        balance = (
            pump["launch_power_w"]
            - pump["residual_power_w"]
            - pump["passive_loss_w"]
            - pump["transferred_to_signal_w"]
            + pump["pump_pump_net_w"]
        )
        assert abs(balance) < 1e-5
    assert np.isclose(
        diagnostics["signal_photon_gain_per_s"],
        diagnostics["pump_photon_transfer_per_s"],
        rtol=1e-9,
    )


def test_pump_pump_raman_requires_depleted_model() -> None:
    """泵浦-泵浦转移只对耗尽模型有定义；未耗尽时必须显式报错而不是静默忽略。"""
    with pytest.raises(ValueError, match="depleted"):
        DistributedRamanFiber(
            pumps=[PumpSpec(1360.0, 0.2, "forward"), PumpSpec(1450.0, 0.2, "forward")],
            pump_pump_raman=True,
            **_fiber_kwargs(),
        )


def test_pump_pump_default_off_reproduces_committed_depleted_waveform() -> None:
    """``pump_pump_raman=False``（缺省）必须逐位复现已提交的耗尽参考波形。"""
    metadata = json.loads(
        (REFERENCE_DIR / "dra_reference_depleted.json").read_text(encoding="utf-8")
    )
    generator = _load_generator()
    tag = "3ch"
    wdm_input, _, _ = generator.build_wdm_dataset(
        tag,
        generator.DATASETS[tag],
        int(metadata["signal"]["num_symbols"]),
        int(metadata["signal"]["seed"]),
    )
    fiber = generator.make_fiber(
        generator.CONFIGS["fwd_0p20w"], nonlinear=True, pump_model="depleted"
    )
    assert fiber.pump_pump_raman is False
    output = fiber.forward(wdm_input)
    assert output.ctx.meta["dra_pump_pump_raman"] is False
    with np.load(REFERENCE_DIR / "dra_reference_depleted.npz") as arrays:
        assert np.array_equal(output.data, arrays[f"nl_{tag}_fwd_0p20w"])

    balance = output.ctx.meta["dra_pump_energy_balance"]
    for pump in balance["pumps"]:
        assert pump["pump_pump_sent_w"] == 0.0
        assert pump["pump_pump_received_w"] == 0.0
        assert pump["pump_pump_net_w"] == 0.0
    assert balance["pump_pump_net_total_w"] == 0.0
    assert "no pump-pump Raman transfer" in output.ctx.meta["dra_pump_model_detail"]


def test_pump_pump_default_off_matches_explicit_off_for_six_backward_pumps() -> None:
    """六泵后向配置下，缺省与显式 False 的波形/剖面必须逐位相同。"""
    generator = _load_generator()
    pumps = generator.WIDEBAND_CONFIGS["cl152"]["bwd_sixpumps"]
    signal = _signal(n_samples=512, sample_rate_hz=100e9, center_freq_hz=190.15e12)
    common = {
        "nonlinear": True,
        "pump_model": "depleted",
        "coarse_step_m": 100.0,
    }
    default = generator.make_wideband_fiber(pumps, pump_pump_raman=False, **common)
    default.depleted_step_m = 100.0
    explicit = generator.make_wideband_fiber(pumps, **common)
    explicit.depleted_step_m = 100.0
    default_out = default.forward(signal)
    explicit_out = explicit.forward(signal)

    assert np.array_equal(default_out.data, explicit_out.data)
    assert np.array_equal(default._depleted_pump_power, explicit._depleted_pump_power)
    assert np.array_equal(default._depleted_signal_power, explicit._depleted_signal_power)
    # 6 个泵都已耦合，打开开关必须改变结果，否则测试没有意义。
    # 泵浦-泵浦转移要求谱域耗尽驱动（标量驱动会低估耗尽，见 §13.9）。
    enabled = generator.make_wideband_fiber(
        pumps, pump_pump_raman=True, depletion_model="spectral", **common
    )
    enabled.depleted_step_m = 100.0
    enabled_out = enabled.forward(signal)
    assert not np.array_equal(default_out.data, enabled_out.data)


def _two_pump_profiles(
    wavelengths_nm: tuple[float, float],
    powers_w: tuple[float, float],
    *,
    pump_pump_raman: bool,
    step_m: float = 50.0,
    alpha_db_per_km: float = 0.2,
    input_power_w: float = 1e-25,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """跑两个前向泵 + 极弱信号的耗尽剖面，返回 (泵浦剖面, z, 能量账)。

    泵浦-泵浦转移只允许谱域耗尽；这里信号退化为单个 1550 nm 频箱，
    因此与标量口径的耦合系数逐点相同。
    """
    pumps = [
        PumpSpec(wavelength, power, "forward", alpha_db_per_km)
        for wavelength, power in zip(wavelengths_nm, powers_w, strict=True)
    ]
    fiber = DistributedRamanFiber(
        pumps=pumps,
        pump_model="depleted",
        pump_pump_raman=pump_pump_raman,
        depletion_model="spectral",
        depleted_step_m=step_m,
        **_fiber_kwargs(length_m=80e3, alpha_db_per_km=alpha_db_per_km),
    )
    z, profile, _, diagnostics = fiber._solve_depleted_profiles(
        input_power_w,
        c / 1550e-9,
        bin_freqs_hz=np.array([c / 1550e-9]),
        bin_weights=np.array([1.0]),
    )
    return profile, z, diagnostics


def test_pump_pump_two_pump_transfer_direction_and_energy_balance() -> None:
    """短波长泵把功率交给长波长泵；每个泵的能量账必须闭合。

    用 1360/1445 nm（频差 13.4 THz，正落在拉曼峰上）作为强耦合对，另用
    任务示例里的 1340/1505 nm（频差 24.5 THz，Gaussian 谱翼 ≈7e-7）验证
    符号方向在弱耦合下同样成立。
    """
    for wavelengths, powers in (((1360.0, 1445.0), (0.63, 0.165)), ((1340.0, 1505.0), (0.63, 0.075))):
        reference, _, _ = _two_pump_profiles(wavelengths, powers, pump_pump_raman=False)
        with_pp, _, diagnostics = _two_pump_profiles(wavelengths, powers, pump_pump_raman=True)
        # 高频（短波长）泵失去功率，低频（长波长）泵获得功率。
        assert with_pp[0, -1] < reference[0, -1]
        assert with_pp[1, -1] > reference[1, -1]

        for pump in diagnostics["pumps"]:
            assert pump["pump_pump_sent_w"] >= 0.0
            assert pump["pump_pump_received_w"] >= 0.0
            balance = (
                pump["launch_power_w"]
                - pump["residual_power_w"]
                - pump["passive_loss_w"]
                - pump["transferred_to_signal_w"]
                + pump["pump_pump_net_w"]
            )
            # 剩余项只有 RK4/梯形积分的离散误差；误差尺度跟总功率（而非单个
            # 泵的标称功率）走，所以按最大标称功率给容差。
            assert abs(balance) < 1e-4 * max(powers)
            assert np.isclose(
                pump["pump_pump_net_w"],
                pump["pump_pump_received_w"] - pump["pump_pump_sent_w"],
                rtol=1e-12,
                atol=1e-18,
            )
        # 全局：净收益为负，差额以声子能量离开光场。
        assert diagnostics["pump_pump_net_total_w"] < 0.0
        assert np.isclose(
            diagnostics["pump_pump_net_total_w"],
            sum(pump["pump_pump_net_w"] for pump in diagnostics["pumps"]),
            rtol=1e-12,
            atol=1e-18,
        )


def test_pump_pump_conserves_pump_photons_and_step_size() -> None:
    """无光纤损耗时成对光子数守恒，且残余功率对 RK4 步长不敏感。"""
    wavelengths, powers = (1360.0, 1445.0), (0.63, 0.165)
    frequencies = np.array([c / (w * 1e-9) for w in wavelengths])
    residuals = {}
    for step_m in (200.0, 100.0, 50.0):
        profile, z, diagnostics = _two_pump_profiles(
            wavelengths, powers, pump_pump_raman=True, step_m=step_m, alpha_db_per_km=0.0
        )
        assert diagnostics["converged"] is True
        photons_launch = float(np.sum(profile[:, 0] / frequencies))
        photons_end = float(np.sum(profile[:, -1] / frequencies))
        # 弱信号（1e-25 W）带走的忽略不计，剩余为不动点/RK4 误差。
        assert abs(photons_end - photons_launch) / photons_launch < 1e-6
        residuals[step_m] = profile[:, -1]
    # 100 m 与 50 m 的残余功率差应远小于功率转移本身。
    assert np.max(np.abs(residuals[100.0] - residuals[50.0])) < 1e-9


def test_pump_pump_six_backward_pumps_converge_and_move_power() -> None:
    """六泵后向：开关打开后不动点仍收敛，功率从短波长泵流向长波长泵。"""
    generator = _load_generator()
    pumps = generator.WIDEBAND_CONFIGS["cl152"]["bwd_sixpumps"]
    signal = _signal(n_samples=512, sample_rate_hz=100e9, center_freq_hz=190.15e12)
    common = {"nonlinear": True, "pump_model": "depleted", "coarse_step_m": 100.0}
    reference = generator.make_wideband_fiber(pumps, **common)
    enabled = generator.make_wideband_fiber(
        pumps, pump_pump_raman=True, depletion_model="spectral", **common
    )
    for fiber in (reference, enabled):
        fiber.depleted_step_m = 100.0
    # 谱域 + pp 的六泵后向不动点需要更多迭代（实测 29 次）。
    enabled.depleted_max_iterations = 128
    reference_out = reference.forward(signal)
    enabled_out = enabled.forward(signal)

    balance = enabled_out.ctx.meta["dra_pump_energy_balance"]
    assert balance["converged"] is True
    assert balance["pump_pump_raman"] is True
    assert 1 < balance["iterations"] <= enabled.depleted_max_iterations
    assert balance["pump_pump_net_total_w"] < 0.0

    # 后向泵在 z=L 发射，残余取 z=0。
    launch_w = np.array([pump.power_w for pump in pumps])
    reference_residual = reference._depleted_pump_power[:, 0]
    enabled_residual = enabled._depleted_pump_power[:, 0]
    pumps_diag = balance["pumps"]
    # 1340/1360 nm 把功率交出去（几乎被抽空），1427/1445/1476/1505 nm 收下。
    for index in (0, 1):
        assert enabled_residual[index] < reference_residual[index] * 1e-2
        assert pumps_diag[index]["pump_pump_sent_w"] > 0.5
        assert pumps_diag[index]["pump_pump_received_w"] < 1e-4
    assert pumps_diag[2]["pump_pump_received_w"] > 0.5  # 1427 nm 接 1340/1360
    for index in (3, 4, 5):  # 1445 / 1476 / 1505 nm
        assert pumps_diag[index]["pump_pump_received_w"] > 0.01
    # 1427 nm 既是接收端也是 1505 nm 的供给端，净收益仍为正。
    assert pumps_diag[2]["pump_pump_net_w"] > 0.0
    # 发射端的标称功率不因泵浦-泵浦项改变。
    assert np.allclose(enabled._depleted_pump_power[:, -1], launch_w, rtol=1e-12)


def test_depleted_reference_dataset_schema() -> None:
    """耗尽参考数据集应可读，并体现泵浦耗尽导致的增益退化。"""
    json_path = REFERENCE_DIR / "dra_reference_depleted.json"
    npz_path = REFERENCE_DIR / "dra_reference_depleted.npz"
    assert json_path.exists(), f"missing depleted metadata: {json_path}"
    assert npz_path.exists(), f"missing depleted waveforms: {npz_path}"

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["model"]["pump_model"].startswith("depleted")
    assert set(metadata["datasets"]) == {
        "3ch",
        "5ch",
        "1ch",
        "3ch_0p5mw",
        "3ch_1p0mw",
        "3ch_2p0mw",
    }

    with np.load(npz_path) as arrays:
        for dataset in metadata["datasets"].values():
            assert dataset["pump_model"] == "depleted"
            assert arrays[dataset["input_key"]].shape[1] == dataset["num_samples"]
            for config in dataset["configs"].values():
                nl = arrays[config["waveform_nonlinear_key"]]
                lin = arrays[config["waveform_linear_key"]]
                assert np.all(np.isfinite(nl)) and np.all(np.isfinite(lin))
                assert len(config["per_channel"]) == dataset["n_channels"]

    three_channel = metadata["datasets"]["3ch"]
    five_channel = metadata["datasets"]["5ch"]
    # 信道数越多，前向泵的开关增益越低（泵浦耗尽）。
    gain_3ch = three_channel["configs"]["fwd_0p20w"]["per_channel"][1]["on_off_gain_db"]
    gain_5ch = five_channel["configs"]["fwd_0p20w"]["per_channel"][2]["on_off_gain_db"]
    assert 0.0 < gain_5ch < gain_3ch < 11.0

    # 1ch 探针：前向放大显著恶化 NLI，后向几乎不变。
    one_channel = metadata["datasets"]["1ch"]["configs"]
    snr_nopump = one_channel["nopump"]["per_channel"][0]["snr_nli_db"]
    snr_forward = one_channel["fwd_0p20w"]["per_channel"][0]["snr_nli_db"]
    snr_backward = one_channel["bwd_0p20w"]["per_channel"][0]["snr_nli_db"]
    assert snr_nopump - snr_forward > 13.0
    assert snr_nopump - snr_backward < 1.0

    # 3ch 前向 0.2 W 在 0.5/1/2 mW 下 NLI 单调增加。
    nli_by_power = [
        metadata["datasets"][tag]["configs"]["fwd_0p20w"]["per_channel"][1]["nli_power_w"]
        for tag in metadata["power_sweep"]["dataset_tags"]
    ]
    assert nli_by_power[0] < nli_by_power[1] < nli_by_power[2]


def test_converged_reference_dataset_schema_and_step_metadata() -> None:
    """62.5 m 收敛数据集应完整，并记录步长收敛元数据与关键 SNR。"""
    json_path = CONVERGED_DIR / "dra_reference.json"
    npz_path = CONVERGED_DIR / "dra_reference.npz"
    assert json_path.exists(), f"missing converged metadata: {json_path}"
    assert npz_path.exists(), f"missing converged waveforms: {npz_path}"

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    legacy = json.loads((REFERENCE_DIR / "dra_reference.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["coarse_step_m"] == 62.5
    assert metadata["pump_model"] == "undepleted"
    assert "nli_method" in metadata
    assert set(metadata["datasets"]) == {"3ch", "5ch", "7ch"}
    assert metadata["step_criterion"]["target_walkoff_symbols"] == 0.15

    with np.load(npz_path) as arrays:
        for dataset in metadata["datasets"].values():
            assert dataset["coarse_step_m"] == 62.5
            assert dataset["pump_model"] == "undepleted"
            assert dataset["step_converged"] is True
            assert isinstance(dataset["step_artifact_500m_db"], float)
            convergence = dataset["step_convergence"]
            assert convergence["coarse_step_m"] == 62.5
            assert abs(convergence["last_increment_db"]) < 0.01
            assert convergence["recommended_step_m_for_target"] > 0.0

            input_key = dataset["input_key"]
            assert arrays[input_key].ndim == 2
            assert arrays[input_key].shape[1] == dataset["num_samples"]
            assert set(dataset["configs"]) == {
                "nopump",
                "fwd_0p10w",
                "fwd_0p20w",
                "bwd_0p10w",
                "bwd_0p20w",
            }
            for config in dataset["configs"].values():
                assert (
                    arrays[config["waveform_nonlinear_key"]].shape
                    == arrays[config["waveform_linear_key"]].shape
                    == arrays[input_key].shape
                )
                assert len(config["per_channel"]) == dataset["n_channels"]

        # 5ch/7ch 无泵的中心信道 SNR 应收敛到约 36.33/36.27 dB。
        five = metadata["datasets"]["5ch"]
        seven = metadata["datasets"]["7ch"]
        snr_5ch = five["configs"]["nopump"]["per_channel"][five["center_channel_index"]]["snr_nli_db"]
        snr_7ch = seven["configs"]["nopump"]["per_channel"][seven["center_channel_index"]][
            "snr_nli_db"
        ]
        assert abs(snr_5ch - 36.33) < 0.05
        assert abs(snr_7ch - 36.27) < 0.05

    # 3ch 在 500 m 已收敛，新旧数据差应 < 0.1 dB。
    three_legacy = legacy["datasets"]["3ch"]
    three_converged = metadata["datasets"]["3ch"]
    legacy_snr = three_legacy["configs"]["nopump"]["per_channel"][
        three_legacy["center_channel_index"]
    ]["snr_nli_db"]
    converged_snr = three_converged["configs"]["nopump"]["per_channel"][
        three_converged["center_channel_index"]
    ]["snr_nli_db"]
    assert abs(converged_snr - legacy_snr) < 0.1
    assert three_converged["step_artifact_500m_db"] < 0.1


def test_converged_depleted_reference_dataset_schema() -> None:
    """62.5 m 耗尽数据集应复用 WP3 逻辑并记录收敛字段。"""
    json_path = CONVERGED_DIR / "dra_reference_depleted.json"
    npz_path = CONVERGED_DIR / "dra_reference_depleted.npz"
    assert json_path.exists(), f"missing converged depleted metadata: {json_path}"
    assert npz_path.exists(), f"missing converged depleted waveforms: {npz_path}"

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["coarse_step_m"] == 62.5
    assert metadata["pump_model"] == "depleted"
    assert set(metadata["datasets"]) == {
        "3ch",
        "5ch",
        "7ch",
        "1ch",
        "3ch_0p5mw",
        "3ch_1p0mw",
        "3ch_2p0mw",
    }

    with np.load(npz_path) as arrays:
        for dataset in metadata["datasets"].values():
            assert dataset["pump_model"] == "depleted"
            assert dataset["step_converged"] is True
            assert abs(dataset["step_convergence"]["last_increment_db"]) < 0.01
            assert arrays[dataset["input_key"]].shape[1] == dataset["num_samples"]
            for config in dataset["configs"].values():
                nl = arrays[config["waveform_nonlinear_key"]]
                lin = arrays[config["waveform_linear_key"]]
                assert np.all(np.isfinite(nl)) and np.all(np.isfinite(lin))
                assert len(config["per_channel"]) == dataset["n_channels"]

    # 7ch 前向 0.2 W 的中心信道 SNR（62.5 m，耗尽模型）。
    # 2026-11-05 修正 DRA 频变增益为频域施加后，前向泵 NLI 显著下降：
    # 旧时域施加给出 22.384 dB，频域施加给出 24.399 dB。
    seven = metadata["datasets"]["7ch"]
    seven_forward = seven["configs"]["fwd_0p20w"]["per_channel"][seven["center_channel_index"]]
    assert abs(seven_forward["snr_nli_db"] - 24.399) < 0.02

    compare_path = CONVERGED_DIR / "depleted_7ch_compare.json"
    assert compare_path.exists(), f"missing 7ch analytic comparison: {compare_path}"
    compare = json.loads(compare_path.read_text(encoding="utf-8"))
    assert compare["pump_depletion"] is True
    assert compare["pump_model"] == "depleted"
    assert compare["coarse_step_m"] == 62.5
    assert abs(compare["waveform_snr_nli_db"] - seven_forward["snr_nli_db"]) < 1e-9
    assert abs(compare["analytic"]["hybrid_raman_paper"]["snr_nli_db"] - 23.387) < 0.01
    assert abs(compare["analytic"]["pcfm_h3_raman"]["snr_nli_db"] - 23.583) < 0.01
    # 修正后解析侧比波形保守约 1.0/0.8 dB（残差 = 解析 - 波形）。
    assert -1.1 < compare["analytic"]["hybrid_raman_paper"]["residual_db"] < -0.9
    assert -0.9 < compare["analytic"]["pcfm_h3_raman"]["residual_db"] < -0.7
    assert 3.0 < compare["historical_500m"]["step_correction_db"] < 3.15
    assert 1.95 < compare["historical_500m"]["analytic_residual_db"]["hybrid_raman_paper"] < 2.15


def _load_step_checker():
    """把步长收敛检查脚本作为模块加载。"""
    spec = importlib.util.spec_from_file_location("_dra_step_checker", STEP_CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_convergence_checker_core_on_3ch() -> None:
    """检查脚本的核心函数应在 3ch 上给出收敛结果与 500 m 伪影。"""
    checker = _load_step_checker()
    report = checker.run_probes(
        checker.DEFAULT_NPZ,
        probes=(("3ch", "nopump"),),
        steps_m=(500.0, 62.5, 31.25),
    )
    probe = report["probes"]["3ch"]["nopump"]
    assert len(probe["steps"]) == 3
    assert abs(probe["last_increment_db"]) < 0.01
    assert 0.0 < probe["artifact_500m_db"] < 0.1
    assert checker.failed_increments(report, 0.01) == []
    # 人为收紧容差时应能报出失败。
    assert len(checker.failed_increments(report, 0.001)) == 1


# ---------------------------------------------------------------------------
# 全 WDM（c57 / cl152）
# ---------------------------------------------------------------------------


def test_wideband_plan_validation_and_resample_ratio() -> None:
    """两个全 WDM 计划的栅格/采样率/偏移必须自洽，并拒绝非法计划。"""
    generator = _load_generator()
    c57 = generator.WIDEBAND_PLANS["c57"]
    cl152 = generator.WIDEBAND_PLANS["cl152"]

    assert c57.n_channels == 57
    assert cl152.n_channels == 152
    for plan, ratio in ((c57, 18), (cl152, 45)):
        assert plan.channel_spacing_hz == 75e9
        assert plan.symbol_rate_hz == 64e9
        assert plan.roll_off == 0.1
        assert plan.resample_ratio == ratio
        assert plan.channel_sample_rate_hz == 256e9
        assert plan.sample_rate_hz == ratio * plan.channel_sample_rate_hz
        assert plan.offsets_hz.shape == (plan.n_channels,)
        assert np.isclose(plan.offsets_hz[0], -plan.offsets_hz[-1])
        assert plan.samples_per_channel == plan.num_symbols * plan.sps

    def make(**overrides):
        fields = {
            "name": "bad",
            "n_channels": 3,
            "channel_spacing_hz": 75e9,
            "center_frequency_hz": 193.75e12,
            "symbol_rate_hz": 64e9,
            "roll_off": 0.1,
            "sps": 4,
            "sample_rate_hz": 4.608e12,
        }
        fields.update(overrides)
        return generator.WidebandPlan(**fields)

    # 非整数重采样比。
    with pytest.raises(ValueError, match="integer multiple"):
        make(sample_rate_hz=4.7e12)
    # 信道重叠。
    with pytest.raises(ValueError, match="overlap"):
        make(channel_spacing_hz=60e9)
    # 采样率不足以覆盖合波带宽。
    with pytest.raises(ValueError, match="alias"):
        make(n_channels=57, sample_rate_hz=2.304e12)


def test_wideband_nopump_matches_undepleted_on_short_record() -> None:
    """全 WDM 代码路径的 nopump 结果应与 undepleted 逐位一致。"""
    generator = _load_generator()
    plan = generator.WidebandPlan(
        name="probe3",
        n_channels=3,
        channel_spacing_hz=75e9,
        center_frequency_hz=193.75e12,
        symbol_rate_hz=64e9,
        roll_off=0.1,
        sps=4,
        sample_rate_hz=4.608e12,
        num_symbols=32,
    )
    signal, _, _ = generator.build_wideband_dataset(plan, seed=1)
    depleted = generator.make_wideband_fiber(
        [], nonlinear=True, pump_model="depleted", coarse_step_m=500.0
    ).forward(signal)
    undepleted = generator.make_wideband_fiber(
        [], nonlinear=True, pump_model="undepleted", coarse_step_m=500.0
    ).forward(signal)
    assert np.array_equal(depleted.data, undepleted.data)


def test_wideband_step_probe_core_on_short_record() -> None:
    """步长探针核心函数应返回可审计的逐档 SNR/走离/耗时字段。"""
    generator = _load_generator()
    plan = generator.WidebandPlan(
        name="probe3",
        n_channels=3,
        channel_spacing_hz=75e9,
        center_frequency_hz=193.75e12,
        symbol_rate_hz=64e9,
        roll_off=0.1,
        sps=4,
        sample_rate_hz=4.608e12,
        num_symbols=32,
    )
    report = generator.wideband_step_convergence(
        plan,
        steps_m=(500.0, 250.0),
        num_symbols=32,
        seed=1,
        pump_model="depleted",
        extend=False,
    )
    assert len(report["steps"]) == 2
    for row in report["steps"]:
        assert row["step_m"] > 0
        assert row["n_steps"] == int(np.ceil(100e3 / row["step_m"]))
        assert np.isfinite(row["snr_nli_db"])
        assert row["walkoff_per_step_symbols"] > 0
        assert row["elapsed_s"] >= 0
    assert isinstance(report["step_converged"], bool)
    assert report["metric"] == "nopump center-channel snr_nli_db"
    assert report["tolerance_db"] == 0.01


def test_wideband_reference_dataset_schema() -> None:
    """已提交的全 WDM 数据应包含计划、光纤、步长与逐信道指标。"""
    for name, n_channels, ratio, configs in (
        ("c57", 57, 18, {"nopump", "fwd_0p20w", "bwd_0p20w"}),
        ("cl152", 152, 45, {"nopump", "bwd_sixpumps", "fwd_0p20w"}),
    ):
        json_path = WIDEBAND_DIR / f"{name}.json"
        npz_path = WIDEBAND_DIR / f"{name}.npz"
        assert json_path.exists(), f"missing wideband metadata: {json_path}"
        assert npz_path.exists(), f"missing wideband waveforms: {npz_path}"

        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        assert metadata["dataset_family"] == "dra_wideband"
        assert metadata["plan"]["name"] == name
        assert metadata["plan"]["n_channels"] == n_channels
        assert metadata["plan"]["resample_ratio"] == ratio
        assert metadata["nli_analysis_bandwidth_hz"] == 64e9
        assert metadata["signal"]["symbol_rate_hz"] == 64e9
        assert metadata["coarse_step_m"] > 0
        assert isinstance(metadata["step_converged"], bool)
        assert metadata["step_converged"] == metadata["step_convergence"]["step_converged"]
        assert metadata["num_samples"] == (
            metadata["plan"]["samples_per_channel"] * ratio
        )
        assert set(metadata["configs"]) == configs
        assert "fiber" in metadata and "pump" in metadata

        with np.load(npz_path) as arrays:
            input_key = f"input_{name}"
            assert input_key in arrays
            assert arrays[input_key].shape == (2, metadata["num_samples"])
            for config in metadata["configs"].values():
                nl = arrays[config["waveform_nonlinear_key"]]
                lin = arrays[config["waveform_linear_key"]]
                assert nl.shape == lin.shape == arrays[input_key].shape
                assert np.all(np.isfinite(nl)) and np.all(np.isfinite(lin))
                assert len(config["per_channel"]) == n_channels
                first = config["per_channel"][0]
                for key in (
                    "freq_offset_hz",
                    "signal_power_w",
                    "nli_power_w",
                    "snr_nli_db",
                    "on_off_gain_db",
                    "common_gain_db",
                    "common_phase_deg",
                ):
                    assert key in first

        # 每信道频偏相对计划中心，且关于 0 对称。
        offsets = np.asarray(
            [row["freq_offset_hz"] for row in metadata["configs"]["nopump"]["per_channel"]]
        )
        assert np.isclose(offsets[0], -offsets[-1])


def test_wideband_analytic_compare_document() -> None:
    """解析对照文档应存在，且 nopump 中心信道残差在 ~1 dB 量级。"""
    path = WIDEBAND_DIR / "analytic_compare.json"
    if not path.exists():
        pytest.skip("analytic comparison has not been generated")
    document = json.loads(path.read_text(encoding="utf-8"))
    for name in ("c57", "cl152"):
        row = document["plans"][name]
        assert row["plan"] == name
        assert row["analytic"]["model"] == "hybrid_raman_paper"
        assert row["analytic"]["config"] == "nopump"
        assert abs(row["residual_db"]) < 2.5



def test_frequency_dependent_gain_is_applied_in_frequency_domain() -> None:
    """频变拉曼增益必须乘在频谱上：零色散 CW 输入不得被幅度调制。

    旧实现把 ``step_gain`` 逐样本乘在时域，相当于把增益谱映射到时间轴，
    会给 CW 引入约 1% 的虚假幅度起伏；该起伏会污染接收机 EVM，并随记录
    长度改变。这里用零色散、零非线性的 CW 输入锁住频域施加行为。
    """
    n_samples = 512
    sample_rate_hz = 100e9
    center_freq_hz = 193.1e12
    ctx = SignalContext(
        sample_rate=sample_rate_hz,
        center_freq=center_freq_hz,
        n_pol=2,
        sps=4,
        sym_rate=sample_rate_hz / 4,
    )
    constant = np.sqrt(1e-3)
    data = np.full((2, n_samples), constant, dtype=np.complex128)
    dra = DistributedRamanFiber(
        length_m=20e3,
        alpha_db_per_km=0.2,
        beta2_ps2_per_km=0.0,
        beta3_ps3_per_km=0.0,
        pumps=PumpSpec(1450.0, 0.2, "forward"),
        coarse_step_size_m=500.0,
        n2_si=0.0,
        a_eff_um2=80.0,
        pump_model="undepleted",
    )
    output = dra.forward(Signal(data, ctx))
    amplitude = np.abs(output.data)
    assert np.allclose(amplitude, amplitude[0, 0], rtol=1e-10, atol=0.0)
    expected_power_ratio = 10 ** (-0.2 * 20 / 10) * float(dra.on_off_gain(center_freq_hz))
    measured_power_ratio = float(np.mean(np.abs(output.data) ** 2) / constant**2)
    assert np.isclose(measured_power_ratio, expected_power_ratio, rtol=1e-3)
