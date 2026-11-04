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
    DistributedRamanFiber,
    FiberChannel,
    PumpSpec,
    RamanPump,
    raman_gain_spectrum,
)
from physics_nlo_eval.core.signal import Signal, SignalContext

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "examples" / "dra_reference"
CONVERGED_DIR = REPO_ROOT / "examples" / "dra_reference_converged"
GENERATOR_PATH = REPO_ROOT / "examples" / "generate_dra_reference.py"
STEP_CHECKER_PATH = REPO_ROOT / "examples" / "check_dra_step_convergence.py"


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
    seven = metadata["datasets"]["7ch"]
    seven_forward = seven["configs"]["fwd_0p20w"]["per_channel"][seven["center_channel_index"]]
    assert abs(seven_forward["snr_nli_db"] - 22.384) < 0.02

    compare_path = CONVERGED_DIR / "depleted_7ch_compare.json"
    assert compare_path.exists(), f"missing 7ch analytic comparison: {compare_path}"
    compare = json.loads(compare_path.read_text(encoding="utf-8"))
    assert compare["pump_depletion"] is True
    assert compare["pump_model"] == "depleted"
    assert compare["coarse_step_m"] == 62.5
    assert abs(compare["waveform_snr_nli_db"] - seven_forward["snr_nli_db"]) < 1e-9
    assert abs(compare["analytic"]["hybrid_raman_paper"]["snr_nli_db"] - 23.387) < 0.01
    assert abs(compare["analytic"]["pcfm_h3_raman"]["snr_nli_db"] - 23.583) < 0.01
    assert 0.9 < compare["analytic"]["hybrid_raman_paper"]["residual_db"] < 1.1
    assert 1.1 < compare["analytic"]["pcfm_h3_raman"]["residual_db"] < 1.3
    assert 2.0 < compare["historical_500m"]["step_correction_db"] < 2.2
    assert 3.0 < compare["historical_500m"]["analytic_residual_db"]["hybrid_raman_paper"] < 3.2


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

