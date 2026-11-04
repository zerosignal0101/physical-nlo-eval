"""分布式拉曼放大 (DRA) 波形模型的回归测试。

覆盖：

* 无泵 DRA 与现有 NumPy ``FiberChannel`` 逐位一致；
* 泵浦功率单调性、前向/后向剖面差异；
* 未耗尽开关增益与 lumped ``RamanPump`` 的频率形状一致；
* 可选 ASE 与 EDFA ASE 分离记录；
* 已提交的小规模 3/5 信道参考数据集 schema 与 SNR 数据点。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
