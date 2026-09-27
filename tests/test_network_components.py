"""Regression tests for analytical NLI and optical network components."""

import numpy as np
import pytest

from physics_nlo_eval.channel import (
    EDFA,
    ROADM,
    FiberChannel,
    FiberSpan,
    NLIChannel,
    PumpSpec,
    RamanPump,
    assess_fiber_nonlinearity,
)
from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.mimo import MIMO_ChannelEstimation
from physics_nlo_eval.rx_front.adc import ADC
from physics_nlo_eval.tx.optics import PolarizationMultiplexer


def _signal(data: np.ndarray, *, sample_rate: float = 100e9, center_freq: float = 193.1e12) -> Signal:
    ctx = SignalContext(sample_rate=sample_rate, center_freq=center_freq, n_pol=data.shape[0], sps=2)
    return Signal(data.astype(np.complex128), ctx)


def test_nli_is_positive_and_has_cubic_power_scaling() -> None:
    """NLI 随发射功率呈三次方增长 (1 dB 功率对应 3 dB NLI)。"""
    span = FiberSpan(80)
    low = assess_fiber_nonlinearity([NLIChannel(193.1e12, 32e9, 0)], span)[0]
    high = assess_fiber_nonlinearity([NLIChannel(193.1e12, 32e9, 1)], span)[0]
    assert low.nli_power_w > 0
    assert np.isclose(high.nli_power_w / low.nli_power_w, 10 ** 0.3)


def test_polarization_multiplexer_combines_sp_fields() -> None:
    """combine() 把两个单偏振场叠加成 (2, N) 的双偏振信号。"""
    x = _signal(np.ones((1, 32)))
    y = _signal(1j * np.ones((1, 32)))
    output = PolarizationMultiplexer().combine(x, y)
    assert output.data.shape == (2, 32)
    assert output.ctx.n_pol == 2


def test_mimo_ls_recovers_cross_polarization_channel() -> None:
    """LS 估计能还原含偏振串扰的 2x2 信道 (旧的对角近似做不到)。"""
    rng = np.random.default_rng(4)
    training = rng.normal(size=(2, 100)) + 1j * rng.normal(size=(2, 100))
    payload = rng.normal(size=(2, 200)) + 1j * rng.normal(size=(2, 200))
    channel = np.array([[0.8 + 0.1j, 0.3 - 0.2j], [-0.2 - 0.1j, 0.9 + 0.05j]])
    received = channel @ payload
    signal = _signal(received)
    signal.ctx.meta.update({"mimo_training_tx": training, "mimo_training_rx": channel @ training})
    recovered = MIMO_ChannelEstimation("LS")(signal)
    assert np.allclose(recovered.data, payload, atol=1e-6)


def test_adc_rational_resampling_and_amplitude_units() -> None:
    """120→80 GS/s 的非整数倍率走有理重采样，且量化保持物理幅度。"""
    tone = np.exp(2j * np.pi * np.arange(300) / 30)[None, :]
    output = ADC(12, sampling_rate_hz=80e9, clip_ratio=2)(_signal(tone, sample_rate=120e9))
    assert output.data.shape[-1] == 200
    assert np.isclose(np.sqrt(np.mean(np.abs(output.data) ** 2)), 1, rtol=0.02)


def test_edfa_unity_gain_adds_no_ase() -> None:
    """0 dB 增益的 EDFA 不注入任何 ASE (旧公式 G 会凭空产生噪声)。"""
    signal = _signal(np.ones((2, 1024)))
    output = EDFA(0, 5, seed=1)(signal)
    assert np.array_equal(output.data, signal.data)


def test_fiber_numpy_fallback_applies_only_fiber_loss() -> None:
    """PyNLO 缺失时的 NumPy SSFM 回退路径。

    本环境装有 PyNLO，回退分支不会被自动触发，因此这里显式把 ``compute_ctx``
    置空来覆盖它。关闭色散/非线性/PMD 后，输出功率应恰好等于光纤损耗。
    """
    length_km, alpha_db_per_km = 20.0, 0.2
    fiber = FiberChannel(
        length_m=length_km * 1e3,
        alpha_db_per_km=alpha_db_per_km,
        beta2_ps2_per_km=0.0,
        beta3_ps3_per_km=0.0,
        coarse_step_size_m=1e3,
        n2_si=0.0,
        a_eff_um2=80.0,
        d_pmd_ps_per_sqrt_km=0.0,
        compute_ctx_name="numpy",
    )
    fiber.compute_ctx = None  # 强制走回退求解器
    rng = np.random.default_rng(0)
    data = (rng.normal(size=(1, 256)) + 1j * rng.normal(size=(1, 256))).astype(np.complex128)
    output = fiber.forward(_signal(data))

    assert output.ctx.meta["fiber_solver"] == "numpy-ssfm"
    # 单偏振输入会被补零成双偏振，只比较有信号的那一行。
    assert output.data.shape == (2, 256)
    gain_db = 10 * np.log10(np.mean(np.abs(output.data[0]) ** 2) / np.mean(np.abs(data[0]) ** 2))
    assert np.isclose(gain_db, -alpha_db_per_km * length_km, atol=1e-6)


def test_fiber_numpy_fallback_matches_pynlo_dispersion() -> None:
    """两侧求解器的线性传递函数必须一致。

    这条用例同时守住两个曾经出错的约定：

    1. **频率轴**：PyNLO 的 ``Grid.create`` 把 bandwidth 当作 n 个频点的
       inclusive span（间隔 ``fs/(n-1)``），而 ``np.fft.fftfreq`` 给的是
       ``fs/n``。用错会让色散相位带上约 ``2/(n-1)`` 的相对误差。
    2. **色散符号**：写成 ``exp(+1j*beta)`` 会传播出反向色散。

    两者都不会让"仅看幅度/损耗"的测试失败，只有与参考求解器逐点比对才抓得住。
    """
    pytest.importorskip("pynlo")
    rng = np.random.default_rng(11)
    n = 3600
    data = (rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))).astype(np.complex128)
    ctx = SignalContext(sample_rate=1152e9, center_freq=193e12, n_pol=2, sps=36)
    common = dict(
        length_m=80e3,
        alpha_db_per_km=0.2,
        beta2_ps2_per_km=-20.0,
        beta3_ps3_per_km=0.1e-3,
        coarse_step_size_m=200.0,
        n2_si=0.0,  # 线性光纤：两侧分步方式不同，只有线性情形可逐点比较
        a_eff_um2=80.0,
        d_pmd_ps_per_sqrt_km=0.0,
    )

    reference = FiberChannel(compute_ctx_name="cupy", **common).forward(Signal(data, ctx))

    fallback_fiber = FiberChannel(compute_ctx_name="numpy", **common)
    fallback_fiber.compute_ctx = None  # 强制走 NumPy SSFM 回退
    fallback = fallback_fiber.forward(Signal(data, ctx))
    assert fallback.ctx.meta["fiber_solver"] == "numpy-ssfm"

    relative = np.sqrt(np.mean(np.abs(fallback.data - reference.data) ** 2)) / np.sqrt(
        np.mean(np.abs(reference.data) ** 2)
    )
    assert relative < 1e-6, f"回退求解器与 PyNLO 的传递函数不一致: 相对误差 {relative:.3e}"


def test_roadm_and_raman_models_preserve_shape() -> None:
    """ROADM 与 RamanPump 保持波形形状，且泵浦提供净增益。"""
    signal = _signal(np.ones((2, 256)))
    signal.ctx.meta["wdm_channels_info"] = [
        {"channel_id": 7, "freq_offset_hz": 0.0, "original_sample_rate_hz": 50e9}
    ]
    filtered = ROADM([7], insertion_loss_db=3)(signal)
    amplified = RamanPump(PumpSpec(1455, 0.5), 80, add_noise=False)(filtered)
    assert amplified.data.shape == signal.data.shape
    assert np.mean(np.abs(amplified.data) ** 2) > np.mean(np.abs(filtered.data) ** 2)
