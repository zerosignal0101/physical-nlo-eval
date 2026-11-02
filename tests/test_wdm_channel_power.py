"""WDM 逐通道功率测试。

验证 ``ChannelSpec.power_dbm_per_pol`` 能在合波前把每个通道缩放到目标
功率，同时不改变未设置功率时的旧行为。
"""

import numpy as np
import pytest

from physics_nlo_eval.channel.wdm import ChannelSpec, WDMMux, WDMPlan
from physics_nlo_eval.core.signal import Signal, SignalContext


def _make_signal(power_dbm_per_pol: float, n_samples: int = 4096) -> Signal:
    """构造一个平均功率已知的双偏振复基带信号。"""
    rng = np.random.default_rng(7)
    amplitude = np.sqrt(10 ** (power_dbm_per_pol / 10) * 1e-3)
    data = rng.choice([-1.0, 1.0], size=(2, n_samples)) + 1j * rng.choice([-1.0, 1.0], size=(2, n_samples))
    data /= np.sqrt(2.0)
    data *= amplitude
    ctx = SignalContext(
        sample_rate=64e9,
        center_freq=193.1e12,
        n_pol=2,
        sps=2,
        sym_rate=32e9,
        bits_per_symbol=2,
        is_ofdm=False,
    )
    return Signal(data.astype(np.complex128), ctx)


def _per_pol_power_w(signal: Signal) -> np.ndarray:
    """返回每个偏振的平均功率 (W)。"""
    return np.mean(np.abs(signal.data) ** 2, axis=1)


def _mux_one(power_dbm_per_pol: float | None) -> Signal:
    """用单通道 WDMPlan 合波，返回输出信号。"""
    plan = WDMPlan(
        reference_freq_hz=193.1e12,
        channels=[ChannelSpec(channel_id=0, freq_offset_hz=0.0, power_dbm_per_pol=power_dbm_per_pol)],
    )
    mux = WDMMux(wdm_plan=plan, target_sample_rate_hz=64e9)
    return mux.combine([_make_signal(0.0)])


def test_channel_power_scales_muxed_signal():
    """设置 -10 dBm/pol 后，合波输出应为 0.1 mW/pol。"""
    out = _mux_one(-10.0)
    np.testing.assert_allclose(_per_pol_power_w(out), 0.1e-3, rtol=0.02)


def test_channel_power_none_keeps_input_power():
    """未设置功率时保留输入功率，保持向后兼容。"""
    out = _mux_one(None)
    np.testing.assert_allclose(_per_pol_power_w(out), 1.0e-3, rtol=0.02)


def test_channel_power_rejects_non_finite():
    """NaN/Inf 功率必须在合波时被拒绝。"""
    plan = WDMPlan(
        reference_freq_hz=193.1e12,
        channels=[ChannelSpec(channel_id=0, freq_offset_hz=0.0, power_dbm_per_pol=np.nan)],
    )
    with pytest.raises(ValueError, match="finite"):
        WDMMux(wdm_plan=plan, target_sample_rate_hz=64e9).combine([_make_signal(0.0)])


def test_channel_power_rejects_zero_power_signal():
    """零功率输入不能静默缩放成目标功率。"""
    plan = WDMPlan(
        reference_freq_hz=193.1e12,
        channels=[ChannelSpec(channel_id=0, freq_offset_hz=0.0, power_dbm_per_pol=0.0)],
    )
    sig = _make_signal(0.0)
    sig.data[:] = 0.0
    with pytest.raises(ValueError, match="zero-power"):
        WDMMux(wdm_plan=plan, target_sample_rate_hz=64e9).combine([sig])
