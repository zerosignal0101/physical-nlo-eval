"""DRA 接收机 EVM 路径测试。

覆盖三件事：

1. 盲分数样点定时恢复能从已知时延中恢复符号相位（合成 DP-QPSK）；
2. c57 主参考的 EVM 回归（改进后应与旧值相差 <1 dB）；
3. cl152 的 EVM 不再饱和在 ~143%，且与频谱 ``snr_nli_db`` 相差 ≤1 dB。

EVM 链路的修复点：色散补偿必须对**全带宽**记录做，使用与传播器相同的
``n/(n-1)`` 频率网格，然后再 demux。若先 demux、再按单信道样本补偿，
信道绝对频偏会留下群延迟与常数相位；cl152 中心信道在 +37.5 GHz，
群延迟约 32.7 symbol，EVM 会饱和到 √2·100%。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from physics_nlo_eval.core.signal import Signal, SignalContext
from physics_nlo_eval.dsp.timing import (
    apply_fractional_delay,
    blind_timing_recovery,
    estimate_fractional_delay,
)
from physics_nlo_eval.tx.mapper import Mapper
from physics_nlo_eval.tx.optics import IQModulator
from physics_nlo_eval.tx.rrc import RRCFilter

REPO_ROOT = Path(__file__).resolve().parents[1]
WIDEBAND_DIR = REPO_ROOT / "examples" / "dra_reference_wide"
GENERATOR_PATH = REPO_ROOT / "examples" / "generate_dra_reference.py"

SYMBOL_RATE_HZ = 64e9
SPS = 4
ROLL_OFF = 0.1


def _load_generator():
    """把参考数据生成脚本作为模块加载，复用光纤参数与 EVM 链路。"""
    spec = importlib.util.spec_from_file_location("_dra_reference_generator_evm", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clean_dp_qpsk(n_symbols: int = 256, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """返回 (发送端 4-sps 波形, 理想符号)，形状 (n_pol, ...)。"""
    rng = np.random.default_rng(seed)
    mapper = Mapper(modulation="DP-QPSK", power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    bits = rng.integers(0, 2, size=n_symbols * mapper.bits_per_symbol * mapper.n_pol, dtype=np.int8).astype(bool)
    symbols = mapper.forward(bits)
    shaped = RRCFilter(sps=SPS, roll_off=ROLL_OFF).forward(symbols)
    waveform = IQModulator(launch_power_dbm_per_pol=0.0).forward(shaped)
    return waveform.data, symbols.data


def _matched_filter(data: np.ndarray) -> Signal:
    ctx = SignalContext(
        sample_rate=SYMBOL_RATE_HZ * SPS,
        center_freq=193.1e12,
        n_pol=data.shape[0],
        sps=SPS,
        sym_rate=SYMBOL_RATE_HZ,
        bits_per_symbol=2,
    )
    return RRCFilter(sps=SPS, roll_off=ROLL_OFF, is_matched_filter=True).forward(Signal(data=data, ctx=ctx))


def _interior_evm(received: np.ndarray, ideal: np.ndarray, ignore: int = 20, trim: int = 10) -> float:
    """符号域 EVM，跳过两端滤波器暂态。"""
    received = received[:, ignore:]
    ideal = ideal[:, ignore:]
    n_symbols = min(received.shape[1], ideal.shape[1]) - trim
    received = received[:, :n_symbols]
    ideal = ideal[:, :n_symbols]
    gain = np.vdot(ideal.ravel(), received.ravel()) / np.vdot(ideal.ravel(), ideal.ravel())
    return float(np.sqrt(np.mean(np.abs(received - gain * ideal) ** 2) / np.mean(np.abs(gain * ideal) ** 2)) * 100.0)


@pytest.mark.parametrize("delay_samples", [0.37, 1.30, -1.20, 1.60])
def test_blind_timing_recovers_known_fractional_delay(delay_samples: float) -> None:
    """已知分数样点延迟经盲定时恢复后，符号 EVM 应 <1%。"""
    waveform, ideal = _clean_dp_qpsk()
    delayed = apply_fractional_delay(
        Signal(
            data=waveform,
            ctx=SignalContext(
                sample_rate=SYMBOL_RATE_HZ * SPS,
                center_freq=193.1e12,
                n_pol=2,
                sps=SPS,
                sym_rate=SYMBOL_RATE_HZ,
                bits_per_symbol=2,
            ),
        ),
        delay_samples,
    )

    matched = _matched_filter(delayed.data)
    estimate = estimate_fractional_delay(matched, ignore_symbols=20)
    # 估计值是“校正量”，因此应与施加的延迟反号。
    assert estimate == pytest.approx(-delay_samples, abs=0.02)

    corrected = blind_timing_recovery(matched, ignore_symbols=20)
    received = corrected.data[:, ::SPS]
    evm = _interior_evm(received, ideal)
    assert evm < 1.0, f"定时恢复后 EVM={evm:.3f}% 应 <1%"


def test_blind_timing_is_noop_for_aligned_signal() -> None:
    """已对齐的信号不应被盲定时搜索无谓地移动。"""
    waveform, _ = _clean_dp_qpsk()
    matched = _matched_filter(waveform)
    corrected = blind_timing_recovery(matched, ignore_symbols=20)
    assert np.array_equal(corrected.data, matched.data)


def _load_wideband(name: str):
    document = json.loads((WIDEBAND_DIR / f"{name}.json").read_text(encoding="utf-8"))
    arrays = np.load(WIDEBAND_DIR / f"{name}.npz")
    generator = _load_generator()
    plan = generator.WIDEBAND_PLANS[name].with_num_symbols(document["signal"]["num_symbols"])
    wdm_input, bits, constellation = generator.build_wideband_dataset(plan, document["signal"]["seed"])
    # 重建的比特必须与落盘波形一一对应，否则 EVM 的参照比特错位。
    assert np.array_equal(bits, arrays[f"bits_{name}"])
    return document, arrays, generator, plan, wdm_input, constellation


def _recompute_evm(name: str, document, arrays, generator, plan, wdm_input, constellation):
    center_idx = plan.n_channels // 2
    results = {}
    for config in document["configs"]:
        nonlinear = Signal(
            data=arrays[f"nl_{name}_{config}"].astype(np.complex128), ctx=wdm_input.ctx
        )
        linear = Signal(
            data=arrays[f"lin_{name}_{config}"].astype(np.complex128), ctx=wdm_input.ctx
        )
        results[config] = generator.center_channel_evm(
            nonlinear,
            linear,
            center_idx,
            arrays[f"bits_{name}"][center_idx],
            constellation,
            fiber_length_km=generator.WIDEBAND_FIBER_LENGTH_KM,
            beta2_ps2_per_km=generator.WIDEBAND_BETA2_PS2_PER_KM,
            beta3_ps3_per_km=generator.WIDEBAND_BETA3_PS3_PER_KM,
            sps=plan.sps,
            roll_off=plan.roll_off,
        )
    return results


def test_c57_receiver_evm_regression() -> None:
    """c57 三个配置的 EVM 不应被全带宽 CDC 修改恶化。"""
    document, arrays, generator, plan, wdm_input, constellation = _load_wideband("c57")
    results = _recompute_evm("c57", document, arrays, generator, plan, wdm_input, constellation)

    expected = {
        "nopump": 2.164,
        "fwd_0p20w": 3.825,
        "bwd_0p20w": 2.223,
    }
    for config, (evm_percent, snr_db, _, _) in results.items():
        assert evm_percent < 4.0, f"{config}: EVM={evm_percent:.3f}% 异常偏高"
        assert snr_db > 28.0, f"{config}: SNR={snr_db:.3f} dB 异常偏低"
        # 与旧路径落盘值相差应远小于 1 dB（本次修复为轻微改善）。
        old_snr = 20 * np.log10(100.0 / expected[config])
        assert abs(snr_db - old_snr) < 1.0


def test_cl152_receiver_evm_matches_spectral_snr() -> None:
    """cl152 的 EVM 不再饱和，且与频谱 NLI SNR 相差 ≤1 dB。"""
    document, arrays, generator, plan, wdm_input, constellation = _load_wideband("cl152")
    results = _recompute_evm("cl152", document, arrays, generator, plan, wdm_input, constellation)

    center_idx = plan.n_channels // 2
    for config, (evm_percent, snr_db, _, _) in results.items():
        assert evm_percent < 10.0, f"{config}: EVM={evm_percent:.3f}%（旧路径为 ~143%）"
        spectral_snr = document["configs"][config]["per_channel"][center_idx]["snr_nli_db"]
        assert abs(snr_db - spectral_snr) <= 1.0, (
            f"{config}: EVM SNR={snr_db:.3f} dB vs 频谱 {spectral_snr:.3f} dB 相差 >1 dB"
        )
        # 落盘 JSON 必须与当前接收机路径一致。
        assert document["configs"][config]["center_channel_evm_percent"] == pytest.approx(
            evm_percent, rel=0.0, abs=1e-6
        )
        assert document["configs"][config]["center_channel_snr_evm_db"] == pytest.approx(
            snr_db, rel=0.0, abs=1e-6
        )
