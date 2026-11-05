"""DRA 调制口径诊断脚本的单元测试。

覆盖：

* 自建 QPSK 输入与 ``build_wideband_dataset`` 逐位一致；
* 同一 DRA 步长下两条路径的中心信道 SNR 一致（容差 1e-6 dB）；
* 高斯符号的每偏振平均功率与 QPSK 星座归一化功率相同（<1e-9 相对）；
* 诊断 JSON 文档包含必需字段并正确配对 QPSK/高斯 NLI 比。

小 plan 用 3 信道、32 symbols、500 m、单 seed，传播成本可忽略。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from physics_nlo_eval.tx.mapper import Constellation

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
CHECKER_PATH = EXAMPLES_DIR / "check_dra_modulation.py"

STEP_M = 500.0
SEED = 7


def _load_checker():
    """把诊断脚本作为模块加载；它内部会导入参考生成器。"""
    spec = importlib.util.spec_from_file_location("_dra_modulation_checker", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_plan(generator):
    """3 信道、32 symbols 的小型全 WDM 计划。"""
    return generator.WidebandPlan(
        name="modulation_probe",
        n_channels=3,
        channel_spacing_hz=75e9,
        center_frequency_hz=193.75e12,
        symbol_rate_hz=64e9,
        roll_off=0.1,
        sps=4,
        sample_rate_hz=4.608e12,
        num_symbols=32,
    )


def test_qpsk_input_matches_reference_bitwise() -> None:
    """自建 QPSK 波形必须与 build_wideband_dataset 逐位一致。"""
    checker = _load_checker()
    generator = checker.generator
    plan = _tiny_plan(generator)

    ours, bits, _ = checker.build_modulated_dataset(plan, SEED, "qpsk")
    reference, ref_bits, _ = generator.build_wideband_dataset(plan, SEED)

    assert ours.data.shape == reference.data.shape
    assert np.array_equal(ours.data, reference.data)
    assert np.array_equal(bits, ref_bits)
    self_check = checker.qpsk_self_check(plan, SEED, ours)
    assert self_check["bit_exact"] is True
    assert self_check["max_abs_diff"] == 0.0


def test_qpsk_center_snr_matches_reference() -> None:
    """同一波形/步长下自建路径与参考路径的中心 SNR 差 <1e-6 dB。"""
    checker = _load_checker()
    generator = checker.generator
    plan = _tiny_plan(generator)

    ours, _, _ = checker.build_modulated_dataset(plan, SEED, "qpsk")
    reference, _, _ = generator.build_wideband_dataset(plan, SEED)
    metrics_ours = generator.wideband_center_channel_metrics(plan, ours, STEP_M, [], "depleted")
    metrics_ref = generator.wideband_center_channel_metrics(plan, reference, STEP_M, [], "depleted")
    assert abs(metrics_ours["snr_nli_db"] - metrics_ref["snr_nli_db"]) < 1e-6
    assert abs(metrics_ours["nli_power_w"] - metrics_ref["nli_power_w"]) / metrics_ref["nli_power_w"] < 1e-9


def test_gaussian_symbol_power_matches_qpsk() -> None:
    """高斯符号每偏振平均功率与 QPSK 星座归一化功率相对差 <1e-9。

    这里比较两条路径的符号功率定义（float64）：QPSK 的
    ``Constellation(power_norm=True)`` 把平均功率归一化到 1，高斯符号
    也按有限样本归一化到 1。进入 RRC/IQ 时两者都转成 ``complex64``，
    IQModulator 再按 complex64 记录值把每偏振功率标到目标发射功率；
    该 dtype 量化残差约 1e-7，不属于调制口径差。
    """
    checker = _load_checker()
    generator = checker.generator
    plan = _tiny_plan(generator)

    qpsk_power = float(np.mean(np.abs(Constellation("DP-QPSK", power_norm=True).lut) ** 2))
    rng = np.random.default_rng(SEED)
    gaussian = checker.gaussian_symbols(rng, n_pol=2, num_symbols=plan.num_symbols)
    gaussian_power = np.mean(np.abs(gaussian) ** 2, axis=1)
    assert np.all(np.abs(gaussian_power - qpsk_power) / qpsk_power < 1e-9)

    # 两条路径使用同一个 IQModulator 发射功率设置。
    qpsk_channels, _ = checker.build_modulated_channels(plan, SEED, "qpsk")
    gaussian_channels, _ = checker.build_modulated_channels(plan, SEED, "gaussian")
    expected_power_dbm = plan.launch_power_dbm_per_pol
    for signal in (*qpsk_channels, *gaussian_channels):
        assert signal.ctx.meta["tx_launch_power_dbm_per_pol"] == expected_power_dbm
    # 传播前的实测每偏振功率只受 complex64 归一化精度限制。
    qpsk_optical = np.mean(np.abs(qpsk_channels[0].data) ** 2, axis=1)
    gaussian_optical = np.mean(np.abs(gaussian_channels[0].data) ** 2, axis=1)
    assert np.all(np.abs(gaussian_optical - qpsk_optical) / qpsk_optical < 1e-6)


def test_document_schema_and_nli_comparison() -> None:
    """小 plan 上跑一对 QPSK/高斯，文档应含必需字段与 NLI 比。"""
    checker = _load_checker()
    generator = checker.generator
    plan = _tiny_plan(generator)

    rows = []
    for modulation in ("qpsk", "gaussian"):
        row, _ = checker.run_combination(
            plan,
            plan_name=plan.name,
            seed=SEED,
            modulation=modulation,
            config_name="nopump",
            pumps=[],
            step_m=STEP_M,
            sample_rate_factor=1,
            pump_model="depleted",
        )
        rows.append(row)

    document = checker.build_document(
        rows,
        plan=plan,
        config_name="nopump",
        pump_model="depleted",
        sample_rate_factor=1,
        num_symbols=plan.num_symbols,
    )
    for row in document["rows"]:
        for field in checker.REQUIRED_ROW_FIELDS:
            assert field in row, f"missing row field {field!r}"
        assert row["elapsed_s"] >= 0.0
        assert np.isfinite(row["snr_nli_db"])

    assert document["schema_version"] == 1
    assert document["qpsk_self_check"]["all_bit_exact"] is True
    assert document["sample_rate_factor"] == 1
    assert document["num_symbols"] == plan.num_symbols
    assert set(document["modulations"]) == {"qpsk", "gaussian"}
    assert len(document["comparisons"]) == 1
    comparison = document["comparisons"][0]
    assert comparison["seed"] == SEED
    assert comparison["nli_ratio_db"] is not None
    assert np.isfinite(comparison["nli_ratio_db"])
    # 精确关系：SNR = P_sig / NLI，故 nli_ratio = signal_ratio - delta_snr。
    assert abs(comparison["nli_ratio_db"] - (comparison["signal_power_ratio_db"] - comparison["delta_snr_db"])) < 1e-6
    assert document["nli_ratio_db_summary"]["n"] == 1


def test_make_plan_validates_sample_rate_factor() -> None:
    """采样率倍数必须为正，且放大后仍保持整数重采样比。"""
    checker = _load_checker()
    plan = checker.make_plan("cl152", 2, 32)
    assert plan.sample_rate_hz == 2 * checker.generator.WIDEBAND_PLANS["cl152"].sample_rate_hz
    assert plan.resample_ratio == 90
    with pytest.raises(ValueError, match=">= 1"):
        checker.make_plan("cl152", 0, 32)
    with pytest.raises(KeyError, match="unknown plan"):
        checker.make_plan("unknown", 1, 32)
