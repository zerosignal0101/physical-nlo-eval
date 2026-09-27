"""TimingRecovery 和 Resampler 测试。

测试内容:
1. 理想定时（无偏移）下的 round-trip 验证
2. 已知分数时延偏移的补偿验证
3. Resampler + TimingRecovery 联合 pipeline
4. AWGN 信道下的定时恢复性能
5. 高阶 QAM (16/64/256QAM) 定时恢复
6. Oerder-Meyer 相位估计精度验证
"""

import numpy as np

from physics_nlo_eval.core.signal import Signal
from physics_nlo_eval.dsp.time_domain import Resampler, TimingRecovery
from physics_nlo_eval.metrics.ber_evm import Demapper
from physics_nlo_eval.tx.mapper import Mapper


def add_awgn(symbols: np.ndarray, eb_n0_db: float, bits_per_symbol: int) -> np.ndarray:
    """给符号添加 AWGN 噪声."""
    eb_n0 = 10 ** (eb_n0_db / 10)
    es = np.mean(np.abs(symbols) ** 2)
    eb = es / bits_per_symbol
    n0 = eb / eb_n0
    sigma = np.sqrt(n0 / 2)
    noise = sigma * (np.random.randn(*symbols.shape) + 1j * np.random.randn(*symbols.shape))
    return symbols + noise


def apply_fractional_delay(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    """对信号施加分数样点延迟（通过 Farrow 插值）。

    Parameters
    ----------
    sig : np.ndarray
        信号，形状 (n_pol, n_samples)
    delay_samples : float
        延迟量（分数样点），正值表示延迟

    Returns
    -------
    np.ndarray
        延迟后的信号
    """
    n_pol, n = sig.shape
    out = np.zeros_like(sig, dtype=np.complex128)

    # 两端各补 2 个样点（边缘保持），让首尾样点也走同一套插值。
    # 早期版本在这两处直接 `out[p, i] = sig[p, i]`，等于**没有**施加延迟：
    # 第 0 个样点因此比它该在的位置早 0.25 个样点，而定时恢复对首个符号的
    # 取样时刻恰好落在它附近，三次插值把这个台阶摊到了符号 0 上。
    # 实测符号 0 的残差 0.0767 正好是 256QAM 最小判决距离 0.1534 的一半，
    # 于是稳定地翻 1 个比特 —— 与定时恢复算法本身无关。
    pad = 2
    for p in range(n_pol):
        padded = np.pad(sig[p], pad, mode="edge")
        for i in range(n):
            fi = i - delay_samples + pad
            base = int(np.floor(fi))
            mu = fi - base

            xm1 = padded[base - 1]
            x0 = padded[base]
            x1 = padded[base + 1]
            x2 = padded[base + 2]

            c0 = x0
            c1 = -(1 / 3) * xm1 - 0.5 * x0 + x1 - (1 / 6) * x2
            c2 = 0.5 * xm1 - x0 + 0.5 * x1
            c3 = -(1 / 6) * xm1 + 0.5 * x0 - 0.5 * x1 + (1 / 6) * x2

            out[p, i] = ((c3 * mu + c2) * mu + c1) * mu + c0

    return out


# ============================================================================
# 测试 1: 理想定时 — 无偏移 round-trip
# ============================================================================


def test_perfect_timing_no_offset():
    """验证 TimingRecovery 在理想（无定时偏移）情况下的行为。

    输入: Mapper 生成 1-sps 符号，Resampler 上采样到 4 sps，
          TimingRecovery 恢复回 1 sps。
    期望: 定时恢复后 BER ≈ 0（符号应完美对齐）。
    """
    print("\n" + "=" * 60)
    print("测试 1: 理想定时无偏移 round-trip")
    print("=" * 60)

    np.random.seed(42)
    modulation = "SP-QPSK"
    num_symbols = 2000

    mapper = Mapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = num_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    signal_1sps = mapper.forward(tx_bits)

    resampler = Resampler(target_sps=4)
    signal_4sps = resampler.forward(signal_1sps)

    assert signal_4sps.ctx.sps == 4
    assert signal_4sps.data.shape[0] == n_pol

    timing = TimingRecovery(sps_in=4)
    recovered = timing.forward(signal_4sps)

    assert recovered.ctx.sps == 1

    demapper = Demapper(modulation=modulation, power_norm=True)
    rx_bits = demapper.forward(recovered)

    min_len = min(len(tx_bits), len(rx_bits))
    bit_errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    ber = bit_errors / min_len

    print(f"  调制: {modulation}, 符号数: {num_symbols}")
    print(f"  比特错误: {bit_errors} / {min_len}")
    print(f"  BER: {ber:.2e}")

    assert bit_errors == 0, f"理想定时下有 {bit_errors} 个比特错误，期望 0"
    print("  [PASS] 理想定时无偏移测试通过")


# ============================================================================
# 测试 2: 已知分数时延偏移的补偿
# ============================================================================


def test_fractional_timing_offset_compensation():
    """验证 TimingRecovery 能补偿已知的分数样点定时偏移。

    方法:
    1. 生成 SP-QPSK 符号，上采样到 4 sps
    2. 施加 0.3 样点分数延迟
    3. TimingRecovery 恢复
    4. 验证 BER ≈ 0

    注意: 直接对延迟信号 demap 的 BER ≈ 50%（定时完全失效），
          定时恢复后 BER = 0 证明算法有效。
    """
    print("\n" + "=" * 60)
    print("测试 2: 分数时延偏移补偿 (delay=0.3 样点)")
    print("=" * 60)

    np.random.seed(123)
    modulation = "SP-QPSK"
    num_symbols = 2000
    delay = 0.3

    mapper = Mapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = num_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    signal_1sps = mapper.forward(tx_bits)

    resampler = Resampler(target_sps=4)
    signal_4sps = resampler.forward(signal_1sps)

    # 施加分数延迟
    delayed_data = apply_fractional_delay(signal_4sps.data, delay)
    signal_delayed = Signal(data=delayed_data, ctx=signal_4sps.ctx)

    # 基准: 不做定时恢复，直接 demap
    demapper = Demapper(modulation=modulation, power_norm=True)
    rx_direct = demapper.forward(signal_delayed)
    min_len_direct = min(len(tx_bits), len(rx_direct))
    ber_direct = np.sum(tx_bits[:min_len_direct] != rx_direct[:min_len_direct]) / min_len_direct
    print(f"  直接 demap (无定时恢复): BER = {ber_direct:.4f} (期望 ≈ 0.5, 定时失效)")

    # 定时恢复
    timing = TimingRecovery(sps_in=4)
    recovered = timing.forward(signal_delayed)
    rx_bits = demapper.forward(recovered)

    min_len = min(len(tx_bits), len(rx_bits))
    bit_errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    ber = bit_errors / min_len

    print(f"  调制: {modulation}, 延迟: {delay} 样点, 符号数: {num_symbols}")
    print(f"  比特错误: {bit_errors} / {min_len}")
    print(f"  BER: {ber:.2e}")

    # 直接 demap 的 BER ≈ 50% 证明延迟破坏了采样
    # 定时恢复后 BER = 0 证明算法有效
    assert ber_direct > 0.40, f"延迟信号基准 BER ({ber_direct:.2f}) 不足以证明定时已破坏"
    assert bit_errors == 0, f"分数时延补偿后有 {bit_errors} 错误，期望 0"
    print("  [PASS] 分数时延偏移补偿测试通过")


# ============================================================================
# 测试 3: 双偏振 DP-16QAM 定时恢复
# ============================================================================


def test_dp_16qam_timing_recovery():
    """验证 DP-16QAM 的定时恢复."""
    print("\n" + "=" * 60)
    print("测试 3: DP-16QAM 双偏振定时恢复")
    print("=" * 60)

    np.random.seed(777)
    modulation = "DP-16QAM"
    num_symbols = 1000
    delay = 0.25

    mapper = Mapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol
    n_pol = mapper.n_pol
    total_bits = num_symbols * bits_per_symbol * n_pol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    signal_1sps = mapper.forward(tx_bits)

    resampler = Resampler(target_sps=4)
    signal_4sps = resampler.forward(signal_1sps)

    delayed_data = apply_fractional_delay(signal_4sps.data, delay)
    signal_delayed = Signal(data=delayed_data, ctx=signal_4sps.ctx)

    timing = TimingRecovery(sps_in=4)
    recovered = timing.forward(signal_delayed)

    demapper = Demapper(modulation=modulation, power_norm=True)
    rx_bits = demapper.forward(recovered)

    min_len = min(len(tx_bits), len(rx_bits))
    bit_errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    ber = bit_errors / min_len

    print(f"  调制: {modulation}, 延迟: {delay} 样点, 符号数: {num_symbols}")
    print(f"  比特错误: {bit_errors} / {min_len}")
    print(f"  BER: {ber:.2e}")

    assert bit_errors == 0, f"DP-16QAM 定时恢复有 {bit_errors} 错误"
    print("  [PASS] DP-16QAM 定时恢复测试通过")


# ============================================================================
# 测试 4: AWGN 信道下的定时恢复
# ============================================================================


def test_timing_recovery_with_awgn():
    """验证 AWGN 信道下定时恢复的稳定性。

    场景: SP-QPSK, Eb/N0=10dB, 0.2 样点延迟
    期望: BER 应与无定时偏移的 baseline 相近（定时恢复有效）。
    """
    print("\n" + "=" * 60)
    print("测试 4: AWGN 信道下定时恢复 (Eb/N0=10dB)")
    print("=" * 60)

    np.random.seed(999)
    modulation = "SP-QPSK"
    num_symbols = 2000
    eb_n0_db = 10.0
    delay = 0.2

    mapper = Mapper(modulation=modulation, power_norm=True)
    bits_per_symbol = mapper.bits_per_symbol
    total_bits = num_symbols * bits_per_symbol

    tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
    signal_1sps = mapper.forward(tx_bits)
    signal_4sps = Resampler(target_sps=4).forward(signal_1sps)

    noisy_data = add_awgn(signal_4sps.data, eb_n0_db, bits_per_symbol)
    sig_noisy = Signal(data=noisy_data, ctx=signal_4sps.ctx)

    # Case A: 有偏移 + 噪声
    delayed_data = apply_fractional_delay(sig_noisy.data, delay)
    sig_a = Signal(data=delayed_data, ctx=sig_noisy.ctx)

    # Case B: 无偏移 + 噪声
    sig_b = sig_noisy

    timing = TimingRecovery(sps_in=4)

    rec_a = timing.forward(sig_a)
    rec_b = timing.forward(sig_b)

    demapper = Demapper(modulation=modulation, power_norm=True)
    rx_a = demapper.forward(rec_a)
    rx_b = demapper.forward(rec_b)

    min_a = min(len(tx_bits), len(rx_a))
    min_b = min(len(tx_bits), len(rx_b))
    ber_a = np.sum(tx_bits[:min_a] != rx_a[:min_a]) / min_a
    ber_b = np.sum(tx_bits[:min_b] != rx_b[:min_b]) / min_b

    print(f"  调制: {modulation}, Eb/N0={eb_n0_db}dB, 延迟={delay} 样点")
    print(f"  BER (偏移+噪声): {ber_a:.2e}")
    print(f"  BER (无偏移+噪声): {ber_b:.2e}")

    assert ber_a < ber_b * 3 or ber_a < 1e-3, f"定时恢复后 BER ({ber_a:.2e}) 显著差于无偏移基线 ({ber_b:.2e})"
    print("  [PASS] AWGN 信道定时恢复稳定性测试通过")


# ============================================================================
# 测试 5: 高阶 QAM (16/64/256QAM) 定时恢复
# ============================================================================


def test_high_order_qam_timing_recovery():
    """验证高阶 QAM 调制格式的定时恢复有效性。

    高阶 QAM 对采样时刻更敏感，是更严格的测试场景。
    同时对比有无定时恢复的差异。
    """
    print("\n" + "=" * 60)
    print("测试 5: 高阶 QAM 定时恢复")
    print("=" * 60)

    np.random.seed(42)
    modulations = ["SP-16QAM", "SP-64QAM", "SP-256QAM"]
    num_symbols = 2000
    delay = 0.25
    eb_n0_db = 18.0  # 高 SNR 以突出定时误差而非噪声

    for mod in modulations:
        mapper = Mapper(modulation=mod, power_norm=True)
        demapper = Demapper(modulation=mod, power_norm=True)
        bits_per_symbol = mapper.bits_per_symbol
        total_bits = num_symbols * bits_per_symbol

        tx_bits = np.random.randint(0, 2, size=total_bits, dtype=bool)
        sig1 = mapper.forward(tx_bits)
        sig4 = Resampler(target_sps=4).forward(sig1)

        # Case A: 有偏移（无噪声）
        delayed = apply_fractional_delay(sig4.data, delay)
        sig_a = Signal(data=delayed, ctx=sig4.ctx)
        rec_a = TimingRecovery(sps_in=4).forward(sig_a)
        rx_a = demapper.forward(rec_a)
        min_a = min(len(tx_bits), len(rx_a))
        ber_a = np.sum(tx_bits[:min_a] != rx_a[:min_a]) / min_a

        # Case B: 有偏移 + AWGN
        noisy_delayed = add_awgn(delayed, eb_n0_db, bits_per_symbol)
        sig_b = Signal(data=noisy_delayed, ctx=sig4.ctx)
        rec_b = TimingRecovery(sps_in=4).forward(sig_b)
        rx_b = demapper.forward(rec_b)
        min_b = min(len(tx_bits), len(rx_b))
        ber_b = np.sum(tx_bits[:min_b] != rx_b[:min_b]) / min_b

        # Case C: 无偏移 + AWGN (baseline)
        noisy_clean = add_awgn(sig4.data, eb_n0_db, bits_per_symbol)
        sig_c = Signal(data=noisy_clean, ctx=sig4.ctx)
        rec_c = TimingRecovery(sps_in=4).forward(sig_c)
        rx_c = demapper.forward(rec_c)
        min_c = min(len(tx_bits), len(rx_c))
        ber_c = np.sum(tx_bits[:min_c] != rx_c[:min_c]) / min_c

        print(f"\n  {mod} @ Eb/N0={eb_n0_db}dB, delay={delay}")
        print(f"    BER(延迟,无噪): {ber_a:.2e}")
        print(f"    BER(延迟+噪):  {ber_b:.2e}")
        print(f"    BER(无偏移+噪): {ber_c:.2e}")

        # 有定时恢复时，BER 应该远低于无定时恢复的基准
        # (无定时恢复的 BER ≈ 0.5 对随机符号，或由噪声决定)
        assert ber_a == 0.0, f"{mod}: 无噪声延迟下 BER 应为 0，实际 {ber_a:.2e}"
        # 有噪声时，有偏移的 BER 不应比无偏移 baseline 差太多
        assert ber_b < ber_c * 5 or ber_b < 1e-3, f"{mod}: 有偏移+噪声 BER ({ber_b:.2e}) vs 基线 ({ber_c:.2e}) 差异过大"
        print(f"    [PASS] {mod} 定时恢复有效")

    print("\n  [PASS] 高阶 QAM 定时恢复测试通过")


# ============================================================================
# 测试 6: Oerder-Meyer 相位估计精度验证
# ============================================================================


def test_phase_estimation_accuracy():
    """验证 Oerder-Meyer 相位估计的精度和线性响应。

    测试验证:
    1. 相位估计 est 与真实延迟呈线性关系 (|ratio| ≈ 1.0)
    2. sign 的一致性: est / true_UI ≈ +1.0
    3. 估计误差在可接受范围内 (< 0.01 UI)
    """
    print("\n" + "=" * 60)
    print("测试 6: Oerder-Meyer 相位估计精度")
    print("=" * 60)

    np.random.seed(42)
    modulation = "SP-16QAM"
    num_symbols = 5000

    mapper = Mapper(modulation=modulation, power_norm=True)
    sig1 = mapper.forward(np.random.randint(0, 2, size=num_symbols * mapper.bits_per_symbol, dtype=bool))
    sig4 = Resampler(target_sps=4).forward(sig1)

    timing = TimingRecovery(sps_in=4)

    test_delays = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

    print(f"\n  {'延迟(样点)':>10} {'真实UI':>8} {'估计UI':>9} {'ratio':>8} {'|err| UI':>9} {'sign':>8}")
    print("  " + "-" * 58)

    all_passed = True
    for delay in test_delays:
        delayed_data = apply_fractional_delay(sig4.data, delay)
        sig_delayed = Signal(data=delayed_data, ctx=sig4.ctx)

        phases = timing._estimate_phase(sig_delayed.data[0])
        est_ui = phases[0]
        true_ui = delay / 4.0
        ratio = est_ui / true_ui
        abs_error = abs(est_ui - true_ui)

        sign_correct = (est_ui > 0) == (true_ui > 0)
        ratio_ok = abs(abs(ratio) - 1.0) < 0.05
        error_ok = abs_error < 0.01

        status = "✓" if (sign_correct and ratio_ok and error_ok) else "✗"
        if not (sign_correct and ratio_ok and error_ok):
            all_passed = False

        print(f"  {delay:10.2f} {true_ui:8.4f} {est_ui:+9.4f} {ratio:8.3f} {abs_error:9.4f} {status:>8}")

    print()

    # 核心断言: 估计值与真实值成比例，且 sign 一致
    # ratio ≈ 1.0 (而非 -1.0) 才算 sign 正确
    ratios = []
    for delay in test_delays:
        delayed_data = apply_fractional_delay(sig4.data, delay)
        sig_delayed = Signal(data=delayed_data, ctx=sig4.ctx)
        phases = timing._estimate_phase(sig_delayed.data[0])
        true_ui = delay / 4.0
        ratios.append(phases[0] / true_ui)

    mean_ratio = np.mean(ratios)
    print(f"  平均 ratio (est_UI / true_UI): {mean_ratio:.4f}")

    sign_is_positive = mean_ratio > 0.9
    ratio_is_linear = abs(mean_ratio - 1.0) < 0.05
    mean_abs_error = np.mean([abs(r - 1.0) for r in ratios])
    error_is_small = mean_abs_error < 0.01

    print(f"  Sign 验证: ratio ≈ {mean_ratio:.4f} {'✓ (sign 正确)' if sign_is_positive else '✗ (应为正数)'}")
    print(f"  比例线性度: |ratio| = {abs(mean_ratio):.4f} (应接近 1.0)")
    print(f"  平均 |ratio-1|: {mean_abs_error:.4f} {'✓' if error_is_small else '✗'}")

    assert sign_is_positive and ratio_is_linear and error_is_small, (
        f"相位估计精度不足: sign_ok={sign_is_positive}, linear={ratio_is_linear}, "
        f"err_small={error_is_small} (mean_ratio={mean_ratio:.4f}, mean_abs_error={mean_abs_error:.4f})"
    )
    print("  [PASS] 相位估计精度测试通过")


# ============================================================================
# 测试 7: Resampler 与 TimingRecovery 的 ctx 元数据传递
# ============================================================================


def test_ctx_metadata_propagation():
    """验证 Resampler 和 TimingRecovery 正确更新 SignalContext."""
    print("\n" + "=" * 60)
    print("测试 7: SignalContext 元数据传递")
    print("=" * 60)

    np.random.seed(7)
    mapper = Mapper(modulation="SP-BPSK", power_norm=True)
    sig = mapper.forward(np.random.randint(0, 2, size=1000, dtype=bool))

    resampler = Resampler(target_sps=4)
    sig_r = resampler.forward(sig)

    assert sig_r.ctx.sps == 4
    assert sig_r.ctx.meta.get("resampled_sps") == 4
    print(f"  Resampler: sps {sig.ctx.sps} → {sig_r.ctx.sps}, resampled_sps=4")

    timing = TimingRecovery(sps_in=4)
    sig_t = timing.forward(sig_r)

    assert sig_t.ctx.sps == 1
    assert sig_t.ctx.meta.get("timing_method") == "Oerder-Meyer"
    print(f"  TimingRecovery: sps {sig_r.ctx.sps} → {sig_t.ctx.sps}, timing_method=Oerder-Meyer")

    print("  [PASS] SignalContext 元数据传递测试通过")


# ============================================================================
# 运行所有测试
# ============================================================================


def run_all_tests():
    """运行所有 TimingRecovery 和 Resampler 测试."""
    print("\n" + "=" * 60)
    print("TimingRecovery / Resampler 验证测试套件")
    print("=" * 60)

    tests = [
        test_perfect_timing_no_offset,
        test_fractional_timing_offset_compensation,
        test_dp_16qam_timing_recovery,
        test_timing_recovery_with_awgn,
        test_high_order_qam_timing_recovery,
        test_phase_estimation_accuracy,
        test_ctx_metadata_propagation,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            result = test_fn()
            if result is False:
                failed += 1
            else:
                passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
