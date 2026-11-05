r"""盲定时（分数样点延迟）恢复。

DRA 参考数据的接收机 EVM 链路在 RRC 匹配滤波之后做一次盲的分数样点
定时搜索。参考信号是 DP-QPSK，因此采用恒模（CMA/Godard）代价：

.. math:: J(\\tau) = \\frac{E[(|y_\\tau|^2 - E|y_\\tau|^2)^2]}{(E|y_\\tau|^2)^2}

代价在正确的符号采样相位处取极小。搜索范围默认 ±0.5 symbol，先以
1/32 symbol 粗搜、再以 1/256 symbol 细搜，精度优于 0.01 symbol。
整个过程不使用发送比特；比特只在 :func:`BERCalculator.calculate_evm`
中用于最终验证。

该恢复步骤主要用于吸收 demux/重采样链可能残留的分数样点延迟。若
CD 已在全带宽上按传播器的频率网格补偿，残余定时通常为 0，此时函数
原样返回输入。
"""

from __future__ import annotations

import numpy as np

from physics_nlo_eval.core.signal import Signal

#: 粗搜 / 细搜步长（单位：symbol）。
_COARSE_STEP_SYMBOLS = 1.0 / 32.0
_FINE_STEP_SYMBOLS = 1.0 / 256.0

#: 低于该延迟（样点）时认为无需补偿，避免无谓的 FFT 舍入。
_ZERO_DELAY_TOL_SAMPLES = 1e-3

#: 只有恒模代价至少改善该比例时才施加校正，避免把噪声引起的
#: 微小伪定时当作真实偏差（会轻微恶化已经对齐的 c57）。
_MIN_RELATIVE_IMPROVEMENT = 0.02


def apply_fractional_delay(signal: Signal, delay_samples: float) -> Signal:
    """用频域相位斜坡对信号施加理想的分数样点延迟。

    ``delay_samples > 0`` 表示把信号向后延迟。信号按记录长度做循环
    移位；本模块只用于不超过半个符号的残余定时，且 EVM 会忽略前导
    符号，因此循环边界不影响结论。
    """
    if delay_samples == 0.0:
        return signal
    n_samples = signal.data.shape[-1]
    freqs = np.fft.fftfreq(n_samples, d=1.0 / signal.ctx.sample_rate)
    spectrum = np.fft.fft(signal.data, axis=-1)
    shifted = np.fft.ifft(
        spectrum * np.exp(-2j * np.pi * freqs * delay_samples / signal.ctx.sample_rate),
        axis=-1,
    )
    return Signal(data=shifted, ctx=signal.ctx)


def _cma_cost(data: np.ndarray, sps: int, ignore_symbols: int) -> float:
    """恒模代价；越小表示采样相位越接近符号最佳点。"""
    symbols = data[:, ::sps][:, ignore_symbols:]
    if symbols.size == 0:
        return np.inf
    power = float(np.mean(np.abs(symbols) ** 2))
    if power <= 0.0:
        return np.inf
    return float(np.mean((np.abs(symbols) ** 2 - power) ** 2) / power**2)


def estimate_fractional_delay(
    signal: Signal,
    *,
    ignore_symbols: int = 0,
    half_range_symbols: float = 0.5,
) -> float:
    """盲估计使恒模代价最小的定时校正量（单位：样点）。

    Parameters
    ----------
    signal : Signal
        匹配滤波后的多偏振信号，``ctx.sps`` 必须为整数。
    ignore_symbols : int, optional
        代价统计时跳过的起始符号数，通常与 EVM 的 ``ignore_headers`` 一致。
    half_range_symbols : float, optional
        搜索半宽（symbol）。默认 ±0.5 symbol 覆盖一个完整符号周期。

    Returns
    -------
    float
        需要施加的延迟（样点）。正值表示延迟信号。
    """
    sps = int(round(float(signal.ctx.sps)))
    if sps <= 1:
        return 0.0

    def cost(delay_symbols: float) -> float:
        shifted = apply_fractional_delay(signal, delay_symbols * sps)
        return _cma_cost(shifted.data, sps, ignore_symbols)

    coarse = np.arange(
        -half_range_symbols,
        half_range_symbols + _COARSE_STEP_SYMBOLS / 2.0,
        _COARSE_STEP_SYMBOLS,
    )
    best_delay = float(coarse[int(np.argmin([cost(d) for d in coarse]))])

    fine = np.arange(
        best_delay - _COARSE_STEP_SYMBOLS,
        best_delay + _COARSE_STEP_SYMBOLS + _FINE_STEP_SYMBOLS / 2.0,
        _FINE_STEP_SYMBOLS,
    )
    best_delay = float(fine[int(np.argmin([cost(d) for d in fine]))])
    return best_delay * sps


def blind_timing_recovery(signal: Signal, *, ignore_symbols: int = 0) -> Signal:
    """对匹配滤波后的信号做盲分数样点定时校正。

    残余定时小于 :data:`_ZERO_DELAY_TOL_SAMPLES`，或恒模代价的改善
    小于 :data:`_MIN_RELATIVE_IMPROVEMENT` 时直接返回原对象。
    """
    delay_samples = estimate_fractional_delay(signal, ignore_symbols=ignore_symbols)
    if abs(delay_samples) < _ZERO_DELAY_TOL_SAMPLES:
        return signal

    sps = int(round(float(signal.ctx.sps)))
    cost_before = _cma_cost(signal.data, sps, ignore_symbols)
    cost_after = _cma_cost(
        apply_fractional_delay(signal, delay_samples).data, sps, ignore_symbols
    )
    if cost_before <= 0.0 or not np.isfinite(cost_before) or not np.isfinite(cost_after):
        return signal
    if (cost_before - cost_after) / cost_before < _MIN_RELATIVE_IMPROVEMENT:
        return signal
    return apply_fractional_delay(signal, delay_samples)
