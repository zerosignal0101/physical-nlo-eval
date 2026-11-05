# cl152 nopump：2x 采样（alias-free）seed 集成

> **2026-11-05 更新（4-seed 128-symbol）**：把同一 2×、0.244 m 设置扩到
> seeds 20261104/05/06/07 后，128-symbol 中心 SNR 为 33.502/34.247/
> **23.178**/31.422 dB，均值 **30.587 ± 5.082 dB**、中位数 32.462 dB。
> 本文原先的 2-seed **33.87 ± 0.53 dB 只取了两个偏高 seed，不稳健**；
> “128 symbols 把散布收窄到 0.53 dB”的结论作废。64-symbol 4-seed
> 30.06 ± 5.56 dB 与 128-symbol 4-seed 30.59 ± 5.08 dB 一致，说明
> cl152 的绝对 NLI 在 128 symbols、4 seeds 下仍未统计收敛。
> 4-seed 数据在 `examples/dra_reference_wide/diagnostics/cl152_modulation_2x_128sym.json`；
> 详见 `docs/dra-waveform.md` §12.2.1/§12.2.3。

本文补充 `docs/dra-waveform.md` §12.2.1/§12.2.2 的单次实现结果。
`dra-waveform.md` 中的 alias-free 数字来自固定 seed 20261104 的 64-symbol
探针；本文用多个随机种子量化 cl152 nopump 中心信道 SNR 的 seed 散布，
并检查记录长度的影响。

## 结论摘要

- **64 symbols、2x 采样、0.244140625 m 的 4-seed 集成为
  30.06 ± 5.56 dB（样本标准差），范围 24.45–35.63 dB。** 单次实现
  35.635 dB 是这 4 个点中的最大值，比集成均值高 5.57 dB。
- **“1–2 dB seed 散布”不适用于 cl152 的 64-symbol 记录。** 实测
  极差 11.19 dB，样本标准差 5.56 dB。4 个种子远不足以给出 sub-dB
  均值：均值的标准误约 2.78 dB。
- **128 symbols 的 2-seed 33.87 ± 0.53 dB 不稳健。** 扩到 4 seed 后为
  33.50 / 34.25 / 23.18 / 31.42 dB，均值 **30.59 ± 5.08 dB**、
  中位数 32.46 dB；原来的 0.53 dB 只是两个偏高 seed 的形式值。
- **目前没有收敛的单一 alias-free 中心值。** 64 symbols 与 128 symbols
  的 4-seed 均值分别为 30.06 和 30.59 dB，标准差 5.56/5.08 dB；
  绝对 NLI 需要更多 seed 或更长记录，不能引用单次/2-seed 的
  33.9 dB 或 35–36 dB。

## 方法

`examples/check_dra_cl152_ensemble.py`：

1. 由 `WIDEBAND_PLANS["cl152"]` 构造 2x 采样计划：
   `dataclasses.replace(plan, sample_rate_hz=2*11.52e12)`，上采样比
   `resample_ratio = 90`（脚本校验该比值为整数）。
2. `build_wideband_dataset(plan.with_num_symbols(N), seed)` 生成 DP-QPSK
   全 WDM 输入；中心信道为 index 76（相对波段中心 +37.5 GHz）。
3. `wideband_center_channel_metrics(plan, input, 0.244140625, [], "depleted")`，
   即 nopump、耗尽泵模型、固定步长。
4. 每个 seed 在独立进程中传播；合并时对 `snr_nli_db` 与 `nli_power_w`
   计算算术均值、样本标准差（ddof=1）、min/max。

2x 是 alias-free 的依据来自既有诊断：cl152 在 0.977 m 下 2x 与 3x
只差 0.003 dB（`cl152_nopump_3x_step_0p977_64sym.json`）。

## 结果

### 64 symbols（2x，0.244140625 m）

| seed | 中心 SNR (dB) | NLI (W) | 信号功率 (W) |
|---:|---:|---:|---:|
| 20261104 | 35.6347 | 1.26797e-8 | 4.64063e-5 |
| 20261105 | 26.2162 | 1.10757e-7 | 4.63441e-5 |
| 20261106 | 24.4457 | 1.69938e-7 | 4.72994e-5 |
| 20261107 | 33.9616 | 1.88133e-8 | 4.68407e-5 |
| **mean ± std** | **30.0645 ± 5.5556** | **7.8047e-8 ± 7.5928e-8** | — |
| min / max | 24.4457 / 35.6347 | 1.2680e-8 / 1.6994e-7 | — |

### 128 symbols（2x，0.244140625 m）

| seed | 中心 SNR (dB) | NLI (W) | 信号功率 (W) |
|---:|---:|---:|---:|
| 20261104 | 33.5015 | 2.11845e-8 | 4.74427e-5 |
| 20261105 | 34.2472 | 1.79122e-8 | 4.76289e-5 |
| **mean ± std** | **33.8744 ± 0.5273** | **1.9548e-8 ± 2.3139e-9** | — |
| min / max | 33.5015 / 34.2472 | 1.7912e-8 / 2.1185e-8 | — |

### 与既有 1x / 单次 2x 对照

| 配置 | symbols | seed | 中心 SNR (dB) | 说明 |
|---|---:|---:|---:|---|
| 1x，0.244 m | 64 | 20261104 | 34.2264 | 既有单次实现 |
| 2x，0.244 m | 64 | 20261104 | 35.6347 | 既有单次实现，也是本文 4-seed 最大值 |
| 2x，0.244 m | 64 | 4 seeds | 30.0645 ± 5.5556 | 本文集成 |
| 2x，0.244 m | 128 | 20261104 | 33.5015 | 本文 |
| 2x，0.244 m | 128 | 20261105 | 34.2472 | 本文 |
| 2x，0.244 m | 128 | 2 seeds | 33.8744 ± 0.5273 | 本文集成 |

同一 seed 20261104 的 1x→2x 差为 +1.41 dB，与 `dra-waveform.md`
记录的 aliasing 偏差一致；但 64→128 symbols 的差为 −2.13 dB，
量级相当甚至更大，说明在 cl152 上**记录长度与采样率是同一个量级的
系统误差，不能只校正混叠而忽略记录长度**。

## Caveat：为什么 64 symbols 的散布远大于 1–2 dB

cl152 的占用带宽约 11.32 THz。按全带最外信道相对波段中心的频偏
5.6625 THz、`beta2 = -21.7 ps^2/km`、100 km 计算，群速度走离为

```text
2*pi * 5.6625 THz * 21.7 ps^2/km * 100 km * 64 GBd
    ≈ 4.94e3 symbols
```

（`wideband_walkoff_per_step_symbols(cl152_plan, 1.0) * 1e5`）。
64/128-symbol 的周期记录分别只有走离长度的约 1/77 和 1/39；跨信道
FWM 在记录边界处反复“绕回”，其带内分量取决于随机符号图案，因此
同样的步长/采样率下，不同 seed 的带内 NLI 可以相差一个数量级。
这解释了两个现象：

- 64-symbol 的 4-seed 极差 11.19 dB，而 128-symbol 的两个 seed
  收敛到 0.75 dB 以内；
- seed 20261105 在 64 symbols 是 26.22 dB，在 128 symbols 变为
  34.25 dB，不能把 64-symbol 的低值归因于“坏 seed”。

因此：

1. **不要把 64-symbol 的单次 35.635 dB 当作 alias-free 基准。**
   它是高尾实现；本文 4-seed 均值只有 30.06 dB，且均值本身的标准误
   约 2.78 dB。按当前 5.56 dB 的散布，即使 30 个 seed 也只能把
   均值的标准误压到约 1 dB。
2. **4 个 seed 不足以引用 sub-dB 均值。** 若要保持 sub-dB 标准误，
   在 64 symbols 下需要约 30 个独立 seed；更实际的做法是先把记录
   长度增加到至少 128 symbols（走离仍远大于记录，但散布已明显
   收窄），再用 4–6 个 seed。
3. **128-symbol 的 4-seed 均值 30.59 ± 5.08 dB（中位数 32.46）
   才是当前可引用的分布。** 原先的 2-seed 33.87 ± 0.53 dB 是两个
   偏高实现的均值，标准误 0.37 dB 只是 n=2 的形式值；扩到 4 seed
   后标准差为 5.08 dB，说明 128 symbols 仍未统计收敛。
4. 0.244 m 的步长本身对 cl152 尚未严格收敛（`dra-waveform.md`
   §12.2.1 的 64-symbol 外推约 36.05 dB），但该外推同样建立在
   64-symbol 记录上，不能与本文的 128-symbol 结果直接拼接。

## 复现

```bash
cd /home/zsig/Documents/gits/wt/dra-cl152-ensemble
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_cl152_ensemble.py --seed-single 20261104 --symbols 64
# 其余 seed 同理；全部完成后：
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_cl152_ensemble.py --combine \
    --seeds 20261104 20261105 20261106 20261107 --symbols 64

# 128 symbols
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_cl152_ensemble.py --seed-single 20261104 --symbols 128 \
    --output examples/dra_reference_wide/diagnostics/cl152_nopump_2x_ensemble_128sym.json
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_cl152_ensemble.py --combine \
    --seeds 20261104 20261105 --symbols 128 \
    --output examples/dra_reference_wide/diagnostics/cl152_nopump_2x_ensemble_128sym.json
```

也可在单进程内用 `--workers N` 并行多个 seed。

结果文件（均为小 JSON，不含波形）：

- `examples/dra_reference_wide/diagnostics/cl152_nopump_2x_ensemble.json`
- `examples/dra_reference_wide/diagnostics/cl152_nopump_2x_ensemble_seed*.json`
- `examples/dra_reference_wide/diagnostics/cl152_nopump_2x_ensemble_128sym.json`
- `examples/dra_reference_wide/diagnostics/cl152_nopump_2x_ensemble_128sym_seed*.json`
