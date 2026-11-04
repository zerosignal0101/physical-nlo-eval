# 分布式拉曼放大 (DRA) 波形参考

本文档说明 `physics_nlo_eval.channel.dra.DistributedRamanFiber` 的物理模型、
数值近似、参考数据集，以及如何把它作为 `network-nlo-eval` 解析 DRA 模型的
独立波形基准。

## 1. 目的与边界

`network-nlo-eval` 使用解析/半解析的拉曼耦合方程评估 DRA 链路的增益、
功率剖面和噪声。为了验证这些解析模型，本仓提供一条**独立的 SSFM 波形
传播通道**：

- **独立**：不复用解析 Raman ODE，也不把 lumped `RamanPump` 接在光纤后面；
- **可验证**：无泵时与现有 NumPy `FiberChannel` 逐位一致；
- **小规模**：参考数据只覆盖 3/5 信道、50 GBd、100 km 单跨，便于完整复现。

模型支持两种泵浦口径：

| `pump_model` | 泵浦功率 | 用途 |
| --- | --- | --- |
| `"undepleted"`（默认） | 解析衰减，不受信号影响 | 历史数据集，保持逐位兼容 |
| `"depleted"` | 与信号功率耦合求解，泵浦被耗尽 | 与解析 Raman ODE 同口径的参考 |

第一版**不追求**全 WDM 规模（57/479 信道）或泵浦-泵浦耦合。

## 2. 物理模型

### 2.1 传播方程

信号仍由 Manakov/标量 SSFM 传播，包含：

- 色散（β₂、β₃）；
- 光纤损耗 α；
- Kerr 非线性（SPM/XPM，Manakov 因子 8/9）；
- 分布式拉曼增益 `g_R(f_p − f_s)`。

DRA 只增加一个逐 z、逐频率的**线性功率增益率**：

```text
γ_R(f, z) = Σ_p  g_peak · g̃_R(f_p − f) · P_p(z) / A_eff
```

其中 `g̃_R` 是归一化 Gaussian 拉曼谱（峰值 1），`P_p(z)` 是泵浦 p 的局部
功率。每一步 SSFM 在非线性前后各乘上半个幅度因子
`exp(0.25 · γ_R · dz)`（合计 `exp(0.5 · γ_R · dz)` 的幅度增益，
即 `exp(γ_R · dz)` 的功率增益），以保证对称分步。

### 2.2 泵浦传播：未耗尽近似

`pump_model="undepleted"`（默认）显式采用**未耗尽泵浦**模型：

- 前向泵：`P_p(z) = P_p(0) · exp(−α_p z)`；
- 后向泵：`P_p(z) = P_p(L) · exp(−α_p (L − z))`；
- 泵浦损耗由每个 `PumpSpec.attenuation_db_per_km` 单独给出；
- **不考虑**泵浦-信号耗尽、泵浦-泵浦拉曼转移、泵浦间功率交换。

对 0.1–0.2 W 泵浦、mW 级信号，这个近似在低功率下是合理的；但强前向泵会
高估信号增益。

未耗尽近似下，前向和后向泵的**总开关增益相同**，因为二者只差泵浦功率
沿 z 的分布，而 `∫_0^L P_p(z) dz` 相同；局部功率剖面和非线性积累则不同。

### 2.3 泵浦耗尽模型（`pump_model="depleted"`）

耗尽模型在 SSFM 步进之外单独求解一组**总功率耦合方程**：

```text
dP_s/dz = −α_s P_s + P_s · Σ_p γ_p P_p
dP_p/dz = −s_p · [ α_p P_p + (f_p/f_s) γ_p P_p P_s ]
γ_p     = g_peak · g̃_R(f_p − f_s) / A_eff
```

- `P_s(z)` 是**信号带内总功率**（两个偏振合计），不是逐信道功率；
- `s_p = +1` 为前向泵（边界 `P_p(0)` 已知），`s_p = −1` 为后向泵
  （边界 `P_p(L)` 已知）；
- `(f_p/f_s)` 因子保证**光子数守恒**：信号每获得一个光子，泵浦就失去一个；
- `γ_p` 取在信道规划的中心频率。WDM 带宽（≤ 400 GHz）远小于拉曼谱
  σ = 2.12 THz，谱形在带内的变化 < 1.5%，因此不逐频点求解耦合方程；
- **不建模泵浦之间的拉曼转移**（沿用第一版假设），因此多泵浦时各泵浦
  只被信号耗尽，互不影响；
- 信号的实际增益仍按频率逐点施加：`γ_R(f, z) = Σ_p g_peak g̃(f_p−f) P_p(z)/A_eff`，
  其中 `P_p(z)` 来自上面的数值剖面。

求解方式：先用未耗尽剖面估一个初始 `P_s(z)`；后向泵从 z=L 反向积分、
前向泵与信号从 z=0 正向积分，两者交替做不动点迭代直到相对变化小于
`depleted_tolerance`（默认 `1e-8`）。纯前向泵时第一次迭代即收敛；
后向泵通常 3–5 次迭代收敛。求解结果会写进输出元数据
`dra_pump_energy_balance`：

```text
launch_power_w − residual_power_w = passive_loss_w + transferred_to_signal_w
```

并且 `signal_raman_gain_w = Σ_p transferred_p · (f_s/f_p)`，可在
`tests/test_distributed_raman.py` 的 `test_depleted_energy_and_photon_balance`
中审计。

> 由于剖面依赖信号功率，耗尽模型的 `pump_power_w()` / `on_off_gain()` 在
> 调用 `forward()` 之前没有定义，会显式抛出 `RuntimeError`。

### 2.4 拉曼增益谱

采用 Gaussian 近似：

```text
g̃_R(Δf) = exp[ −0.5 · ((Δf − 13.2 THz) / 2.12 THz)² ]
Δf = f_p − f_s
```

峰值频移 13.2 THz、σ = 2.12 THz，与现有 lumped `RamanPump._gain` 的
Gaussian 近似完全一致，因此两者的频率形状可以直接对照。峰值增益系数
默认 `6e-14 m/W`，有效面积默认 80 μm²。

> 负频移（信号频率高于泵浦）侧按 Gaussian 尾部自然趋于零；第一版不额外
> 建模反 Stokes 损耗/增益。

### 2.5 通道间 ISRS

`DistributedRamanFiber` 默认**不建模**通道间受激拉曼散射（ISRS）。
如果传入 `raman_r3`，现有 NumPy SSFM 会额外执行时域拉曼响应卷积，从而
叠加常规 ISRS；此时元数据 `dra_channel_isrs="raman_r3"`。参考数据集使用
`raman_r3=None`，元数据为 `"not_modelled"`。

## 3. 数值实现

- **只走 NumPy SSFM**：`DistributedRamanFiber.compute_ctx` 在构造时强制置
  `None`，`forward()` 直接调用 `FiberChannel._forward_numpy`，绝不走 PyNLO
  的 Manakov 传播器。
- **无泵逐位一致**：`pumps=None` 时不传增益回调，执行路径与
  `FiberChannel._forward_numpy` 完全相同，输出数组逐位相等。
- **增益施加位置**：每个 SSFM 步的增益率在步中点取值，并在非线性前后各
  施加一半幅度增益（对称分步）。当 `coarse_step_size_m=500 m`、泵浦衰减
  尺度约 20 km 时，步内泵浦功率变化小于 2.5%，中点近似误差可忽略。
- **频率轴**：与现有 NumPy 路径一致，采用
  `fftfreq(n, 1/fs) · n/(n−1)` 的 PyNLO inclusive-span 约定。
- **相位/幅度**：增益只改变幅度；色散、损耗、Kerr 的相位处理保持不变。

### 3.1 耗尽模型的数值细节

- 泵浦-信号功率方程用 **RK4** 在固定网格上积分，默认 `depleted_step_m=25 m`
  （与 SSFM 的 `coarse_step_size_m=500 m` 无关）。
- 收敛性（**仅对 1ch、前向 0.2 W、100 km 成立**）：泵浦 ODE 步长 100 m → 5 m
  时输出功率变化 < 1e-6 dB；SSFM 步长 500 m → 60 m 时 `SNR_NLI` 从 23.790 dB
  变到 23.703 dB（0.09 dB）。历史参考数据集固定使用 500 m / 25 m。该 500 m
  步长**只对 1ch/3ch 足够**；5ch/7ch 在 500 m 明显未收敛（5ch 高估 NLI
  1.53 dB、7ch 高估 4.01 dB），不能用“数值误差 < 0.1 dB”概括，详见 §11。
- 后向泵的不动点迭代步数写入元数据 `dra_depletion_iterations`，收敛标志
  写入 `dra_depletion_converged`。参考数据的后向泵均在 ≤ 5 次迭代内收敛。
- 信号功率剖面用的是**中心频率的耦合系数**，不随瞬时信号频谱变化；这使
  `P_s(z)` 与 SSFM 实际波形之间存在 < 1% 的模型差，见 §10 的交叉校验。

## 4. ASE

默认 `add_ase=False`，此时：

```text
dra_ase_model = "none"
dra_ase_power_per_pol_w = 0.0
```

可选开启 `add_ase=True` 时，使用未耗尽近似下的分布式 ASE 模型：

```text
S_ase(f) = h f n_sp · Σ_p ∫_0^L γ_p(f, z) · G(z→L, f) dz
G(z→L, f) = exp(∫_z^L [−α(f) + Σ_p γ_p] dz′)
```

每个偏振独立注入复高斯噪声；`n_sp` 默认取量子极限 1.0。记录键为
`dra_ase_*`，与 EDFA 的 `edfa_ase_*` 完全分开。`pump_model="depleted"` 时
上式中的 `γ_p` 使用数值求解的泵浦剖面，但 ASE 自身仍不回灌泵浦（不重复计入
耗尽），因此该选项依旧只用于可选的噪声注入。

该 ASE 模型仍忽略泵浦噪声传递和后向 ASE，因此仅作为**可选**
波形噪声注入。**参考数据集不启用 ASE**，其中的 SNR 均为 NLI 限制值，
不是含 ASE 的总 SNR。

## 5. API 示例

```python
from physics_nlo_eval.channel import DistributedRamanFiber, PumpSpec

fiber = DistributedRamanFiber(
    length_m=100e3,
    alpha_db_per_km=0.2,
    beta2_ps2_per_km=-20.0,
    beta3_ps3_per_km=0.1e-3,
    pumps=[
        PumpSpec(1450.0, 0.20, "forward"),
        PumpSpec(1450.0, 0.10, "backward"),
    ],
    coarse_step_size_m=500.0,
    pump_model="depleted",     # 默认 "undepleted"
    add_ase=False,
)
output = fiber.forward(wdm_signal)
print(output.ctx.meta["fiber_solver"])          # "numpy-ssfm-dra"
print(output.ctx.meta["dra_on_off_gain_db_max"])
print(output.ctx.meta["dra_pump_energy_balance"])
```

关键元数据：

| 键 | 含义 |
| --- | --- |
| `fiber_solver` | `"numpy-ssfm-dra"` |
| `dra_model` | `"undepleted"` 或 `"depleted"` |
| `dra_n_pumps` / `dra_pumps` | 泵浦数量与逐泵配置 |
| `dra_on_off_gain_db_min/max` | 信号带内开关增益范围（耗尽模型下由数值剖面反算） |
| `dra_channel_isrs` | `"not_modelled"` 或 `"raman_r3"` |
| `dra_ase_model` | `"none"` 或 `"undepleted-raman-ase"` |
| `dra_ase_power_per_pol_w` | 可选 ASE 功率 |
| `dra_pump_model_detail` | 仅耗尽模型：耦合方程与迭代方式说明 |
| `dra_pump_energy_balance` | 仅耗尽模型：逐泵初值/残余/损耗/转移功率与信号增益 |
| `dra_depletion_iterations` / `dra_depletion_converged` | 仅耗尽模型：不动点迭代诊断 |

## 6. 参考数据集

文件：

```text
examples/dra_reference/dra_reference.npz           # 未耗尽（历史 500 m）
examples/dra_reference/dra_reference.json
examples/dra_reference/dra_reference_depleted.npz  # 泵浦耗尽（历史 500 m）
examples/dra_reference/dra_reference_depleted.json

examples/dra_reference_converged/dra_reference.npz           # 未耗尽，62.5 m
examples/dra_reference_converged/dra_reference.json
examples/dra_reference_converged/dra_reference_depleted.npz  # 泵浦耗尽，62.5 m
examples/dra_reference_converged/dra_reference_depleted.json
examples/dra_reference_converged/step_convergence.json       # 步长探针结果
```

生成脚本：

```bash
# 只生成默认（未耗尽）数据集；500 m 输出与历史版本逐位一致
python examples/generate_dra_reference.py

# 追加耗尽数据集（3/5ch、1ch、发射功率扫描）
python examples/generate_dra_reference.py --depleted

# 只重新生成耗尽数据集
python examples/generate_dra_reference.py --depleted --skip-base

# 生成 62.5 m 收敛数据集到独立目录（不覆盖历史数据）
python examples/generate_dra_reference.py \
    --output-dir examples/dra_reference_converged --coarse-step-m 62.5 --depleted

# 可选：--symbols 256 --seed 20261104
```

`--coarse-step-m` 默认 500 m，保持历史行为；传入非 500 m 时元数据会额外写入
`coarse_step_m`、`pump_model`、`step_criterion`，以及每个数据集的
`step_converged` / `step_artifact_500m_db`。

### 6.1 参数

| 项目 | 值 |
| --- | --- |
| 信道数 | 3、5，另加 7 信道压力测试 |
| 调制 | DP-QPSK，50 GBd，RRC roll-off 0.1，sps=4 |
| 信道间隔 | 100 GHz |
| 入纤功率 | 0 dBm / 偏振 / 信道 |
| 光纤 | 100 km，0.2 dB/km，β₂=−20 ps²/km，β₃=0.1×10⁻³ ps³/km，A_eff=80 μm² |
| 泵浦 | 1450 nm，0.10/0.20 W，前向/后向，0.25 dB/km |
| 跨数 | 1 |
| 符号数 | 256 / 信道 |
| SSFM 粗步长 | 历史数据 500 m；收敛数据 62.5 m（7ch 到 31.25 m 仍有 <0.003 dB 余量） |

配置键：

```text
nopump, fwd_0p10w, fwd_0p20w, bwd_0p10w, bwd_0p20w
```

每个配置保存两份波形：

- `nl_<tag>_<config>`：完整非线性传播；
- `lin_<tag>_<config>`：相同泵浦/损耗/色散，但 `n2=0` 的线性参考。

`npz` 还包含 `input_<tag>`、`bits_<tag>`、`offsets_<tag>`。`json` 中的
`waveform_nonlinear_key` / `waveform_linear_key` 指向对应数组。

### 6.2 逐信道指标

- **信号功率**：非线性输出在信道带宽
  `Rs=50 GHz` 的 FFT 带内功率（双偏振合计）；分析带宽固定为 50 GHz，
  与解析 GN 模型一致，不再用含滚降的 55 GHz。
- **NLI 定义（v2）**：对每个信道在 50 GHz 理想矩形带内拟合单一复标量
  `a`，使 `NL ≈ a·LIN`，残差 `NL−a·LIN` 作为加性 NLI；`a` 只吸收公共
  增益/相位，不吸收带内频率相关失真。`nli_definition` 已写入 JSON。
- **NLI 功率**：对每个信道在频域拟合单一复标量 `a`，使
  `NL ≈ a·LIN`，取残差 `NL − a·LIN` 的带内功率。这样剔除的是相干接收机
  本来就会补偿的公共相位/幅度，保留信道内失真。
- **原始 NLI**：不做拟合的 `NL − LIN` 带内功率，单独记录为
  `nli_power_raw_w` / `snr_nli_raw_db`，便于审计公共相位的影响。
- **SNR_NLI**：`10 log10(P_signal / P_NLI)`。
- **开关增益**：同一信道相对 `nopump` 配置的信号功率比。
- **中心信道 EVM/SNR**：demux（理想矩形滤波）→ 公共复标量对齐 → CDC →
  RRC 匹配滤波 → EVM；`snr_evm = −20 log10(EVM)`。EVM 前 30 个符号被忽略
  以避开滤波器暂态。

### 6.3 ASE 字段

参考数据集所有配置均为：

```json
"dra_ase_model": "none",
"dra_ase_power_per_pol_w": 0.0
```

这是有意为之：第一版的 ASE 模型是可选近似，不用于参考数据的 SNR 声明。

### 6.4 耗尽数据集（`dra_reference_depleted.*`）

与默认数据集**共享同一波形网格、比特、种子和 NLI 提取口径**，唯一的区别是
`pump_model="depleted"`。包含四组数据：

| tag | 信道数 | 发射功率 | 配置 | 用途 |
| --- | --- | --- | --- | --- |
| `3ch` | 3 | 0 dBm/pol | 5 个（±0.1/0.2 W） | 与默认 3ch 数据集逐场景对照 |
| `5ch` | 5 | 0 dBm/pol | 5 个 | 信道数依赖 |
| `1ch` | 1 | 0 dBm/pol | `nopump`/`fwd_0p20w`/`bwd_0p20w` | 隔离 SPM，无 XPM |
| `3ch_0p5mw` / `3ch_1p0mw` / `3ch_2p0mw` | 3 | 0.5/1/2 mW/pol | `nopump` + `fwd_0p20w` | NLI ∝ P³ 检验 |

每个数据集元数据额外记录：

```json
"pump_model": "depleted",
"launch_power_dbm_per_pol": 0.0,
"purpose": "..."
```

顶层 `model.pump_model` / `model.pump_model_detail` 与 `power_sweep` 段落
说明口径与扫描用途。数组命名与默认数据集一致（`nl_/lin_/input_/bits_/offsets_`），
因此现有的 `band_nli_metrics` 流程可以直接复用。

> 上述历史耗尽数据同样使用 500 m 步长，5ch 结果受步长伪影影响；收敛版本见
> `examples/dra_reference_converged/dra_reference_depleted.*`（62.5 m）。

分析脚本（不依赖解析仓）：

```bash
python examples/analyze_dra_depleted.py --json /tmp/dra_analysis.json
```

它输出 1ch SPM 增强、功率扫描拟合指数，以及耗尽/未耗尽逐配置对照。

## 7. 验证

`tests/test_distributed_raman.py` 覆盖：

1. **无泵逐位一致**：`DistributedRamanFiber(pumps=None)` 与
   `FiberChannel(compute_ctx_name="numpy", compute_ctx=None)` 的
   `np.array_equal` 为真；0 W 泵浦同样逐位一致。
2. **泵浦功率单调**：0.1 W → 0.2 W，输出信号功率单调增加。
3. **前向/后向剖面不同**：前向泵在输入端最强，后向泵在输出端最强；
   未耗尽总开关增益相同。
4. **与 lumped RamanPump 对照**：`on_off_gain` 的频率形状与
   `RamanPump._gain` 在 10⁻⁹ dB 内一致。
5. **可选 ASE**：`add_ase=True` 时记录 `dra_ase_*`，同 seed 可复现；
   与 EDFA 的 `edfa_ase_*` 不混用。
6. **3 信道波形 SNR 数据点**：参考数据中 `3ch` 的
   `nopump`、`bwd_0p20w` 中心信道 `snr_nli_db` 与 `snr_evm_db` 均 > 20 dB；
   前向 0.2 W 因入纤即被放大，NLI 明显更差，体现前后向泵的物理差异。
7. **7 信道压力测试**：`7ch` 用于观察有限信道数与 NLI 提取的收敛性；
   历史 500 m 数据未收敛，只作趋势参考，收敛数据见 §11。
8. **未耗尽路径逐位兼容**：用固定种子重算 `3ch/fwd_0p20w`，与已提交的
   `dra_reference.npz` 做 `np.array_equal`，锁住 `"undepleted"` 数值路径。
9. **耗尽模型能量/光子账**：`launch − residual = passive_loss + transferred`，
   且 `signal_raman_gain_w = transferred · f_s/f_p`；同时用 SSFM 实测的
   开关增益交叉校验由耗尽剖面反算的 `on_off_gain`（≤ 0.05 dB）。
10. **泵浦耗尽趋势**：1→3→5 信道、前向 0.2 W 时 on-off gain 单调下降；
    后向泵在 1e-7 W 信号下与未耗尽结果相差 < 1e-3 dB。
11. **耗尽参考数据集 schema**：3/5/1ch 与三档功率扫描齐全，1ch 前向增强
    与后向近零增强、扫描 NLI 单调递增。
12. **收敛参考数据集 schema**：62.5 m 的未耗尽/耗尽数据齐全，每个数据集
    记录 `coarse_step_m`、`step_converged`、`step_artifact_500m_db` 与走离判据；
    5ch/7ch 无泵中心 SNR 分别约 36.33/36.27 dB，3ch 与 500 m 差 < 0.1 dB。
13. **步长收敛检查脚本**：`examples/check_dra_step_convergence.py` 的核心函数
    在 3ch 上验证最后两档步长增量 < 0.01 dB，并给出 500 m 伪影。

## 8. 作为解析 DRA 模型的参考

推荐的对照方式：

1. **开关增益谱**：解析模型对每个泵浦给出 `G_on/off(f)`；与
   `DistributedRamanFiber.on_off_gain(f)` 逐频点比较。二者使用同一
   Gaussian 谱，差异应主要来自解析模型的积分近似。
2. **逐信道功率/SNR**：用 `json` 中的 `signal_power_w`、`nli_power_w`、
   `snr_nli_db` 对照解析/EGN 输出；注意本参考的 NLI 已做单复标量对齐，
   与未做公共相位补偿的原始差值不同。
3. **波形级对照**：读取 `nl_*`/`lin_*` 数组，可在相同接收 DSP 下比较
   EVM、误差矢量相关或逐符号失真；不要只比较总功率。
4. **前后向差异**：前向泵在输入端提供增益，NLI 显著高于后向泵；这是
   解析模型必须复现的关键趋势，而不仅是总开关增益。
5. **口径对齐**：解析 `RamanPaperSpanSolver` 默认解的就是耦合信号/泵浦
   ODE，因此应与 `dra_reference_depleted.json` 对照；与未耗尽数据集对照
   会系统性高估波形增益（见 §10）。

### 8.1 与解析默认模型的残差（2026-11-04 快照）

解析侧取 `network-nlo-eval` 的 `hybrid_raman_paper`（等价于
`buglia2024_ra_gn`，仓库 HEAD `c69ba98`），`scripts/dra_waveform_compare.py`
同口径，中心信道 `SNR_NLI`：

| 场景 | 未耗尽波形 | 耗尽波形 | 解析默认 | 残差（未耗尽） | 残差（耗尽） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3ch 无泵 | 36.778 | 36.778 | 36.328 | −0.450 | −0.450 |
| 3ch 前向 0.1 W | 30.034 | 30.379 | 30.575 | +0.541 | +0.196 |
| 3ch 前向 0.2 W | 21.747 | 23.095 | 23.803 | +2.056 | +0.708 |
| 5ch 无泵 | 34.794 | 34.794 | 35.706 | +0.912 | +0.912 |
| 5ch 前向 0.1 W | 28.230 | 28.797 | 30.061 | +1.831 | +1.264 |
| 5ch 前向 0.2 W | 19.644 | 21.801 | 23.491 | +3.848 | +1.690 |
| 3ch 后向 0.2 W | 36.514 | 36.527 | 36.156 | −0.358 | −0.371 |
| 5ch 后向 0.2 W | 34.511 | 34.531 | 35.545 | +1.034 | +1.014 |

残差定义：解析 − 波形，正数表示解析更乐观。对齐泵浦口径后，前向残差从
0.54–3.85 dB 收敛到 0.20–1.69 dB；后向与无泵场景不变（本来就与泵浦耗尽
无关）。

> **步长警告**：上表基于历史 500 m 数据。3ch 在 500 m 已收敛，5ch 未收敛
> （无泵高估 NLI 1.53 dB，前向/后向同样受影响）。表中 5ch 行不能作为最终
> 残差；62.5 m 收敛后的对照见 §11.4。7ch 行未列入本表，收敛后的无泵
> hybrid−wave 为 −0.95 dB。

## 9. 已知限制

- **泵浦-泵浦拉曼转移未建模**：多泵浦时各泵浦只被信号耗尽，彼此不交换功率；
- 耗尽模型的信号只保留**带内总功率**一个标量剖面，中心频率耦合系数近似；
- 拉曼谱为 Gaussian 近似，不是实测 `RamanGain.dat`；
- 通道间 ISRS 默认关闭；
- ASE 为可选近似，参考数据未启用，且不回灌泵浦；
- 只验证单跨 100 km、1/3/5 信道、50 GBd；不适用于全 WDM 规模；
- PMD 关闭（`d_pmd=0`），参考数据聚焦 DRA/色散/Kerr 的确定性对照。

## 10. 泵浦耗尽参考的分析结论

数据来自 `dra_reference_depleted.*`（3/5/1ch 与功率扫描，256 symbols，
0 dBm/pol），解析侧为 `hybrid_raman_paper` 默认口径。

### 10.1 单信道 SPM 探针

| 量 | 波形（未耗尽） | 波形（耗尽） | 解析 GN |
| --- | ---: | ---: | ---: |
| 1ch 无泵 `SNR_NLI` | 38.130 | 38.130 | 37.869 |
| 1ch 前向 0.2 W | 23.350 | 23.790 | 24.836 |
| 前向 NLI 增强 | 14.780 | 14.340 | 13.033 |

耗尽把 1ch 的增强从 14.78 dB 降到 14.34 dB，但相对解析 GN 仍多出
**1.31 dB**（未耗尽时多 1.75 dB）。3ch 场景的同类超额为 1.16 dB、5ch 为
0.78 dB：三者相差不到 0.55 dB，且随信道数**不增长**，说明剩余偏差主要来自
**单信道 SPM 在强前向增益下的非线性积累**，而不是 XPM 或跨信道项。

### 10.2 NLI ∝ P³ 标度

3ch 前向 0.2 W，发射功率 0.5/1/2 mW/pol，`NLI` 拟合指数：

| 模型 | 配置 | 逐段斜率 | 整体拟合 |
| --- | --- | --- | ---: |
| 耗尽 | 无泵 | 2.998 / 2.995 | 2.996 |
| 耗尽 | 前向 0.2 W | 2.589 / 2.323 | 2.456 |
| 未耗尽 | 前向 0.2 W | 2.984 / 2.950 | 2.967 |

无泵对照严格给出 P³（2.996），说明 NLI 提取本身没有问题。耗尽 + 前向泵下
指数降到 **2.46** 且随功率继续下降：发射功率升高 → 泵浦被耗尽更多 → 实际
增益下降 → 带内信号功率每个倍频程只增长 1.6–1.75×（而非 2×），因此以
**发射功率**为自变量时 NLI 呈亚三次。这直接说明：在强前向泵场景里，GN 的
`NLI ∝ P³` 前提在参考侧本身就不成立，不能把斜率偏差归因于解析实现。

### 10.3 剩余偏差的归因

| 候选 | 证据 | 结论 |
| --- | --- | --- |
| SSFM 数值 | 步长 500→60 m 只让 **1ch** 前向 SNR 变 0.09 dB；泵浦 ODE 100→5 m 变化 < 1e-6 dB（该结论不适用于 5/7ch，见 §11） | 否 |
| 参考口径（泵浦耗尽） | 前向残差 0.54–3.85 dB → 0.20–1.69 dB | 已闭合大部分 |
| GN 适用域 | 1ch（无 XPM）仍有 1.31 dB 超额，且该超额不随信道数增长；P³ 在耗尽 + 前向泵下失效 | 剩余偏差的主因 |

因此：**剩余 0.2–1.7 dB 是 GN 适用域问题**（强前向增益下高局部功率的 SPM
积累），不是 SSFM 数值误差，也不是剩下的参考口径问题。建议解析模型在
前向 0.2 W、≥3 信道时保留 1–1.7 dB 的口径界，而不要再通过调参去消除。

> 负结果保留：把泵浦口径对齐后仍有非零残差；本工作不去拟合这部分，只把它
> 记录为 GN 的适用域边界。5ch 无泵的 +0.91 dB 与 7ch 无泵的 ~3 dB 缺口属于
> 另一条线索（GN 的 XPM/波形口径），不在本轮范围内。后续 WP5a 已确认这两处
> 缺口主要来自 SSFM 步长未收敛：62.5 m 下 5ch/7ch 无泵残差降到 −0.62/−0.95 dB，
> 不再是 3 dB 级问题，见 §11。

## 11. 步长收敛与重生成（WP5a）

### 11.1 证据：500 m 的 5/7ch 未收敛

同一份 `input_*` 波形只改变 `coarse_step_size_m` 重跑 SSFM，中心信道
`SNR_NLI`（无泵）：

| 步长 | 3ch | 5ch | 7ch |
|---:|---:|---:|---:|
| 500 m | 36.778 | 34.794 | 32.267 |
| 250 m | 36.743 | 36.127 | 34.279 |
| 125 m | 36.720 | 36.344 | 35.334 |
| 62.5 m | 36.708 | 36.331 | 36.274 |
| 31.25 m | 36.702 | 36.325 | 36.274 |
| 15.625 m | 36.698 | 36.321 | 36.271 |
| **500 m 相对收敛值的 SNR 差** | **+0.08 dB** | **−1.53 dB** | **−4.01 dB** |

负值表示 500 m 高估 NLI（SNR 偏低）。最后一步（31.25→15.625 m）的变化都
<0.0031 dB；以 31.25 m 为参考，62.5 m 的增量也只有 0.0001–0.0062 dB。

7ch 泵浦配置（同一 `input_7ch`，0.2 W）：

| 配置 | 500 m | 62.5 m | 31.25 m | 500 m 伪影 |
|---|---:|---:|---:|---:|
| 前向 0.2 W | 17.636 | 19.321 | 19.315 | −1.68 dB |
| 后向 0.2 W | 31.952 | 35.868 | 35.868 | −3.92 dB |

因此：**旧 `examples/dra_reference/` 的 5ch/7ch 数据（含 WP3 耗尽数据集）
是未收敛的，只能作为历史/趋势参考，不能用于基准或残差定标**。3ch 在 500 m
已经收敛（0.08 dB），旧 3ch 数据仍可用。

### 11.2 走离判据

每步的群速度走离按

```text
Δτ = 2π · Δf · |β2| · dz        （seconds）
walkoff_symbols = Δτ · Rs
```

计算，`Δf` 取最外信道相对中心信道的频偏。经验判据是每步走离
≲0.12–0.15 symbol：

| 数据集 | 最外偏移 | 500 m 走离 | 62.5 m 走离 | 0.15 symbol 建议步长 |
|---|---:|---:|---:|---:|
| 3ch | 100 GHz | 0.314 | 0.039 | 239 m |
| 5ch | 200 GHz | 0.628 | 0.079 | 119 m |
| 7ch | 300 GHz | 0.942 | 0.118 | 80 m |

62.5 m 对所有数据集都满足判据；7ch 若需要额外余量可用 31.25 m，但
62.5 m 的最后一步增量已 <0.01 dB。

### 11.3 收敛数据集

| 文件 | 内容 | 步长 |
| --- | --- | --- |
| `examples/dra_reference_converged/dra_reference.{json,npz}` | 3/5/7ch、5 配置、未耗尽 | 62.5 m |
| `examples/dra_reference_converged/dra_reference_depleted.{json,npz}` | 3/5ch、1ch、功率扫描、耗尽 | 62.5 m |
| `examples/dra_reference_converged/step_convergence.json` | 步长探针原始结果 | 500→31.25 m |

复制命令：

```bash
python examples/generate_dra_reference.py \
    --output-dir examples/dra_reference_converged --coarse-step-m 62.5 --depleted

# 只重跑步长探针（读历史 input_*，不修改任何数据）
PYTHONPATH=src python examples/check_dra_step_convergence.py
```

数据集元数据新增：

| 键 | 含义 |
| --- | --- |
| `coarse_step_m` | 顶层与逐数据集的实际 SSFM 步长 |
| `pump_model` | `"undepleted"` 或 `"depleted"` |
| `nli_method` | 与历史数据集相同的 NLI 提取口径 |
| `step_criterion` | 走离公式、0.15 symbol 目标、逐数据集建议步长 |
| `step_convergence` | 逐数据集的无泵中心 SNR、最后两档增量、500 m 伪影、走离 |
| `step_converged` / `step_artifact_500m_db` | 便于直接引用的扁平字段 |

`step_artifact_500m_db = SNR(500 m) − SNR(收敛步长)`，负值表示 500 m
高估 NLI。所有新数据集均为 `step_converged: true`。

### 11.4 收敛后的解析对照

用 `network-nlo-eval` 的 `hybrid_raman_paper`（仓库 HEAD `c69ba98`，
`scripts/dra_waveform_compare.py` 只读调用）对照 62.5 m 波形：

| 场景 | 波形（62.5 m） | 解析 hybrid | 残差（解析−波形） |
| --- | ---: | ---: | ---: |
| 3ch 无泵 | 36.708 | 36.328 | −0.380 |
| 5ch 无泵 | 36.331 | 35.706 | −0.624 |
| 7ch 无泵 | 36.274 | 35.326 | −0.948 |
| 7ch 前向 0.2 W | 19.321 | 23.387 | +4.066 |
| 7ch 后向 0.2 W | 35.868 | 35.171 | −0.697 |

无泵残差从“随信道数增长到 3 dB”降到 ≤1 dB，说明旧 5/7ch 的 N 依赖缺口
主要是步长伪影。**前向 0.2 W 的 +4.07 dB 残差没有被步长解释**（500 m→31.25 m
只闭合 1.68 dB），仍属于强前向增益下 SPM 积累的 GN 适用域问题，按 §10.3
保留为口径界，不通过调参消除。

### 11.5 验证与质量门

```bash
cd /home/zsig/Documents/gits/wt/physical-dra-step
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python -m pytest tests/test_distributed_raman.py -q
/home/zsig/miniconda3/envs/nlo/bin/python -m ruff check \
    examples/generate_dra_reference.py examples/check_dra_step_convergence.py
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python examples/check_dra_step_convergence.py
```

验收数据点：

- 3ch 收敛后与旧 500 m 差 0.07 dB（<0.1 dB）；
- 5ch/7ch 无泵中心 SNR 为 36.331/36.274 dB，与 WP4 探针一致；
- 7ch 无泵与 hybrid 的差降到 −0.95 dB（约 1 dB 量级）；
- 历史 `undepleted` 路径的逐位兼容测试继续通过。
