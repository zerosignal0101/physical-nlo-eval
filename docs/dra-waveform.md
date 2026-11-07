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

> 上面的 `P_s(z)` 是默认 `depletion_model="scalar"` 的口径。可选
> `depletion_model="spectral"` 把 `P_s` 换成逐频箱功率向量，使泵浦 ODE
> 与 SSFM 的逐频率增益同口径；方程、谱箱和能量账见 §13.9。默认仍是
> `"scalar"`，参考数据逐位不变。

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
- **增益施加位置**：每个 SSFM 步的增益率在步中点取值；增益是频率域线性
  算符（`step_gain(z, dz, ω)` 按 `ω` 逐频点返回），与色散一起乘在频谱上，
  非线性前后各施加半程幅度增益（对称分步）。**2026-11-05 修正**：旧实现
  把这个频变因子逐样本乘在时域，等价于把增益谱映射到时间轴，会给 CW 输入
  引入约 1% 的虚假幅度起伏；该起伏在非线性与线性参考中同时存在，频谱 NLI
  差分会部分抵消，但接收机 EVM 会完整暴露它，并随记录长度变化。修正后
  零色散 CW 输出严格平坦（`tests/test_distributed_raman.py` 的
  `test_frequency_dependent_gain_is_applied_in_frequency_domain`）。
  当 `coarse_step_size_m=500 m`、泵浦衰减尺度约 20 km 时，步内泵浦功率
  变化小于 2.5%，中点近似误差可忽略。
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
    depletion_model="scalar",  # 可选 "spectral"，见 §13.9
    spectral_bins=256,
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
| `dra_depletion_model` | 仅耗尽模型：`"scalar"` 或 `"spectral"`（§13.9） |
| `dra_spectral_bins` | 仅谱域耗尽：有效频箱数；scalar 为 `null` |
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
- **中心信道 EVM/SNR**：全带宽 CDC（与传播器相同的 `n/(n-1)` 频率网格）→
  demux（理想矩形滤波）→ 公共复标量对齐 → RRC 匹配滤波 → 盲定时恢复
  （恒模准则的分数样点搜索）→ EVM；`snr_evm = −20 log10(EVM)`。EVM 前
  30 个符号被忽略以避开滤波器暂态。色散补偿必须在 demux 之前对全带宽
  记录做，否则信道绝对频偏会留下群延迟与常数相位（见 §12.4）。

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
> `examples/dra_reference_converged/dra_reference_depleted.*`（62.5 m），
> 并在 3/5ch 之外加入 7ch（历史 500 m 的 7ch 未收敛，因此不进入默认数据）。

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
| `examples/dra_reference_converged/dra_reference_depleted.{json,npz}` | 3/5/7ch、1ch、功率扫描、耗尽 | 62.5 m |
| `examples/dra_reference_converged/depleted_7ch_compare.json` | 7ch 前向 0.2 W 与解析 hybrid/H3 的对照 | — |
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

用 `network-nlo-eval` 的 `hybrid_raman_paper`（仓库 HEAD `3756e4f`，
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
保留为口径界，不通过调参消除。上表 7ch 前向/后向行使用**未耗尽**波形，
而解析默认是耗尽口径；前向 0.2 W 的同口径（耗尽波形）对照见 §11.5。

### 11.5 7ch 耗尽前向泵的同口径对照

`dra_reference_converged/dra_reference_depleted.json` 现在包含 7ch。取
7ch 前向 0.2 W 的中心信道（62.5 m、耗尽模型）：

| 量 | SNR_NLI (dB) | 相对波形残差 |
| --- | ---: | ---: |
| 波形（耗尽，62.5 m） | 24.399 | — |
| 解析 hybrid（`pump_depletion=True`） | 23.387 | −1.012 |
| 解析 H3（`pump_depletion=True`） | 23.583 | −0.816 |
| `numerical_dra`（耗尽） | 23.334 | −1.065 |

残差定义为“解析 − 波形”，负值表示解析更保守。机器可读结果在
`examples/dra_reference_converged/depleted_7ch_compare.json`。

> **2026-11-05 修正**：旧表使用把频变增益逐样本乘在时域的波形，7ch 前向
> 0.2 W 的 62.5 m 结果为 22.384 dB、残差 +1.00/+1.20 dB（解析更乐观）。
> 修正增益施加方式后波形 NLI 下降 2.02 dB，残差反号。后向/无泵参考几乎
> 不变（3ch 后向 0.002 dB、无泵逐位相同），见 §3 的增益施加说明。

- 历史 500 m 耗尽波形的同一场景为 21.328 dB（用同一输入与耗尽模型重跑），
  62.5 m 为 24.399 dB，**步长修正 +3.07 dB**；相对 hybrid 的残差从 +2.06 dB
  降到 **−1.01 dB**，H3 从 +2.26 dB 降到 −0.82 dB。
- 剩余 ~0.8–1.1 dB 与前向场景的 GN 适用域边界同量级，但方向变为解析
  保守；这不是步长或泵浦口径问题。
- H3 只比 hybrid 多乐观 0.20 dB，与 WP4 的 0.1–0.3 dB 结论一致。

### 11.6 验证与质量门

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
- 7ch 耗尽前向 0.2 W 中心 SNR 22.384 dB，相对 hybrid/H3 残差 +1.00/+1.20 dB；
- 历史 `undepleted` 路径的逐位兼容测试继续通过。

## 12. 全 WDM 参考（WP7：c57 / cl152）

本节把参考从 3/5/7ch、100 GHz 间隔、50 GBd 扩展到
`network-nlo-eval` 的 `c_57ch` / `cl_152ch`：75 GHz 栅格、64 GBd、
C 或 C+L 全 WDM。数据在 `examples/dra_reference_wide/`，不覆盖旧目录。

### 12.1 计划与数据文件

| 项目 | c57 | cl152 |
| --- | --- | --- |
| 信道数 | 57 | 152 |
| 间隔 | 75 GHz | 75 GHz |
| 中心频率 | 193.75 THz | 190.15 THz |
| 符号率 / RRC | 64 GBd / 0.1 | 64 GBd / 0.1 |
| 单信道采样率 | 256 GHz (4 sps) | 256 GHz (4 sps) |
| 合波采样率 | 4.608 THz | 11.52 THz |
| 整数重采样比 | 18 | 45 |
| 占用带宽 / NLI 分析带宽 | 70.4 / 64 GHz | 70.4 / 64 GHz |
| 单信道样本数 | 1,024 (256 symbols) | 512 (128 symbols) |
| 总样本数 | 18,432 | 23,040 |
| 配置 | `nopump`、`fwd_0p20w`、`bwd_0p20w` | `nopump`、`bwd_sixpumps`、`fwd_0p20w` |
| 主数据步长 | 0.48828125 m | 0.244140625 m |
| 收敛标记 | `step_converged: true` | `step_converged: false` |

文件：

| 文件 | 内容 |
| --- | --- |
| `examples/dra_reference_wide/c57.{json,npz}` | 57ch、256 symbols、3 配置 |
| `examples/dra_reference_wide/cl152.{json,npz}` | 152ch、128 symbols、3 配置 |
| `examples/dra_reference_wide/step_convergence.json` | 64-symbol 步长探针原始结果 |
| `examples/dra_reference_wide/analytic_compare.json` | 只读调用 `hybrid_raman_paper` 的自检 |

`npz` 数组命名与旧数据一致：`input_<tag>`、`bits_<tag>`、`offsets_<tag>`、
`nl_<tag>_<config>`、`lin_<tag>_<config>`；波形为 `complex64` 压缩。
每个计划的 npz 均 <10 MB。JSON 顶层包含 `plan`、`fiber`、`signal`、`pump`、
`nli_analysis_bandwidth_hz`、`coarse_step_m`、`step_converged`、
`step_artifact_500m_db`、`step_convergence`、`num_samples` 与 `configs`；
每个 `per_channel` 行包含 `freq_offset_hz`（相对计划中心）、
`signal_power_w`、`nli_power_w`、`snr_nli_db`、`on_off_gain_db`、
`common_gain_db`、`common_phase_deg`。

### 12.2 步长收敛：全 WDM 远比 3/5/7ch 苛刻

用 64 symbols、seed 20261104、`nopump`、中心信道 `SNR_NLI` 做探针
（`step_convergence.json`）：

| 步长 (m) | c57 SNR (dB) | cl152 SNR (dB) |
|---:|---:|---:|
| 500 | 12.4754 | 3.3002 |
| 250 | 14.3634 | 7.3105 |
| 125 | 17.4960 | 9.7827 |
| 62.5 | 19.1984 | 11.7409 |
| 31.25 | 24.1318 | 15.1253 |
| 15.625 | 25.8553 | 18.3583 |
| 7.8125 | 29.1510 | 21.0737 |
| 3.90625 | 31.5150 | 24.1045 |
| 1.953125 | 32.0342 | 27.1552 |
| 0.9765625 | 32.3698 | 30.1200 |
| 0.48828125 | **32.3697** | 33.2165 |
| 0.244140625 | — | 34.2264* |

`*` 0.244 m 的 cl152 点是同一 64-symbol 输入的补充诊断，不在
`step_convergence.json` 的自动档位内；它说明 0.488 m 仍未收敛。

- **c57 在 0.48828125 m 收敛**：最后两档增量 −2.2×10⁻⁵ dB，满足
  `<0.01 dB`；`converged_step_m` = 0.48828125 m。
- **cl152 在 0.48828125 m 未收敛**：最后两档增量 +3.10 dB；补充探针
  继续加密到 0.244 / 0.122 m，得到 34.2264 / 34.6793 dB，增量
  +1.01 / +0.45 dB。按 `NLI = N_p + C·dz^p` 三点拟合（p≈1.43）外推
  收敛值约 **34.98 dB**，0.122 m 距外推值约 0.30 dB。严格 0.01 dB
  判据需要 ~0.002 m，运行时间不可接受；主数据用 0.244140625 m，
  并明确记录 `step_converged: false`，不伪造收敛。
- **500 m 的伪影是灾难性的**：`step_artifact_500m_db` 为
  −19.89 dB (c57) / −29.92 dB (cl152)。全 WDM 的 500 m 结果只能作为
  历史趋势，不能用于基准或校准。

#### 12.2.1 采样率/混叠：1× 采样率会系统性高估 NLI

152ch 的占用带宽 11.32 THz 与 11.52 THz 采样率的 Nyquist 带宽只差
~62 GHz；三次非线性产物会折叠回中心信道，污染 NLI。用同一 64-symbol
输入、同一 0.244 m 步长，把合波采样率提高到 2× / 3×：

| 计划 | 步长 (m) | 1× SNR | 2× SNR | 3× SNR | 1×→2× 偏差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| c57 | 0.488 | 32.3697 | 32.7826 | — | +0.41 dB |
| c57 (256 sym) | 0.488 | 34.5879 | 35.4759 | — | +0.89 dB |
| cl152 | 0.977 | 30.1200 | 31.4344 | 31.4316 | +1.31 dB |
| cl152 | 0.488 | 33.2165 | 34.4813 | — | +1.27 dB |
| cl152 | 0.244 | 34.2264 | 35.6347 | — | +1.41 dB |

- 2× 与 3× 在 0.977 m 只差 **0.003 dB**（同一 seed），说明 2× 在
  **固定输入**下已足够消除混叠；3× 没有再降低 NLI。
- **但 64-symbol 记录的 seed 散布极大**：cl152 nopump、2×、0.244 m 的
  4 个 seed 中心 SNR 为 35.635 / 26.216 / 24.446 / 33.962 dB，
  均值 **30.06 ± 5.56 dB**（极差 11.19 dB）。同 seed 的 1×→2× 偏差
  +1.4 dB 仍然成立，但单次 64-symbol 2× 值不能当作收敛估计。
- 128-symbol、2×、0.244 m 最初的两个 seed 为 33.502 / 34.247 dB，
  均值 33.87 ± 0.53 dB；但把同一固定参数扩到 4 个 seed
  （20261104–07）后为 33.502 / 34.247 / **23.178** / 31.422 dB，
  均值 **30.59 ± 5.08 dB**（极差 11.07 dB）。seed 20261106 是可复现的
  低实现：同一 seed 的 64-symbol 集合也给出最低值 24.446 dB。
  2-seed 的 33.87 ± 0.53 dB 是两个偏高实现的均值，不能当作收敛基准。
- **修正**：cl152 的 alias-free 中心估计不能取单一的 ~33.8–33.9 dB。
  4-seed 的中位数约 32.46 dB、均值 30.59 dB，单次实现的散布可达
  ~11 dB；1× 128-symbol 主数据 32.473 dB 与 2× 中位数一致，但
  ~1.3 dB 的 1×→2× 偏差仍只在同 seed 下成立。
- 详细的 128-symbol 4-seed 数据见 §12.2.3 的调制诊断
  （`cl152_modulation_2x_128sym.json`）；64-symbol 4-seed 表见
  `docs/dra-ensemble.md` 与
  `examples/dra_reference_wide/diagnostics/cl152_nopump_2x_ensemble*.json`。

**泵浦配置的 2× 采样率对照（2026-11-05 新增）。** 用同一 seed、同一步长，
把 `fwd_0p20w` / `bwd_*` 也做 1×/2× 对照（`--mode oversample`），
可以在扣掉混叠偏差后检查解析残差。c57 为 256 symbols、0.488 m；
cl152 为 128 symbols、0.244 m。解析值仍为 hybrid（c57 bwd 32.930、
fwd 24.670；cl152 bwd 30.708、fwd 29.900 dB），残差 = 解析 − 波形：

| 计划 | 配置 | seed | 1× SNR | 2× SNR | 1×→2× | 2× 残差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| c57 | `bwd_0p20w` | 20261104 | 34.373 | 35.206 | +0.83 | **−2.28** |
| c57 | `bwd_0p20w` | 20261105 | 32.479 | 32.988 | +0.51 | **−0.06** |
| c57 | `fwd_0p20w` | 20261104 | 28.732 | 29.264 | +0.53 | **−4.59** |
| c57 | `fwd_0p20w` | 20261105 | 27.028 | 27.377 | +0.35 | **−2.71** |
| cl152 | `bwd_sixpumps` | 20261104 | 32.010 | 33.010 | +1.00 | **−2.30** |
| cl152 | `bwd_sixpumps` | 20261105 | 32.979 | 34.247 | +1.27 | **−3.54** |
| cl152 | `fwd_0p20w` | 20261104 | 28.376 | 29.160 | +0.78 | **+0.74** |
| cl152 | `fwd_0p20w` | 20261105 | 27.531 | 28.131 | +0.60 | **+1.77** |

- 同 seed 的 1×→2× 偏差在泵浦下**不是常数**：cl152 bwd 为
  +1.00/+1.27 dB，与 nopump 的 +1.3…+1.4 dB 接近；但 cl152 fwd 只有
  +0.60/+0.78 dB。c57 同样如此：bwd +0.51/+0.83 dB 接近 nopump 的
  +0.89 dB（256 sym、seed 20261104），fwd 仅 +0.35/+0.53 dB。
  强前向增益下 1× 高估 NLI 的幅度明显小于 nopump，不能直接把
  nopump 的 ~1.4 dB 混叠修正套到前向。
- 2× 残差（两 seed 均值）：c57 bwd **−1.17**（单 seed −2.28/−0.06）、
  c57 fwd **−3.65**（−4.59/−2.71）；cl152 bwd **−2.92**（−2.30/−3.54）、
  cl152 fwd **+1.26**（+0.74/+1.77）。因此 alias-free 下 c57 bwd 在
  ±1.5 dB 内；c57 fwd 仍约 −3.7 dB（解析保守），单 seed 可达 −4.6 dB；
  cl152 bwd 的均值已超出 ±1.5 dB，cl152 fwd 均值在 ±1.5 dB 内，但
  seed 20261105 达 +1.77 dB。旧的 cl152 alias-free 外推 −2.7/+0.1 dB
  作废，见 §12.3。
- **这不是收敛估计**：每个 (计划, 配置) 只有 2 个 seed，2× 只去掉混叠，
  不解决统计散布。cl152 已知的 128-symbol nopump 4-seed 标准差
  **5.08 dB** 仍然存在；本批泵浦 2× 单 seed 残差也已跨 1–2 dB。
  这些数字只能作为**同 seed 的 alias-free 锚点**，不能当作绝对 NLI 或
  0.1 dB 级基准。原始数据见
  `examples/dra_reference_wide/diagnostics/{c57,cl152}_pumped_2x_spectral.json`。

#### 12.2.2 记录长度与 seed 集成：绝对 NLI 有 1–2 dB 统计散布

步长探针用 64 symbols 足以判断**步长趋势**，但不足以给出**绝对 NLI**。
固定 c57 nopump、0.488 m，改变记录长度/seed：

| symbols | 64 | 128 | 256 (seed 20261104) | 256 (seed 20261105) | 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 中心 SNR (dB) | 32.370 | 32.825 | 34.588 | 32.709 | 33.722 |

- 五个点没有单调趋势，均值 **33.24 dB、样本标准差 0.90 dB**；
  单次 256-symbol 实现的散布约 ±1–2 dB，远大于 3/5/7ch 的
  0.3–0.5 dB。全 WDM 的波形残差必须用多 seed 集成，不能拿单次
  实现做 0.1 dB 级比较。
- **前向泵的记录长度依赖是增益施加 bug，已在 2026-11-05 修正**：
  旧实现把频变拉曼增益逐样本乘在时域，使增益谱映射到时间轴，产生
  随记录长度变化的虚假幅度/NLI。修正后 c57 `fwd_0p20w`、0.488 m、
  seed 20261104 的中心 SNR 为 64/128/256/512/1024 symbols =
  **26.93 / 26.91 / 28.73 / 27.95 / 27.41 dB**（EVM 25.97 / 26.11 /
  28.35 / 27.73 / 27.25 dB），没有单调发散。旧值
  26.78/23.50/20.13/14.15 dB（64→512）与 EVM 22.44→13.65 dB 已作废，
  对应的 `c57_forward_record_evm.json`、`c57_forward_long_record_*`
  在 JSON 中标记为 superseded。
- 后向泵同样受该 bug 影响（57ch 全带宽内增益谱变化约 5 dB，时间域
  调制不可忽略）：c57 `bwd_0p20w` 两 seed 中心 SNR 从
  32.22/30.68 dB 修正为 **34.37/32.48 dB**；3ch 窄带（200 GHz）内
  增益近似平坦，后向只变化 0.002 dB。无泵逐位不变。
- 64-symbol 步长探针的 `snr_nli_db` 只用于比较相邻步长；主数据的
  `per_channel` 用 128/256 symbols，仍建议多 seed 复算。

**EVM 对照与修正后的前向参考。**

`examples/check_dra_forward_record.py` 在同一批 0.488 m、seed 20261104
的 c57 波形上同时测频谱 NLI（`snr_nli_db`）和接收机 EVM SNR
（全带宽 CDC → demux → 公共增益对齐 → RRC → 盲定时 → EVM）。修正后两者在
0.4–0.5 dB 内一致，且不随记录长度发散：

| symbols | fwd 频谱 (dB) | fwd EVM (dB) | nopump EVM (dB) |
| ---: | ---: | ---: | ---: |
| 64 | 26.928 | 25.972 | 29.843 |
| 128 | 26.914 | 26.110 | 30.671 |
| 256 | 28.732 | 28.346 | 33.295 |
| 512 | 27.947 | 27.726 | 32.949 |
| 1024 | 27.413 | 27.252 | 32.683 |

**cl152 接收机 EVM 已修复（2026-11-05）。** 旧 EVM 链路先 demux、再按
单信道样本做 CDC。信道绝对频偏 `f_c` 在传播器的全带宽色散相位里产生
线性项 `−β₂·ω_c·ω·L` 和常数项 `−β₂·ω_c²·L/2`，单信道基带 CDC 只补偿
`ω²` 项，无法表示这两项。cl152 中心信道 `f_c=+37.5 GHz`，线性项对应
群延迟 `τ=−β₂·ω_c·L≈511 ps≈32.7 symbol≈131` 个 256 GBd 样点，判决点被
整体打乱，EVM 饱和在 `√2·100%`，且与泵浦配置无关；c57 中心信道
`f_c=0`，所以旧路径只在 cl152 上出错。修复方式是把 CDC 移到 demux
之前、对全带宽记录使用与传播器相同的 `n/(n-1)` 网格，并在 RRC 后加
一次盲的恒模分数样点定时恢复。重算（同一批落盘 `nl/lin` 波形，未重跑
传播）得到：

| cl152 配置 | 频谱 `snr_nli_db` | EVM | EVM SNR | 差值 |
| --- | ---: | ---: | ---: | ---: |
| nopump | 32.473 | 2.599% | 31.705 dB | 0.77 dB |
| bwd_sixpumps | 32.010 | 2.701% | 31.370 dB | 0.64 dB |
| fwd_0p20w | 28.376 | 3.898% | 28.182 dB | 0.19 dB |

c57 回归同时轻微改善：nopump/fwd/bwd 的 EVM 从 2.164/3.825/2.223% 变为
2.086/3.781/2.148%（`c57.json` 中的落盘值仍是修复前的历史快照）。

修正后的 256-symbol 主参考（`c57.json` / `c57_seed20261105.json`）：
前向中心 SNR **28.732 / 27.028 dB**（EVM 28.35 / 26.49 dB），后向
**34.373 / 32.479 dB**。前向两 seed 均值 27.88 dB，与解析 hybrid
24.670 dB 的残差为 **−3.21 dB**（解析保守）；后向两 seed 均值
33.43 dB 对解析 32.930 dB 的残差为 **−0.50 dB**。前向的负残差随
信道数增大（3ch +0.54、5ch +0.25、7ch −1.01、57ch −3.21 dB），
是新的开放问题（泵浦耗尽驱动/强前向增益 GN 适用域），不能用旧的
“周期记录伪影”解释。详见 `docs/dra-open-issues.md` §1.1 与 §5。

走离判据（`2π·Δf_max·|β2|·dz·Rs`）：

| 计划 | Δf_max | 500 m | 62.5 m | 31.25 m | 0.488 m | 0.15-symbol 建议步长 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c57 | 2.10 THz | 9.16 | 1.145 | 0.573 | 0.0089 | 8.19 m |
| cl152 | 5.66 THz | 24.71 | 3.088 | 1.544 | 0.0241 | 3.04 m |

**0.15-symbol 的走离目标对全 WDM 不适用**：c57 实测要到 0.009 symbol
才收敛，cl152 在 0.024 symbol 仍未收敛。全 WDM 的步长由跨信道 FWM
相位（而不是最外信道与中心的简单群速度走离）控制；这解释了为什么
62.5/31.25 m 的“可行性”计时探针不能当作收敛步长。

#### 12.2.3 调制格式口径（收敛的高斯控制）

波形是 DP-QPSK（恒模），GN 解析模型假设高斯符号。诊断脚本
`examples/check_dra_modulation.py` 复用参考生成器的 `WidebandPlan`、
RRC/IQ/合波、`make_wideband_fiber` 与 `band_nli_metrics`，只在符号
生成处切换调制：QPSK 分支与 `build_wideband_dataset` 的输入波形逐位
一致（自检 `max|diff| = 0`，见下）；高斯符号为每偏振独立复高斯
`(randn+1j·randn)/√2`，按有限样本归一化到单位每偏振功率，RRC 参数
（sps=4、roll_off=0.1）与 QPSK 完全相同。

每个 (计划, 采样率, 步长, symbols, seed, 调制) 先传播一次非线性、一次
`n2=0` 线性参考。下表是 nopump 中心信道、2× 采样的 4-seed 结果；
NLI 比为 `10·log10(NLI_gauss / NLI_qpsk)`，正值表示高斯控制的 NLI 更高。

**cl152，2×，0.244140625 m，128 symbols（alias-free）**

| seed | QPSK SNR (dB) | QPSK NLI (W) | 高斯 SNR (dB) | 高斯 NLI (W) | NLI 比 (dB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20261104 | 33.502 | 2.118×10⁻⁸ | 26.349 | 1.099×10⁻⁷ | +7.15 |
| 20261105 | 34.247 | 1.791×10⁻⁸ | 30.096 | 4.577×10⁻⁸ | +4.08 |
| 20261106 | **23.178** | **2.280×10⁻⁷** | 29.313 | 5.558×10⁻⁸ | **−6.13** |
| 20261107 | 31.422 | 3.416×10⁻⁸ | 30.632 | 4.078×10⁻⁸ | +0.77 |
| 均值 | 30.587 ± 5.082 | — | 29.097 ± 1.911 | — | **+1.47 ± 5.70** |

中位数 NLI 比 +2.42 dB；按 NLI 均值之比为 −0.78 dB（被 seed
20261106 的 QPSK 低实现主导）。剔除这一个 QPSK 实现后，其余三点为
+7.15 / +4.08 / +0.77 dB，均值 **+4.00 ± 3.19 dB**，NLI 均值之比
**+4.28 dB**。

**c57，2×，0.48828125 m，256 symbols**

| seed | QPSK SNR (dB) | QPSK NLI (W) | 高斯 SNR (dB) | 高斯 NLI (W) | NLI 比 (dB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20261104 | 35.476 | 1.332×10⁻⁸ | 26.404 | 1.070×10⁻⁷ | +9.05 |
| 20261105 | 33.253 | 2.237×10⁻⁸ | 28.846 | 6.104×10⁻⁸ | +4.36 |
| 20261106 | 29.361 | 5.461×10⁻⁸ | 27.547 | 8.295×10⁻⁸ | +1.82 |
| 20261107 | 33.516 | 2.093×10⁻⁸ | 29.926 | 4.782×10⁻⁸ | +3.59 |
| 均值 | 32.901 ± 2.560 | — | 28.181 ± 1.533 | — | **+4.70 ± 3.09** |

中位数 +3.97 dB，NLI 均值之比 **+4.29 dB**。

**cl152，1×，0.244140625 m，128 symbols（仅用于与旧探针衔接）**

| seed | QPSK SNR (dB) | 高斯 SNR (dB) | NLI 比 (dB) |
| ---: | ---: | ---: | ---: |
| 20261104 | 32.473 | 26.268 | +6.20 |
| 20261105 | 32.968 | 29.590 | +3.29 |
| 均值 | 32.720 ± 0.350 | 27.929 ± 2.349 | **+4.75 ± 2.06** |

**自检与步长检查**

- QPSK 路径逐位复现参考输入：`max|diff| = 0`；传播值也与已提交数据
  完全一致——cl152 2× 0.244 m 的 33.5015/34.2472 dB 即 §12.2.1 的
  33.502/34.247 dB，cl152 1× seed 20261104 的 32.4730 dB 即 1× 主
  数据，c57 2× 256-symbol seed 20261104 的 35.4759 dB 即 2× 主数据
  （§12.2.1 表）。因此两条路径除符号分布外没有口径差。
- 高斯控制的步长（cl152 2×、128 symbols、seed 20261104）：
  0.244 m 为 26.349 dB（NLI 1.099×10⁻⁷ W），0.488 m 为 26.172 dB
  （NLI 1.146×10⁻⁷ W），**差 +0.177 dB < 0.5 dB**，无需补 0.122 m。
  高斯控制的峰值更高但 NLI 本身更大，相对步长误差反而小于 QPSK；
  步长这一项已经收敛。
- 统计上仍未收敛：固定 128 symbols 时，单次实现的 NLI 比可在
  −6.1…+9.0 dB 间变化，标准差 3–6 dB。原因是公共复增益拟合后剩下的
  带内 NLI 残差是随机符号实现的相干叠加，QPSK 同样有重尾（seed
  20261106 的 QPSK SNR 比另一个 seed 低 11 dB，且该低值在 64-symbol
  集合中可复现）。**因此不能把任何一个单次值当作调制口径常数。**

**结论**

- 旧的 1×、64-symbol 单次探针（DP-QPSK 34.226 dB / 复高斯 27.220 dB，
  +7.0 dB）是重尾分布中的一个高实现，**未收敛，仅能作为方向性证据**：
  高斯符号确实可能产生高得多的 NLI。它与 4-seed 配对均值
  +1.47 dB 相差 5.5 dB，与中位数 +2.42 dB 相差 4.6 dB，均 >1 dB，
  绝对值不可引用。
- 去掉那个 QPSK 低实现后，cl152 的中央估计 +4.0 ± 3.2 dB；c57 的
  4-seed 均值为 +4.70 ± 3.09 dB，NLI 均值之比 +4.29 dB。两者都与
  “恒模 QPSK vs 复高斯”的四阶矩预期 **+3.0 dB** 同量级（SPM 项
  `E|X|⁴` 翻倍），而不是 7 dB。**当前最佳中央估计取 ~4 dB**，但
  128-symbol、4-seed 的样本标准差仍有 3–6 dB，统计上远未收敛。
- 高斯控制本身（cl152 2× 128 symbols）4-seed 中心 SNR 为
  **29.10 ± 1.91 dB**；旧的 27.220 dB 落在该散布内。
- 这是 nopump 口径；泵浦下增益/功率剖面改变，调制差可能不同，
  不能外推。若要把调制项做成 0.1 dB 级预算，需要显著更多 seed
  （至少数十个）和/或更长的记录，而不是继续加密步长。
- **是否进入默认校准预算**：默认对照是 GN 解析 vs DP-QPSK 波形，
  调制口径已经隐含在参考波形里，其 nopump 残差本身受同一重尾支配
  （§12.3：4-seed 中位数 +0.3 dB、均值 +2.2 dB）。不应把 5–7 dB
  作为固定修正量塞进默认预算。若对照对象换成高斯符号波形，才需要
  单列一个调制项，建议取 ~4 dB 且带 3–6 dB 的实现不确定度。

### 12.3 解析自检（hybrid_raman_paper）

用 `network-nlo-eval` 的 `hybrid_raman_paper` 只读对照 nopump 中心信道。
解析侧使用与波形相同的平坦 α=0.162 dB/km、β₂=−21.7 ps²/km、
n2 与 A_eff；`symbol_rate_hz=64e9`、
`channel_bandwidth_source="symbol_rate"`、`pump_depletion=True`。

**色散斜率不能取 0**：波形用 β₃=1e-4 ps³/km，按 network 的
`_calc_beta3_from_s_d` 反解等价于 **S ≈ −0.0219 ps/(nm²·km)**。
`check_dra_wideband_analytic.py` 和
`dra_wideband_compare.fiber_from_reference` 都优先从
`beta3_ps3_per_km` 反解 S；用默认 S=0.06 会把 cl152 中心 SNR 抬高
约 0.46 dB（32.765 → 33.225），造成口径错配。

结果在 `analytic_compare.json`，残差定义为“解析 − 波形”（正值表示解析更乐观）。

| 计划 | 波形 | 波形 SNR (dB) | 解析 SNR (dB) | 残差 (dB) |
| --- | --- | ---: | ---: | ---: |
| c57 | 1×、256 sym、seed 20261104 | 34.588 | 33.133 | **−1.455** |
| c57 | 1×、256 sym、seed 20261105 | 32.709 | 33.133 | **+0.424** |
| c57 | 1× 两 seed 均值 | 33.649 | 33.133 | **−0.516** |
| c57 | 2×、256 sym、seed 20261104 | 35.476 | 33.133 | **−2.343** |
| cl152 | 1×、128 sym、0.244 m 主数据 | 32.473 | 32.765 | **+0.292** |
| cl152 | 2×、128 sym、0.244 m、2 seed 均值（**不稳健，勿引用**） | 33.874 | 32.765 | −1.109 |
| cl152 | 2×、128 sym、0.244 m、4 seed 均值 | 30.587 | 32.765 | **+2.178** |
| cl152 | 2×、128 sym、0.244 m、4 seed 中位数 | 32.462 | 32.765 | +0.303 |
| c57 | 2×、256 sym、`bwd_0p20w`、2 seed 均值 | 34.097 | 32.930 | **−1.167** |
| c57 | 2×、256 sym、`fwd_0p20w`、2 seed 均值 | 28.320 | 24.670 | **−3.650** |
| cl152 | 2×、128 sym、0.244 m、`bwd_sixpumps`、2 seed 均值 | 33.629 | 30.708 | **−2.921** |
| cl152 | 2×、128 sym、0.244 m、`fwd_0p20w`、2 seed 均值 | 28.645 | 29.900 | **+1.255** |
| cl152 | 1×、64 sym、0.122 m 探针（单次） | 34.679 | 32.765 | −1.914 |
| cl152 | 2×、64 sym、0.244 m 探针（4 seed 中最大） | 35.635 | 32.765 | −2.870 |

- c57 的 1× 单 seed 残差在 −1.46…+0.42 dB 之间，两 seed 均值 −0.52 dB，
  与 3/5/7ch 的 ±1 dB 量级一致；但单次实现的散布有 ~1.9 dB，不能只看
  一个 seed。
- c57 的 2× 数据（同一 seed）把波形抬高 0.89 dB，残差变为 −2.34 dB；
  说明 1× 的 +0.42…−1.46 dB 里包含约 0.4–0.9 dB 的混叠偏差。
- **修正**：cl152 的 2× 128-symbol 两 seed 均值 33.874 ± 0.53 dB 是两个
  偏高实现的均值。扩到 4 个 seed 后为 33.502 / 34.247 / 23.178 /
  31.422 dB，均值 **30.587 ± 5.082 dB**，中位数 32.462 dB。按均值算
  对解析 32.765 dB 的残差是 **+2.18 dB**（解析更乐观），按中位数是
  +0.30 dB。单次实现的 SNR 可跨 ~11 dB，因此 cl152 的解析残差只能
  报“约 −1…+2 dB、单次实现可跨数 dB”，不能当作 0.1 dB 级基准。
- 64-symbol 的 34.68 / 35.64 dB 是**单次实现**：同一 cl152 2× 设置的
  4 个 64-symbol seed 跨 24.45…35.64 dB（均值 30.06、标准差 5.56 dB）。
  之前的 “3.2 dB 跨 32.47…35.64” 主要来自这个 64-symbol seed 散布，
  不是解析模型的系统偏差；36.05 dB 的三点外推正是取了 4 个 seed 中
  最大的一个，不能引用。
- §12.2.3 的收敛高斯控制显示调制格式是 152ch 的一个真实口径项，但
  不是固定的 7 dB：剔除一个 QPSK 低实现后的中央估计约 +4 dB
  （c57 4-seed 均值 +4.70 dB，NLI 均值之比 +4.29 dB），与四阶矩
  预期的 +3 dB 同量级；旧的 7.0 dB 单次探针不可引用。步长已经收敛
  （0.177 dB），统计上仍需更多 seed/更长记录才能把调制项压到 1 dB
  以内。152ch 的绝对 nopump 残差按 4-seed 中位数约 +0.3 dB，按均值
  +2.2 dB，不再引用 −1.1 dB 这个 2-seed 值。

**派生：factor=2（双偏振总功率驱动）× 实测 2× 波形。** 解析值与采样率
无关，所以可以把网络侧 factor=2 的解析 SNR 与这里实测的 2× 波形 SNR
组合（不是直接跑 factor=2 的 2× 对照）：

| 计划 | 配置 | 2× 波形 (dB) | 解析 factor=2 hybrid | 残差 hybrid | H3 残差 |
| --- | --- | ---: | ---: | ---: | ---: |
| c57 | bwd_0p20w | 34.097 | 32.997 | **−1.10** | −1.15 |
| c57 | fwd_0p20w | 28.320 | 26.637 | **−1.68** | −1.94 |
| cl152 | bwd_sixpumps | 33.629 | 31.880 | **−1.75** | −1.99 |
| cl152 | fwd_0p20w | 28.645 | 30.676 | **+2.03** | +1.85 |

加上总功率驱动后，2× 残差落在约 −1.1…−1.8（backward）和
−1.7/+2.0（forward）的带内；剩余 0.3–0.5 dB 超出 ±1.5 dB 的部分
与单 seed 1–2 dB（cl152 nopump 更达 5 dB）的统计散布同量级，不能
当作已分离的模型偏差。3/5/7ch 的 factor=2 残差为 +1.01/+0.96/−0.11
（1×）；这些窄带配置的 1×→2× 偏差很小，结论不变。

- 因此：**57ch 的 nopump 与 backward 的两 seed 均值在 ~1.2 dB 内与
  解析一致（2× bwd 均值 −1.17 dB；单 seed 可到 −2.28 dB）；
  152ch 的 2-seed −1.1 dB 已被 4-seed 数据推翻（中位数 +0.3 dB、
  均值 +2.2 dB，单次实现散布数 dB），只能报“~±2 dB 量级、统计未
  收敛”。64-symbol 探针只能看同 seed 的步长趋势，不能给绝对 NLI。**
  修正增益施加后，前向也可以进入绝对
  对照：c57 前向两 seed 均值 27.88 dB 对解析 24.670 dB 为 **−3.21 dB**
  （解析保守），后向两 seed 均值 33.43 dB 对 32.930 dB 为 **−0.50 dB**。
  cl152 1× 128-symbol 的修正值为 nopump 32.473 (+0.29)、bwd 32.010
  (−1.30)、fwd 28.376 (+1.52 dB)；旧 bwd 29.720/fwd 16.146 作废。
  1× 数据仍含混叠偏差（同 seed 1×→2×：cl152 bwd +1.0/+1.3、
  fwd +0.6/+0.8 dB；c57 bwd +0.5/+0.8、fwd +0.35/+0.53 dB）。
  实测 2× alias-free 残差为 cl152 bwd **−2.92 dB**（两 seed 均值；
  单 seed −2.30/−3.54，已超出 ±1.5 dB）、fwd **+1.26 dB**
  （+0.74/+1.77，均值在 ±1.5 dB 内、seed 20261105 略超）；c57 为
  bwd **−1.17 dB**（±1.5 dB 内；单 seed −2.28/−0.06）、fwd
  **−3.65 dB**（单 seed −4.59/−2.71）。旧的 cl152 alias-free 外推
  −2.7/+0.1 dB **作废**：它套用了 nopump 的 ~1.4 dB 混叠偏差，但
  泵浦 fwd 的同 seed 1×→2× 偏差实际只有 +0.6…+0.8 dB，因此
  alias-free 的 fwd 残差仍是约 +1.3 dB，而不是 +0.1 dB。
  前向残差随信道数增大（3ch +0.54、5ch +0.25、7ch −1.01、57ch −3.21 dB），
  是泵浦耗尽驱动/强前向增益 GN 适用域的新开放问题。

### 12.4 限制

- **平坦 α**：所有信道用 1550 nm 的 0.162 dB/km；`SMF_100KM` 多项式在
  C+L 边缘约 0.162–0.178 dB/km，L 波段损耗被低估。
- **β₃ 口径已对齐**：波形 β₃=1e-4 ps³/km 对应 S≈−0.0219；解析侧必须
  用这个斜率。用 S=0.06（β₃≈0.133）会让 cl152 中心 SNR 高约 0.46 dB。
- **耗尽模型**：用计划中心频率的带内总功率驱动泵浦耗尽（默认
  `depletion_coupling="center"`；可选的 `"band_average"` 见 §13.5，
  未设为默认）；不建模泵浦-泵浦拉曼转移、泵浦 ASE；通道间 ISRS 未建模
  （`raman_r3=None`）。§13 给出了这些近似导致的逐点剖面差。
- **规模/记录长度**：c57 256 symbols、cl152 128 symbols。全 WDM 的
  NLI 统计散布远大于 3/5/7ch：c57 nopump 64/128/256/512 symbols 与
  两个 256-symbol seed 的均值为 33.24 dB、标准差 0.90 dB。单次实现
  不能用于 0.1–0.2 dB 级比较；需要多 seed 集成。
- **前向泵记录长度（已修正）**：旧实现把频变增益乘在时域，导致
  c57 `fwd_0p20w` 的频谱 NLI 与 EVM 随记录长度单调下降；2026-11-05
  改为频域施加后，64→1024 symbols 的频谱 SNR 为
  26.93/26.91/28.73/27.95/27.41 dB，无单调发散（EVM 同步）。
  修正后的前向主参考（256 symbols）为 28.732/27.028 dB（两 seed），
  解析 hybrid 残差 −3.21 dB（解析保守）。旧 `c57_forward_record_evm`
  与 `c57_forward_long_record_*` 诊断已标记 superseded，不能再引用。
  前向的负残差随信道数增大（3ch +0.54 → 57ch −3.21 dB），是当前
  最大的开放口径问题，见 `docs/dra-open-issues.md` §1.1/§5。
- **EVM 的 CDC 顺序与频率网格口径**：色散补偿必须对**全带宽**记录做，
  再 demux；频率网格用与传播器相同的 `n_full/(n_full-1)` 缩放。旧实现
  在 demux 后按 `n_ch` 个信道样本补偿，需要额外的 `cdc_freq_scale`
  修正，且无法表示信道绝对频偏 `f_c` 引入的线性/常数相位。c57 中心
  信道 `f_c=0` 掩盖了这个问题；cl152 中心信道在 `+37.5 GHz`，群延迟
  `−β₂·ω_c·L≈511 ps≈32.7 symbol`，旧 EVM 对所有配置都饱和在 ~143%
  （`√2·100%`），与频谱 `snr_nli_db` 相差约 35 dB。全带宽 CDC 后
  cl152 三个配置的 EVM SNR 为 31.71/31.37/28.18 dB，与频谱相差
  0.77/0.64/0.19 dB；c57 回归不变或轻微改善。旧 `cdc_freq_scale`
  参数已删除。
- **cl152 步长/统计**：`step_converged: false`。2× 128-symbol 的
  2-seed 均值 33.87 ± 0.53 dB 是两个偏高实现；4-seed 为 33.502 /
  34.247 / 23.178 / 31.422 dB，均值 **30.59 ± 5.08 dB**、中位数
  32.46 dB（§12.2.1/§12.2.3）。2× 64-symbol 的 4-seed 均值
  **30.06 ± 5.56 dB**（极差 11.19 dB）；单次 64-symbol 的 35.64 dB 是
  4 个 seed 中最大，三点外推 36.05 dB 不可引用。步长本身（0.244 与
  0.122 m 同 seed）只差 0.45 dB，3× 与 2× 同 seed 只差 0.003 dB。
  高斯控制的步长已收敛：cl152 2×、128 symbols 的 0.244 vs 0.488 m
  同 seed 只差 0.177 dB。
- **全 WDM 统计预算**：c57 nopump 5 次测量标准差 0.90 dB；cl152
  64-symbol 4-seed 标准差 5.56 dB，128-symbol 4-seed 标准差
  **5.08 dB**（2-seed 的 0.53 dB 不可用）。绝对 NLI 必须用 128-symbol
  （或更长）多 seed 集成；64-symbol 只用于同 seed 的步长趋势。
- **2× 泵浦诊断（2026-11-05）**：c57 `bwd_0p20w`/`fwd_0p20w` 与
  cl152 `bwd_sixpumps`/`fwd_0p20w` 各做了 2 个 seed 的同 seed 1×/2×
  对照（c57 256 sym、0.488 m；cl152 128 sym、0.244 m），结果见
  `diagnostics/{c57,cl152}_pumped_2x_spectral.json`。alias-free（2×）
  两 seed 均值残差：c57 bwd −1.17、fwd −3.65；cl152 bwd −2.92、
  fwd +1.26 dB。2× 只去掉混叠，**不解决 seed 散布**：每个配置只有
  2 个 seed，单 seed 残差已跨 1–2 dB；cl152 已知 nopump 128-symbol
  4-seed 标准差 5.08 dB，泵浦配置没有理由更小。因此这些值只能作为
  同 seed 的 alias-free 锚点，不能当作收敛的绝对 NLI。fwd 在强增益下
  的 1×→2× 偏差较小（cl152 +0.60…+0.78、c57 +0.35…+0.53 dB），
  不能套用 nopump 的 ~1.4 dB 修正。
- **调制口径**：波形是恒模 DP-QPSK，GN 假设高斯；cl152 2× 128-symbol
  4-seed 的配对 NLI 比为 +1.47 ± 5.70 dB（中位数 +2.42 dB），
  c57 2× 256-symbol 为 +4.70 ± 3.09 dB。旧的 7.0 dB 单次探针未收敛、
  仅方向；当前中央估计约 **+4 dB**（与四阶矩预期 +3 dB 同量级），
  统计上仍被 3–6 dB 的实现散布主导。跨调制/跨模型比较必须固定
  调制口径；默认校准不额外加入 5–7 dB 固定修正。

### 12.5 复现

```bash
cd /home/zsig/Documents/gits/wt/physical-dra-wideband

# 1) 步长探针（64 symbols，约 3.5 min + 8.5 min）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/generate_dra_reference.py --wideband --wideband-probe-only \
    --wideband-plans c57 --wideband-dir /tmp/dra_wide_c57
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/generate_dra_reference.py --wideband --wideband-probe-only \
    --wideband-plans cl152 --wideband-dir /tmp/dra_wide_cl152
# 合并两个 step_convergence.json 的 plans 段到 examples/dra_reference_wide/

# 2) 主数据（复用探针；c57 约 32 min，cl152 约 75 min）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/generate_dra_reference.py --wideband --reuse-probe \
    --wideband-plans c57 --wideband-step-map c57=0.48828125
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/generate_dra_reference.py --wideband --reuse-probe \
    --wideband-plans cl152 --wideband-symbols-map cl152=128 \
    --wideband-step-map cl152=0.244140625
# 第二个统计 seed（可选，约 32 min）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/generate_dra_reference.py --wideband --reuse-probe \
    --wideband-plans c57 --wideband-step-map c57=0.48828125 --seed 20261105

# cl152 2× 多 seed 集成（128 symbols；2-seed 的 33.87 不稳健，用 4 seed）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_cl152_ensemble.py --symbols 128 \
    --seeds 20261104 20261105 20261106 20261107 --workers 4

# 调制口径（QPSK vs 复高斯）：每个 (seed, modulation) 一次非线性 + 一次线性
# 传播；建议用 nohup 把 4 个组合并行，再用 --merge 配对
for seed in 20261104 20261105 20261106 20261107; do
    for mod in qpsk gaussian; do
        PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
            examples/check_dra_modulation.py --plan cl152 --modulations "$mod" \
            --symbols 128 --seeds "$seed" --step-m 0.244140625 \
            --sample-rate-factor 2 \
            --output "/tmp/cl152_mod_${seed}_${mod}.json"
    done
done
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_modulation.py --merge /tmp/cl152_mod_*.json \
    --output examples/dra_reference_wide/diagnostics/cl152_modulation_2x_128sym.json

# 3) 解析自检（只读调用 network-nlo-eval）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_wideband_analytic.py

# 4) 前向泵记录长度：频谱 NLI vs 接收机 EVM（约 70 min）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_forward_record.py

# 4b) 修正后的前向记录长度诊断（0.488 m）：64/128/256/512/1024 symbols
#     2026-11-05 之前的结果使用旧时域增益实现，已标记 superseded。
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_forward_record.py --symbols 64 128 256 512 1024 \
    --configs fwd_0p20w --step-m 0.48828125 \
    --output examples/dra_reference_wide/diagnostics/c57_fwd_record_evm_gainfix_0p488.json

# 2× 采样率混叠检查（可选）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_forward_record.py --symbols 256 \
    --configs fwd_0p20w --step-m 0.48828125 --sample-rate-factor 2

# 5) 质量门
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python -m pytest tests/test_distributed_raman.py -q
/home/zsig/miniconda3/envs/nlo/bin/python -m ruff check \
    examples/generate_dra_reference.py examples/check_dra_wideband_analytic.py \
    examples/check_dra_wideband_convergence.py examples/check_dra_forward_record.py \
    tests/test_distributed_raman.py
```

§12.2.1/§12.2.2 的原始探针 JSON 在
`examples/dra_reference_wide/diagnostics/`。采样率/记录长度诊断可用
`examples/check_dra_wideband_convergence.py` 复现：

```bash
# 混叠：cl152 nopump，0.488/0.244 m，1×/2×/3×
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_wideband_convergence.py --plan cl152 --mode oversample \
    --config nopump --steps 0.48828125 0.244140625 --factors 1 2 3 --symbols 64

# 记录长度/seed：c57 nopump，0.488 m
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_wideband_convergence.py --plan c57 --mode record \
    --config nopump --steps 0.48828125 --symbols 64 128 256 512 \
    --seeds 20261104 20261105
```

泵浦配置的 1×/2× 同 seed 对照（§12.2.1/§12.3）用同一脚本复现。
注意：`oversample` 模式目前固定使用 `WIDEBAND_SEED=20261104`，不读取
`--seeds`；seed 20261105 的行是在导入生成器后覆盖
`generate_dra_reference.WIDEBAND_SEED` 得到的（未改源码）：

```bash
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_wideband_convergence.py --plan cl152 --mode oversample \
    --config fwd_0p20w --steps 0.244140625 --symbols 128 --factors 1 2
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_wideband_convergence.py --plan cl152 --mode oversample \
    --config bwd_sixpumps --steps 0.244140625 --symbols 128 --factors 1 2
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_wideband_convergence.py --plan c57 --mode oversample \
    --config fwd_0p20w --steps 0.48828125 --symbols 256 --factors 1 2
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/check_dra_wideband_convergence.py --plan c57 --mode oversample \
    --config bwd_0p20w --steps 0.48828125 --symbols 256 --factors 1 2
```

## 13. 剖面差异诊断（物理波形 vs 解析 Raman ODE）

§12.3 的残差是"解析 − 波形"，只给出标量 SNR 差。本节把两侧在**同一泵浦
配置、同一输入波形**下的泵浦/信号剖面逐点比出来，判断残差到底来自哪一段
物理。所有数字由
`examples/diagnose_dra_profile.py`（只读调用 network-nlo-eval）跑出，
机器可读结果在
`examples/dra_reference_wide/diagnostics/profile_diagnosis.json`（物理侧
`depletion_coupling="center"`）与
`..._band_average.json`（可选口径，见 §13.5）。中心信道 SNR 的 A/B 结果在
`..._profile_snr_ab.json`；`c57:bwd` 用 center 口径重跑得到
34.372986 dB，与参考 34.372986 dB 一致（差 3e-7 dB），可作为整条对照链的
自检。

### 13.1 口径与观测量

- **物理侧**：`DistributedRamanFiber(pump_model="depleted")`，输入功率取
  参考数据 `input_<plan>` 波形的带内总功率（与 `generate_dra_reference.py`
  逐点同口径），步长、泵浦功率、A_eff 全部取参考元数据。
- **解析侧**：`hybrid_raman_paper` 的 `RamanPaperSpanSolver`，`spp_steps=32`、
  `pump_depletion=True`、`raman_ase=False`，`signal_polarizations` 取 1/2。
  通过包装求解器的 `_integrate_forward/_integrate_backward` 记录逐 z 的
  泵浦与逐信道信号剖面（不修改 network 仓）。
- 比较量：泵浦总功率 P_p(z)（按发射端标称功率归一化）、中心信道信号功率
  P_s(z)/P_s(0)、以及由剖面反算的 on-off 增益
  （`10log10(P_s(L)/P_s(0)) + αL`，αL = 16.2 dB）。

物理侧的耗尽模型用**一个标量**（带内总功率、中心频率耦合）驱动泵浦 ODE，
所以它的"中心信道剖面"就是它唯一的信号剖面；解析侧逐信道，中心信道有
自己的耦合。

### 13.2 开关增益（中心 / 带边）

单位 dB。带边取计划的最低/最高信道频率。

| 计划 | 配置 | 物理 中心/低/高 | 解析 f=1 中心/低/高 | 解析 f=2 中心/低/高 |
| --- | --- | ---: | ---: | ---: |
| cl152 | `bwd_sixpumps` | 6.84 / 3.31 / 13.62 | 10.97 / 13.58 / 24.95 | 9.71 / 12.86 / 22.02 |
| cl152 | `fwd_0p20w` | 1.54 / 0.00 / 3.16 | 1.52 / 2.29 / 1.71 | 1.05 / 3.39 / −0.92 |
| c57 | `bwd_0p20w` | 9.39 / 6.30 / 5.24 | 10.35 / 6.87 / 5.84 | 9.70 / 6.44 / 5.48 |
| c57 | `fwd_0p20w` | 3.78 / 2.54 / 2.11 | 5.82 / 3.86 / 3.28 | 4.13 / 2.75 / 2.33 |

波形参考自身的逐信道 on-off（§12.1 的 JSON）与物理侧一致到 0.01 dB 量级
（例：cl152 `bwd_sixpumps` 中心 6.85 dB，物理 6.84 dB），说明"物理侧 =
参考波形"这一侧没有口径错配。

**带内增益形状**：解析侧比物理侧更"倾斜"——cl152 六泵在带边高 13 dB 的
位置，解析 f=1 报 24.95 dB。解析侧的带边增益里含**泵浦-泵浦拉曼转移**
（1340/1360 nm 的 0.63 W 把功率交给 1427/1445 nm 泵），物理侧不建模该项
（§13.4 的消融给出直接证据）。

### 13.3 逐点剖面差

归一化剖面的相对差（max / rms，%），`center` 口径。z 从 1.56 km 起
（解析侧第一个 RK4 样本点），共 21 点。

| 计划 | 配置 | 泵浦 f=1 | 信号中心 f=1 | 泵浦 f=2 | 信号中心 f=2 | 最大差所在 z |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cl152 | `bwd_sixpumps` | 18.5 / 7.8 | 19.5 / 5.7 | 20.6 / 8.8 | 11.5 / 3.3 | 泵浦 ~90 km，信号 100 km |
| cl152 | `fwd_0p20w` | 3.3 / 1.0 | 0.7 / 0.3 | 15.3 / 5.5 | 4.8 / 2.6 | 泵浦 ~1.6 km，信号 ~11 km |
| c57 | `bwd_0p20w` | 3.7 / 1.9 | 5.5 / 1.6 | 1.5 / 0.7 | 1.7 / 0.5 | 泵浦 85–90 km，信号 100 km |
| c57 | `fwd_0p20w` | 21.3 / 7.4 | 32.7 / 17.6 | 4.4 / 1.6 | 5.1 / 2.7 | 泵浦 ~6.5 km，信号 16–26 km |

读法：

- **前向泵**的差异集中在 z 的前 10–45 km：信号在这里最强，泵浦耗尽的
  积分最重，耦合系数差被指数放大（c57 `fwd` 在 11 km 处物理泵浦只剩
  标称功率的 13%，解析还有 28%）。
- **后向泵**的差异集中在 z 的最后 5–25 km（泵浦从 z=L 进入的地方），
  出口段的信号剖面差可达 20%（cl152 六泵：物理 0.116、解析 0.300）。
- `signal_polarizations=2` 系统性地把前向配置的差异减半
  （c57 fwd 32.7%→5.1%，cl152 bwd 19.5%→11.5%），因为物理侧的泵浦消耗是
  用双偏振总功率驱动的。这与 §12.3 里 factor=2 把 c57 fwd 残差改善约
  2 dB 的结论一致（光子因子修正后：f=1 −2.85 dB → f=2 −0.87 dB）。

### 13.4 主因判定

**(a) cl152 `bwd_sixpumps`：主因是物理侧缺少泵浦-泵浦拉曼转移，不是中心频率耦合。**

带内功率加权的平均耦合与中心频率耦合在总量上几乎相同（按泵浦功率加权
只差 **−0.01 dB**；逐泵虽差 −3.3…+3.9 dB，但 1476 nm 的 −3.3 dB 与
1445/1505 nm 的 +3.8 dB 相互抵消）。消融实验把解析侧耦合矩阵的
**泵浦-泵浦块清零**（保留泵浦-信号耦合），中心信道剖面与物理侧的差从
max 19.5% / rms 5.7% 降到 **3.3% / 1.1%**，中心 on-off 从 10.97 dB 降到
**7.86 dB**（物理 6.84 dB）。也就是说，这一个缺失项解释了该配置约 83%
的剖面差。

**(b) 单泵配置（c57、cl152 `fwd`）：主因是"每偏振 vs 双偏振"驱动 + 带内平均耦合。**

消融对单泵无效（没有泵浦-泵浦项）。差异由两部分组成：

1. 物理侧用双偏振总功率耗尽，解析 `signal_polarizations=1` 只用每偏振
   功率——`f=2` 把 c57 fwd 的信号剖面差从 32.7% 压到 5.1%；
2. 中心频率耦合 vs 带内平均：1450 nm 泵在 c57 的中心正好落在 13.2 THz
   拉曼峰上，4.2 THz 带宽两侧都在峰外，功率加权平均比中心低 **0.68 dB**；
   cl152（152 信道、11.3 THz）的中心对 1427/1445 nm 泵明显偏离峰、而带内
   高/低频段更靠近峰，平均比中心高 **1.63 dB**。

**(c) 两侧光子因子曾互为倒数，已在 network 侧修正（不是剩余差）。**

物理侧泵浦行是 `dP_p/dz = −(f_p/f_s)·γ·P_p·P_s`，解析侧旧实现是
`−(f_s/f_p)·γ·P_p·P_s`（逐位核对：c57 的单泵行比值为 0.9371 =
f_s/f_p）。两者都"自洽"，但相对彼此差 `(f_s/f_p)² ≈ 0.878`：同样的信号
增益下，解析侧少消耗约 12% 的泵浦功率。network `a0982cd` 把
`RamanPaperSpanSolver._build_coupling_matrix` 的负耦合因子改成标准光子数
守恒的 `f_receiver/f_source`，并加了 `C[j,k]/f_j + C[k,j]/f_k = 0` 的
回归测试；物理侧没有改，能量账保持不变。修正后 c57 fwd 的 1× 残差从
−3.21 改善到 −2.85 dB，c57 bwd 几乎不变；cl152 六泵的改善与物理侧缺
泵浦-泵浦混在一起，不能单独归因。详见 network `docs/dra-open-issues.md` §8。

**(d) 后向不动点与 5 参数剖面拟合不是主因。**

- 物理侧后向不动点在 8 次迭代内没到 1e-8，但第 8→13 次迭代的泵浦剖面
  只变 1.6e-6（相对标称功率），信号输出功率只变 1.5e-7 W。把上限提到
  16 次即收敛，剖面不变。因此 §12 的参考数据没有被不动点截断污染。
- 解析侧的 5 参数拟合误差没有单独量化成本节的对比量；§13.2 的 on-off 是
  从 ODE 剖面本身反算的，不含拟合。拟合对 NLI 的影响需要另做，
  这里不作结论。

### 13.5 可选的 `depletion_coupling="band_average"`

物理侧新增参数（默认 `"center"`，与既有结果逐位一致）：

- `"center"`：泵浦 ODE 用信号中心频率的 γ_p（历史口径）。
- `"band_average"`：用**输入信号功率谱加权**的带内平均 γ_p 驱动泵浦 ODE，
  使泵浦耗尽量与逐频率施加的信号增益用同一个平均耦合；信号增益仍逐频率
  施加。权重由 `forward()` 里输入波形的 FFT 功率谱给出，因此对任意信号
  谱（含带边滚降）都成立，不需要额外的频点配置。

注意这**只**修正 (b) 里的耦合平均，不涉及 (a) 的泵浦-泵浦转移；光子因子
已在 network 侧修正（§13.4(c)），本节表中解析侧使用修正后的口径。

**中心信道 SNR 残差变化**（残差 = 解析 − 波形；f=1/f=2 是解析侧的
`signal_polarizations`）。c57 在参考步长 0.488 m 上直接测得；cl152 的参考
步长是 0.244 m，受单次运行时限约束，A/B 在 **0.488 m 同 step** 上测得
（center 与 band_average 用同一步长，故差值有效；0.244 m 的绝对值见 §12.3）。

| 计划 | 配置 | step | center 残差 f1 / f2 | band_average 残差 f1 / f2 | 改善（f1 / f2） |
| --- | --- | ---: | ---: | ---: | ---: |
| c57 | `bwd_0p20w` | 0.488 | −1.431 / −1.363 | −1.386 / −1.317 | +0.045 / +0.046 |
| c57 | `fwd_0p20w` | 0.488 | −3.701 / −1.721 | −3.003 / −1.023 | **+0.698 / +0.698** |
| cl152 | `bwd_sixpumps` | 0.488 | +0.100 / +0.850 | +0.016 / +0.766 | +0.084 / +0.084 |
| cl152 | `fwd_0p20w` | 0.488 | +2.542 / +3.311 | +1.590 / +2.359 | **+0.952 / +0.952** |

> **口径说明（2026-11-07）**：上表的解析侧已含 network 光子因子修正
> （`a0982cd`，负耦合改为光子数守恒的 `f_receiver/f_source`）。波形侧
> SNR 与旧 A/B 相同；解析 SNR 按实测 ΔSNR 平移：c57 bwd +0.01、
> c57 fwd +0.36、cl152 bwd f1 +0.75 / f2 +0.32、cl152 fwd +0.16/+0.15 dB。
> 残差绝对值因此与修正前的版本不同，但"改善"列（只来自波形侧）不变。

- 波形 SNR 的同 step 变化：c57 bwd −0.046、c57 fwd −0.698、cl152 bwd +0.084、
  cl152 fwd +0.952 dB。残差改善 = −ΔSNR，且 f=1/f=2 两条几乎相同，
  说明这是**波形侧**的系统变化，与解析侧的偏振口径无关。
- **判定：仍不把它设为默认。** 修正后它对四个配置都改善 |残差|
  0.04–0.95 dB（方向一致），但 c57 fwd 仍有约 **3.0 dB** 负残差，
  cl152 bwd 只改善 0.08 dB 且残差已接近零；它只修正耦合平均，不解决
  泵浦-泵浦转移与偏振驱动，因此是可选实验口径，不是校准闭合。
- 若用 §12.3 的 2× 实测残差做基准，band_average 后按同 step ΔSNR
  外推为（f=1）：c57 bwd −1.11、c57 fwd −2.59、cl152 bwd −2.09、
  cl152 fwd +0.46 dB。这是外推，不是直接测量，只用于判断方向。

### 13.6 下一步（按预期收益排序）

1. **物理侧加泵浦-泵浦拉曼转移**（只对 `pump_model="depleted"`、可开关）。
   消融显示它能把 cl152 六泵的中心剖面差从 19.5% 压到 3.3%，是唯一
   能直接改善该配置剖面差的项。代价是物理 ODE 从"每泵独立"变成
   泵浦-泵浦耦合矩阵，后向泵不动点迭代的收敛性要重新验证。
   **已实现并实测，见 §13.8**：剖面差确实按预期下降，但增益形状与
   SNR 明显恶化，因此保持默认关闭。谱域耗尽驱动补上后再测，见 §13.9：
   SNR 崩坏已修复，但 pp 仍有约 3.2–3.8 dB 的代价，默认仍关闭。
2. **解析侧光子因子已修正**：network `a0982cd` 把 `−(f_s/f_p)` 改成
   光子数守恒的 `−(f_p/f_s)`，并加了守恒回归测试。物理侧不需要迁就；
   全 WDM 残差的最新值见 network `docs/dra-open-issues.md` §8。
3. **把 `signal_polarizations=2` 作为泵浦配置对照的默认口径**（不改
   物理侧）：它在 c57 fwd 上把剖面差从 32.7% 压到 5.1%，1× 残差从
   −2.85 改到 −0.87 dB（f=1/f=2），且不影响 nopump。这与 §12.3 的
   建议一致。
4. 分离解析侧 5 参数拟合对 NLI 的贡献（本节未量化）。
5. **把泵浦耗尽的信号驱动从单标量改成逐信道/逐频率功率**，这是启用
   泵浦-泵浦项的前提；否则 §13.8 的 SNR 崩坏会一直存在。
   **已实现，见 §13.9**：`depletion_model="spectral"` 修复了非物理放大，
   但 pp 仍使 SNR 比 no-pp 低 3.2–3.8 dB，因此不作为默认。

### 13.7 复现

```bash
# 剖面诊断（快，秒级；写 center 与 band_average 两份 JSON）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --coupling center \
    --output examples/dra_reference_wide/diagnostics/profile_diagnosis.json
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --coupling band_average \
    --output examples/dra_reference_wide/diagnostics/profile_diagnosis_band_average.json

# 中心信道 SNR A/B（每个用例一次非线性 SSFM，c57 约 5 min、cl152 约 10 min；
# 线性参考用频域精确解，实测与参考 lin_* 波形相对误差 4e-8）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases c57:fwd_0p20w --snr \
    --coupling band_average
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases cl152:fwd_0p20w --snr \
    --coupling band_average --step-m 0.48828125
```

`--snr` 的物理侧非线性传播仍走 `DistributedRamanFiber.forward()`；只有
线性参考用频域一次乘完（色散/损耗/拉曼增益都是频域对角算符、彼此对易，
分步乘积严格等于 `exp((A+B)L)`），因此不改变口径。

### 13.8 物理侧的泵浦-泵浦拉曼转移（可选，默认关闭）

§13.4(a) 的消融说明"物理侧不建模泵浦-泵浦转移"是 cl152 六泵剖面差的主因。
本节把它补进 `DistributedRamanFiber`，开关是构造参数
`pump_pump_raman: bool = False`，只对 `pump_model="depleted"` 有效
（与 `"undepleted"` 同时使用会显式抛 `ValueError`：未耗尽剖面是解析衰减，
没有可耦合的泵浦状态量）。

#### 实现口径

对每一对泵 `(i, j)`（`f_i > f_j`，i 为短波长/高频泵），按**成对光子数守恒**
写入泵浦行的 RHS：

```
R_ij = g_R(f_i − f_j) / A_eff · P_i · P_j          # W/m
dP_i += −(f_i / f_j) · R_ij                        # 高频泵失去一个泵浦光子
dP_j += +R_ij                                      # 低频泵获得功率
```

于是 `d(P_i/f_i)/dz + d(P_j/f_j)/dz = 0`，差额 `(f_i − f_j)/f_j · R_ij` 作为
声子能量离开光场。`g_R` 与信号增益共用同一 Gaussian 谱
（`raman_gain_spectrum` + `raman_peak_gain_m_per_w` + `raman_peak_shift_hz` +
`raman_fwhm_hz`）与同一个 `a_eff_si`；矩阵形式见
`DistributedRamanFiber._pump_pump_coupling()`。

积分上把它写成"沿传播方向的泵浦损耗率"`−P_i·Σ_j C_ij P_j`，与 `α_p P_p`、
耗尽项同号，一起乘 `−s_p`（`s_p = ±1`）。这样前向泵得到 `+dP_i/dz`，后向泵
因为沿 −z 传播、同一份增益对应 `dP_i/dz` 取反。前向/后向 RK4 都包含该项：

* `forward_rhs`：总泵浦向量 = 当前前向泵状态 + 上一轮的后向泵剖面；
* `backward_rhs`：总泵浦向量 = 上一轮的前向泵剖面 + 当前后向泵状态。

交叉项（前向泵 ↔ 后向泵）在两个方向各用对方的上一轮剖面，沿用原有不动点
结构；每对只由各自的 ODE 写自己的 `dP`，不重复计入。

#### 能量账

`_depletion_diagnostics["pumps"]` 的旧键不变，新增三个（默认关闭时恒为 0）：

| 键 | 含义 |
| --- | --- |
| `pump_pump_sent_w` | 该泵在泵浦-泵浦项中失去的功率积分（正数） |
| `pump_pump_received_w` | 该泵在泵浦-泵浦项中获得的功率积分（正数） |
| `pump_pump_net_w` | `received − sent`（有符号）；顶层另有 `pump_pump_net_total_w` = Σ |

`net` 是"增益"口径，所以逐泵能量账写作
`launch − residual − passive_loss − transferred_to_signal + pump_pump_net ≈ 0`
（§13.4(c) 的 `launch − residual − passive − transferred` 形式加上 pp 项时
符号取正）。

cl152 `bwd_sixpumps`、起点功率 0.304 W 的输入波形、25 m RK4、32 次迭代实测：

| 泵 (nm) | launch | residual | passive | →信号 | sent | recv | net | 账余 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1340 | 0.63 | 2.81e-6 | 0.09401 | 1.2e-21 | 0.53598 | 0 | −0.53598 | +1.4e-6 |
| 1360 | 0.63 | 6.05e-7 | 0.09398 | 1.1e-15 | 0.53602 | 7.9e-6 | −0.53602 | +1.1e-6 |
| 1427 | 0.165 | 8.15e-5 | 0.31927 | 2.0e-3 | 0.39067 | 0.54703 | +0.15635 | −1.4e-6 |
| 1445 | 0.131 | 4.41e-4 | 0.44801 | 0.05639 | 0.06748 | 0.44132 | +0.37384 | −1.1e-6 |
| 1476 | 0.125 | 2.78e-7 | 0.07604 | 0.07199 | 1.0e-5 | 0.02303 | +0.02302 | +3.8e-8 |
| 1505 | 0.075 | 1.17e-3 | 0.45995 | 0.04570 | 0 | 0.43183 | +0.43183 | −1.6e-7 |

单位 W。账余是 `launch − residual − passive − →信号 + net`，量级 1e-6 W
（相对标称功率约 2e-6），来源是 25 m RK4 与梯形积分的离散误差：两泵算例里
步长 200→100→50→25 m 的账余为 3.9e-5→9.7e-6→2.4e-6→6.0e-7 W，按 ~h² 收敛。
**全局 `Σ net = −0.08695 W < 0`**，|Σ net| 即泵浦-泵浦项交给声子的功率。
1340/1360 nm 各送出 0.536 W（其中约 85% 的入射功率），1427 nm 收 0.547 W
再转给 1505 nm 0.391 W，链式 1340/1360 → 1427 → 1505 与 §13.2 的判断一致。

独立校验：用未耗尽剖面粗估 1340→1427 的转移
`R ≈ (6e-14/80e-12)·0.98·0.63·0.165 = 7.6e-5` W/m，泵浦损耗 0.25 dB/km 对应
`∫exp(−2α(L−z))dz ≈ 8.7 km`，得 ≈0.66 W；实际 0.54 W（1340 自身被耗尽后
速率下降）与之同量级，说明实现没有漏乘/多乘。

两泵极限（80 km、0.2 dB/km、信号 1e-25 W，即只有泵浦-泵浦项在动）：

* 1360/1445 nm（频差 13.4 THz，正落在拉曼峰）：残余 0.015825 / 0.004145 W
  变成 **2.2e-7 / 0.019038 W**，send 0.536 W / receive 0.504 W；
* 1340/1505 nm（频差 24.5 THz，Gaussian 谱翼 ≈7e-7）：残余
  0.01582488 → 0.01582487 与 0.00188391 → 0.00188393 W，即短波长泵降、
  长波长泵升，量级只有 1e-8 W——与谱翼一致，符号仍然正确；
* 两个泵都把 `alpha_db_per_km` 设 0 时，`Σ P/f` 在 200/100/50/25 m 四种步长
  下都守恒到 −6.1e-8（相对值，残差与步长无关，说明是 1e-8 的不动点阈值而非
  积分误差）；100 m 与 50 m 的残余功率最大差 2.2e-13 W（相对 1.5e-7）。

#### 逐位兼容

默认 `pump_pump_raman=False` 时 `_pump_pump_coupling()` 返回 `None`，RHS 走与
此前完全相同的分支（不做任何额外求和），因此：

* `tests/test_distributed_raman.py::test_pump_pump_default_off_reproduces_committed_depleted_waveform`
  用 3ch 耗尽参考做 `np.array_equal`，并断言三个新键恒为 0、元数据里仍是
  `no pump-pump Raman transfer`；
* `...::test_pump_pump_default_off_matches_explicit_off_for_six_backward_pumps`
  锁定六泵后向配置的缺省/显式 False 逐位相同，同时确认打开开关确实改变结果；
* 重跑 §13.7 的默认剖面诊断，与已提交 `profile_diagnosis.json` 的所有数值
  逐项相同，只多出新字段（`pump_pump_raman` 与三项能量账）；
* 另用 `git archive b7e7c7a`（本分支的基点）的代码树独立重算同一六泵后向
  配置（512 样点、100 m 步长）的 `nl_*` 波形与泵浦剖面，与当前默认路径
  `np.array_equal` 逐位相同（max |Δ| = 0）。

#### cl152 `bwd_sixpumps` 剖面 A/B

`--pump-pump --coupling center --step-m 0.48828125 --iterations 32`
（`examples/dra_reference_wide/diagnostics/profile_diagnosis_pump_pump.json`）。
相对差为 max / rms（%），解析侧仍含它自己的泵浦-泵浦项，所以是"物理补上后"
的真实对照：

| 比较 | 关闭（§13.3） | 打开 | 解析 f=1 现状 |
| --- | ---: | ---: | ---: |
| 泵浦总剖面 f=1 | 18.5 / 7.8 | **12.4 / 5.0** | — |
| 泵浦总剖面 f=2 | 20.6 / 8.8 | **14.6 / 6.0** | — |
| 中心信号剖面 f=1 | 19.5 / 5.7 | **9.0 / 2.5** | — |
| 中心信号剖面 f=2 | 11.5 / 3.3 | 17.0 / 4.8 | — |

中心信道 f=1 的剖面差从 19.5/5.7% 降到 9.0/2.5%，与 §13.4(a) 的消融预期
（3.3/1.1%）同向，但物理侧补上 pp 后仍差约 2.7 倍；f=2 反而略差
（11.5→17.0%），因为解析 f=2 的泵浦耗尽更强，剩余差主要是泵浦 `A_eff`
口径（解析侧逐泵 6.95e-11 m²，物理侧标量 8e-11 m²）与偏振驱动。
光子因子已在 network 侧修正（§13.4(c)），不再是这里的剩余项。

**但增益形状明显变差**：中心/带边 on-off 从 6.84 / 3.31 / 13.62 dB 变成
**12.05 / 20.36 / 35.43 dB**（解析 f=1 为 10.97 / 13.58 / 24.95，f=2 为
9.71 / 12.86 / 22.02）。中心接近解析，带边则远高于解析，也远高于参考波形
（6.85 dB 中心）。原因是 1427/1445/1505 nm 三泵被抬高 0.44–0.55 W 后，
靠近它们的短波长信道获得的开关增益远大于中心信道。

#### cl152 `bwd_sixpumps` 中心信道 SNR A/B

同 step 0.48828125、32 次迭代，`--snr`（每个用例一次非线性 SSFM，约 6.5 min）：

| 物理侧 | 波形 `snr_nli_db` | 残差 f=1 | 残差 f=2 |
| --- | ---: | ---: | ---: |
| 关闭（复现 §13.5） | 31.354 | +0.100 | +0.850 |
| 打开 | **5.889** | **+25.564** | **+26.314** |

解析侧（`hybrid_raman_paper`，始终含它的泵浦-泵浦项）两次运行完全相同：
光子因子修正前（network `e1ee4af`）为 f=1 30.707 dB、f=2 31.880 dB；
修正后（network `a0982cd`，光子数守恒的 `f_receiver/f_source`）为
f=1 **31.454** dB、f=2 **32.204** dB。因此上表"打开"行的残差为
**+25.564 / +26.314 dB**（仍为约 25 dB 的崩坏，结论不变）。关闭时波形
31.353959 dB 与已提交 `profile_snr_ab.json` 的 31.353958 dB 差 1e-6 dB
（已提交那次用默认 8 次迭代；该 JSON 的解析残差字段是光子因子修正前
口径，引用时以本表为准）。

**SNR 不是改善，而是崩坏 25 dB。** 直接原因不是 pp 项本身，而是它与现有
"用一个标量带内总功率驱动泵浦耗尽"的近似不相容：打开 pp 后各信道开关增益
拉开到 20–35 dB，但泵浦 ODE 只知道输入波形的带内总功率（0.304 W，见
`signal_out_w` 只有 0.117 W），SSFM 却把这一增益逐频率施加到每个信道。
用频域精确解（n2 = 0）实测同样的剖面：152 信道总输出从 **0.054 W 涨到
5.88 W**，逐信道功率跨 275–502（相对刻度）。于是中心信道的 NLI 被相邻
高增益信道通过 XPM/FWM 推高约 25 dB，`snr_nli_db` 塌到 5.9 dB。也就是说：

* 泵浦剖面这一层，pp 是对的（与解析侧总泵浦剖面差 12.4%，与未耗尽粗估一致）；
* 但"单标量信号功率 + 逐频率增益"这套近似在增益斜率变陡后失效，pp 只是把
  这个既有缺陷放大到不可忽略。

#### 结论与建议

1. `pump_pump_raman` 保持**默认 False**：它保持参考数据逐位不变，而打开后
   虽然中心剖面差减半，但带边增益与 SNR 都严重偏离，不构成可用的改进。
2. 若要把它变成默认，需要**同时**做两件事，属于另一次改动：
   (a) 重新生成 `examples/dra_reference_wide/` 等参考 npz/json；
   (b) 把泵浦耗尽的驱动从"单一标量带内总功率"改成逐信道/逐频率功率，
   否则 §13.8 的 SNR 崩坏会一直存在。
3. 解析侧 `signal_polarizations` 仍是独立的系统差，pp 打开后依然存在；
   Eq. (2) 频率因子已在 network 侧修正，不再是剩余项。

#### 复现

```bash
# 剖面 A/B（秒级）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases cl152:bwd_sixpumps --pump-pump \
    --coupling center --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_diagnosis_pump_pump.json

# 中心信道 SNR A/B（约 6.5 min）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases cl152:bwd_sixpumps --snr --pump-pump \
    --coupling center --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_snr_pump_pump.json
```

> **注意（2026-11-07）**：上面的命令是在 §13.9 之前记录的。现在
> `pump_pump_raman=True` 与 `depletion_model="scalar"` 已显式禁止，
> `--pump-pump` 未指定 `--depletion-model` 时会默认 `spectral`；直接重跑
> 会得到 §13.9 的新结果，而不是本节的历史 scalar+pp 数字。历史 JSON
> （`profile_diagnosis_pump_pump.json`、`profile_snr_pump_pump.json`）保留
> 作对照，不要用当前 CLI 覆盖。复现历史 scalar+pp 需要 §13.9 之前的
> 代码版本。

不动点收敛（cl152 全波形、25 m RK4、阈值 1e-8）：打开 pp 后 8 次不收敛，
16 次仍不收敛，31 次收敛。8→16 次中心 on-off 只变 0.024 dB（12.030→12.054），
16→32 次只变 0.0002 dB，剖面差与能量账在 4 位有效数字内不变；因此 §13.8 的
结论不受迭代截断影响。上表的 8 次数据仅用于说明"必须提高迭代上限"。

> **历史文件说明**：`profile_snr_ab.json` 的波形 `snr_nli_db` 是实测且仍
> 有效；其解析残差字段生成于 network 光子因子修正（`a0982cd`）之前，
> 引用时按 §13.5 的修正后解析值平移，不要直接把旧残差当现值。本次重新
> 生成的 `profile_diagnosis*.json` 与 `profile_snr_pump_pump.json` 已用
> 修正后的解析侧。

> **后续（2026-11-07）**：§13.8 的负结果保留作历史对照。它的直接原因
> （"单标量信号功率 + 逐频率增益"口径失配）已在 §13.9 用逐频箱泵浦耗尽
> 驱动修复：cl152 六泵 + pp 的总输出从 5.88 W 回到 0.446 W，SNR 从
> 5.889 dB 回到 28.190 dB。但 pp 仍带来约 3.2–3.8 dB 的 SNR 代价，因此
> **默认仍然关闭**；结论见 §13.9.7。

## 13.9 逐信道/逐频率的泵浦耗尽驱动

§13.8 证明：打开 `pump_pump_raman` 后，物理侧的"单标量带内总功率 + 逐频率
增益"近似与 SSFM 不再同口径，导致 0.304 W 输入出现 5.88 W 的非物理线性
输出和 5.9 dB 的 SNR 崩坏。本节把泵浦 ODE 的信号驱动改成**逐频箱功率**，
让泵浦耗尽量与 SSFM 在每一频点施加的增益使用同一个 γ，再重跑 §13.8 的
A/B。默认仍是 `"scalar"`，本节的谱域口径是**可选实验路径**；重新生成
`examples/dra_reference_wide/*.npz` 是另一步，本次没有做。

### 13.9.1 API 与约束

`DistributedRamanFiber.__init__` 新增：

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `depletion_model` | `"scalar"` | `"scalar"`（历史口径）或 `"spectral"`（逐频箱） |
| `spectral_bins` | `256` | 谱域模型的频箱数，必须 >0 |

导出常量 `DEPLETION_MODELS = ("scalar", "spectral")`。

* 默认 `"scalar"` 的数值路径**未改动**：3ch 耗尽参考 `np.array_equal`、
  六泵后向配置与 `dra-dev` 逐位一致（全量测试通过）。
* `pump_pump_raman=True` 只允许 `depletion_model="spectral"`；与
  `"scalar"` 同时使用显式抛 `ValueError`（提示先启用谱域驱动）。
  与 `pump_model="undepleted"` 的旧报错保持不变。
* `depletion_coupling` 只对 `"scalar"` 有意义。`"spectral"` 时忽略它，
  元数据 `dra_depletion_coupling` 记为 `null`，`dra_pump_model_detail`
  明确写 "depletion_coupling is not used"；显式给非默认值会发
  `RuntimeWarning`，不静默混用。

### 13.9.2 频箱与耦合系数

`forward()` 从输入 `opt_signal` 构造频域功率分布，频率轴与 SSFM 完全一致：

```
freqs    = center_freq + fftfreq(n, 1/fs) · n/(n−1)
spectrum = mean(|FFT(data)|², axis=polarization)
```

把频率轴等宽划分为 `spectral_bins` 个箱，只保留有功率的箱（空箱权重 0）。
每箱用**功率加权平均频率**作为代表：

```
f_b = Σ_{k∈b} S_k f_k / Σ_{k∈b} S_k
w_b = Σ_{k∈b} S_k / Σ_all S_k          # Σ_b w_b = 1
```

信号总功率仍取 `input_power_w = Σ_pol mean_t |A|²`，
`P_{s,b}(0) = input_power_w · w_b`，因此 `Σ_b P_{s,b}(0) = input_power_w`。
输入波形本身不被修改；这些频箱只驱动泵浦 ODE。

逐箱耦合系数：

```
γ_{p,b} = g_peak · g̃_R(f_p − f_b) / A_eff
```

复用 `raman_gain_spectrum`、`raman_peak_gain_m_per_w`、`raman_fwhm_hz`、
`raman_peak_shift_hz` 与 `a_eff_si`，与 `_make_step_gain` 的 SSFM 逐频率
增益同源。信号损耗沿用 `alpha_np_inv_m` 的均值（参考数据只有平坦损耗），
光子因子 `f_p/f_b` **逐箱**计算。

### 13.9.3 耦合 ODE 与数值实现

状态量 `P_{s,b}`（形状 `(n_bins,)`）与 `P_p`（形状 `(n_pumps,)`）：

```
dP_{s,b}/dz = (Σ_p γ_{p,b} P_p − α_s) P_{s,b}
dP_p/dz     = −s_p [ α_p P_p
                     + P_p Σ_b (f_p/f_b) γ_{p,b} P_{s,b}
                     + P_p Σ_j C_{pj} P_j ]
```

`C` 与 §13.8 的 `_pump_pump_coupling()` 完全相同（成对光子数守恒，符号和
频率因子未改）。前向/后向 RK4 结构沿用标量实现：前向把 `P_{s,b}` 和前向泵
一起作为状态积分；后向 RHS 用 `ps(:, z)` 的逐箱线性插值。

初值用未耗尽泵浦剖面和逐箱速率积分出完整的 `P_{s,b}(z)`，供后向泵第一次
迭代使用，不退回标量初值。

**后向泵的欠松弛。** cl152 六泵全为后向泵。谱域 + pp 打开后，从 z=L 反向
积分 `dP/dz = +λP` 的条件数很大；直接把 RK4 结果整条替换会让不动点出现
周期 2 振荡（实测 `pump_err` 在 2.2–3.2 之间来回跳，`signal_out` 在
0.13–3.4 W 之间振荡，256 次仍不满足 1e-8）。因此对**后向泵剖面**做
`profile ← (1−θ)·previous + θ·updated`，取 θ=0.5；θ=0.25 收敛到同一
不动点（on-off 与 `signal_out` 在 8 位有效数字内相同）。没有 pp 项时
θ=1（保持原样，13 次收敛）。θ 只影响不动点迭代的收敛速度，不改变方程。

cl152 六泵 + pp 在 25 m RK4、阈值 1e-8 下 **26 次迭代收敛**。注意
`depleted_max_iterations` 的类默认值仍是 8；谱域 + pp 需要调用方显式提高
上限（本节的诊断统一用 `--iterations 32`），并检查
`dra_depletion_converged`。

### 13.9.4 能量账

诊断字典新增 `signal_out_w`、`spectral_bins`、`depletion_model`，以及
两个光子账标量：

* `signal_photon_gain_per_s = ∫ Σ_b (Σ_p γ_{p,b} P_p) P_{s,b} / (h f_b) dz`
* `pump_photon_transfer_per_s = Σ_p transfer_p / (h f_p)`

其中 `transfer_p = ∫ P_p Σ_b (f_p/f_b) γ_{p,b} P_{s,b} dz`。逐泵能量账仍是

```
launch − residual − passive − transferred + pump_pump_net ≈ 0
```

cl152 `bwd_sixpumps`、0.304 W 输入、25 m RK4、32 次上限（26 次收敛）实测：

| 泵 (nm) | launch | residual | passive | →信号 | sent | recv | net | 账余 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1340 | 0.630 | 1.12e-5 | 0.11753 | 2.5e-12 | 0.51245 | 0 | −0.51245 | +3.0e-7 |
| 1360 | 0.630 | 1.11e-5 | 0.12313 | 4.4e-8 | 0.50687 | 9.2e-6 | −0.50686 | −9.7e-8 |
| 1427 | 0.165 | 1.42e-4 | 0.29228 | 0.21401 | 0.20614 | 0.54757 | +0.34143 | +2.4e-7 |
| 1445 | 0.131 | 7.07e-5 | 0.24345 | 0.25352 | 0.01952 | 0.38556 | +0.36604 | +5.4e-7 |
| 1476 | 0.125 | 1.52e-5 | 0.10386 | 0.04987 | 8.2e-6 | 0.02875 | +0.02875 | +7.8e-8 |
| 1505 | 0.075 | 1.38e-4 | 0.21588 | 0.06991 | 0 | 0.21092 | +0.21092 | −1.1e-7 |

单位 W；账余最大 5.4e-7 W。**Σ net = −0.07217 W < 0**，差额以声子能量
离开光场。光子账：
`signal_photon_gain_per_s = pump_photon_transfer_per_s = 4.281777634e18 s⁻¹`
（相对差 1.4e-16），逐箱光子数严格守恒。

同一次求解的总信号功率：

| 量 | 值 (W) |
| --- | ---: |
| ODE `signal_out_w` | 0.445872 |
| 非线性 SSFM 总输出（n2=2.6e-20） | 0.445825 |
| 频域线性参考总输出（n2=0） | 0.445871 |

ODE 与 SSFM 相对差 **1.1e-4**，与线性参考 **3.8e-6**；三者都在
"输入 0.304 W + 泵浦总注入 1.756 W = 2.060 W" 以下，§13.8 的 5.88 W
非物理放大已经消失。

### 13.9.5 scalar vs spectral

**窄带/CW 一致性**：100 GHz 采样、单 1450 nm 前向泵、每偏振 1 mW，
scalar 与 spectral 的末泵浦残余相对差 **2.5e-5**、输出功率相对差
**1.8e-5**，均远小于 0.1%。

**c57 `fwd_0p20w`（单泵，pp 不参与）**，step 0.48828125：

| 物理侧 | 波形 `snr_nli_db` | 解析残差 f=1 / f=2 | ODE 总输出 / SSFM 总输出 |
| --- | ---: | ---: | ---: |
| scalar | 28.732 | −3.701 / −1.721 | 0.006539 / 0.005797（差 12.8%） |
| spectral | 28.072 | −3.040 / −1.060 | 0.006402 / 0.006402（差 6e-7） |

spectral 让波形 SNR 下移 0.660 dB，与 §13.5 的 `band_average` 结果
（−0.698 dB）同向同量级；解析残差同时改善约 0.66 dB，因此这不是谱域
引入的独立退化，而是"用真实信号谱驱动泵浦"后的预期变化。更重要的是，
**scalar 的 ODE 总输出与 SSFM 相差 12.8%，spectral 只有 6e-7**，说明
谱域驱动才与 SSFM 同口径。

**cl152 `bwd_sixpumps` 不带 pp**（0.48828125 m）：scalar 列取
`profile_diagnosis_scalar_32iter.json`（32 次迭代，`converged=True`），
与 spectral 使用同一迭代上限；它与 §13.5 的 8 次版本在 on-off 上只差
5e-5 dB 量级。

| 物理侧 | 波形 `snr_nli_db` | 解析残差 f=1 / f=2 | 泵浦剖面差 f=1 / f=2 | 中心信号剖面差 f=1 / f=2 |
| --- | ---: | ---: | ---: | ---: |
| scalar | 31.354 | +0.100 / +0.850 | 18.5 / 7.8%, 20.6 / 8.8% | 19.5 / 5.7%, 11.5 / 3.3% |
| spectral | 31.453 | +0.001 / +0.751 | 18.0 / 7.5%, 20.2 / 8.5% | 17.8 / 5.1%, 9.7 / 2.7% |

不带 pp 时 spectral 基本中性（+0.099 dB），剖面差略降；ODE 总输出与 SSFM
相对差 5.2e-5（0.048131 / 0.048128 W）。

### 13.9.6 cl152 六泵 spectral + pp 的 A/B

step 0.48828125 m、25 m RK4、32 次上限。旧 scalar+pp 负结果取自已提交的
`profile_diagnosis_pump_pump.json` / `profile_snr_pump_pump.json`（该组合
现在被 API 禁止，不再重跑）；新结果在
`profile_diagnosis_spectral_pump_pump.json` /
`profile_snr_spectral_pump_pump.json`。

**剖面与开关增益：**

| 物理侧 | 中心/低/高 on-off (dB) | 泵浦剖面差 f=1 / f=2 | 中心信号剖面差 f=1 / f=2 | ODE `signal_out_w` |
| --- | ---: | ---: | ---: | ---: |
| scalar + pp（旧） | 12.05 / 20.36 / 35.43 | 12.5 / 5.0%, 14.6 / 6.0% | 9.0 / 2.5%, 17.0 / 4.8%* | 0.1171 W |
| spectral + pp（新） | **9.83 / 9.70 / 25.24** | **2.2 / 0.9%, 0.6 / 0.2%** | **7.4 / 2.2%, 0.7 / 0.2%** | **0.4459 W** |
| 解析 f=1 | 10.97 / 13.58 / 24.95 | — | — | — |
| 解析 f=2 | 9.71 / 12.86 / 22.02 | — | — | — |

\* 旧 scalar 路径没有频谱信息，它的"中心信号剖面"实际是带内总功率，与
spectral 的中心信道提取不是同一观测量；旧值只作历史对照。

spectral+pp 的中心 on-off（9.83 dB）与解析 f=2（9.71 dB）相差 0.12 dB，
高带边（25.24 dB）与解析 f=1（24.95 dB）相差 0.29 dB，低带边（9.70 dB）
仍比解析 f=1/f=2 低 3.2–3.9 dB。相对于旧 scalar+pp 的 35.43 dB 高带边，
增益形状已经回到解析量级。泵浦与中心信号剖面差也从 12–17% 降到 0.2–7.4%。

**中心信道 SNR：**

| 物理侧 | 波形 `snr_nli_db` | 解析残差 f=1 / f=2 |
| --- | ---: | ---: |
| scalar 无 pp（§13.5 同 step） | 31.354 | +0.100 / +0.850 |
| spectral 无 pp（本次） | 31.453 | +0.001 / +0.751 |
| scalar + pp（旧负结果） | **5.889** | +25.564 / +26.314 |
| spectral + pp（新） | **28.190** | +3.263 / +4.013 |

spectral+pp 比旧 scalar+pp 恢复 **+22.301 dB**，不再出现 5.9 dB 的崩坏；
总输出也从 5.88 W 回到 0.446 W。但相对同 step 的 no-pp 波形
（31.354/31.453 dB），spectral+pp 仍低 **3.16–3.82 dB**，解析残差
+3.26/+4.01 dB，超过任务设定的 2 dB 门槛。

### 13.9.7 结论与建议

1. **默认仍是 `depletion_model="scalar"` + `pump_pump_raman=False`。**
   scalar 路径逐位不变；谱域口径是可选的实验路径。
2. **谱域耗尽本身是成功的**：窄带与标量一致到 2.5e-5；单泵 c57 与六泵
   cl152 的 ODE 总输出都与 SSFM 对齐到 1e-4 量级（标量路径差 12.8%）；
   逐泵能量账闭合到 1e-6 W，逐箱光子数守恒到 1e-16。它解决了 §13.8 的
   "单标量 + 逐频率增益"口径失配。
3. **仍不建议启用 `pump_pump_raman=True`。** 虽然非物理放大和 SNR 崩坏
   已经修复，但 pp 把 1340/1360 nm 的约 0.47 W 转给 1427/1445/1505 nm 后，
   带边信道开关增益仍达 25.2 dB（中心 9.83 dB），中心信道承受来自被放大
   带边信道的 XPM/FWM，波形 SNR 比 no-pp 低 3.16–3.82 dB。解析侧
   `hybrid_raman_paper` 的 GN 口径给出 31.45/32.20 dB，残差 +3.3/+4.0 dB，
   说明剩余差不是 ODE/SSFM 口径错配（两者总输出已对齐到 1e-4），而是
   pp 造成的强增益倾斜下 NLI 口径的系统差。
4. **重生成参考数据是另一步。** 本次只提交代码、测试、诊断脚本和诊断
   JSON；`examples/dra_reference_wide/*.npz` 未改。若要把 spectral 设为
   默认或启用 pp，需要同时重跑参考数据集并复核对解析模型的 NLI 口径。

### 13.9.8 复现

```bash
# 谱域剖面诊断（秒级）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases cl152:bwd_sixpumps \
    --depletion-model spectral --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_diagnosis_spectral.json

# 谱域 + pp 剖面诊断（约 15 s）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases cl152:bwd_sixpumps --pump-pump \
    --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_diagnosis_spectral_pump_pump.json

# 谱域 / 谱域+pp 的中心信道 SNR（各约 6 min）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases cl152:bwd_sixpumps --snr \
    --depletion-model spectral --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_snr_spectral.json
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases cl152:bwd_sixpumps --snr --pump-pump \
    --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_snr_spectral_pump_pump.json

# c57 fwd 单泵对照（各约 5 min）
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases c57:fwd_0p20w --snr \
    --depletion-model scalar --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_snr_c57_fwd_scalar.json
PYTHONPATH=src /home/zsig/miniconda3/envs/nlo/bin/python \
    examples/diagnose_dra_profile.py --cases c57:fwd_0p20w --snr \
    --depletion-model spectral --step-m 0.48828125 --iterations 32 \
    --output examples/dra_reference_wide/diagnostics/profile_snr_c57_fwd_spectral.json
```
