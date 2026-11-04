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

第一版**不追求**全 WDM 规模（57/479 信道）或泵浦-信号强耗尽场景。

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

第一版显式采用**未耗尽泵浦**模型：

- 前向泵：`P_p(z) = P_p(0) · exp(−α_p z)`；
- 后向泵：`P_p(z) = P_p(L) · exp(−α_p (L − z))`；
- 泵浦损耗由每个 `PumpSpec.attenuation_db_per_km` 单独给出；
- **不考虑**泵浦-信号耗尽、泵浦-泵浦拉曼转移、泵浦间功率交换。

对 0.1–0.2 W 泵浦、mW 级信号，这个近似在参考数据的功率范围内是合理的；
但强泵浦/长距离/高信号功率场景需要后续的耦合 RK4 版本。

未耗尽近似下，前向和后向泵的**总开关增益相同**，因为二者只差泵浦功率
沿 z 的分布，而 `∫_0^L P_p(z) dz` 相同；局部功率剖面和非线性积累则不同。

### 2.3 拉曼增益谱

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

### 2.4 通道间 ISRS

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
`dra_ase_*`，与 EDFA 的 `edfa_ase_*` 完全分开。

该 ASE 模型仍忽略泵浦耗尽、泵浦噪声传递和后向 ASE，因此仅作为**可选**
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
    add_ase=False,
)
output = fiber.forward(wdm_signal)
print(output.ctx.meta["fiber_solver"])          # "numpy-ssfm-dra"
print(output.ctx.meta["dra_on_off_gain_db_max"])
```

关键元数据：

| 键 | 含义 |
| --- | --- |
| `fiber_solver` | `"numpy-ssfm-dra"` |
| `dra_model` | `"undepleted"` |
| `dra_n_pumps` / `dra_pumps` | 泵浦数量与逐泵配置 |
| `dra_on_off_gain_db_min/max` | 信号带内开关增益范围 |
| `dra_channel_isrs` | `"not_modelled"` 或 `"raman_r3"` |
| `dra_ase_model` | `"none"` 或 `"undepleted-raman-ase"` |
| `dra_ase_power_per_pol_w` | 可选 ASE 功率 |

## 6. 参考数据集

文件：

```text
examples/dra_reference/dra_reference.npz
examples/dra_reference/dra_reference.json
```

生成脚本：

```bash
python examples/generate_dra_reference.py
# 可选：--symbols 256 --seed 20261104 --output-dir ...
```

### 6.1 参数

| 项目 | 值 |
| --- | --- |
| 信道数 | 3 与 5 |
| 调制 | DP-QPSK，50 GBd，RRC roll-off 0.1，sps=4 |
| 信道间隔 | 100 GHz |
| 入纤功率 | 0 dBm / 偏振 / 信道 |
| 光纤 | 100 km，0.2 dB/km，β₂=−20 ps²/km，β₃=0.1×10⁻³ ps³/km，A_eff=80 μm² |
| 泵浦 | 1450 nm，0.10/0.20 W，前向/后向，0.25 dB/km |
| 跨数 | 1 |
| 符号数 | 256 / 信道 |

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

## 9. 已知限制

- 泵浦未耗尽，未建模泵浦-泵浦耦合和泵浦耗尽；
- 拉曼谱为 Gaussian 近似，不是实测 `RamanGain.dat`；
- 通道间 ISRS 默认关闭；
- ASE 为可选近似，参考数据未启用；
- 只验证单跨 100 km、3/5 信道、50 GBd；不适用于全 WDM 规模或强耗尽；
- PMD 关闭（`d_pmd=0`），参考数据聚焦 DRA/色散/Kerr 的确定性对照。
