# 话题：onetrans-d · 复刻 OneTrans-S（训练侧）

> 状态：立项（2026-11-16）· 论文已归档 `papers/OneTrans_2510.26104v3.pdf/.txt`（arXiv v3）
> 范围：**只做训练**。不做推理优化、不做在线 A/B、不做 serving（论文里 KV cache / p99 那套一律不碰）。
> 目标：在昇腾 950DT 上把 **OneTrans-S** 复刻到与论文同结构、同 batch 口径，把**训练性能与利用率**与论文的 **H100** 数字并列对比。
> 本文件是话题的唯一真源：论文数字只在这里落一次，报告引用本文件。

---

## 1. 论文事实（唯一出处：arXiv:2510.26104v3）

### 1.1 模型与训练口径

| 项 | 值 | 出处 |
|---|---|---|
| 模型家族 | OneTrans：统一 tokenizer 把序列与非序列特征变成**一条** token 序列；序列 token 共享参数、NS-token 用专有参数；因果注意力 | Abstract / §3.2 |
| 变体 | **OneTransS**（91 M params，2.64 TFLOPs）、OneTransL（330 M，8.62 TFLOPs） | Table 2 |
| **S 的结构（复刻目标）** | **6 个堆叠 OneTrans block**、宽度 **d=256**、**H=4** heads；金字塔把序列 query token 从 **1190 → 12** 线性收缩，每层取 32 的整数倍，顶层对齐 NS-token 数 | **§4.1.4** |
| TFLOPs 口径 | 「training compute in TFLOPs **at batch size 2048**」（未说明是 per-sample / per-step / per-batch） | §4.1.2 |
| batch 口径 | 「**per-GPU batch size is set to 2048** during training」 | §4.1.2 |
| 训练硬件 | **data-parallel all-reduce on 16 × H100**（型号/互连/精度论文未写） | §4.1.2 |
| 优化器 | 双优化器：sparse 用 Adagrad(0.1, 1.0)，dense 用 RMSProp(lr 0.005)；Pre-Norm；全局 grad-norm clip 90(dense)/120(sparse) | §4.1.2 |
| 数据 | 工业大表：29.1B impressions / 27.9M users / 10.2M items，**不公开** | Table 1 |
| 精度口径 | CTR/CVR 的 AUC 与 UAUC，next-batch evaluation，按天宏平均 | §4.1.2 |

### 1.2 性能与利用率的全部论文数字（就这些，没有别的）

| 项 | 值 | 出处 | 备注 |
|---|---|---|---|
| 未优化 OneTransS 训练 runtime | **407 ms** | Table 4 首行 | 唯一一处直接给 S 的训练耗时 |
| 训练显存（峰值 GPU） | 未优化 **53.13 GB** | Table 4 | OneTransL 的 32 GB 见 Table 5 |
| 逐项优化收益（相对未优化基线） | pyramid −28.7% / KV cache −30.2% / FlashAttention −50.1% / mixed precision+recomp −32.9% | Table 4 | 见 §1.3 读法 |
| **MFU** | OneTransL **30.8%**、基线 DCNv2+DIN **13.4%** | Table 5 | **S 的 MFU 论文没给** |
| pyramid 的算力贡献 | 关 pyramid（full-length）→ 2.64T 变 **8.08 TFLOPs**（3.06×） | Table 3 第 6 行 | 口径无关的比值，可用来自证结构 |
| 吞吐（samples/s） | **论文一处都没给** | — | 只能由 TFLOPs/MFU 回推，或我们自测 |

### 1.3 Table 4 的读法（先说清，否则数字会被误用）

四行收益是**单点消融**（未优化基线 + 只加某一项），**不是累乘链**。判据：末行 −32.9% 优于第三行 −50.1% 的一半却不单调，若是逐行累加/累乘不可能出现；且 §4.4 正文对四项是并列叙述（pyramid、KV caching、FA、mixed precision 各自「reduces / improves」）。

按单点消融回算绝对值（H100 SXM：bf16 dense 989.4 TF、fp32 67 TF，官方 spec）：

| 形态 | 步时 | 实达算力 | 占 bf16 峰值 | 占 fp32 峰值 |
|---|---:|---:|---:|---:|
| 未优化（基线，实际精度未知） | 407 ms | 6.49 TF/s | 0.66% | 9.68% |
| +pyramid | 290 ms | 9.10 TF/s | 0.92% | 13.6% |
| +KV cache | 284 ms | 9.29 TF/s | 0.94% | 13.9% |
| +FlashAttention | 203 ms | 13.0 TF/s | 1.31% | 19.4% |
| +mixed precision（含重算） | 273 ms | 9.67 TF/s | 0.98% | 14.4% |

**读法结论**：407 ms 这一档只可能是 **fp32/未优化** 形态（9.7% of fp32 是个合理的「没优化」数字；若当成 bf16 则 0.66%，低到不像 H100 上能跑出的训练）。而 Table 5 的 30.8% MFU 属于 **OneTransL + bf16**（反推 8.62T ÷ (989.4T×0.308) = **28.3 ms/步**）。**两者不同口径，不能互相推**。

> **自纠（2026-11-16）**：本文件上一版 §1.2 用「S 也 30.8% MFU」推出 S = 8.7 ms/步，与 Table 4 的 407 ms 差 47×。那是把 L 的 bf16 MFU 套到 S 的 fp32 基线上，**口径混用，已撤回**。修正后的结论：**论文没有 OneTrans-S 的 bf16/MFU 数字，S 侧利用率只能自测**。

### 1.4 论文没给的东西（决定对比方式）

1. **S 的 MFU / 实达算力 / 吞吐**（只给 L 的 30.8%，且是 bf16）。
2. H100 具体型号（SXM/PCIe）、互连、训练精度（bf16 还是 fp32）、MFU 峰值取自哪张 spec。
3. 公开数据集 → **精度不可对表**，只能对结构、batch 口径、性能、利用率。
4. 407 ms 是「每步」还是「每 sample」——Table 4 列名只写 `Training Runtime (ms)`，见 §4.1 待核项。

---

## 2. 复刻定义（把「OneTrans-S」写成可判定的配置）

| 维度 | 目标值 | 判据 |
|---|---|---|
| 层数 / 宽度 / heads | 6 blocks（+1 个全长期 base block，见 §4.2）/ d=256 / 4 heads | 构造日志打印 schedule 与每层 token 数 |
| token 布局 | seq **1190** → 12，ns **12**，[SEP] 1 → 总 1203，每层对齐 32 | `linear_pyramid_schedule` 输出 = `[1203, …, 12]` |
| 参数 | 90~91 M（dense，不含稀疏表） | 本仓实测 **90.44 M**（ffn=1536） |
| 每卡 batch | **2048**（论文 per-GPU） | 日志 `per_gpu_batch=2048` |
| 精度 | bf16 autocast + bool mask（落 `FlashAttentionScoreV4`） | profiler `kernel_details.csv` 里出现 FA 算子 |
| 数据 | 论文数据不可得 → 用 TAAC2026 样本 / 合成 token | 报告里明写「数据不同 ⇒ 精度不可对表」 |
| 卡数 | 1 die 建基线 → 8 die（本地 8 die = 全局 16,384）；H100 侧是 16 卡 = 全局 32,768 | 对比时**只并列绝对值**，不换算 |

---

## 3. 现有资产（已在本仓）

| 资产 | 内容 | 与目标的关系 |
|---|---|---|
| `GPU_Reference/` | 上游 `WestbrookLong/OneTrans_Pytorch` @ `8febd7e`（submodule，零改动） | 代码参照；结构对账见 §4.2 |
| `OneTrans_NPU/` | 950DT 移植：`npu_compat.py` + bool/float mask 开关 + DDP 接线 | 可复用接线层 |
| 精度三段 A/B/C | fp32 1.49e-07 / bf16 2.59e-03 / 10 步 loss 轨迹 5.96e-08，全 PASS | 复刻后可复用同套判据 |
| 多卡扩展 | 1/2/4/8 die = 829.5 / 1,303.6 / 2,348.3 / 3,686.2 sps（4.44×） | ⚠️ 绑定 toy 配置（d=128 / seq 16 / 本地 batch 64） |
| 端到端 | 1,000 行 2 epoch、78.9 ms/步 | ⚠️ 同上，不代表论文规模 |

**关键差距：现在跑通的是「最小可用配置」，不是 OneTrans-S**（论文 S：91 M、1190 token、per-GPU 2048）。

---

## 4. 缺口清单（按阻塞顺序）

| # | 缺口 | 判据 | 状态 |
|---|---|---|---|
| A | **S 结构对账**：上游实现是否 = 论文 S | 参数 ~91 M ±5% **且** 算力与论文口径自洽 | **进行中｜参数面已对到，算力面 24× 不符（§4.2）** |
| A′ | **论文 TFLOPs 口径**（per-sample / per-step / per-batch 2048） | 能给出唯一自洽读法，或明确「不可判定」并改用比值判据 | **未闭合｜阻塞 S1** |
| B | batch 口径：per-GPU 2048 × 8 die = 16,384 | 日志打印 `per_gpu_batch=2048` / `global=16384` | 待做（配置面） |
| C | 数据：论文工业数据不可得 | 明确替代并在报告标注「精度不可对表」 | 待做 |
| D | MFU 探针：950DT 侧自测 | `FlopCounterMode` 出每步 GFLOP；峰值 950DT Cube bf16 **432 TF**（32 AI core × 1650 MHz，真机已校核 429.8 TF=99.5%） | 待做（探针已在 zeus 工程验证可用） |
| E | 利用率三项：AICore / HBM / 通信 | 一次运行同时采 `npu-smi` util + HBM 峰值 + HCCL 占比 | 待做 |
| F | 扩展性 1 / 8 die | 同配置同池同 batch，报绝对吞吐 + 扩展比 | 待做 |

### 4.1 待核：论文 407 ms 与 2.64 TFLOPs 是否同源

2.64 TFLOPs 若按 batch 2048 摊到每 sample = 1.29 GFLOP/sample（**在 91 M 模型上是每 token 1.1 MFLOP，比「一个 91 M 参数的 token」（≈182 MFLOP）低两个数量级**）⇒ 无论怎么读，Table 2 的 TFLOPs 都**不可能**是「1190 token 走满 91 M 参数」这条链上的每样本计算。三种可能，需一条条排除：

1. TFLOPs 是「每步、每卡、batch 2048」的**有效计算**（已按 pyramid + 共享参数折算），而 407 ms 里含大量非计算开销（fp32 物化 attention、host 侧等）；
2. TFLOPs 的 batch 2048 指**全局**而非 per-GPU；
3. 印刷/单位问题（TFLOPs 实际是 GFLOPs 或每 batch 值）。

处理原则：**不替论文选口径**。S1 的结论只允许写「按 X 读法自洽 / 不自洽」，并改用 §4.3 的**比值判据**（口径无关）做结构自证。

### 4.2 实测证据（2026-11-16，CPU，无需上卡）

命令（CPU 即可，无需上卡；本仓 `tools/probes/`，torch 2.10.0+cpu 实测）：

```bash
cd onetrans-d && export ONETRANS_NPU_DIR=$PWD/OneTrans_NPU
python3 tools/probes/param_count_paper.py     # 参数量 + 网格扫
python3 tools/probes/flops_paper_shape.py     # 每 sample 训练 FLOPs（前向+loss+反向）
python3 tools/probes/flops_pyramid_ratio.py   # pyramid 开/关的 FLOPs 比值
```

| 量 | 论文 | 本仓实测（论文形状：d=256/h=4/ns=12/seq=1190/pyramid 6 段） | 判定 |
|---|---|---|---|
| 参数量 | 91 M | **90.44 M**（ffn=1536） | ✅ 差 −0.65%，在 ±5% 内 |
| 每样本训练算力 | 2.64 TFLOPs（口径不明，见 §4.1） | **31.53 GFLOP/sample**（batch 2 实测，线性外推 batch 2048 = 64.6 TFLOP） | ❌ 差 **24.5×** |
| pyramid 开/关比值 | 8.08 / 2.64 = **3.06×** | **1.68×** | ❌ 结构不等价 |

**参数面能对上不等于结构对上**：`Δ=0.6 M` 的「91 M」在网格里至少有三组解（ffn=1536/ns=12 → 90.4 M；ffn=512/ns=32 → 92.2 M；ffn=1024/ns=8/multi=2 → 92.2 M），单靠参数量**不可判定**；而算力比值 1.68× vs 3.06× 说明本仓 pyramid **只剪 query、不剪 K/V/FFN**（节省只来自注意力分数矩阵），论文的 3.06× 需要更彻底的剪枝。

口径提醒：`FlopCounterMode` 只统计 aten 可见算子（SDPA 若走融合后端会计漏），三行自测数字**只在同一口径内互比**，不跨口径换算。

### 4.3 结论

- **S 结构对账未闭合**：参数落在 91 M 附近属「多解命中」，算力与 pyramid 比值两条独立判据都不符 ⇒ 现在这套上游实现**可以当参照骨架，但不能宣称「= 论文 OneTrans-S」**。
- 因此报告口径先定成：**「同形状（d=256/6 层/1190 token/91 M）的 OneTrans 训练，950DT vs 论文 H100」**，把「结构等价」单列为待证项。

---

## 5. 对比协议（写进报告的硬规则）

1. **只并列，不相除**：950DT 与 H100 不同平台，收益% 一律不出；给绝对值 + 「占各自口径的百分比」并注明分母。
2. **MFU 同公式**：`每步训练 FLOPs ÷ (卡数 × 峰值 × 步时)`。峰值来源必须写清：H100 官方 spec（bf16 dense 989.4 TF）/ 950DT 真机 432 TF（Cube 32 核 × 1650 MHz，matmul 校核 429.8 TF）。
3. **论文侧 S 的 bf16 数字缺失** ⇒ 对比表里 H100 一栏只允许两档：`407 ms`（Table 4 未优化基线）与 `203 ms`（Table 4 单点消融 +FlashAttention，按 2.64 TFLOPs 回算），**其余一律标「论文未给」**，不许用 OneTransL 的 30.8% MFU 顶替。
4. **吞吐标口径**：per-GPU batch、全局 batch、卡数、步数窗（稳态剔除前 10 步）、是否含数据搬运。
5. **利用率分列**：MFU（算力）× AICore util × HBM 占比 × 通信暴露时间，不合成单一指数。
6. **精度不与论文对表**（数据不同），只报相对 GPU 参照实现的数值一致性（现有 A/B/C 三段判据）。

---

## 6. 计划

| 阶段 | 内容 | 产出 | 判据 |
|---|---|---|---|
| S0 | 论文事实固化 + 归档 | 本文件 + `papers/` | 每条有出处 ✅ |
| S1 | 结构对账（A/A′/B/C） | 本节 §4 + `docs/ONETRANS_S_SPEC.md` | 参数 ±5% **且** pyramid 比值在 3.06×±20% 内，或给出「论文口径不可判定」的证明 |
| S2 | 把 S 形状拉到 950DT（1 die → 8 die） | `results/s_train_*.txt` | `EXIT=0` ∧ 稳态步数 > 0 ∧ 每 rank 有 step 行 |
| S3 | 性能 + 利用率（D/E） | MFU 表 + util/HBM/HCCL | 三项利用率齐全且同轮同源 |
| S4 | 与 H100 并列 + 报告 | `results/ONETRANS_S_VS_H100_v1_YYYYMMDD.{md,html}` | 对比协议 §5 六条全满足 |

**下一步（S1 收尾，仍不上卡）**：
1. 逐层读 `MultiOneTransBlock`/`OneTransBlock` 的 `pyramid_stack_len` 语义，确认「只剪 query」还是「剪 token」，定位 1.68× vs 3.06× 的差；
2. 用 1 组 `k`（NS-token 数）+ FFN 比重的形状扫，把「参数 91 M」与「ratios 3.06×」两个约束同时满足的配置找出来（现在只满足第一个）；
3. 三条口径读法各算一次 407 ms 的自洽性，写进 §4.1，得出「可判定/不可判定」的结论。
