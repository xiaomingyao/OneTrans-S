# 详细对账：本仓 vs 论文 OneTrans-S

> 论文：`arXiv:2510.26104v3`（ByteDance + NTU，2026-02）
> 本仓：`github.com/xiaomingyao/OneTrans-S`（昇腾 950DT，16 die）
> 上游代码基准：`WestbrookLong/OneTrans_Pytorch` @ `8febd7e`（★125，其自身溯源到 TF demo `KiraYeetar/OneTrans_Tensorlfow`）
> 对账日期：2026-11-17

## 0 结论（六条）

| # | 结论 | 强度 |
|---|---|---|
| 1 | **结构全对齐**：Mix 参数化、金字塔收缩、[SEP]、底右因果、Pre-Norm、SiLU×2、1 base + N pyramid — 逐项可在代码里指出落点 | ✅ 代码级 |
| 2 | **参数量对齐**：本仓 **91,026,946** vs 论文 91 M（差 0.03%） | ✅ 实测 |
| 3 | **计算量同口径对齐**：nominal 6ND **2.6489 T** vs 论文 **2.6481 T**（差 0.03%） | ✅ 反解复现 |
| 4 | **FFN 宽度 1536 有交叉证据**：论文 Table 3「Shared parameters = 24 M」按本仓拆分模型反解得 **24,816,130**（FFN=1536 唯一命中；256→20.2 M、1024→23.0 M、2048→26.7 M 均偏） | ✅ 交叉验证 |
| 5 | **训练配方有 3 处实质差异**：优化器（AdamW vs 双优化器 RMSProp/Adagrad）、**梯度裁剪未实现**、无稀疏 embedding | ⚠️ 差异 |
| 6 | **不能声称"官方实现对齐"**：论文未开源任何代码，上游是第三方二手重实现，且 Layer 数表述（6 vs 7 pass）与 tokenizer 规模存在无法闭合的差异 | ⚠️ 可信边界 |

## 1 身份与来源

```
论文 OneTrans（ByteDance + NTU，无开源代码）
   └─ KiraYeetar/OneTrans_Tensorlfow   TF 单文件 demo（第三方，★19）
        └─ WestbrookLong/OneTrans_Pytorch   PyTorch 重写 + 训练外壳（第三方，★125）  ← 本仓代码基准
             └─ OneTrans_NPU（本仓）       配到 S 形状 + 昇腾移植 + 13 个优化开关
```

| 层 | 是否有官方代码 | 说明 |
|---|---|---|
| 论文 | ❌ | 全文 `grep -i "github\|code is available"` 零命中 |
| 上游 GPU 参考 | ❌（第三方） | README 首句即 "a PyTorch **reading-and-reimplementation** project based on the original TensorFlow demo" |
| 本仓 | — | 以上游为基准，非论文官方实现 |

## 2 结构对齐（代码级）

| 论文要素（出处） | 论文描述 | 本仓落点 | 判定 |
|---|---|---|---|
| 统一序列（§4.1.4） | S 序列 1190 + NS 12 + [SEP] 1 | `total = 1203`，`use_sep_token=True` | ✅ |
| 金字塔收缩（§4.1.4） | 每层线性收缩，取 32 的倍数，顶层 = NS 数 | `linear_pyramid_schedule(1203, 12, 6, 32)` = `[1203,960,736,480,256,12]` | ✅ |
| 只剪 query（§3.4） | 每层只让尾部 L 个位置做 query，K/V 为该层全长 | `q = x[-Lq:]`，`k/v = x` 全长 | ✅ |
| 底右对齐因果（§3.4） | 金字塔尾部截断 ⇒ 因果掩码右对齐 | `mask_type="paper_causal"`：`col ≤ row + (Lk−Lq)` | ✅ |
| Mix 参数化（§3.2） | S 共享一套 QKV/FFN；每个 NS token 各一套 | `kqv_list[13]` / `ffn_list[13]`，S 用 index `ns_len` | ✅ |
| Pre-Norm（§4.1.4） | Pre-Norm | `rms_0 → attn → +res → rms_1 → ffn → +res` | ✅ |
| FFN 激活 | 论文图示 SiLU | `act(proj_1(x))` + `act(proj_2(h))`，两次 SiLU | ✅ |
| 层堆叠 | "6 stacked OneTrans blocks" | 1 base block + 6 pyramid block = **7 次 block-pass** | ⚠️ 见 §9.1 |
| 位置信息（Fig.2b） | 图中标 Positional Information | **代码无可训练位置编码参数**（`sep_token` 仅 256 维向量） | ⚠️ 未闭合 |

## 3 超参与训练配置

| 项 | 论文 | 上游默认值 | 本仓实际值 | 判定 |
|---|---|---|---|---|
| d_model | 256（§4.1.4） | 128 | **256** | ✅ |
| heads | 4（§4.1.4） | 4 | **4** | ✅ |
| FFN 宽度 | **未给** | 256 | **1536**（=6×d） | ⚠️ 推定 + §4 交叉证据 |
| 序列长度 | 1190 → 12 | 16 → 4 | **1190 → 12** | ✅ |
| NS token | 12 | 4 | **12** | ✅ |
| pyramid layers | 6 | 6 | **6** | ✅ |
| align | 32 | 32 | **32** | ✅ |
| multi_num | 1（每 stage 1 block） | 4 | **1** | ✅ |
| per-GPU batch | 2048 | 32 | **2048**（档 A）/ 1024×2（档 B） | ✅ |
| 全局 batch | 32,768（16 卡） | — | **32,768** | ✅ |
| 精度 | 基线 fp32；优化档 BF16/FP16 | fp16 | **bf16** | ✅ 对应优化档 |
| 优化器 | 双优化器、无 weight decay：sparse **Adagrad(β₁=0.1, β₂=1.0)**、dense **RMSProp(lr=0.005, α=0.99999, momentum=0)** | `AdamW(lr=1e-3, wd=1e-4)` | **AdamW(lr=5e-3, wd=0)** | ⚠️ 实质差异 |
| 梯度裁剪 | global grad-norm，**dense 90 / sparse 120** | 无 | **未实现** | ❌ 缺失 |
| 稳定性 | Pre-Norm + grad-norm clip | Pre-Norm | Pre-Norm | ✅ / ⚠️ |
| 数据管线 | 工业全量特征 | TAAC2026 1000 行样本 | 合成 token 常驻 device（纯计算口径） | ⚠️ 口径不同 |
| 结构性 dropout | 论文未提 | 无 | 无 | ✅ |

## 4 参数量逐模块拆分（实测，`tools/probes/param_breakdown.py`）

| 模块 | 参数量 | 占比 |
|---|---:|---:|
| stack_blocks（6 个 pyramid block） | 77,274,624 | 84.89% |
| base_block（第 1 个 block） | 12,879,104 | 14.15% |
| non_seq_tokenizer（256 → 12×256 = 3072） | 789,504 | 0.87% |
| head（LayerNorm + 256→256→2） | 66,818 | 0.07% |
| seq_tokenizer（64 → 256） | 16,640 | 0.02% |
| sep_token（[SEP] 向量） | 256 | 0.00% |
| **合计** | **91,026,946** | 100% |

按算子族：**FFN 71,728,384（78.8%）**｜QKV 17,961,216（19.7%）｜其余 1,337,346（1.5%）。
每 block 完全一致：QKV 13 组 2,565,888 + FFN 13 组 10,246,912 = 12,812,800。

### 4.1 用论文消融表交叉验证（新增证据）

| 论文 Table 3 行 | 论文 Params | 本仓拆分模型下的对应量 | 判定 |
|---|---:|---:|---|
| 参考（完整 S） | 91 M | 91,026,946 | ✅ |
| **Shared parameters** | **24 M** | 只共享 FFN（QKV 仍 13 组）⇒ **24,816,130** | ✅ **唯一命中**：FFN=256→20.2 M、512→21.1、1024→23.0、1536→24.82、2048→26.7 |
| Group-wise Tokenizer | 78 M | 无法复现（差 13 M 必须在 tokenizer 内，本仓 tokenizer 仅 0.81 M） | ⚠️ 说明论文 tokenizer 比本仓重得多 |
| w/o pyramid stack | 92 M | 参数应与 S 相同（91 M） | ⚠️ 论文自身小数点级不一致（+1 M） |

**结论**：`Shared parameters = 24 M` 是本报告找到的最强独立证据，把 FFN=1536 从"凑 91 M 的推定"升级为"推定 + 交叉佐证"；同时暴露 **tokenizer 实现与论文存在实质差异**（论文 Group-wise 变体省 13 M，本仓 tokenizer 总共只有 0.81 M）。

## 5 计算量对齐

| 口径 | 公式 | 论文 | 本仓 | 差 |
|---|---|---|---:|---:|
| **nominal（6ND）** | `6 × N × Σ(block-pass token)` | 2.6481 TFLOP/样本 | **2.6489 T** | +0.03% |
| Σ token | 各 block query 长度累加 | 4850（= 1203+1203+960+736+480+256+12） | 4850 | ✅ |
| **actual（按分组形状）** | 逐层 QKV/FFN/attention 实际矩阵乘 | 论文未给 | **40.91 GFLOP/样本** | — |
| 每 token-pass | — | 546.2 MFLOP（nominal） | 8.44 MFLOP（actual） | 64.7× |

**两点必须说明**：
1. nominal 把 91 M 参数全乘到每个位置，而实际序列 token 只用 13 组中的 1 组 ⇒ **nominal = actual × 64.7**，nominal 不可当 MFU 分子。
2. Σ 只有按 **7 次 block-pass** 才对上论文 2.64 T ⇒ 论文的算力口径是 7 次，与"6 stacked"的文字不一致（见 §9.1）。

## 6 性能对齐

| 指标 | 论文 | 本仓 | 性能比（本仓/论文） |
|---|---:|---:|---:|
| 开箱（无优化）步时 | 407 ms | **1,942.1 ms** | 21.0% |
| 开箱显存峰值 | 53.13 GB | 52.30 GB | 101.6% |
| 优化后步时（同精度档） | 273 ms | **614.7 ms**（档 B）/ 752.0 ms（档 A） | **44.4%** / 36.3% |
| 优化后显存峰值 | 27.10 GB | 58.48 GB（档 B）/ 39.17 GB（档 A） | 215.8% / 144.5% |

> 两侧不是同栈：CANN 9.1.0 + torch_npu 2.10 + aclnn FA v4 vs H100 + FA2 + 未披露框架 ⇒ **只能并列步时，不能相除算收益%**。

## 7 精度验收（本仓自证，论文侧无对应数据）

| 层级 | 判据 | 结果 |
|---|---|---|
| L1 布局类（ns-layout / 去 cat / 全量投影 / FA BSH / SDPA causal） | 与上游逐位或 1 ULP | logits 逐位 0.0；dq 1 ULP（9.766e-4） |
| L2 精度档变更（融合 rms_norm / 显式 bf16 干路） | 对 fp64 参考的相对误差 | rms_norm 3.891e-3（优于手写 7.704e-3）；bf16 干路 `mean rel(Δg)=0.61%` |
| 全模型三关 | 张量 1.49e-7 / loss Δ 5.96e-8 / 梯度 4.88e-4 | 全 PASS |

## 8 与上游的逐位等价（代码基准自证）

| 校验项 | 结果 |
|---|---|
| 参数量 | 91,026,946（两边相同） |
| state_dict | 949 个张量，key 集合与形状 **全等** |
| 前向输出 md5 | **`c4392b1c57956629…` 完全相同** |
| pyramid schedule | `[1203, 960, 736, 480, 256, 12]` 相同 |

> 注意：这一节证明的是"本仓 = 上游"，**不是**"上游 = 论文"。后者只能由 §2–§4 的逐项对账支撑。

## 9 不可对齐 / 未闭合

### 9.1 层数表述（6 vs 7）

论文正文写 "6 stacked OneTrans blocks"，但：
- 代码结构是 **1 base block + 6 pyramid block**（上游亦如此）；
- 论文自己的 TFLOPs **只有按 7 次 block-pass 才对得上 2.64 T**（按 6 次得 1.99 T，差 −24.6%）。

⇒ 推断论文的"6"指 pyramid 层数，base block 未计入文字表述。**本仓沿用 7 次**，并在报告里注明这是口径推定。

### 9.2 tokenizer 规模

论文 Table 3 的 Group-wise Tokenizer 变体 Params **78 M**（比 S 少 13 M），说明 tokenizer 变体对总参数影响达 13 M 量级；本仓 tokenizer 仅 **0.81 M**。⇒ 论文的 tokenizer 内部结构（尤其 timestamp-aware fusion 的投影方式）与本仓不同，无法闭合。

### 9.3 位置编码

论文 Fig.2(b) 标了 Positional Information，但未给形式；本仓**没有可训练位置编码参数**。差异存在但影响未知。

### 9.4 训练配方三项差异

| 项 | 影响 | 是否影响步时口径 |
|---|---|---|
| 无双优化器（AdamW 代替 RMSProp/Adagrad） | 收敛轨迹不同 | 否（优化器开销已计入，但算法不同） |
| 无 global grad-norm clip | 长跑稳定性 | 否 |
| 无稀疏 embedding | 参数量构成不同 | 否（但少了表的 gather/optimizer 开销） |

### 9.5 数据口径

论文用工业全量特征与真实样本；本仓用**合成 token 常驻 device**，故步时只含计算+通信，不含取数与 H2D。这是刻意选择（可比性优先），但**使步时与论文不完全同义**。

### 9.6 论文自身的不自洽（不属于本仓问题，但影响"对齐"的可判定性）

- TFLOPs 列：2.64 T/2048 样本 = 1.29 GFLOP/样本，而一次前向就需 216.6 GFLOP/样本 ⇒ 该列只能是某种归一化，不能作吞吐基准；
- `w/o pyramid stack` Params 92 M vs 参考 91 M（参数与序列长度无关）；
- 203 ms 与 273 ms 两行百分比都相对 407 ⇒ 单点消融，不是累乘链。

## 10 可信边界（对外表述纪律）

**可以说**：
1. 结构与论文逐项对齐（附代码落点）；
2. 参数量 91,026,946 与论文 91 M 一致，且 FFN 宽度由论文消融表交叉佐证；
3. nominal 计算量与论文同口径一致（2.649 vs 2.648 T）；
4. 与本仓代码基准（上游 PyTorch 重实现）逐位等价（949 张量 / md5）；
5. 同 batch 口径下的步时对比：开箱 4.77×，优化后 2.14×（性能比 44.4%）。

**不能说**：
1. "复刻了论文官方实现"（论文未开源）；
2. "与论文实现同栈对比"（框架/算子库/硬件全不同）；
3. "MFU 与论文持平/落后 X%"（论文未给 S 的 MFU，且两侧 FLOP 记账口径不同）；
4. "参数量完全相同"（数值同、构成不同：论文含稀疏表）。

**补齐清单（按性价比排序）**：
1. 补 global grad-norm clipping（dense 90）—— 一行代码级改动，论文明确要求；
2. 标定 tokenizer 结构（若能拿到论文实现细节，否则在报告里固定为"已知差异"）；
3. 位置编码形式确认；
4. 可选：用 RMSProp 复跑一次 loss 轨迹，验证收敛无系统性偏差。
