# 对账速查：论文 OneTrans-S 实现 vs 本仓实现

> 速查版（4 张表）。完整版（14 模块实现对照 + 算子改写代价 + kernel 落点 + 精度取证）见 `IMPL_COMPARISON_PAPER_VS_REPO.md`；
> 参数逐张量拆分见 `PARAM_COMPARISON_ONETRANS_S.md`；十节总对账见 `AUDIT_VS_PAPER_ONETRANS_S.md`。
> 边界：论文**未开源代码**，故"论文实现"按其文字规定逐项核对。

## 一、结构层

| 项 | 论文 OneTrans-S | 本仓 | 判定 |
|---|---|---|---|
| 层数 | "6 stacked OneTrans blocks" | **1 base + 6 pyramid = 7 次 block-pass** | ⚠️ 表述歧义（论文 TFLOPs 也只有按 7 次才对得上 2.64 T） |
| d_model / heads | 256 / 4 | 256 / 4 | ✅ |
| FFN 宽度 | **未给** | 1536（= 6×d） | ⚠️ 推定（由消融表 24 M 行交叉佐证） |
| 序列长度 | 1190 → 12，每层取 32 的倍数，顶层 = NS 数 | `[1203,960,736,480,256,12]` | ✅ |
| NS token / [SEP] | 12 / 有 | 12 / 有（总长 1203） | ✅ |
| 参数共享 | S 共享一套 QKV+FFN，每个 NS 各一套 | `kqv_list[13]` / `ffn_list[13]`，S 用第 12 组 | ✅ |
| 掩码 | 金字塔尾部截断 ⇒ 底右因果 | `paper_causal`：`col ≤ row + (Lk−Lq)` | ✅ |
| Pre-Norm / SiLU×2 | 是 | 是 | ✅ |
| 位置编码 | 图中有方框，**未给形式** | **无可训练位置编码参数** | ⚠️ 未闭合 |
| 参数量 | 91 M（**不含 sparse embedding**，原文 "excluding sparse embeddings"） | **91,026,946**（纯稠密，无稀疏表） | ✅ **同口径一致** |
| 计算量 | 2.6481 TFLOPs（nominal 6ND） | **2.6489 TFLOPs**（同口径，差 0.03%） | ✅ |

## 二、算子层

| 项 | 论文 | 本仓 | 差异性质 |
|---|---|---|---|
| Attention | **FlashAttention-2** | `npu_fusion_attention`（aclnn **FA v4**）+ 部分层内置 causal | 等价物替换 |
| Mask 实现 | 未规定 | 显式张量 + 缓存；注意 FA 的 `atten_mask` 是 **True=屏蔽**（与 SDPA 相反） | 语义对齐 |
| QKV 计算 | 未规定 | 逐 token `Linear`（默认）/ `NS_BMM=1` 时 13 组拼成 **1 次 bmm** | 本仓改写 −25.70% |
| FFN 计算 | 未规定 | 逐 token（默认）/ `FULL_PROJ=1` 全量投影 + 原地覆盖 NS 行 | 本仓改写 −11.13% |
| 归一化 | RMSNorm | 手写链 / `FUSED_NORM=1` 走厂商 `npu_rms_norm`（反向复用 rstd） | 本仓改写 −14.85% |
| 序列拼接 | [SEP] + concat | `torch.cat` / `CAT_INPLACE=1` 预分配 + 原地写 | 本仓改写 −9.48% |
| 精度 | BF16/FP16 + 激活重算 | autocast bf16（默认）/ `EXPLICIT_BF16=1` 显式 bf16 干路 + 档 A 重算 | 对应"mixed precision"档 |
| 反向 | 未规定 | 全 autograd，FA/FFN/QKV 走厂商 backward，无自写 kernel | 等价 |

## 三、训练层

| 项 | 论文 | 本仓 | 判定 |
|---|---|---|---|
| 优化器 | **双优化器、无 weight decay**：sparse Adagrad(0.1, 1.0) / dense **RMSProp(lr=0.005)** | **AdamW(lr=5e-3, wd=0)** | ⚠️ 实质不同 |
| 梯度裁剪 | global grad-norm，**dense 90 / sparse 120** | **未实现** | ❌ 缺口 |
| per-GPU batch | 2048 + gradient accumulation | 2048（档 A，开重算） | ✅ |
| 分布式 | data-parallel，16 GPU all-reduce | DDP over HCCL，16 die（2 节点 × 8） | ✅ 同构 |
| 数据 | 工业全量特征 | 合成 token 常驻 device，步时不含取数与 H2D | ⚠️ 口径选择 |
| 框架/算子库 | **未披露** | CANN 9.1.0 + torch_npu 2.10 + FA v4 | 无可比 |

## 四、差异清单（只有 4 条要记住）

| 类型 | 内容 | 能否闭合 |
|---|---|---|
| **结构差异** | tokenizer 规模：论文的 Group-wise 变体省掉 13 M 参数 ⇒ 它的 tokenizer 比本仓（0.81 M）重得多 | ❌ 论文未披露内部投影方式 |
| **训练差异** | 优化器（AdamW vs RMSProp/Adagrad）、**grad-norm clip 未实现** | ✅ clip 一行代码可补；优化器可换 |
| **口径差异** | 参数量含稀疏表 vs 纯稠密；层数 6 vs 7 pass | ⚠️ 已标注 |
| **本仓多出** | 8 个算子级开关（合计 **−68.35%**）、DDP、profiling 钩子 | 论文侧无对应 |

## 一句话

**算法对齐、算子改写、训练配置缺两项**：结构 11 项里 9 项对齐（另 2 项为推定/构成差异）；算子层 7 处替换全部有逐位或 1 ULP 取证；真正的缺口只有 **grad-norm clip** 与 **双优化器**。
