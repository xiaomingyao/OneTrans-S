# 论文 4 项优化在昇腾 950DT 的适用性评估（2026-11-17）

对照 `docs/TOPIC_ONETRANS_S.md` §1.3 的 Table 4 单点消融：pyramid 290 ms / KV cache 284 ms /
FlashAttention 203 ms / mixed precision+重算 273 ms（均相对 407 ms 基线）。

## 0 一句话结论

**4 项里 3 项已经实施并计入现有成绩（pyramid、FA 的昇腾等价物、bf16+重算），1 项（cross-request KV cache）在本仓无从实施。**
所以我们与 273/203 ms 的差距**不是"少做了某项论文优化"**，而是**同一套手段在两套栈上的执行效率差**
（cube 50.9%、FA 反向 35%、vector 侧已贴峰值）。

| 论文手段 | 昇腾适用性 | 我们的状态 | 实测 |
|---|---|---|---|
| Pyramid Stack | ✅ 完全适用 | **已实施** | `pyramid_schedule=[1203,960,736,480,256,12]`；开/关 = 1.68×（论文表格 3.06× 口径待核，正文写法 1.66× 与我们自洽） |
| Cross-Request KV Caching | ❌ 不适用 | 无对象 | 我们无"请求/用户/历史"维度，无 KV 可复用 |
| FlashAttention-2 | ✅ 等价物适用 | **已实施**（FA v4） | fwd 44.8% / bwd 35.0%（对 432 TF）；是我们最大单项收益来源 |
| Mixed Precision + Recomputation | ✅ 适用 | **已实施**（bf16 + 全 block 重算） | O1 显式 bf16：档 B −15.79%、档 A −16.01%；重算使每卡 2048 可行（39.17 GB） |

## 1 Pyramid Stack —— 适用，已实施，是当前成绩的前提

- 论文化：逐层把 query 长度从 1190 压到 12（k/v 保持层输入长度），6 层 pyramid。
- 昇腾侧依赖：**零**。它只改形状与 schedule，不依赖任何专用算子 ⇒ 在 Ascend 上完全等价可用。
- 我们现状：`[1203, 960, 736, 480, 256, 12]` 已生效，形状复核 `params=91,026,946` 通过；
  所有已有数字（档 A 752.0 / 档 B 614.7）**都建立在这一项之上**。
- 口径警示：论文 Table 3 说关掉 pyramid 变 3.06×（8.08 T vs 2.64 T），我们实测开/关只有 **1.68×**；
  按论文正文写法（只剪 query、k/v 仍为层输入全序列）算出来是 **1.66×**，与实测自洽 ⇒ 判为**表格口径待核**，不为它改结构。
- 剩余增量：更激进的 schedule（等于改结构，**用户已冻结**）。

## 2 Cross-Request KV Caching —— 不适用（不是"难"，是没有对象）

两个独立原因：

1. **项目范围**：本任务只做训练，不做推理；而 cross-request KV caching 本质是**推理侧**跨请求复用历史 KV。
2. **数据/接口没有这一维**：它进 Table 4 的"训练 runtime"，说明论文里它改变的是**一个 batch 内的特征复用/组织**
   （同一 user 的多条请求共享 KV）。我们的复刻输入是**稠密特征 + 12 个 NS token 的合成张量**，
   既没有 request id、也没有 user 历史序列，**没有可复用的 KV 结构**。
   要引入就等于换一个模型（变长历史序列 + KV 存储管理 + 特征服务接口）。

⇒ **不适用**，也无法通过"改 kernel / 调 schedule"模拟。

## 3 FlashAttention-2 —— 等价物适用且已实施；FA2 本体不可复刻，量化版 FA 在本栈缺失

- FA2 是 CUDA/Hopper 实现，不能直接搬。昇腾的等价物是 **`aclnnFlashAttentionScore`（V4）+ `FlashAttentionScoreGradV4`**，
  我们已经在用（每步 14 次前向调用）。
- 实测效率：**前向 44.8%**（193.8 TF/s）、**反向 35.0%**（151.3 TF/s），是我们最大的一项单项收益（G1 SDPA-causal / G3 FA BSH）。
- 已确认不可再动的旋钮：`inner_precise`、mask dtype（bool/uint8）、`sparse_mode` 替代 mask（161001）、
  Triton 自写 attention（慢 2.32–3.12×，理论上限 0.6%）。
- **新证据（2026-11-17，`probe_quant_fa.py`）**：`torch_npu.npu_quant_fusion_attention` 与 `…_backward` 的 schema 存在，
  但底层 **`aclnnQuantFlashAttentionScore` 不在 `libopapi.so`** —— 连 **bf16 输入都报算子缺失**，
  int8 / fp8 更是无从谈起 ⇒ **注意力没有量化入口**，"FA 走 int8 换 2× 速度"这条路封死。
- 判定：**适用且已实施**；残余只有反向 35%→45%（约 −29 ms），只能等厂商或 AscendC 硬写。

## 4 Mixed Precision + Recomputation —— 适用，已实施到本栈的 bf16 上限

| 子项 | 我们做了什么 | 实测 | 备注 |
|---|---|---|---|
| 混合精度 | **显式 bf16 干路（O1）**：关 autocast，残差流/激活/权重全走 bf16，权重用每步一次的 bf16 影子，优化器仍 fp32 | 档 B 730.0 → **614.7 ms（−15.79%）**；档 A 895.3 → **752.0 ms（−16.01%）**；HBM 69.60→58.48 / 47.85→39.17 GB | **属精度档变更**：`mean rel\|Δg\| = 0.61%`、worst 3.25%，已单独记账（README §3.12） |
| 重算 | `ONETRANS_CKPT_SCOPE` 全 block 激活重算 | 使**每卡 2048** 可行：档 A 752.0 ms / 39.17 GB | 免重算的 2048 会 OOM（E5：还需 1.43 GiB） |
| 再降精度 | fp8：**不可用** | `torch._scaled_mm` 2166 ms（0.3 TF/s 回退）、`torch.mm`/`npu_quant_matmul` 均 161002 | 栈限制，不是我们没做 |
| | int8：**可用但代价大** | GEMM **802 TF/s = 1.99×** bf16；量化误差 per-token 3.10% / per-32block 2.46%（bf16 是 0.61%） | 只能换 GEMM；FA 无 int8 入口 |

⇒ **适用，且已经把 bf16 这条路吃到上限**；再往下是"换训练算法"，不是"开个开关"。

## 5 由此得到的差距归因（重要）

- 论文 407 → 203/273 走的是「pyramid → FA → mixed precision+重算」这套；我们 1,942 → 614.7 走的是**同一套的昇腾版本**
  （ns-layout、融合 rms_norm、去 cat、SDPA-causal、FA BSH、全量投影、显式 bf16），累计 **−68.35%**。
- 所以**剩余 2.25–2.75×（对 273 ms）不在"手段清单"上，在"执行效率"上**：
  cube 50.9%（FA 反向 35%、GEMM 64.3%）、vector 侧 240 ms 已贴 HBM 峰值、cube∥vector 重叠仅 15%。
- 若判据取 273×1.05 = **287 ms**：按当前 kernel 结构，vector 240 + FA 跑满 bf16 峰值 64 = **304 ms 就已超过它**；
  加上 FA 无量化入口（本节 §3 的新证据）⇒ **287 ms 在现行算法 + 可用算子下不可达**。
