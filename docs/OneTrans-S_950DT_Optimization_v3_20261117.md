# OneTrans-S 在昇腾 950DT 的训练性能复核报告 v3

- 日期：2026-11-17（v3：本轮新增 3 项有效优化 + 1 次自我纠错，档 B 从 892.5 → **730.0 ms**）
- 目标：按论文 arXiv:2510.26104 的 OneTrans-S 形状（91 M 参数、d=256、H=4、seq 1190→12、6 层 pyramid、per-GPU batch 2048）在
  **16 卡 950DT（2×8 die，122 + 111）** 上跑起来，实测步时/吞吐/MFU/显存，与论文 16×H100 的 407 ms **并列**对比（只做训练，不含推理）
- 代码与原始数据：`github.com/xiaomingyao/OneTrans-S`（`main`，含 `results/prof*_kernel_details.csv` 全部 profile 原始件）

## 1 结论摘要

| 档 | 对齐轴 | 步时 | 全局 sps | 每卡 sps | MFU | HBM | vs 论文 407 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** 每卡 2048 + 激活重算 | 全局 batch **和** per-GPU batch 都对齐 | **895.3 ms** | 36,601.7 | 2,287.6 | **16.70%** | 47.85 GB | 2.20× |
| **B** 每卡 1024×累加2、无重算 | 只对齐全局 batch 32,768 / 16 卡 | **730.0 ms** | 44,889.1 | 2,805.6 | **20.48%** | 69.60 GB | 1.79× |

相对本轮起点（1,942.1 ms / 16,872.3 sps / MFU 7.70% / 52.30 GB）：
档 A **−53.90%** 步时、**+116.94%** 吞吐；档 B **−62.41%** 步时、**+166.05%** 吞吐。

六步有效优化（每步都有 kernel 级证据）：ns-layout → 融合 rms_norm → 去 cat（档 A/B 分档）→ SDPA 内置 causal → FA `BSH` 布局 → 全量投影。
另有 11 项实测无效 / 不可用，全部留档（§4）。

## 2 口径纪律（先读这段，否则数字会被误读）

1. **不与论文相除算「收益%」**：论文 407 ms 是 fp32 未优化基线，本仓是 bf16 + FA 融合 + 融合 norm 档；只能并列。
2. **档 B 的 per-GPU micro-batch 是 1024×2**，论文是 2048 ⇒ 档 B 只在「全局 batch / 卡数」这一轴上对齐，**不可与档 A 混读**。
3. 档 A 与档 B 的数学目标相同（micro-batch 均值 ÷ 累加数 = 整批均值），但 bf16 重排 + AdamW 非线性会让 loss 轨迹分叉
   ⇒ **不可逐值比对、不可当精度回归**。
4. **布局类改动的精度记账**：`cat_inplace` / 全量投影 / FA `BSH` 都不改变写入的**值**，但会改变张量的地址、对齐与 kernel 分块，
   从而改变 fp32 累加顺序 ⇒ loss 相对差 **≤2e-4**、`max|Δg|` 落在 **1 个 bf16 ULP** 量级 ⇒ 记为 **bf16 噪声级等价**：
   可用于性能档，**不可当精度基线**。
5. 训练数据是合成 token 常驻 device，步时**不含取数/H2D**（纯计算口径）。
6. **跨节点实验必须先 md5 校验两节点代码一致**（v3 期间踩过：只推一个节点 ⇒ rank0 跑旧码，收益被误判为 0，见 §3.7）。

## 3 优化逐项（六步有效）

### 3.1 ns-layout：NS-token 专有 QKV/FFN 由「逐 token Linear」合成一次 bmm

- 动机：12 个 NS token 各有自己的 QKV/FFN 权重，原实现每 token 发一个 1-token matmul（每 block 12×3 次）。
- 开关 `ONETRANS_NS_BMM=1`；CPU UT（`tools/probes/ut_ns_bmm_equiv.py`）：参数量 471,938 一致、前向 `max|Δout| 5.96e-08`、
  反向 `worst rel|Δg| 1.34e-06`（判据 ≤1e-5）⇒ 等价通过。
- 实测（16 die，全局 32,768）：**1,942.0 → 1,442.9 ms（−25.70%）**，吞吐 16,873.0 → 22,710.2 sps（+34.60%），MFU 7.70% → 10.36%。
- 独立复跑三次一致（−25.7% / −26.1% / −25.7%）。

### 3.2 融合 rms_norm：手写 norm 链换成 `torch_npu.npu_rms_norm`

- 动机：profile 显示手写链（`float→pow→mean→rsqrt→mul→改dtype→乘 fp32 weight`）实际占 **≈280 ms/步（19%）**，
  其中 `Mul` 单项 151.6 ms（`bf16 × fp32 weight` 会把张量提成 fp32 再走一遍向量运算）。
- 开关 `ONETRANS_FUSED_NORM=1`；精度反而更好：对 fp64 参考 `max_rel 3.891e-03 / rms 1.655e-03`，手写链是 `7.704e-03 / 2.347e-03`。
- 实测（16 die）：**1,443.9 → 1,228.6 ms（−14.90%）**，吞吐 +17.52%，MFU 10.35% → 12.17%，HBM 52.27 → 47.57 GB。
- 逐算子：`Mul 151.6→0`、`Square 27.9→0`、`Divs 22.2→0`、`Pow 14.4→0`、`Mean 13.1→0`、`ReduceSum 32.2→19.1`、`Add 130.8→109.0`；
  新增 `RmsNorm 29.7 + RmsNormGrad 35.1` ⇒ 净 ≈ −213 ms。

### 3.3 去 cat：`_project_one` 的 seq/ns 合并改「预分配 + 原地写」

- 开关 `ONETRANS_CAT_INPLACE=1`；只改内存写路径，写入值相同（地址/对齐变化 ⇒ §2.4 的噪声级等价）。
- 逐算子（8 die 同档 profile）：`Cat_ConcatD 51.4→7.8`、`Cat_CastAiCore 42.2→6.4`、`InplaceCopy_Cast 143.1→78.5`、
  `InplaceCopy_Transpose 76.6→26.6`；整步 1,235.1 → 1,121.7 ms。
- 实测（16 die）：档 A **1,228.6 → 1,112.1 ms（−9.48%）**；档 B **959.2 → 892.5 ms（−6.95%）**。

### 3.4 SDPA 内置 causal（`ONETRANS_SDPA_CAUSAL=1`）

- 事实：`paper_causal` 是**底右对齐**因果；SDPA 的 `is_causal=True` 在 `q≠k` 时是**左上对齐**（不等价，`max|Δ| 4.26`），
  但在 **`q_len == key_len`** 的层里两者是同一张掩码 ⇒ 可省掉整张掩码张量、让 FA 走内置 causal 分支。
- 适用：base block + 第一个 pyramid block（都是 1203×1203，占 attention 前向算力约一半）。
- 精度：模型 logits **逐位相同（max|Δ| = 0.0）**。
- 实测（16 die，n=8）：档 A **1,112.1 → 1,053.5 ms（−5.27%）**、档 B **892.5 → 847.3 ms（−5.06%）**；MFU 14.19% / 17.64%。

### 3.5 FA `BSH` 布局（`ONETRANS_FA_BSH=1`）

- 做法：`npu_fusion_attention(..., input_layout="BSH")` 的输入输出都是 `(B, L, d)` 连续张量，
  省掉 `output.transpose(1,2).contiguous()` 这次整张量往返以及 FA 内部的部分转置。
- 实测：单算子 fwd（B512·H4·480×1203）**2.215 → 1.513 ms（−31.7%）**；16 die 档 B **847.3 → 823.5 ms（−2.81%）**，
  吞吐 39,792.6 sps，MFU 18.15%，HBM 68.99 GB。
- 精度：单算子前向 **逐位相同**、`dk/dv` 反向逐位、`dq` 1 ULP；模型级 2 步 `max|Δg| = 2.44e-4`（0.5 ULP）。
- 两个必须知道的坑：
  1. **`atten_mask` 语义是「True = 屏蔽」**，与 SDPA 的「True = 保留」相反 —— 传反了**不报错**但结果全错（`max|Δ| 4.53`）；
  2. `Lq == ns_len` 的最后一层（12 行）q 是 **fp32 且非连续**（bmm + fp32 bias 的 type-promotion），FA 直接报 `161002`
     ⇒ 该分支限定 `ns_len < Lq < Lk and q.is_contiguous() and q.dtype==k.dtype`，那层留在 SDPA 上。

### 3.6 全量投影 + 原地覆盖 NS 行（`ONETRANS_FULL_PROJ=1`）

- 做法：每个投影原为「seq 段 `F.linear` + NS 段 `bmm` + 新建缓冲合并」。改为**全量**投影（NS 的 12 行也走共享权重），
  再把 NS 行整块覆盖成独立权重的结果 ⇒ 省掉一次整块 alloc + 一次全量写，代价是多算 `12/L ≈ 1%` 的投影算力。
- 实测：单 die batch 1024 **421.2 → 374.9 ms（−11.0%）**；16 die 档 B **847.3 → 753.0 ms（−11.13%）**，
  吞吐 43,518.3 sps，MFU 19.85%。（单卡与 16 卡一致，是本次唯一"单卡可预测"的项。）
- 精度（单 die 2 步，81 参数全量梯度）：loss rel ≤ 9e-5；`max|Δg| = 4.88e-4` = 最大梯度元素（0.154）的 **1 个 bf16 ULP**，
  各参数 `max|Δg|` 中位数 `7.6e-6`。

### 3.7 两项叠加（当前最优）与一次自我纠错

`ONETRANS_FULL_PROJ=1 ONETRANS_FA_BSH=1` 同时开启：

| 档（16 die · n=8） | 步时 | 全局 sps | 每卡 sps | MFU | HBM |
|---|---:|---:|---:|---:|---:|
| 档 B（1024×2 无重算） | **730.0 ms**（−13.84% vs 847.3） | **44,889.1** | 2,805.6 | **20.48%** | 69.60 GB |
| 档 A（2048+重算） | **895.3 ms**（−15.02% vs 1,053.5） | **36,601.7** | 2,287.6 | **16.70%** | 47.85 GB |

合并配置精度（单 die 2 步，81 参数全量梯度）：`max|Δg| = 4.88e-4`（1 ULP）、`|g|≥1e-2` 上 mean rel **6.2e-4**、
worst rel 1.15e-2、无 NaN ⇒ 与单项同级的噪声级等价。单 die 合并版 362.0 ms（−14.0%）。

**自我纠错（必须记录）**：这两项**第一次测 16 卡时都得出「0.00%，无效」**，并据此写下"16 卡墙钟不由这些 kernel 决定"的结论。
真因是 **rank0（122）上的 `main_pytorch.py` 没有同步**（只推给了 111），rank0 跑的是旧代码、开关在旧文件里根本不存在
⇒ 那次测量作废，结论撤回。双节点 md5 校验后复测为 −11.13% / −2.81%，与单卡一致。
**纪律：跨节点实验开跑前必须 md5 双校验。**

## 4 无效 / 不可用项（同口径实测，全部留档，开关保留）

| 编号 | 改动 | 结果 | 判定 |
|---|---|---|---|
| E1a | RMSNorm 输出 dtype 归位 | 1,443.7 vs 1,443.9（+0.03%），精度不降 | 无效 |
| E4 | `--ddp-strict` | +0.01% | 无效 |
| E5 | 仅 FFN 重算（`CKPT_SCOPE=ffn`） | OOM（需 1.43 GiB，已占 82.48/91.75） | 不可行 |
| E9/E10 | `--input-dtype bf16` | −0.23% / 0.0%，只省 0.45~0.58 GB | 无效 |
| E11 | 权重 bf16 影子 | 档 A −0.15%、档 B +0.06%；Cast 84.8→84.5 ms | 无效 |
| E13 | 掩码缓存 | −0.04% / −0.03% | 无效 |
| E14 | `npu_add_rms_norm` | 无 autograd kernel，三个输入梯度全 `None`（静默错梯度） | **高危不可用** |
| E15a/E15b | `gradient_as_bucket_view` / `jit_compile` | −0.15% / +0.01% | 无效 |
| E16 | `torch.compile(inductor)` | `NoTritonConfigsError: No valid triton configs` | 封死 |
| E17/E18/E19 | `NPUGraph` 整步 / 仅前向 / `make_graphed_callables` | `rtStreamEndCapture ... stream not joined`（前向内部有旁路 stream）；捕获期 OOM | 封死 |
| G-a | FA `sparse_mode` 覆盖 `q<k` 层 | `atten_mask=None` 时 `sparse_mode` 被忽略（与无 mask 逐位相同）；给了 mask 则 `sparse_mode=3` 报 `161001` | 不可行 |
| Triton-h | 用 Triton 融合「残差加 + RMSNorm」 | fwd 1.450→1.527 ms（+5.3%）、bwd 2.915→2.888（−0.9%）：与厂商融合算子**打平** | 无效 |

**图模式封死的证据**：连纯前向捕获都过不了 stream-join 检查 ⇒ 是某个算子内部起了旁路 stream（嫌疑 FA v4 / `npu_rms_norm`），
属专项工程，不是调参。**Triton 打平的原因**：baseline 已在 ~1.7 TB/s（贴 HBM 上限），2-op 融合只省一次 0.63 GB 往返 ≈0.4 ms。

## 5 优化后 profile 画像（16 die · 当前配置 · 3 稳态步）

- **kernel 并集 844.3 ms ≈ 墙钟 848.9 ms**，气泡 **0.8 ms**、Free 4~6 ms ⇒ **kernel-bound，没有空洞**；
  4,522 个 kernel/步，并集与总和（882.3 ms）只差 4.5% ⇒ 重叠极少。
- 构成（ms/步，括号为次数）：

| 项 | ms/步 | 次数 |
|---|---:|---:|
| FA 反向（`FlashAttentionScoreGradV4`） | 127.4 | 14 |
| MatMul（`Mm`） | 108.9 | 168 |
| **Add** | **101.4** | **471** |
| InplaceCopy-Cast | 57.8 | 924 |
| Addmm | 51.9 | 86 |
| InplaceCopy-Slice | 46.8 | 190 |
| ViewCopy | 45.4 | 318 |
| FA 前向 | 40.6 | 14 |
| SiluGrad / RmsNormGrad | 35.3 / 35.1 | 54 / 28 |
| InplaceZero | 30.7 | 588 |
| TensorMove / Transpose | 24.9 / 23.3 | 232 / 334 |
| Silu / ReduceSum | 22.7 / 20.8 | 54 / 158 |

- 下一步候选（按收益 ÷ 风险）：① attention 尾段已做（§3.5）；② 切片/散播链（`Slice 46.8 + SliceBwd + zeros/zero_ 60 ≈ 110 ms`）
  可用自定义 `autograd.Function` 把「切 + 投影 + 回填」合成一个 node；③ ns 的 `bmm + bias` 换 `npu_baddbmm`；
④ `InplaceZero 588 次/步` 需先定位来源（in-place 写的 autograd 副产品），未定位前不下结论。

## 6 与论文 16×H100 并列对比

| 项 | 论文 OneTrans-S（16×H100） | 本仓 950DT 档 A | 本仓 950DT 档 B |
|---|---:|---:|---:|
| 稳态步时 | **407 ms** | **895.3 ms** | **730.0 ms** |
| 全局吞吐 | 80,511 sps（**推算**，论文未给） | 36,601.7 sps | **44,889.1 sps** |
| 每卡吞吐 | 5,032 sps（2048 ÷ 0.407 s，**推算**） | 2,287.6 sps | 2,805.6 sps |
| MFU | **S 未给**（OneTransL 30.8% bf16、DCNv2+DIN 13.4%） | **16.70%** | **20.48%** |
| 显存 | 53.13 GB（fp32 未优化档） | 47.85 GB | 69.60 GB |
| 口径 | fp32、未优化基线 | bf16 + FA v4 + 融合 norm + 六步优化 | 同左，per-GPU batch 1024×2 |

**并列读数**：档 A 是 **2.20×** 论文 407 ms，档 B 是 **1.79×**。这个比值只表示"同形状下两套栈的稳态步时之比"，
**不是**"昇腾比 H100 慢 N 倍"—— 口径（精度、融合程度、batch 划分）不同，按 §2.1 不得相除求收益。

## 7 未闭合项（明确未做/未定，不含推测）

1. **论文 Table 3 的 pyramid 算术倍率口径**：本仓 pyramid 开/关实测 **1.68×**，论文表格口径为 3.06×；按论文正文写法
   （§3.4 只剪 query、K/V 为层输入全序列；§4.1 线性 schedule、每层 32 对齐、顶层 = NS 数）计算为 **1.66×**，与实测自洽
   ⇒ 判为**论文表格口径待核**，不为它改码。
2. **`ReduceSum 20.8 ms / 158 次`** 与 **`InplaceZero 30.7 ms / 588 次`** 的来源：全仓无 `clip_grad_norm_`、无 `.sum(` 调用，
   归因未定位，不作结论。
3. 每卡 batch 4096 与"每卡 2048 去掉重算"两条路均被显存/算子封死（E5、`FlashAttentionScoreGradV4` 失败），未评估。
4. 图模式需先定位"算子内部旁路 stream"，属专项工程，未排期。
