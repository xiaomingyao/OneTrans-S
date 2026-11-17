# OneTrans-S

**OneTrans-S 在昇腾 950DT 上的训练复核，目标：性能与论文的 H100 持平** —— 按论文（arXiv:2510.26104v3）OneTrans-S 的形状把模型拉起来，量真实训练步的**步时 / 吞吐 / 显存 / MFU**，把 950DT 追到 **≤ 427 ms/步**（论文 407 ms 的 ±5% 内）。

> 范围：**只做训练**。推理、cross-request KV cache、p99 延迟那套一律不做。
> 状态（2026-11-17）：形状已对齐（参数量 `91,026,946` vs 论文 91 M，+0.03%），**1 / 8 / 16 die（跨机 122+111）已实测**。
> 当前最好成绩：**档 A 752.0 ms（论文口径，1.85×）/ 档 B 614.7 ms（全局对齐，1.51×）**，MFU 19.88% / 24.32%；
> 距「持平」判据（≤427 ms）还需 **−43.2% / −30.5%** ⇒ 见 §1.1；结构等价性仍未闭合（§6）。
> ⚠️ 最后一步（显式 bf16）是**精度档变更**（不是布局等价的微调），见 §3.12 的精度记账。
> 论文归档 `papers/OneTrans_2510.26104v3.pdf`；话题与证据链 `docs/TOPIC_ONETRANS_S.md`；原始日志 `results/`。

**交付文档（全部 Markdown，GitHub 直接渲染）**

| 文档 | 内容 |
|---|---|
| [`docs/OneTrans-S_950DT_Optimization_v4_20261117.md`](docs/OneTrans-S_950DT_Optimization_v4_20261117.md) | **总报告 v4（当前）**：双档主表（614.7 / 752.0 ms）、**七步有效优化 + 16 项无效留档**、**TOP-DOWN 四层分阶段诊断**、利用率（FA 反向 35% / GEMM 64.3% / cube 50.9%）、剩余空间三档、与 H100 并列、未闭合项 |
| [`docs/OneTrans-S_950DT_Optimization_v3_20261117.md`](docs/OneTrans-S_950DT_Optimization_v3_20261117.md) | 总报告 v3（历史）：六步有效 + 11 项无效留档，档 B 730.0 ms 那一版 |
| [`docs/OPT_ROUND1_RESULTS.md`](docs/OPT_ROUND1_RESULTS.md) | Round1 逐项实验记录（E1a/E1b/E4/E5/E7/E9/E10/E11，含无效项留档与归因更正） |
| [`docs/PLAN_OPT_ROUND1.md`](docs/PLAN_OPT_ROUND1.md) | 实验方案与验收门槛（单变量、同节点同档 A/B、≥1% 且大于极差、精度门禁先过） |
| [`docs/PROFILE_16DIE_BOTTLENECK.md`](docs/PROFILE_16DIE_BOTTLENECK.md) | 16 卡 profile 全景与 4 条瓶颈（含 B3 低估 6 倍的更正） |
| [`docs/PROFILE_ROUND2_AFTER_OPT.md`](docs/PROFILE_ROUND2_AFTER_OPT.md) | **优化后版本的深度 profile**：类别构成、算力账、op 级归因、效率核算、6 个可优化方向（含收益/风险） |
| [`docs/OPT_PORT_FROM_ZEUS.md`](docs/OPT_PORT_FROM_ZEUS.md) | zeus-onetrans 8 项优化的逐项准入判定 |
| [`docs/TOPIC_ONETRANS_S.md`](docs/TOPIC_ONETRANS_S.md) | 论文事实唯一真源（形状、Table 4/5 数字、口径） |

## 1. 这个仓做什么

1. **复刻形状**：论文 OneTrans-S = 6 个堆叠 OneTrans block、d=256、H=4、序列 token 1190 → 12（每层对齐 32）、NS-token 12、per-GPU batch 2048。
2. **复核训练**：在 950DT 上按这个形状跑训练步，量四个数（稳态步时 / 吞吐 / HBM 峰值 / MFU），留可复跑探针与原始日志。
3. **追平 H100**：把 950DT 的稳态步时压到论文 H100 的同一量级（**≤ 427 ms/步**），过程中只并列绝对值与各自口径的百分比，
   **不跨平台相除算收益**（跨平台纪律）。差距与路径见 §1.1，完成度见表。

### 1.1 目标：性能与 H100 持平（2026-11-17 更新）

**目标定义（唯一口径，不再改）**

| 项 | 取值 | 说明 |
|---|---|---|
| 基准 | 论文 Table 4 首行 **407 ms/步** | 16×H100、per-GPU batch 2048、全局 32,768（fp32 未优化档） |
| **达标判据** | **稳态步时 ≤ 427 ms**（407 × 1.05） | 稳态 = 跳前 3 步后取 8 步中位；抖动 > 1% 需复测 |
| 达标吞吐 | 全局 ≥ **76,700 sps**、每卡 ≥ **4,792 sps** | 对 427 ms；严格对 407 ms 则是 80,511 / 5,032 |
| 达标 MFU | ≥ **35.0%**（427 ms）/ **36.7%**（407 ms） | 分母固定 432 TF（单 die Cube bf16，32 核 × 1650 MHz） |
| 对齐轴 | 16 卡、per-GPU batch 2048、全局 32,768 | **档 A 是唯一可称"持平"的口径**；档 B 只对齐全局 batch |
| 判定层级 | 「持平」≤427 ms；「接近」427~500 ms；「未达」>500 ms | 只认 16 卡实测，单卡不作判定依据 |

**当前距离目标**

| 口径 | 当前步时 | 与 407 ms 之比 | 还需再降（到 427 ms） | 当前 MFU | 达标 MFU |
|---|---:|---:|---:|---:|---:|
| 档 A（论文 per-GPU 2048 + 重算） | **752.0 ms** | **1.85×** | **−43.2%** | 19.88% | 36.7% |
| 档 B（全局对齐，1024×2 无重算） | **614.7 ms** | **1.51×** | **−30.5%** | 24.32% | 36.7% |

**算力账：为什么还差一半**（每卡每步 2048 样本 × 31.529 GFLOP = **64.57 TFLOP**）

- 目标 407 ms ⇒ 需要 **158.6 TF/s/卡（MFU 36.7%）**；当前 730 ms ⇒ 实际 **88.5 TF/s/卡（20.48%）**。
- 16 卡 profile 显示：kernel 并集 844.3 ms ≈ 墙钟 848.9 ms、气泡仅 0.8 ms ⇒ **不是等通信、不是空转，是 kernel 本身慢**；
  其中 cube（FA fwd/bwd + MatMul）约 **39%**，非算力（cast/拷贝/切片/逐元素/归零）约 **61%**。
- 所以达标必须**同时**做到两件事（任一单独都到不了 407）：

| 组合 | cube 效率 | 非算力占比 | 折算步时 | 判定 |
|---|---|---|---:|---|
| 现状 | ≈40% | 61% | 730 ms | 1.79× |
| 只把非算力砍半 | ≈40% | 30% | ≈507 ms | 仍不达标（29.5% MFU） |
| 只提 cube +10 pp | 50% | 61% | ≈673 ms | 仍不达标（22.2% MFU） |
| **两者同时**（非算力砍 2/3 + cube +10 pp） | 50% | 20% | **≈376 ms（39.7% MFU）** | **达标** |

- 对应工作项：① cube 侧——FA 反向效率（当前最差，127.4 ms/步）、MatMul 分块与 bf16 累加；② 非算力侧——剩余 cast/拷贝/切片/归零的融合与消除
  （清单见 `docs/PROFILE_ROUND2_AFTER_OPT.md` §6 与报告 v3 §5）；③ 结构性减算力（更激进的 pyramid / K-V 剪枝）**与论文冲突，需用户决策后才动**。
- **不在目标内的路**：加卡（用户明确只有 16 卡）、改 batch 口径（会把"持平"变成不可比）、降低精度档（fp8 等，需单独精度论证）。

上游 GPU 实现 `WestbrookLong/OneTrans_Pytorch` 以 **git submodule** 锁定 commit，**上游源码零改动**：

```bash
git clone --recurse-submodules https://github.com/xiaomingyao/OneTrans-S
git -C GPU_Reference rev-parse HEAD        # 8febd7e…
git -C GPU_Reference status --porcelain    # 空 = 未改动
```

## 2. 形状对账（论文 vs 本仓实测）

| 项 | 论文 OneTrans-S | 本仓实测 | 判定 |
|---|---|---|---|
| 参数量（dense，不含稀疏表） | 91 M（Table 2） | **91,026,946 = 91.03 M** | +0.03%，PASS |
| 层数 | 6 stacked blocks（§4.1.4） | 7 = 6 个 pyramid 段 + 1 个全长期 base block | 实现多 1 个 base block，结构对账里要说明 |
| 宽度 / heads | d=256 / H=4 | 256 / 4 | PASS |
| pyramid | 1190 → 12，每层取 32 的整数倍（§4.1.4） | `[1203, 960, 736, 480, 256, 12]`（起点 1203 = 1190 + 12 + 1 个 `[SEP]`） | 形状一致，起点含 SEP |
| per-GPU batch | 2048（§4.1.2） | `per_card_batch=2048`，8 die 时全局 16,384 | PASS |
| 每样本训练算力 | 2.64 TFLOPs @batch 2048（口径未说明） | **31.53 GFLOP/sample**（CPU `FlopCounterMode` 实测） | ❌ 差 24.5×，见 §5 |

参数只是必要条件：能落到 91 M 的形状**至少三组解**，所以本仓不拿参数量当结构等价的证据。

## 3. 950DT 实测（2026-11-16）

环境（本轮 S 复核）：`Ascend950DT_9582` / CANN 9.1.0 / **torch 2.10.0+cpu + torch_npu 2.10.0** / 容器 `zeus-950-v2`。
口径：bf16 autocast + **bool mask**（落 `FlashAttentionScoreV4`）+ AdamW；**token 常驻 device ⇒ 步时不含取数与 H2D**（纯计算口径）；MFU 分母 = 单 die Cube bf16 **432 TF**（32 AI core × 1650 MHz）。

### 3.0 收益总表（起点 → 当前，16 die · 全局 batch 32,768 · n=8 中位，全部实测）

| 阶段 | 档 | 打开什么 | 稳态步时 | 相对起点 | 相对上一档 | 全局 sps | 每卡 sps | MFU | HBM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 起点（无优化） | — | — | 1,942.1 ms | — | — | 16,872.3 | 1,054.5 | 7.70% | 52.30 GB |
| +ns-layout | A | `ONETRANS_NS_BMM=1` | 1,442.9 ms | −25.70% | −25.70% | 22,710.2 | 1,419.4 | 10.36% | 52.27 GB |
| +融合 rms_norm | A | `ONETRANS_FUSED_NORM=1` | 1,228.6 ms | −36.74% | −14.85% | 26,671.4 | 1,667.0 | 12.17% | 47.57 GB |
| **+去 cat** | **A** | `ONETRANS_CAT_INPLACE=1` | **1,112.1 ms** | **−42.74%** | −9.48% | **29,465.3** | **1,841.6** | **13.44%** | 47.57 GB |
| 换档 B（每卡 1024×2、关重算） | B | 同上三项，去掉 `--checkpoint-act` | 959.2 ms | −50.61% | **−21.92%**（对同阶段档 A） | 34,161.9 | 2,135.1 | 15.58% | 70.66 GB |
| +去 cat | B | 同上 | 892.5 ms | −54.04% | −6.95% | 36,716.4 | 2,294.8 | 16.75% | 70.17 GB |
| +SDPA 内置 causal | A | `ONETRANS_SDPA_CAUSAL=1` | 1,053.5 ms | −45.75% | −5.27% | 31,104.8 | 1,944.1 | 14.19% | 47.57 GB |
| +SDPA 内置 causal | B | 同上 | 847.3 ms | −56.37% | −5.06% | 38,672.9 | 2,417.1 | 17.64% | 70.17 GB |
| +FA BSH 布局 | B | `ONETRANS_FA_BSH=1` | 823.5 ms | −57.60% | −2.81% | 39,792.6 | 2,487.0 | 18.15% | 68.99 GB |
| +全量投影 | B | `ONETRANS_FULL_PROJ=1` | 753.0 ms | −61.23% | −11.13% | 43,518.3 | 2,719.9 | 19.85% | 70.78 GB |
| **BSH + 全量投影（当前最快）** | **B** | 两者同开 | **730.0 ms** | **−62.41%** | −3.05% | **44,889.1** | **2,805.6** | **20.48%** | 69.60 GB |
| BSH + 全量投影（档 A） | A | 两者同开 + 重算 | 895.3 ms | −53.90% | −15.02% | 36,601.7 | 2,287.6 | 16.70% | 47.85 GB |
| +显式 bf16 干路 | B | `ONETRANS_EXPLICIT_BF16=1` | **614.7 ms** | **−68.35%** | −15.79% | **53,307.4** | **3,331.7** | **24.32%** | 58.48 GB |
| **+显式 bf16 干路（当前最快，档 A）** | **A** | 同上 + 重算 | **752.0 ms** | **−61.28%** | −16.01% | **43,575.4** | **2,723.5** | **19.88%** | **39.17 GB** |

> **profile 复核（2026-11-17）**：当前最优配置（末两行）另跑 3 个稳态步的设备侧 profile，步时 **614.3 ms**（n=8 中位 614.7，差 0.07%）、设备 kernel 合计 636.1 ms、`step_trace_time` 显示 计算 605.5 / 通信 19.5（暴露 4.7）/ 气泡 4.1 ⇒ **与 A/B 口径一致**。

读表纪律：① 「相对上一档」按本表相邻行相除，与 §3.3–§3.6 各节报的「当场对照臂」有 ±0.1 pp 差异（各节分母是同期实测的对照臂）；
② 档 A 与档 B 不可混读——档 B 的 per-GPU micro-batch 是 1024×2（论文是 2048），只对齐全局 batch / 卡数，见 §4.2；
③ 所有步时都不含取数与 H2D，与论文 407 ms 的对比只能并列、不能相除算收益。

**优化前基线扫描**（1 / 8 / 16 die，为上面「起点」行的出处；全部实测）

| 配置 | 每卡 batch | 全局 batch | 稳态步时 | 每卡 sps | 全局 sps | HBM 峰值 | MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 die | 256 | 256 | 233.7 ms | 1,095.6 | 1,095.6 | 20.79 GB | 8.00% |
| 1 die | 512 | 512 | 434.6 ms | 1,178.2 | 1,178.2 | 40.39 GB | 8.60% |
| 1 die | 1024 | 1024 | 837.0 ms | 1,223.4 | 1,223.4 | 79.55 GB | 8.93% |
| 1 die | 1024（+激活重算） | 1024 | 966.4 ms | 1,059.6 | 1,059.6 | 26.64 GB | 7.73% |
| **1 die** | **2048（+重算）** | 2,048 | **1,901.8 ms** | 1,076.9 | 1,076.9 | 51.96 GB | 7.86% |
| **8 die** | **2048（+重算）** | 16,384 | **1,934.2 ms** | 1,058.9 | **8,470.9** | 52.30 GB | 7.73% |
| **8 die** | 2048 × accum 2（+重算） | **32,768** | **3,870.8 ms** | 1,058.2 | **8,465.4** | 52.95 GB | 7.72% |
| **16 die**（122+111） | **2048（+重算）** | **32,768** | **1,942.0 ms** | 1,054.6 | **16,873.0** | 52.30 GB | 7.70% |

**跨机 16 die 已实测跑通**（`world_size=16`，两端各 8 die，代码两端 md5 一致）：全局 batch 32,768 用**一次前向**覆盖（8 die 那档要 `accum 2`）。

同全局 batch=32,768 的两种实现方式对照（同机、同口径，只差卡数）：

| 实现 | 稳态步时 | 全局 sps | 每卡 sps | 卡数 ×2 的效果 |
|---|---:|---:|---:|---|
| 8 die × 2048 × accum 2 | 3,870.8 ms | 8,465.4 | 1,058.2 | — |
| **16 die × 2048** | **1,942.0 ms** | **16,873.0** | 1,054.6 | 吞吐 **1.993×**（跨机效率 **99.7%**）、步时 0.502×、每卡 −0.34% |

⇒ 跨机 UB allreduce 在该形状下**不是**瓶颈（每步梯度量 91 M 参数 ≈ 364 MB）。

三条实测结论：

1. **显存是硬约束**：1 die 每卡 2048 不开激活重算直接 OOM（已分配 86.63 GiB、还需 2.34 GiB）；**每卡 4096 任何组合都上不去**（见 §3.1）⇒ 论文 per-GPU 2048 是这台机器的实际可达上沿，全局 32,768 只能靠 `--grad-accum 2`。
2. **吞吐在 batch ≥ 512 后进入平台**（每卡 1.06k–1.22k sps）：batch 256 时固定开销占比高；把全局 batch 从 16,384 抬到 32,768 只把步时线性拉长 2.001×，吞吐 8,470.9 → 8,465.4（−0.06%）⇒ 该形状每卡算力已饱和。
3. **8 卡扩展 7.87×（效率 98.3%）**，每卡 sps 仅降 1.7% ⇒ 该形状下 DDP 梯度同步**不是**瓶颈；瓶颈在单卡 kernel 效率（MFU 7.7–8.9%）。

### 3.1 每卡 batch=4096 为什么上不去（2026-11-16 排查）

论文全局 32,768 ÷ 8 die = 每卡 4096 是**算术值**，但物理上装不下。逐级压显存，缺口从 11.11 GiB 收到 18 MiB，最后卡在 **FlashAttention 反向的 workspace** 而不是激活量：

| 步骤 | 手段 | 结果 |
|---|---|---|
| 1 | b4096 + 激活重算（基线） | OOM，需再 11.11 GiB（已分配 80.01 / 91.75） |
| 2 | `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512` | OOM（80.01 / 需 11.11）——不是碎片问题 |
| 3 | `expandable_segments:True` | OOM（79.91 / 需 11.11） |
| 4 | `ONETRANS_FFN_CHUNK=131072`（分块 FFN） | OOM，需再 386 MiB（89.66） |
| 5 | 预分配写分片（去掉 `torch.cat` 的整张量复制），chunk=32768 | OOM，需再 22 MiB（90.20） |
| 6 | 输入 token 存 bf16（`--input-dtype bf16`） | OOM，需再 18 MiB（89.93） |
| 7 | `garbage_collection_threshold:0.5` | 显存过了，算子失败：`aclnnFlashAttentionScoreGradV4` / `ERR00100 PTA call acl api failed` |

两个可复用的实验旋钮（默认关闭，不影响已记录结果）：

- `ONETRANS_FFN_CHUNK=<rows>`：FFN 按行分块，**数学逐元素等价**，但 bf16 下**非逐位一致**（b256 两步 loss `0.68401/1.74561` vs `0.68401/1.74525`，相对差 2.1e-4）。
- `--input-dtype bf16`：常驻 device 的输入 token 改 bf16（4096 batch 下省 0.6 GiB）。

⇒ 结论：**96 GiB/die 上 OneTrans-S（seq 1190）单卡上沿 = 2048**；要 per-card 4096 必须走 16 die（122+111，全局 32,768 一次前向）或改造注意力分块（未做，标待办）。

原始日志：`results/s_train_950_1die_8die.txt`（含每步 ms / loss / OOM 原始报错）。

### 3.2 算子落点核对：attention 走 **FAv4 融合算子**（2026-11-16）

用 profiler 抓 S 形状下一个训练步（batch 16，仅为抓算子名，耗时不可与 batch 2048 的步时比），同一份 `kernel_details.csv` 里数出来：

| 算子 | 每步次数 | 每步耗时 |
|---|---:|---:|
| `aclnnFlashAttentionScoreV4`（前向） | 7.0 | 486.8 µs |
| `aclnnFlashAttentionScoreGradV4`（反向） | 7.0 | 1,418.6 µs |
| 附带的 `Transpose`（FA 前后） | 49 | 519.4 µs |
| **attention 合计** | — | **2,424.9 µs** |

判据：① 前向/反向各 **7 次/步 = 7 个 block**（1 base + 6 pyramid 段）各一次；② CSV 里 `Softmax` 只出现在交叉熵（`LogSoftmax` 1 次/步、3.2 µs），**不存在 `BatchMatMul+Softmax` 的注意力回退链** ⇒ 不是 float 加性 mask 的手工路径；③ 该 CSV 内 attention 占总 kernel 时间 **7.0%**，反向占主导（1,418.6 vs 486.8 µs）。

复现命令与原始 CSV：`tools/probes/probe_fa_kernel_s.py` + `results/kernel_details_fa_probe_b16.csv` + `results/fa_kernel_probe_950.txt`。

**顺带发现（未优化，仅记录）**：同一份 profile 里 `Cast`（`aclnnInplaceCopy_CastAiCore`）5,939 µs/步、占总 kernel 17%，是第一大项；`MatMulV3` 其次 3,894 µs/步。要做 kernel 级优化，靶子在这里，而不是 attention。

**embedding 说明**：本仓**没有稀疏嵌入表**（全仓零 `nn.Embedding` / `torchrec` / `hkv` / `tbe`），输入特征经 `nn.Linear` tokenizer 变 token，与论文 Table 2「Params 不含稀疏嵌入」的口径一致 ⇒ 不存在「TBE 还是 HKV」的选择（那两者只服务稀疏表；HKV/融合 EC 属于 `zeus-onetrans` 生产模型那条线）。

### 3.3 从 zeus-onetrans 移植优化：ns-layout（`ONETRANS_NS_BMM=1`）

zeus-onetrans 的 8 项优化按本仓准入四条（数学等价 / loss-梯度一致 / 不引入新依赖硬编码 / 不改 dtype-mask 语义）逐项筛过，
**只有 ns-layout 能干净移植**：把「每个 NS-token 一组专有 QKV/FFN」的逐 token `nn.Linear` 合成一次 `bmm`
（数学逐元素等价，每个 token 仍用自己的权重，只是把 12×3 次 1-token 小 matmul 变成 3 次 bmm）。逐项判定表见 `docs/OPT_PORT_FROM_ZEUS.md`。

等价性（CPU UT `tools/probes/ut_ns_bmm_equiv.py`，无需卡）：参数量 `471,938 == 471,938`、前向 `max|Δout| = 5.96e-08`、
一步反向 `worst rel|Δg| = 1.34e-06`（判据 ≤1e-5）⇒ **EQUIVALENT_PASS**。

| 档 | 基线步时 | +ns_bmm 步时 | Δ | 吞吐 | MFU |
|---|---:|---:|---:|---:|---:|
| 1 die · batch 2048（12 步中位） | 1,908.9 ms | **1,437.8 ms** | **−24.7%** | 1,072.9 → **1,424.4** sps（+32.8%） | 7.83% → **10.40%** |
| 16 die · 全局 32,768 | 1,942.0 ms | **1,442.9 ms** | **−25.7%** | 16,873.0 → **22,710.2** sps（+34.6%） | 7.70% → **10.36%** |

参数量不变（91,026,946）、HBM 持平（51.96 → 51.93 GB）。默认仍关（`ONETRANS_NS_BMM=0`），本文档所有主对比档都已开。

### 3.4 Round1 追加：融合 rms_norm（`ONETRANS_FUSED_NORM=1`，有效）

手写 RMSNorm 链（`float→pow→mean→rsqrt→mul→改dtype→乘 fp32 weight`）在 16 die 上实测占 **≈280 ms/步（19%）**，
其中 `Mul` 单项 151.6 ms。换成 `torch_npu.npu_rms_norm`（反向 `RmsNormGrad`）后：

| 臂（16 die · 全局 32,768 · n=8） | 步时 | 全局 sps | 每卡 sps | MFU | HBM | kernel/步 |
|---|---:|---:|---:|---:|---:|---:|
| 仅 ns_bmm | 1,443.9 ms | 22,694.9 | 1,418.4 | 10.35% | 52.27 GB | 3,083 |
| **+fused_norm** | **1,228.6 ms** | **26,671.4** | **1,667.0** | **12.17%** | **47.57 GB** | **2,762** |
| Δ | **−14.90%** | **+17.52%** | +17.5% | +1.82 pp | −4.70 GB | −321 |

精度：与 fp64 参考比，融合算子 `max_rel 3.891e-03 / rms 1.655e-03`，**优于**手写链 `7.704e-03 / 2.347e-03`
（`tools/probes/probe_fused_norm.py`，b64/seq1203，0.063 vs 0.205 ms/call）。逐项结果、失败项与阶段拆分见 `docs/OPT_ROUND1_RESULTS.md`。

### 3.5 第二步：关掉激活重算（每卡 1024 × 累加 2，全局仍 32,768）

profile 对照（`--no-backward` 诊断档）证明整步 MFU 12% 只是「同一算力被计了 4 遍」：

| 段 | 耗时/步 | 单遍效率 |
|---|---:|---:|
| 前向 | 309.2 ms（25%） | **MFU 48.24%** |
| 激活重算（backward 内的二次前向） | 309.2 ms（25%） | 同上 |
| 真反向（≈2× 前向 FLOPs） | 616.7 ms（50%） | **48.5%** |
| 合计 | 1,235.1 ms | 12.17% ≈ 48.4% ÷ 4 |

⇒ 单遍 kernel 效率已经是 48%，真正的浪费是「重算那一遍」。重算在每卡 2048 上无法去掉（E5 实测 OOM），
所以改为**每卡 1024 + 梯度累加 2**（全局 batch、卡数、loss 目标都不变，仅 per-GPU micro-batch 与重算策略不同）：

| 档（16 die · 全局 32,768 · n=8） | 步时 | 全局 sps | 每卡 sps | MFU | HBM | 重算 |
|---|---:|---:|---:|---:|---:|:--:|
| 每卡 2048（对齐论文 per-GPU batch） | 1,228.6 ms | 26,671.4 | 1,667.0 | 12.17% | 47.57 GB | 开 |
| **每卡 1024 × 累加 2（无重算）** | **959.2 ms** | **34,161.9** | **2,135.1** | **15.58%** | 70.66 GB | 关 |
| Δ | **−21.92%** | **+28.09%** | +28.1% | +3.41 pp | +23.09 GB | — |

两条读法纪律：① 两档的**全局 batch 与卡数完全一致**，但 paper 写的是 per-GPU batch=2048，所以「每卡 1024」属**改变了论文的
per-GPU 轴**，只能在「全局 batch 对齐」这一轴上比；② 数学目标相同（各 micro-batch 取均值再除累加数 = 整批均值），
但 bf16 重排 + AdamW 非线性会让 loss 轨迹分叉（实测 step4 0.93893 vs 0.92159），**不可逐值对比、也不可当精度回归用**。

### 3.6 第三步：`_project_one` 的 seq/ns 合并去 cat（`ONETRANS_CAT_INPLACE=1`，有效）

`torch.cat([seq_part, ns_part])` 换成「预分配 + 两段原地写」。kernel 侧证据（8 die · 每卡 2048 · 3 稳态步）：

| 算子 | cat 前 | cat 后 | Δ |
|---|---:|---:|---:|
| `aclnnCat_ConcatD` | 51.4 ms / 47 次 | **7.8 ms / 7 次** | −43.6 |
| `aclnnCat_CastAiCore` | 42.2 ms / 48 次 | **6.4 ms / 8 次** | −35.8 |
| `InplaceCopy_CastAiCore` | 143.1 ms / 677 次 | 78.5 ms / 657 次 | −64.7 |
| `InplaceCopy_TransposeAiCore` | 76.6 ms / 211 次 | 26.6 ms / 231 次 | −50.0 |
| FA 附属转置（新增） | 0 | 27.9 ms / 40 次 | +27.9 |
| `InplaceCopy_ViewCopy` | 32.1 / 99 | 54.7 / 199 | +22.6 |
| `InplaceCopy_TensorMove` | 2.1 / 76 | 24.0 / 116 | +21.9 |
| **整步** | **1,235.1 ms / 2,762 kernel** | **1,121.7 ms / 2,862 kernel** | **−113.4 ms** |

两档实测（16 die · 全局 32,768 · n=8）：

| 档 | 加 cat_inplace 前 | 加之后 | Δ | 吞吐 | MFU |
|---|---:|---:|---:|---:|---:|
| A（每卡 2048+重算） | 1,228.6 ms | **1,112.1 ms** | **−9.48%** | 26,671.4 → **29,465.3** sps（+10.48%） | 12.17% → **13.44%** |
| B（每卡 1024×2 无重算） | 959.2 ms | **892.5 ms** | **−6.95%** | 34,161.9 → **36,716.4** sps（+7.48%） | 15.58% → **16.75%** |

**精度口径（不许说成逐位等价）**：只改内存写路径，写入值相同；但新缓冲的地址/对齐变了，下游 kernel 的分块与 fp32 累加顺序随之变，
实测 loss 相对差 ≤2e-4（step1 1.26852 vs 1.26844、step3 1.06913 vs 1.06892，step0 完全一致 0.69804）⇒
属 **bf16 噪声级等价**，可用作性能档，不可用作精度基线。

### 3.7 第四步：权重 bf16 影子（`ONETRANS_WEIGHT_SHADOW=1`）—— 逐位等价，但**无收益**

autocast 每次 `F.linear`/`bmm` 都把 fp32 权重现 cast 一遍（重算 + 累加下同一权重每步要被 cast 三次）。
`ONETRANS_WEIGHT_SHADOW=1` 按 `(param, 切片)` 缓存一份 bf16 权重，每步每权重只 cast 一次；缓存的张量仍接在原
Parameter 的计算图上（`p[idx].transpose(...).to(bf16)`），因此 DDP 梯度同步、激活重算、梯度累加语义都不变。

**精度**：UT（`tools/probes/ut_weight_shadow_equiv.py`，NPU 上跑 2 步 × 2 micro-batch，含 AdamW）
loss 与全部 41 个参数梯度**逐位相同（rel = 0）**。
踩坑一条：bmm 路径的 **bias 必须保持 fp32** —— baseline 的 `bf16 权重乘积 + fp32 bias` 走 type-promotion
在 fp32 域相加，把 bias 一起 cast 成 bf16 会改成 bf16 域加法，实测差 1 ULP（9.766e-04）。

**性能（16 die，n=8）**：

| 档 | 不开 | 开 | Δ | HBM |
|---|---:|---:|---:|---:|
| A（每卡 2048+重算） | 1,112.1 ms | 1,110.4 ms | **−0.15%** | 47.57 → 47.74 GB |
| B（每卡 1024×2 无重算） | 892.5 ms | 893.0 ms | **+0.06%** | 70.17 → 69.93 GB |

**判定：无效（低于 1% 门槛），默认关。** kerenel 侧证据（8 die 同档 profile）：kernel 总数 2,862 → 2,693（−169/步）、
名字含 `Cast` 的算子 665 → 552 次，但 **Cast 总时长只从 84.8 ms 掉到 84.5 ms（−0.3 ms）**
⇒ 被删掉的是「权重级小 cast」（每次约 0.02 ms），剩下的 ~78 ms 全是大张量**激活** cast（544 次、平均 0.14 ms），
与权重无关。**这条实测否掉了「影子权重能吃掉反向 Cast」的预估（原先估上限 ~7%，实际 ~0.03%）。**

### 3.8 第五步：query_len == key_len 的层用 SDPA 内置 causal（`ONETRANS_SDPA_CAUSAL=1`，有效）

`paper_causal` 掩码是**底右对齐**因果（`col <= row + (key_len − query_len)`）。实测 PyTorch SDPA 的 `is_causal=True`
在 `q_len != k_len` 时用的是**左上对齐**，两者不等价（`max|Δ| 4.26`）；但在 **`q_len == k_len` 的层里两种对齐是同一个掩码**，
可以直接省掉整张掩码张量、让 FA 走内置 causal 分支。

适用层：base block 与第一个 pyramid block（都是 1203×1203），合计占 attention 前向算力约一半。其余 `q<k` 的层继续走显式 mask。

- 精度：模型 logits **逐位相同（max|Δ| = 0.0）**（`tools/probes/probe_is_causal.py` + 模型级对照）。
- 单算子：同形状 (512,4,480,1203) 下单次 1.493 → 0.406 ms（**−72.8%**，但这是 q<k 的对照；q==k 层省的是掩码张量与 FA 分支）。
- 16 die（n=8）：档 A **1,112.1 → 1,053.5 ms（−5.27%）**、档 B **892.5 → 847.3 ms（−5.06%）**；
  吞吐 29,465.3 → **31,104.8** / 36,716.4 → **38,672.9** sps；MFU 13.44% → **14.19%** / 16.75% → **17.64%**。

### 3.9 第六步：全量投影 + 原地覆盖 NS 行（`ONETRANS_FULL_PROJ=1`，**有效，16 卡 −11.13%**）

每个投影原本是「seq 段走 `F.linear` + NS 段走 `bmm` + 新建缓冲合并」。改成先**全量**投影（NS 的 12 行也走共享权重），
再把 NS 行整块覆盖成独立权重的结果：省掉一次整块 alloc + 一次全量写，代价是多算 `12/L ≈ 1%` 的投影算力。

| 场景 | baseline | FULL_PROJ | Δ |
|---|---:|---:|---:|
| 单 die · batch 1024 | 421.2 ms | **374.9 ms** | **−11.0%** |
| 16 die · 档 B（1024×2 无重算） | 847.3 ms | **753.0 ms** | **−11.13%** |
| 16 die · 档 A（2048+重算） | 1,053.5 ms | （与 BSH 合并后测，见 §3.11） | — |

精度（单 die，2 步，81 个参数全量梯度对拍）：loss rel ≤ **9e-5**，`max|Δg| = 4.88e-4` 恰好是最大梯度元素（0.154）的
**1 个 bf16 ULP**，各参数 `max|Δg|` 中位数 `7.6e-6` ⇒ 布局/分块导致的 **bf16 噪声级等价**，不可当精度基线。

> **自纠（重要）**：第一次测 16 卡时为 "−0.02%（无效）"，原因是 rank0（122）上的 `main_pytorch.py` **没有同步**
> （只推给了 111），rank0 跑的是旧代码 ⇒ 该次测量作废。同步后（md5 双节点一致）复测为 **−11.13%**，与单卡 −11.0% 一致。
> 教训：**跨节点实验必须先 md5 校验两节点代码一致，再开始跑**。

### 3.10 第七步：attention 走 FA 的 `BSH` 布局（`ONETRANS_FA_BSH=1`，**有效，16 卡 −2.81%**）

`npu_fusion_attention` 支持 `input_layout="BSH"`：输入 `(B,L,d)`、输出也是 `(B,L,d)` 连续，省掉
`output.transpose(1,2).contiguous()` 这次整张量往返，以及 FA 内部的部分转置。

| 场景 | baseline | BSH | Δ |
|---|---:|---:|---:|
| 单算子 fwd（B512·H4·480×1203） | 2.215 ms | **1.513 ms** | **−31.7%** |
| 单 die · batch 1024 | 421.1 ms | **409.8 ms** | **−2.7%** |
| 16 die · 档 B | 847.3 ms | **823.5 ms** | **−2.81%** |

精度：单算子前向 **逐位相同（max|Δ| = 0.0）**、`dk/dv` 逐位、`dq` 1 ULP；模型级 2 步 `max|Δg| = 2.44e-4`
（最大梯度 0.154 的 0.5 ULP）⇒ 噪声级等价。

两个必须知道的坑：
1. **`atten_mask` 语义是「True = 屏蔽」**，与 SDPA 相反；传反了不报错但结果全错（max|Δ| 4.53）。
2. `Lq == ns_len` 的最后一层（12 行）q 是 **fp32 且非连续**（bmm + fp32 bias 的 type-promotion），FA 直接报
   `161002` ⇒ BSH 分支要限定 `ns_len < Lq < Lk and q.is_contiguous() and q.dtype==k.dtype`，那层留在 SDPA 上。

> **自纠**：第一次测 16 卡为 "0.00%"，同 §3.9 —— rank0 代码未同步所致，该次测量作废。

### 3.11 第八步：G2 + G3 合并（当前最优）

两项叠加（`ONETRANS_FULL_PROJ=1 ONETRANS_FA_BSH=1`）：

| 档（16 die · n=8） | 步时 | 全局 sps | 每卡 sps | MFU | HBM |
|---|---:|---:|---:|---:|---:|
| 档 B（1024×2 无重算） | **730.0 ms**（−13.84% vs 847.3） | **44,889.1** | 2,805.6 | **20.48%** | 69.60 GB |
| 档 A（2048+重算） | **895.3 ms**（−15.02% vs 1,053.5） | **36,601.7** | 2,287.6 | **16.70%** | 47.85 GB |

单 die 合并版 362.0 ms（vs baseline 421.1，−14.0%）；合并配置精度：`max|Δg| = 4.88e-4`（1 ULP），
|g|≥1e-2 上 worst rel 1.15e-2、mean rel 6.2e-4，无 NaN ⇒ 与单项同级的 bf16 噪声级等价。

### 3.12 第九步：显式 bf16 干路（`ONETRANS_EXPLICIT_BF16=1`，**有效，档 B −15.79%**，但**改变训练精度档**）

做法：**关掉 autocast**，让残差流/激活/权重全走 bf16（RMSNorm 输出与 block 入口强制 bf16、NS 的 fp32 bias 落 bf16），
权重用每步一次的 bf16 影子（图仍接在 fp32 参数上，优化器仍算 fp32），只有 loss 前把 logits 转回 fp32。

目的：消灭 autocast 在「bf16 matmul ↔ fp32 逐元素」之间反复插入的激活 cast（16 die 上约 58 ms/步 + 大量 LayoutCopy）。

| 场景 | baseline | 显式 bf16 | Δ |
|---|---:|---:|---:|
| 单 die · batch 1024 | 363.5 ms | **305.1 ms** | **−16.1%** |
| 16 die · 档 B | 730.0 ms | **614.7 ms** | **−15.79%** |
| 16 die · 档 A | 895.3 ms | **752.0 ms** | **−16.01%** |
| HBM（档 B / 档 A） | 69.60 / 47.85 GB | **58.48 / 39.17 GB** | −11.1 / −8.7 GB |

**精度记账（重要，与前面所有优化不同级）**：单 die、batch 256、**1 步**（同一初始权重）对拍 81 个参数全梯度：

| 指标 | 结果 |
|---|---|
| loss 相对差 | **9.8e-5** |
| `max|Δg|` | **5.33e-4** |
| `mean rel|Δg|`（|g|≥1e-2，173 个元素） | **6.1e-3（0.61%）** |
| `worst rel|Δg|` | 3.25e-2（3.2%） |
| NaN / Inf | 无 |

⇒ 这是**训练精度档的变更**（干路逐元素运算从 fp32 降到 bf16），**不是**布局类的 1-ULP 噪声，**不能拿来当精度基线**；
10 步 loss 轨迹两种模式都稳定（0.70 → 1.15 → 1.60 → … → 0.69），无发散。默认关。

**踩坑（两次，都记下来）**：
1. 显式 bf16 下 `Lq == ns_len` 那层的 NS 路径 fp32 bias 会把 q 提成 fp32，SDPA 直接报 dtype 不一致 ⇒ NS bias 一起落 bf16；
2. **影子缓存必须每步清**：探针里漏了 `clear_weight_shadow_cache()`，第 2 步用了第 1 步的陈旧权重，梯度 absmax 掉到 1/4，
   差点被误判成"bf16 训练坏了"。凡是用影子的脚本，每步都要清。

### 3.13 第十~十二步：三项「看起来能赢」的尝试，实测**全部否决**（2026-11-17）

不是没做，是做完发现不行。逐项留证据，避免以后重复投入。

| 尝试 | 做了什么 | 实测结果 | 判定 |
|---|---|---|---|
| **cube 侧三个杠杆** | qkv 投影 3 次 `linear(256→256)` 合成 1 次 `linear(256→768)`；FA `inner_precise=1`；mask dtype bool→uint8 | 合并后单算子 **−8.7%** 且输出**逐位相同**，但每步 qkv 前向总共只有 ~5.0 ms ⇒ 每步只省 **0.44 ms（0.07%）**；`inner_precise` 与 mask dtype **完全无差别**（2.999/2.995、14.766/14.769 ms） | **否决** |
| **图模式** | 手动 `torch.npu.graph` 逐算子二分；`auto_dispatch_capture` / `capture_error_mode` 三种组合；TorchAir（`torch.compile(backend="npu")`）粒度二分 | **attention 算子全部不可捕获**（`capture_end … 107025`）：SDPA(is_causal/掩码)、FA(fwd/fwd+bwd) 全 FAIL，而 `linear`/`npu_rms_norm`/`npu_add_rms_norm`/`silu`/`mean`/`zeros` 全 OK；TorchAir 卡在 GE `InferShape: k-axis a(1191) vs b(256)`（= 1203−12，我们的投影写法对图编译器不友好）；**气泡实测只有 4.13 ms（0.7%）⇒ 即便打通，消 launch 的上限就是 0.8%** | **否决**（要写得 trace 友好才有 GE 融合的未知增量） |
| **自写 attention（Triton 验证）** | 写 Triton-Ascend flash-attention 前向，与本仓真实形状对比厂商 FA v4 | `tl.dot`/标量/online-softmax 都能跑，但前向 **慢 2.32~3.12×**（Lq480/Lk1203 3.08×、1203/1203 2.32×；tile 调到 64×128 最好也只有 2.32×）；**反向还没写**，而 FA 反向本身占 65.8 ms/步 | **否决**（AscendC 硬写同理不建议） |
| **RMSNorm 两招** | ① N1：`_RMSNormFn` 存 rstd + `npu_rms_norm_backward` 直接调；② `npu_add_rms_norm` 融合前向 | ① 真模型**毫无变化**（`rmsnorm` 家族 39.95 → 40.01 ms，步时 304.8 → 304.7）——eager autograd 本来就用 rstd，探针里的「−49%」是我 A 臂多算了一次 602 MiB 的 `clone()`；② 融合前向 **1.448 → 0.966 ms（−33.3%）**、`h` 与 `x1+x2` **逐位相同**、`y` 差 1 bf16 ULP，但**反向无 autograd**，且代码里只有一半的 add→norm 配对能融 | ① **否决**；② 上限 **0.7%（intra-block）~1.35%（跨 block 改造）** |

两处自纠也记在这里：Triton 版第一稿把底右因果的 `n_start` 写反（跳过了「允许的前段列」），
跑出「快 29~51%」是假的，修正后才是「慢 2.3~3.1×」；RMSNorm 探针的 −49% 来自 A 臂的 `clone()` 拷贝开销。

## 4. 与论文 H100 的差距（目标：持平，按**全局 batch = 32,768** 对齐）

> 主对比档已叠加 `ONETRANS_NS_BMM=1`（从 zeus-onetrans 移植的 ns-layout 优化，数学等价、UT 通过；逐项判定见 `docs/OPT_PORT_FROM_ZEUS.md`，实测见 §3.3）。

### 4.1 对齐轴（哪些对齐、哪些必须并列说明）

| 轴 | 论文 H100 | 本仓 950DT | 是否对齐 |
|---|---|---|---|
| **全局 batch** | 16 卡 × 2048 = **32,768** | 16 die × 2048 = **32,768** | ✅ **本对比的对齐轴** |
| per-GPU batch | 2048 | 2048 | ✅ 对齐 |
| 卡数 | 16 | **16（122 + 111 各 8 die）** | ✅ 对齐（主对比档） |
| 并行方式 | 16 卡数据并行，一次前向覆盖 32,768 | 16 die 数据并行，一次前向覆盖 32,768 | ✅ 对齐（8 die 那档需 accum 2，仅作附档） |
| 精度 / 优化档 | **fp32、未优化基线**（Table 4 首行） | **bf16 + bool mask(FA) + 激活重算** | ❌ 不一致，只能并列不能比快慢 |
| 数据 | 工业自有数据（29.1B impressions） | 合成 token 常驻 device（不含取数与 H2D） | ❌ 不一致，算力不可对表 |
| 优化器 | Adagrad(sparse) + RMSProp(dense) | AdamW（密度参数统一） | ❌ 不一致，步时含优化器开销 |

⇒ 全局 batch 与 per-GPU batch 两条对齐后，**剩下的差异是平台、精度档、数据来源**，所以下表只并列绝对值 + 口径百分比，**不做收益%**；
判「是否持平」只看 §1.1 的步时判据（≤427 ms，档 A 口径）。

### 4.2 主对比表（双档并列：档 A 严格对齐论文 per-GPU 轴；档 B 只对齐全局 batch）

两个档都是 16 die / 全局 32,768，差别只在「每卡 micro-batch 与是否重算」。**两档都要报，不许只挑快的那个。**

**档 A：每卡 2,048 + 激活重算（与论文 per-GPU batch 完全对齐，可逐轴比）**

| 指标 | 论文 16 × H100 | 本仓 16 die × 950DT（档 A） | 差（仅描述，非收益） |
|---|---:|---:|---|
| 全局 batch / per-GPU batch | 32,768 / 2,048 | 32,768 / 2,048 | **对齐** |
| 卡数 / 节点 | 16 / 2（节点数论文未写，按 8 卡/机推断） | 16 / **2**（122 + 111） | 对齐 |
| **稳态步时** | **407 ms** | **752.0 ms** | 1.85×（口径不同，见 §4.1） |
| **每卡吞吐** | 5,032 sps（2048 ÷ 0.407 s，**推算**） | **2,723.5 sps**（实测） | 1.85× |
| 全局吞吐 | 80,511 sps（**推算**，论文未给） | **43,575.4 sps**（实测） | 1.85× |
| 训练显存峰值 | 53.13 GB（fp32 未优化档） | **39.17 GB**（bf16 + 重算 + 显式 bf16 干路） | 并列，精度档不同，不作因果 |
| MFU | **S 未给**；OneTransL 30.8%（bf16）；DCNv2+DIN 13.4% | **19.88%**（分母 432 TF；分子 31.53 GFLOP/sample） | 无可比项（S 侧论文未给） |

**档 B：每卡 1,024 × 累加 2、无重算（只对齐全局 batch 32,768 与 16 卡）**

| 指标 | 论文 16 × H100 | 本仓 16 die × 950DT（档 B） | 差（仅描述，非收益） |
|---|---:|---:|---|
| 全局 batch / per-GPU micro-batch | 32,768 / 2,048 | 32,768 / **1,024×2** | 全局对齐，**per-GPU 轴不同** |
| 卡数 / 节点 | 16 / 2（推断） | 16 / **2** | 全局对齐 |
| **稳态步时** | **407 ms** | **614.7 ms** | 1.51× |
| **每卡吞吐** | 5,032 sps（**推算**） | **3,331.7 sps**（实测） | 1.51× |
| 全局吞吐 | 80,511 sps（**推算**） | **53,307.4 sps**（实测） | 1.51× |
| 训练显存峰值 | 53.13 GB | **58.48 GB**（无重算，换显存） | 并列，不作因果 |
| MFU | S 未给 | **24.32%** | 无可比项 |

四条读法纪律：

1. **1.85× / 1.51× 都不能读成「950DT 比 H100 慢 N 倍」**：论文那档是 fp32 未优化基线，本仓是 bf16+FA+融合 norm 档；要判平台差距，得等论文给出 S 的 bf16 档数字，**目前论文未给**。
2. **每卡吞吐的倍数只可与同档互读**：档 A 与论文 per-GPU 对齐，档 B 的 per-GPU micro-batch 是 1,024（论文是 2,048），跨档互比或用它算收益都属禁用操作。
3. 档 B 的数学目标与档 A 相同（micro-batch 均值 ÷ 累加数 = 整批均值），但 bf16 重排 + AdamW 非线性会让 loss 轨迹分叉（step4：0.93893 vs 0.92159）⇒ **不可逐值比对，不可当精度回归**。
4. 训练算力/样本：论文 2.64 TFLOPs @batch 2048（口径未说明）vs 本仓 31.53 GFLOP/sample（CPU 探针）⇒ 24.5× 差异属结构未闭合，见 §6。

### 4.3 卡数与全局 batch 的完整对照（全部实测）

| 档 | 卡数 / 节点 | 每卡 batch | 全局 batch | 稳态步时 | 全局 sps | 每卡 sps |
|---|---|---:|---:|---:|---:|---:|
| 1 die | 1 / 1 | 2,048 | 2,048 | 1,901.8 ms | 1,076.9 | 1,076.9 |
| 8 die | 8 / 1 | 2,048 | 16,384 | 1,934.2 ms | 8,470.9 | 1,058.9 |
| 8 die × accum 2 | 8 / 1 | 2,048×2 | **32,768** | 3,870.8 ms | 8,465.4 | 1,058.2 |
| **16 die** | **16 / 2** | **2,048** | **32,768** | **752.0 ms**（+ns_bmm+fused_norm+cat_inplace+sdpa_causal+bsh+full_proj+explicit_bf16） | **43,575.4** | **2,723.5** |
| **16 die（无重算档）** | 16 / 2 | 1,024×2 | **32,768** | **614.7 ms** | **53,307.4** | **3,331.7** |
| 论文（参考） | 16 / 2（推断） | 2,048 | 32,768 | **407 ms** | 80,511（推算） | 5,032（推算） |

> 前三行都是「每卡 2048 + 重算」口径（对齐论文 per-GPU batch）；末行是关掉重算的替代档（per-GPU micro-batch 1024×2，全局 batch 不变），
> 两档不可混读。全开前的起点是 1,942.0 ms / 16,873.0 sps（§3.3 / §3.4 对照）。

三条引用纪律（详见 `docs/TOPIC_ONETRANS_S.md` §5）：① 跨平台不相除；② 论文 H100 那档是 fp32 未优化基线，不能与本仓 bf16 优化档直接比快慢，只能并列；③ 论文没给 S 的 bf16/MFU，「按 OneTransL 的 30.8% 推 S」属口径混用（本仓曾算错过一次，已在话题文档自纠）。

### 4.4 剩余空间清单（2026-11-17，按"可动性"分三档）

当前瓶颈画像（3 稳态步 profile，见 §3.13 与 `docs/PROFILE_ROUND2_AFTER_OPT.md` §10）：
**计算占 98.6%**，其中 cube（FA 前向 44.8% / FA 反向 35.0% / GEMM 64.3%）合计 348.4 ms，
向量侧（Add/SiLU/搬运/RMSNorm/ReduceSum/Zero/Cast）合计 ~240 ms，通信暴露 4.7 ms，气泡 4.1 ms。
**修正**：GEMM 曾被我按「只算前向 FLOPs」误报为 21.4%，正确是 **64.3%**（线性层反向是 2× 前向）。

| 档 | 项 | 预估 | 前提 / 阻塞 |
|---|---|---:|---|
| **一（确定性）** | `ReduceSum 20.8 ms / 158 次` 定位并消除 | **−20** | 至今未定位（Python 层每步只发 1 次 `mean`） |
| | `Zero 14.0 ms / 580 次` | −14 | autograd 的 `zeros_like`；参数按组拆 + 自定义反向 |
| | 搬运/重排残余（`TensorMove 26.0`、`Transpose 11.6`） | −15 | G3 已吃大头，剩 NS 路径与 narrow |
| | NS 12 行小 GEMM 合并（210 次 × 18 µs） | −3.8 | 已验证逐位相同 |
| | add+norm 融合（intra-block 14 对） | −4.3 | `npu_add_rms_norm` 已验证 |
| | 图模式消 launch | −4.1 | 就是气泡上限 |
| | 通信暴露部分 | −4.7 | — |
| | **小计** | **≈ −67** | ⇒ **548 ms** |
| **二（cube 效率）** | FA 反向 35% → 45%（追平前向） | −29 | API 旋钮全灭、Triton 自写更慢 |
| | GEMM 64.3% → 75% | −25 | 形状天生小（M=12/256, N=256） |
| | **小计** | **≈ −54** | ⇒ **493 ms（MFU 30.3%）** |
| **三（要决策）** | 自定义反向（压反向那 ~195 ms 向量算子海） | **−90** | 等于重写全部 backward |
| | FP8 训练 | cube 理论 ×2 | 需 scale recipe，精度风险最高 |
| | 减算力（K/V 剪枝 / 更激进 pyramid / ns_len） | 每 −10% 算力 ≈ −35 ms | **与论文结构冲突（已冻结）** |
| | 档 A 专项：分块重算 | 档 A 752 → ~700 | 只改善论文口径那一列 |

**结论（必须直说）**：不碰算法与精度，天花板 **493 ms（MFU 30.3%）**，**到不了 427 ms**。
要「持平 H100」，只能在**自定义反向**或 **FP8 / 减算力**上做一次大改；零碎优化凑不出来。

## 5. 16 卡 profiling：瓶颈在哪（2026-11-16，rank0 profile，3 个稳态步）

完整分析见 **`docs/PROFILE_16DIE_BOTTLENECK.md`**，原始数据 `results/prof16_*.csv`，复算脚本 `tools/probes/analyze_profile.py`。

| 事实 | 值 | 含义 |
|---|---:|---|
| compute kernel 并集占步时 | **99.8%**（1,440.6 / 1,443.2 ms） | 设备侧**没有空隙**（气泡仅 2.6 ms），不靠提并行度取胜 |
| 通信（HCCL）并集 | 15.6 ms（1.08%），69% 被计算掩盖 | **通信不是瓶颈**，净暴露 ≈4.8 ms（0.33%） |
| kernel 数 | **3,083 个/步**，中位 15.9 µs，52% <20 µs | 碎片化，下发间隙直接落在关键路径 |
| cube 类（FA+MatMul） | 484.2 ms（33.1%），同口径理论 149.5 ms | **cube 内效率 30.9%** |
| 非 cube（vector+layout+comm） | **978.6 ms（66.9%）** | 三分之二时间不在矩阵乘上 |

构成（ms/步）：`vector 逐元素 443.9（30.3%）` / `FA 融合 270.8（18.5%）` / `MatMul 213.4（14.6%）` /
`Cast 185.4（12.7%，677 次）` / `切片拷贝 146.5（10.0%）` / `转置 78.0（5.3%）` / `拼接 52.1（3.6%）` / `归约 45.6（3.1%）` / `通信 15.7（1.1%）`。

**四条瓶颈（按可动性排序）**：① layout 搬运占 32%（Cast 677 次/步 + pyramid 切片/拼接/转置）；
② cube 内只跑 30.9%（FA 前向 14 次/步 = 7 block × 激活重算的二次前向）；③ 手写 RMSNorm 链 **≈280 ms/步（19%；初版误估 42.6 ms，已更正）**，
换 `npu_rms_norm` 后实测 **−14.90%**（§3.4）；④ kernel 碎片 3,083 个/步。
可动候选与预估收益见该文档 §4。**实验方案已定稿：`docs/PLAN_OPT_ROUND1.md`**（E1a/E1b/E3/E4/E5 单变量设计、判据、采集协议、收益门槛）。

## 6. 已知差距：结构对账未闭合

| 判据 | 论文 | 本仓 | 结论 |
|---|---|---|---|
| 参数量 | 91 M | 91.03 M | ✅ 但属**多解命中**（ffn/ns 至少三组组合都能落到 91 M） |
| pyramid 开/关的算力比值 | 8.08 / 2.64 = **3.06×**（Table 3 vs Table 2） | **1.68×** | ❌ 本仓 pyramid 只剪 query、不剪 K/V/FFN |
| 每样本训练算力 | 2.64 TFLOPs（口径未说明；摊到 batch 2048 ≈ 1.29 GFLOP/sample） | 31.53 GFLOP/sample | ❌ 24.5× |

⇒ 现阶段只能宣称「**同形状**的 OneTrans 训练」，不能宣称「= 论文 OneTrans-S」。复现脚本：`tools/probes/`（参数量 / 每样本 FLOPs / pyramid 比值，CPU 可跑）。

## 7. 训练入口

```bash
cd OneTrans_NPU

# 1 die 冒烟（形状对齐、跑通即可）
python3 scripts/run_onetrans_s.py --device npu --batch-size 64 --steps 10 --warmup 3

# 1 die 论文 per-GPU batch（必须开激活重算，否则 OOM）
python3 scripts/run_onetrans_s.py --device npu --batch-size 2048 --steps 20 --warmup 5 \
  --checkpoint-act --flops-per-sample 31529400000

# 8 die（参数面不变，只换拉起方式）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  scripts/run_onetrans_s.py --device npu --batch-size 2048 --steps 20 --warmup 5 \
  --checkpoint-act --flops-per-sample 31529400000 --out-json ../results/s_8die.json

# 8 die 且全局 batch = 32,768（论文口径）：micro 2048 × accum 2
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  scripts/run_onetrans_s.py --device npu --batch-size 2048 --grad-accum 2 --steps 20 --warmup 5 \
  --checkpoint-act --flops-per-sample 31529400000 --out-json ../results/s_8die_g32768.json
```

常用旋钮只有 `--batch-size / --grad-accum / --steps / --checkpoint-act / --flops-per-sample` 五个；形状默认值就是论文 S，不要手改模型代码。
判据：`[shape] params=91,026,946` ∧ `[shape] pyramid_schedule=[1203, 960, 736, 480, 256, 12]` ∧ `[perf] steady_median_step_ms` 有值 ∧ 退出码 0。

## 8. 迁移与精度验证（历史资产，仍然有效）

接线改动只在 `npu_compat.py` + 两个入口脚本，`models/*`、`utils/*` 零改动（上游 submodule 亦零改动）；触碰上游 2 文件 147 行，占 1,233 行活跃 Python 的 11.9%。逐项对照 →
[`docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md`](docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md)、[`docs/MIGRATION.md`](docs/MIGRATION.md)、[`patches/npu_changes.diff`](patches/npu_changes.diff)。

| # | 修改点 | 修改说明 |
|---|---|---|
| 1 | 新增兼容层 `npu_compat.py`（160 行） | 设备初始化 / AMP 能力门 / no-op scaler / autocast / HCCL / 收尾集中一处；`models`、`utils` 保持 0 行 |
| 2 | attention mask 形态开关 | `ONETRANS_ATTN_MASK=float\|bool`（+9 行）。float 加性 mask 让 SDPA 退回 Cast+Muls+BatchMatMul+Softmax（45 kernel / 139 µs）；bool mask 落 `FlashAttentionScoreV4`（6 kernel / 36.9 µs，3.77×） |
| 3 | AMP 能力门 | NPU 只放行 bf16（平台 autocast 默认 fp16），GradScaler 改 no-op |
| 4 | 多卡装配 | HCCL + `device_ids=[local_rank]` + `broadcast_buffers=False` + `find_unused_parameters=True`（最深 pyramid 块有 16 个参数本步不参与前向）；循环内不做跨卡归约 |
| 5 | 计时与收尾 | 计时前 `torch.npu.synchronize()`；收尾 `shutdown()` + `os._exit(0)`，避免 teardown 抖动被误报成失败 |

精度三段（NPU vs CPU fp32 golden，全部 PASS，日志 `results/accuracy_950_rerun.txt`）：fp32 前向 `max|Δ|=1.49e-07`（cos 0.99999988）；bf16 前向 `max|Δ|=2.59e-03`（cos 0.99998170）；10 步训练轨迹 `Δloss=5.96e-08`、`Δacc=0.00 pp`。

> ⚠️ 环境差异：以上历史验证跑在 **torch 2.7.1+cpu / torch_npu 2.7.1.post4 / Python 3.11**（`/opt/buildtools/torch_v2_pt2.7.1`，SoC `Ascend950DT` V120 板卡 `IT23HMDB_D2`、CANN 9.1.0）下；§3 的 S 复核跑在同一节点另一容器（**torch 2.10.0+cpu / torch_npu 2.10.0 / Python 3.12**）。两套数字各自绑定自己的环境，**不跨环境比较**。toy 配置下的功能/多卡历史结果（冒烟 76.9 ms、mask kernel 3.77×、DDP `worst rel|Δg|=1.21e-06`、1/2/4/8 die 扩展 4.44×）见 `results/` 与 `docs/MIGRATION.md`。

## 9. 目录

| 目录 | 内容 |
|---|---|
| `GPU_Reference/` | submodule → 上游 GPU 实现 @ `8febd7e`（零改动，diff 与数值基准） |
| `OneTrans_NPU/` | NPU 实现：`npu_compat.py` 兼容层 + `scripts/run_onetrans_s.py`（本仓主入口）+ 数据版入口 + `tests/` 探针 |
| `papers/` | 论文 PDF + 文本（v3） |
| `docs/` | 话题与证据链（`TOPIC_ONETRANS_S.md`）、代码实现对比、五阶段迁移对照 |
| `tools/probes/` | 参数量 / 每样本 FLOPs / pyramid 比值三个 CPU 探针 |
| `results/` | 实测日志（每行可追溯到命令与环境） |
| `patches/` | 完整改动 diff |

## 10. 快速开始（历史探针）

```bash
# 冒烟（前向 + 反向 + 3 步）
python OneTrans_NPU/tests/smoke_npu.py --device npu

# 精度三段（A: fp32 / B: bf16 / C: 10 步训练轨迹）
python OneTrans_NPU/tests/accuracy_npu.py --steps 10

# 与参考实现对照（两侧各起子进程，比结构 + 数值）
python OneTrans_NPU/tests/compare_ref_vs_npu.py

# 真实数据集端到端训练（toy 形状；bool mask 变体加 ONETRANS_ATTN_MASK=bool）
HF_ENDPOINT=https://hf-mirror.com python OneTrans_NPU/scripts/run_taac2026_sample.py \
  --device npu --batch-size 32 --epochs 2 --seq-len 32 --d-model 64 --multi-num 2 \
  --num-pyramid-layers 3 --amp --amp-dtype bf16 --output-dir /tmp/onetrans_out
```

多卡（DDP/HCCL）、mask 与 AMP 单项 A/B、HCCL dtype 探针、一键批量脚本 → [`docs/MIGRATION.md`](docs/MIGRATION.md) §6.4。
