# OneTrans-S 在昇腾 950DT 的训练性能复核报告 v1

- 日期：2026-11-16
- 目标：按论文 arXiv:2510.26104 的 OneTrans-S 形状（91 M 参数、d=256、H=4、seq 1190→12、6 层 pyramid、per-GPU batch 2048）在
  **16 卡 950DT（2×8 die，122 + 111）** 上跑起来，实测步时/吞吐/MFU/显存，与论文 16×H100 的 407 ms **并列**对比（只做训练，不含推理）
- 代码与原始数据：`github.com/xiaomingyao/OneTrans-S`（`main`，含 `results/prof*_kernel_details.csv` 全部 profile 原始件）

## 1 结论摘要

| 档 | 对齐轴 | 步时 | 全局 sps | 每卡 sps | MFU | HBM | vs 论文 407 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** 每卡 2048 + 激活重算 | 全局 batch **和** per-GPU batch 都对齐 | **1,112.1 ms** | 29,465.3 | 1,841.6 | **13.44%** | 47.57 GB | 2.73× |
| **B** 每卡 1024×累加2、无重算 | 只对齐全局 batch 32,768 / 16 卡 | **892.5 ms** | 36,716.4 | 2,294.8 | **16.75%** | 70.17 GB | 2.19× |

相对本轮起点（1,942.1 ms / 16,872.3 sps / MFU 7.70% / 52.30 GB）：
档 A **−42.74%** 步时、**+74.63%** 吞吐；档 B **−54.04%** 步时、**+117.62%** 吞吐。

## 2 口径纪律（先读这段，否则数字会被误读）

1. **不与论文相除算「收益%」**：论文 407 ms 是 fp32 未优化基线，本仓是 bf16 + FA 融合 + 融合 norm 档；只能并列。
2. **档 B 的 per-GPU micro-batch 是 1024×2**，论文是 2048 ⇒ 档 B 只在「全局 batch / 卡数」这一轴上对齐，**不可与档 A 混读**。
3. 档 A 与档 B 的数学目标相同（micro-batch 均值 ÷ 累加数 = 整批均值），但 bf16 重排 + AdamW 非线性会让 loss 轨迹分叉
   （step4：0.93893 vs 0.92159）⇒ **不可逐值比对、不可当精度回归**。
4. `cat_inplace` 这一步只改内存写路径，写入值相同；新缓冲的地址/对齐变化会改变下游 kernel 的分块与 fp32 累加顺序
   ⇒ loss 相对差 **≤2e-4**（step0 完全一致 0.69804）⇒ 记为 **bf16 噪声级等价**，可用于性能档，不可当精度基线。
5. 训练数据是合成 token 常驻 device，步时**不含取数/H2D**（纯计算口径）。

## 3 三波优化（每项都有 kernel 级证据）

### 3.1 ns-layout：NS-token 专有 QKV/FFN 由「逐 token Linear」合成一次 bmm

- 动机：12 个 NS token 各有自己的 QKV/FFN 权重，原实现每 token 发一个 1-token matmul（每 block 12×3 次）。
- 开关 `ONETRANS_NS_BMM=1`；CPU UT（`tools/probes/ut_ns_bmm_equiv.py`）：参数量 471,938 一致、前向 `max|Δout| 5.96e-08`、
  反向 `worst rel|Δg| 1.34e-06`（判据 ≤1e-5）⇒ 等价通过。
- 实测（16 die，全局 32,768）：**1,942.0 → 1,442.9 ms（−25.70%）**，吞吐 16,873.0 → 22,710.2 sps（+34.60%），MFU 7.70% → 10.36%。
- 独立复跑三次一致（−25.7% / −26.1% / −25.7%）。

### 3.2 融合 rms_norm：手写 norm 链换成 `torch_npu.npu_rms_norm`

- 动机：profile 显示手写链（`float→pow→mean→rsqrt→mul→改dtype→乘 fp32 weight`）实际占 **≈280 ms/步（19%）**，
  其中 `Mul` 单项 151.6 ms（因为 `bf16 × fp32 weight` 会把张量提成 fp32 再走一遍向量运算）。
- 开关 `ONETRANS_FUSED_NORM=1`；精度反而更好：对 fp64 参考 `max_rel 3.891e-03 / rms 1.655e-03`，手写链是 `7.704e-03 / 2.347e-03`
  （单算子探针 `tools/probes/probe_fused_norm.py`，b64/seq1203，0.063 vs 0.205 ms/call）。
- 实测（16 die）：**1,443.9 → 1,228.6 ms（−14.90%）**，吞吐 +17.52%，MFU 10.35% → 12.17%，HBM 52.27 → 47.57 GB，kernel 3,083 → 2,762/步。
- 逐算子：`Mul 151.6→0`、`Square 27.9→0`、`Divs 22.2→0`、`Pow 14.4→0`、`Mean 13.1→0`、`ReduceSum 32.2→19.1`、`Add 130.8→109.0`；
  新增 `RmsNorm 29.7 + RmsNormGrad 35.1` ⇒ 净 ≈ −213 ms。

### 3.3 去 cat：`_project_one` 的 seq/ns 合并改「预分配 + 原地写」

- 开关 `ONETRANS_CAT_INPLACE=1`；只改内存写路径。
- 逐算子（8 die 同档 profile）：`Cat_ConcatD 51.4→7.8`、`Cat_CastAiCore 42.2→6.4`、`InplaceCopy_Cast 143.1→78.5`、
  `InplaceCopy_Transpose 76.6→26.6`；新增 `FA 附属转置 27.9`、`ViewCopy +22.6`、`TensorMove +21.9`；整步 1,235.1 → 1,121.7 ms。
- 实测（16 die）：档 A **1,228.6 → 1,112.1 ms（−9.48%）**；档 B **959.2 → 892.5 ms（−6.95%）**。

## 4 失败项与无效项（同口径实测，全部留档）

| 编号 | 改动 | 结果 | 判定 |
|---|---|---|---|
| E1a | RMSNorm 输出 dtype 归位（手写链不变） | 1,443.7 vs 1,443.9（+0.03%）；精度不降（3.888e-03→3.891e-03） | 无效 |
| E4 | `find_unused_parameters=False` | 1,228.7 vs 1,228.6（+0.01%） | 无效 |
| E5 | 只重算 FFN、保留 attention 激活 | OOM：82.48 / 91.75 GiB（差 1.43 GiB） | 每卡 2048 不可行 |
| E9 | 档 A + `--input-dtype bf16` | 1,109.5 vs 1,112.1（−0.23%，抖动内）；HBM −0.58 GB | 性能无效，仅省显存 |
| E10 | 档 B + `--input-dtype bf16` | 892.4 vs 892.5（0.0%）；HBM −0.45 GB | 性能无效，仅省显存 |

## 5 瓶颈分析证据链

### 5.1 整步被「算 4 遍」，不是 kernel 效率低

用 `--no-backward` 只跑前向做对照（8 die · 每卡 2048）：

| 段 | 耗时/步 | kernel/步 | 单遍效率 |
|---|---:|---:|---:|
| 仅前向 | 309.2 ms | 643 | **MFU 48.24%** |
| 完整步 | 1,235.1 ms | 2,762 | 12.17% |
| ⇒ backward(+opt) | 925.9 ms | 2,119 | — |

分解：真反向 `925.9 − 309.2 = 616.7 ms` ≈ 2 × 309.2（反向 FLOPs 是前向 2 倍），单遍效率 48.5% 与前向一致
⇒ **前向 1 遍 + 重算 1 遍 + 真反向 2 遍 = 4 遍算力**，整体 MFU 12.17% ≈ 48.4% ÷ 4。

### 5.2 通信与分布式侧不是瓶颈（已隔离）

| 配置 | 步时 | 每卡 sps | MFU |
|---|---:|---:|---:|
| 1 die（world_size=1，无 DDP） | 1,227.5 ms | 1,668.4 | 12.18% |
| 16 die（DDP + 跨机 UB allreduce） | 1,228.6 ms | 1,667.0 | 12.17% |
| Δ | **+0.09%** | −0.08% | — |

profile 侧同样只有 13 次 HCCL/步、并集 15.6 ms（1.08%），其中 69% 被计算掩盖 ⇒ 通信净暴露 ≈0.33%；
`opt.step` 仅 3.4 ms（0.2%）⇒ 「反向里的 Cast 与优化器无关」。

### 5.3 当前算子构成（8 die，去 cat 后，1,121.7 ms/步、2,862 kernel/步）

| 类别 | ms/步 | 占比 |
|---|---:|---:|
| cube·FA 融合（含附属转置） | 312.8 | 27.9% |
| cube·MatMul | 213.7 | 19.1% |
| vector·逐元素（Mul/Add/Silu…） | 192.9 | 17.2% |
| layout·切片/拷贝 | 157.8 | 14.1% |
| layout·Cast | 84.8 | 7.6% |
| norm 融合（RmsNorm + Grad） | 65.2 | 5.8% |
| layout·转置 | 26.7 | 2.4% |
| vector·归约 | 20.4 | 1.8% |
| comm | 9.3 | 0.8% |
| layout·拼接 | 7.8 | 0.7% |

cube 类合计 526.5 ms（46.9%），非 cube 595.3 ms（53.1%）；单遍 cube 效率 48% 说明**每遍内部**已接近一半峰值。

### 5.4 attention 落点核对（FAv4 生效）

`aclnnFlashAttentionScoreV4` 前向 7 次/步 + `aclnnFlashAttentionScoreGradV4` 反向 7 次/步（7 个 block 各一次，前向因重算出现 14 次），
kernel 列表里无 `BatchMatMul + Softmax` 的注意力回退链（`Softmax` 只出现在交叉熵）⇒ bool mask 路径确实落到 FA 融合算子。

## 6 未闭合项

1. **结构对账**：本仓每样本 31.53 GFLOP（CPU 探针），论文 Table 2 折算 1.29 GFLOP/sample ⇒ 24.5×；pyramid 开关比 1.68× vs 论文 3.06×。
   说明本仓只剪 query token、不剪 K/V/FFN，结构与论文不完全一致（README §6）。
2. **Cast 仍 84.8 ms/步（666 次）**，来自 autograd 内部的 dtype 往返；要吃掉需给权重做 bf16 shadow（结构性改动 + 梯度精度语义变化），
   当前上限约 7%，收益已不占优。
3. **FA 反向 158.9 ms/步（7 次）**，按理论 ≈36% 效率，是 cube 里效率最低的一项；进一步要靠 attention 分块/算法级改动。

## 7 产物与复现

```bash
# 档 A（对齐论文 per-GPU batch）
ONETRANS_NS_BMM=1 ONETRANS_FUSED_NORM=1 ONETRANS_CAT_INPLACE=1 \
torchrun --nnodes=2 --node_rank=N --nproc_per_node=8 \
  --master_addr=192.168.0.82 --master_port=296xx scripts/run_onetrans_s.py \
  --device npu --batch-size 2048 --steps 8 --warmup 3 --checkpoint-act --flops-per-sample 31529400000

# 档 B（关重算，每卡 1024×累加 2）
... --batch-size 1024 --grad-accum 2 --steps 8 --warmup 3   # 不加 --checkpoint-act

# profile（rank0，3 个稳态步）
... --profile-dir /tmp/prof --profile-steps 3 [--profile-with-stack]
# 分析
python3 tools/probes/analyze_profile.py results/prof_8die_cat_inplace_kernel_details.csv \
  --step-ms 1111.2 --tflops-per-step 32.29
```

- 原始 profile：`results/prof16_kernel_details.csv`（16 die 基线）、`prof_8die_fused_norm_kernel_details.csv`、
  `prof_8die_cat_inplace_kernel_details.csv`、`prof_8die_fwdonly_kernel_details.csv`
- 逐项实验记录：`docs/OPT_ROUND1_RESULTS.md`；实验方案：`docs/PLAN_OPT_ROUND1.md`；瓶颈分析：`docs/PROFILE_16DIE_BOTTLENECK.md`
- 探针：`tools/probes/{analyze_profile,probe_phase_split,probe_fused_norm,ut_ns_bmm_equiv,ut_rms_dtype_equiv,attribute_by_stack}.py`
