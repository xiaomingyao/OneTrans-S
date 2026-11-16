# 16 卡 OneTrans-S profiling 与瓶颈分析

- 采集：2026-11-16 21:31，122+111 各 8 die，`torchrun --nnodes=2 --nproc_per_node=8`
- 配置：全局 batch 32,768（每卡 2048）、bf16 autocast、bool mask（FAv4）、激活重算、`ONETRANS_NS_BMM=1`
- 工具：`torch_npu.profiler`（rank0 单卡抓，`--profile-dir` 已在 `scripts/run_onetrans_s.py` 内建；`schedule(wait=warmup, active=3)` 只抓 3 个稳态步）
- 同进程非 profile 窗口实测：`steady_median_step_ms = 1443.7`（profile 窗口 1443.2，开销 <0.1%）
- 原始数据：`results/prof16_kernel_details.csv`、`results/prof16_operator_details.csv`、`results/prof16_breakdown.txt`

## 1 全局图景：没有气泡，也没有通信瓶颈

| 指标 | 值（每步） | 读法 |
|---|---:|---|
| wall time | 1,443.2 ms | 三轴对齐档 |
| compute kernel 区间并集 | 1,440.6 ms（**99.8%**） | 设备侧几乎被 kernel 排满，**空隙只有 2.6 ms（0.2%）** |
| 通信 kernel 区间并集 | 15.6 ms（1.08%） | 13 次 HCCL，其中 **69% 被计算掩盖** |
| 通信净暴露 | ≈ 4.8 ms（**0.33%**） | 跨机 allreduce 不是瓶颈 |
| kernel 数 | 3,083 个/步 | 中位数 15.9 µs；**52% 短于 20 µs** |
| MFU（432 TF 分母） | 10.36% | 与 profile 一致 |

结论先说：**瓶颈不在通信、不在空闲，而在 kernel 本身**——设备 99.8% 的时间都在执行 kernel，
其中三分之二不是矩阵乘。

## 2 算子构成（3 个稳态步平均，`ms/步`）

| 类别 | ms/步 | 占比 | kernel/步 | 其中 <20 µs |
|---|---:|---:|---:|---:|
| vector · 逐元素（Mul/Add/Silu/Pow/…） | 443.9 | 30.3% | 838 | 55% |
| cube · FA 融合（含其前后 Transpose） | 270.8 | 18.5% | 28 | 4% |
| cube · MatMul（Mm + Addmm） | 213.4 | 14.6% | 306 | 30% |
| layout · Cast | 185.4 | 12.7% | 726 | 66% |
| layout · 切片/拷贝（Slice/ViewCopy/Zero） | 146.5 | 10.0% | 496 | 66% |
| layout · 转置 | 78.0 | 5.3% | 257 | 27% |
| layout · 拼接（Cat/ConcatD） | 52.1 | 3.6% | 47 | 0% |
| vector · 归约（ReduceSum/LogSoftmax） | 45.6 | 3.1% | 157 | 38% |
| comm | 15.7 | 1.1% | 13 | 0% |
| other | 11.3 | 0.8% | 213 | 58% |
| norm 融合算子 | 0.0 | 0.0% | 2 | — |
| **合计** | **1,462.9** | 100% | **3,083** | — |

- **cube 类合计 484.2 ms/步（33.1%）**；同口径理论 cube 时间 **149.5 ms/步**（64.57 TFLOP ÷ 432 TF）⇒ **cube 内部效率 30.9%**。
- **非 cube 类合计 978.6 ms/步 = 66.9%**。

### 单算子 TOP15（ms/步）

| 算子 | ms/步 | n/步 |
|---|---:|---:|
| `aclnnFlashAttentionScoreGradV4` | 158.8 | 7 |
| `aclnnMul_MulAiCore` | 151.6 | 140 |
| `aclnnInplaceCopy_CastAiCore_Cast` | 143.6 | 677 |
| `aclnnAdd_AddAiCore` | 130.8 | 251 |
| `aclnnFlashAttentionScoreV4` | 108.1 | **14** |
| `aclnnMm_MatMulV3Common` | 106.2 | 84 |
| `aclnnAddmm_MatMulV3Common` | 102.8 | 82 |
| `aclnnInplaceCopy_SliceAiCore` | 84.2 | 122 |
| `aclnnInplaceCopy_TransposeAiCore` | 77.0 | 211 |
| `aclnnCat_ConcatD` | 52.1 | 47 |
| `aclnnSilu_SiluAiCore` | 45.1 | 46 |
| `aclnnCat_CastAiCore_Cast` | 41.8 | 48 |
| `aclnnSiluBackward_SiluGradAiCore` | 34.4 | 27 |
| `aclnnInplaceCopy_ViewCopyAiCore` | 32.9 | 99 |
| `aclnnReduceSum_ReduceSumOpAiCore` | 32.2 | 107 |

## 3 瓶颈排序（每条都带证据）

**B1. layout 与逐元素搬运吃掉 1/3 的步时（Cast + 切片 + 转置 + 拼接 = 462 ms，32.0%）**
- Cast **677 次/步**、143.6 ms，其中 **66% 的 Cast 短于 20 µs** ⇒ 典型的 autocast 边界小算子洪流，不是有效计算。
- Slice 122 次/步（84.2 ms）+ ViewCopy 99 次/步（32.9 ms）+ Cat 47 次/步（52.1 ms）⇒ pyramid 每层「切子序列 → 算完 → 拼回去」的内存来回；`Cat_Cast` 那 48 次/步（41.8 ms）说明拼完还要再转精度。
- Transpose 211 次/步（77.0 ms）⇒ 进 FA 前的 Q/K/V 布局变换，与 FA 自身附属的 49 次转置叠加。

**B2. cube 类只跑到峰值的 30.9%（484.2 ms vs 理论 149.5 ms）**
- 其中 FA 前向 **14 次/步**（= 7 个 block × 激活重算的第二次前向），FA 反向 7 次/步；matmul 306 次/步（同样含重算翻倍）。
- 单个 FA 反向 22.7 ms、前向 7.7 ms/次（batch 2048）；batch 16 时前向仅 0.49 ms ⇒ 放大 128 倍 batch 只摊到 15.7 倍时间，说明**大 batch 下 FA 反而更“值”**，但反向仍是 attention 的大头。

**B3. RMSNorm 是手写链，没走融合算子（≈43 ms/步 纯加性开销）**
- `Square 27.9 + Pow 14.4 + Rsqrt 0.3` = **42.6 ms/步、n=28/步**（28 = 7 block × 2 norm × 2 次前向），另有 ReduceSum/Mul 中的对应份额。
- 容器内 `torch_npu` 已有 `npu_rms_norm` / `npu_rms_norm_backward` / **`npu_add_rms_norm`（残差加 + norm 一起融合）**，本仓一个都没用。

**B4. 激活重算把前向整体算了两遍**
- 直接证据：FA 前向 14 次/步 vs 7 个 block；matmul 306 次/步（84 Mm + 82 Addmm 的核心数 ×2 量级）。
- 代价可量化（1 die · batch 1024 同形状）：不重算 837.0 ms → 重算 966.4 ms，**+15.5%**；2048 档不开重算会 OOM，所以这是显存换时间的既定选择。

**B5. kernel 碎片化（3,083 个/步）**
- 因为 compute 并集已占 99.8%，**没有可被掩盖的 slack**：每一个 <20 µs 的小 kernel 的下发/调度间隙都直接落在关键路径上。
- 排序结论：想提 MFU，必须**减少 kernel 数**（融合/合并/去掉拷贝），而不是"提高并行度"。

## 4 优化候选（按预估收益/风险排序，收益为上限估算，须实测）

| # | 措施 | 预估 | 准入检查 |
|---|---|---:|---|
| 1 | 手写 RMSNorm → `torch_npu.npu_add_rms_norm`（残差+norm 融合，反向用 `npu_rms_norm_backward`） | 60~80 ms/步（4~5.5%），kernel 数 −200 级 | 数学等价（同 eps、fp32 内部计算），需 UT 对 loss/梯度 |
| 2 | 权重/常量的 bf16 cast 一次性缓存（autocast 每步重复 cast 同一个权重） | 30~60 ms/步（2~4%） | 不改 dtype 语义（cast 结果与现状逐值相同），需 UT |
| 3 | pyramid 切片/拼接改 view 或原地写（避免每层 materialize 子序列） | 上限 150~200 ms/步（10~14%），风险高 | 依赖 FA/线性对非连续输入的支持，要先小规模验证 |
| 4 | `find_unused_parameters=False`（profile 已证本步无未用参数） | host 侧小幅，未计 | 零风险，bmm 模式成立 |
| 5 | 选择性重算（只重算 FFN，不重算 attention） | 取决于显存，未计 | 显存优先，需逐档试 |
| 6 | 通信侧 | **无收益空间**（净暴露 0.33%） | — |

## 5 复现

```bash
# 16 卡 + rank0 profile（3 个稳态步）
ONETRANS_NS_BMM=1 torchrun --nnodes=2 --node_rank=N --nproc_per_node=8 \
  --master_addr=192.168.0.82 --master_port=29571 scripts/run_onetrans_s.py \
  --device npu --batch-size 2048 --steps 6 --warmup 2 --checkpoint-act \
  --flops-per-sample 31529400000 --profile-dir /tmp/prof16 --profile-steps 3

# 解析（本文件第 1~2 节的表就是它出的）
python3 tools/probes/analyze_profile.py results/prof16_kernel_details.csv \
  --step-ms 1443.7 --tflops-per-step 64.5722
```

> 口径纪律：profile 窗口 ≠ 稳态（本次两者差 <0.1%，可互证）；本页所有百分比都是**同一份 CSV 内部**的相对占比，
> 不与论文 H100 数字相除（论文 407 ms 是 fp32 未优化档）。
