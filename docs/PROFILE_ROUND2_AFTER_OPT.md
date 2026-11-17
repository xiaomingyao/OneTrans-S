# 优化后版本的深度 profiling：瓶颈在哪、下一步能动什么

- 版本：优化后（`ONETRANS_NS_BMM=1` + `ONETRANS_FUSED_NORM=1` + `ONETRANS_CAT_INPLACE=1`）
- 采集：8 die（111 单节点）· rank0 profiler · 3 个稳态步 · `ASCEND_RT_VISIBLE_DEVICES=0..7`
- 对象：**档 B（每卡 1024×2、关重算、全局 batch 16,384@8die）** 与 **档 A（每卡 2048、开重算）** 各一份
- 原始件：`results/prof_8die_trackB_kernel_details.csv`、`results/prof_8die_trackB_operator_details.csv`、
  `results/prof_8die_cat_inplace_kernel_details.csv`（档 A）
- 复算脚本：`tools/probes/analyze_profile_deep.py`（按/算子口径）、`tools/probes/probe_aten_by_phase.py`（py 层计数）

## 0 一句话结论

档 B 的 892.5 ms 里，**cube（FA + MatMul）只占 41.6%（377.6 ms，效率 39.6%）**，
**非算力部分占 58.4%（530.7 ms）**：其中 layout（拷贝/转换/转置）237.9 ms（26.6%）、逐元素 162.2 ms（17.9%）、归约与 norm 71.2 ms（7.8%）。
逐元素与 layout 已接近 HBM 带宽上限（`Add` 平均 215 µs ≈ 2.9 TB/s），**只能靠「消掉/融合」而不是「调优」**；
再做三次零风险小改（掩码缓存、权重影子、输入 bf16）实测全部无效 ⇒ 下一步必须动**布局重排、算子融合或图模式**。

## 1 全景：两档类别构成（ms/步）

| 类别 | 档 A（1,121.7 ms） | 档 B（892.5 ms） | 说明 |
|---|---:|---:|---|
| cube（FA 融合 + MatMul） | 480.7（42.9%） | **377.6（41.6%）** | 档 A 多出的 103 ms 就是重算那一遍前向 |
| vector·逐元素 | 193.0（17.2%） | 162.2（17.9%） | Add/Mul/Silu/Sub，带宽受限 |
| layout·拷贝 | 181.0（16.1%） | 141.4（15.6%） | `InplaceCopy` 系：select/slice/view/cast 的落地 |
| vector·归约与 norm | 85.4（7.6%） | 71.2（7.8%） | RmsNorm/Grad、ReduceSum、grad-clip |
| layout·Cast | 84.8（7.6%） | 64.1（7.1%） | dtype 边界往返（**几乎全是激活**，权重只占 0.3 ms） |
| layout·转置 | 46.1（4.1%） | 32.4（3.6%） | bmm 前的 transpose、FA 附属转置 |
| other | 29.5（2.6%） | 30.7（3.4%） | `zeros_like`、事件等 |
| comm（HCCL） | 13.4（1.2%） | 20.8（2.3%） | 2.3%，且大部分被计算掩盖 |
| layout·拼接/切分 | 7.8（0.7%） | 7.9（0.9%） | cat 已基本被摘掉 |
| **合计** | **1,121.7** | **892.5**（挂钟） | kernel 并集 908.3 ms ⇒ 多流有 ~1.8% 重叠 |

kernel 数：档 A 2,862/步、档 B 4,534/步（多一遍 micro-batch 的 elementwise）。
尺寸分布（档 B）：≥200 µs 的 kernel 只占 28.2% 的数量却吃 **76.7%** 的时间；<20 µs 的占 14.9% 数量只占 0.8% 时间
⇒ **不是「kernel 太碎」问题**（这也是掩码缓存、权重影子都无效的原因）。

## 2 算力账：一遍前向 57%、反向一遍只有 47%

| 量 | 数值 | 依据 |
|---|---:|---|
| 每卡每步算力 | 64.57 TFLOP | 31.529 GFLOP/sample × 2048 sample |
| 单遍理想（432 TF） | 149.5 ms | 峰值口径 |
| 档 B 实际遍数 | 3 遍（前向 1 + 反向 2） | 关重算 |
| 3 遍理想 | 448.5 ms | — |
| **实测 892.5 ms** | **= 理想 3 遍的 50.2%** | — |
| 仅前向（`--no-backward`，no_grad） | **261.7 ms = 单遍 57.1%** | 实测 |
| 反推「反向 1 遍」 | 315.4 ms = 单遍 47.4% | (892.5−261.7)/2 |

结论：**反向比前向每遍贵 20.5%**，而且两者都只到 47%~57%，不是「算 4 遍」的问题（档 B 已关重算），是**每遍内部 40~50% 的时间在做非 cube 的事**。

## 3 算子级归因（从 profiler db 关联 op → device 时间，档 B）

| op | 设备时间 ms/步 | task/步 | 归属判断 |
|---|---:|---:|---|
| `aclnnInplaceCopy` | **173.5** | 411 | 各类 `copy_` 的总和：cast 反传、select/slice 落地、原地写 |
| `aclnnInplaceAdd` | **67.8** | 97 | 梯度累加（DDP bucket / `p.grad.add_`）、残差原地加 |
| `aten::select` | **64.4** | 76 | NS 权重切片 `kqv_w[:12, proj]` 与 NS 输入切片，NPU 上会真落地拷贝 |
| `ToCopyBackward0` | **64.1** | 101 | dtype 边界的反传 cast（bf16→fp32） |
| `aclnnInplaceZero` | 40.0 | 156 | `zero_()` / 掩码与 buffer 初始化（每步重复） |
| `aclnnBatchMatMul` | 45.3 | 49 | NS 专有 QKV/FFN 的 bmm |
| `TransposeBackward0` | 36.9 | 33 | bmm/FA 前的 transpose 反传 |
| `aten::slice` + `SliceBackward0` | 34.0 + 32.2 | 73 + 69 | 去 cat 的原地写与其反传切片 |
| `aten::addmm` + `AddmmBackward0` | 28.4 + 25.4 | 22 + 22 | seq 主干 Linear |
| `aten::linear` | 24.4 | 41 | tokenizer / head |
| `ViewBackward0` | 22.1 | 53 | view 反传（split_heads 等） |
| `aclnnReduceSum` + `aclnnMuls` | 20.8 + 12.3 | 42 + 38 | **grad-norm 裁剪** |
| `npu_rms_norm` + `RmsNorm` | 17.6 + 13.9 | 5 + 6 | 融合 norm（已优化） |
| `aclnnArange` | 12.1 | 10 | 掩码构造（缓存后无收益，见 §5） |

## 4 各类效率核算（档 B）

| 部分 | 算力/时间 | 效率 |
|---|---|---|
| cube 全体 | 64.57 TFLOP / 377.6 ms = **171 TF/s** | 峰值 432 TF 的 **39.6%** |
| FA 前向（14 次/步） | 12.23 TFLOP / 54.1 ms = 226 TF/s | **52.3%** |
| FA 反向（14 次/步） | 24.46 TFLOP / 158.7 ms = 154 TF/s | **35.7%** ← cube 里最差 |
| MatMul（`Mm` 109.0 + `Addmm` 52.0） | 27.88 TFLOP / 161.0 ms = 173 TF/s | **40.1%** |
| 逐元素（举例 `Add`） | 1024×1203×256 bf16 ≈ 630 MB / 215 µs | ≈ **2.9 TB/s**，已贴 HBM 上限 ⇒ 只能去掉 |

> 口径：attention FLOPs 按 `4·Lq·Lk·d` 累加（pyramid 的 K/V 仍是全长 1203），fwd ≈ 5.97 GFLOP/sample；bwd 按 2× 计。

## 5 本轮又试了 3 个「零风险」候选，全部无效（留档）

| 编号 | 改动 | 结果 | 判定 |
|---|---|---|---|
| E13 | attention 掩码缓存（同形状只算一次，逐位等价） | 档 A 1,111.7 vs 1,112.1（−0.04%）；档 B 892.2 vs 892.5（−0.03%） | 无效 |
| E11 | 权重 bf16 影子（见 `OPT_ROUND1_RESULTS.md` §11） | −0.15% / +0.06% | 无效 |
| E9/E10 | `--input-dtype bf16` | −0.2% / 0.0%，只省 0.5 GB 显存 | 无效 |

**共同教训**：这三项都是从「kernel 数量 / 小 kernel 太多」出发的，而实测 <20 µs 的 kernel 只占 0.8% 时间、掩码类 arange 只值 12 ms。
**瓶颈在「大张量的搬运与格式转换」和「cube 内效率」，不在小算子数量。**

## 5.1 按 §6 的清单又验了 3 项（结论：全部不可用）

| 编号 | 对应方向 | 做法 | 结果 | 判定 |
|---|---|---|---|---|
| E14 | D1 融合残差加 + rms_norm | `torch_npu.npu_add_rms_norm(x1, x2, w, eps)` | **该版本 torch_npu 没有注册 autograd kernel**：前向可用，但 dW/dX 全是 `None`（只报 warning，静默错梯度） | **不可用（危险）** |
| E15a | D2 grad/桶拷贝 | DDP `gradient_as_bucket_view=True` | 档 B 891.2 vs 892.5（−0.15%，抖动内） | 无效 |
| E15b | D5 图编译（轻量版） | `torch.npu.set_compile_mode(jit_compile=True)` | 档 B 892.6 vs 892.5（+0.01%） | 无效 |

E14 的细节值得留档：`npu_add_rms_norm` 返回三元组 `(y, rstd, x_sum)`，前向与 `npu_rms_norm(a1+a2)` 相比 max rel 8.8e-3（说明它的 RMS 在 bf16 域内算，比现在的 fp32 域差一个量级），
且反传三个输入梯度全为 `None`——**如果只在训练脚本里"顺手"换上去，loss 会照常下降但梯度是错的**，属必须避免的坑。

## 5.2 路线 A（图模式）实测：两条路都被卡住

| 编号 | 尝试 | 结果 |
|---|---|---|
| E16 | `torch.compile(model, backend="inductor")` | **编译失败**：`InductorError: NoTritonConfigsError — No valid triton configs`（Triton codegen 在 FuseKernel 的 mask 分支上找不到合法 config） |
| E17 | ACL 图捕获 `torch.npu.NPUGraph` 整步 capture + replay | **捕获失败**：batch 1024 时 `OutOfMemoryError capture_end`（68.4/91.75 GiB 已占，捕获还要额外 pool）；降到 batch 512 后换错：`rtStreamEndCapture execution failed, reason=capture model contains a stream that was not joined to the original stream`（步内有算子走内部旁路 stream） |
| E18 | 只捕获**前向**（`no_grad` + `torch.npu.graph`） | 仍失败，同一个 `rtStreamEndCapture`（旁路 stream 出在前向内部，与 DDP/HCCL 无关） |
| E19 | `torch.npu.make_graphed_callables(model, ...)` 捕获 fwd+bwd | **捕获期 OOM**：需 2.78 GiB，只剩 671 MiB（capture pool 使 reserved 已到 89.98/91.75 GiB，且前一次失败留下碎片） |

**图模式的结论是「起不来」，不是「收益小」**：连纯前向都过不了 stream-join 检查，说明是**某个算子内部起了旁路 stream**（嫌疑：FA v4 或 `npu_rms_norm`）。
若真要推，第一步应做「逐算子的最小捕获二分」定位到具体 op，再决定是绕开它还是换实现——这已属专项工程。

顺带得到的 eager 标度（1 die、pyramid 6、关重算）：`batch 512` 前向 no_grad **67.4 ms**、整步 **229.2 ms**；
`batch 1024` 整步 **443.2 ms**（前一次探针）⇒ 512→1024 近似 1.93×，**线性标度良好**，说明没有隐藏的固定开销拐点。

**附带得到的交叉验证（很有价值）**：单 die、pyramid 6 层、batch 1024（单 micro-batch）、关重算的 eager 稳态步时 = **443.2 ms**；
翻倍到 1024×2 即 ≈886 ms，与 16 die 实测 **892.5 ms** 只差 **+0.7%** ⇒ 跨机 DDP + allreduce 的净开销确实 ≈0（与档 A 的 +0.09% 结论一致）。

⇒ 路线 A 在这套软件栈上**不是「调个开关」能拿到的**：要可用图模式，需要先把步内算子改成可捕获（FA/HCCL 的内部 stream、capture 期内存池），属专项工程。

## 5.3 低风险候选 (a)：FA 内置 causal 覆盖 `q<k` 的层 —— 不可行

动机：G1 只覆盖了 `q_len == key_len` 的 2 个 block；剩余 5 个 `q<k` 的 block 仍在建整张 bool mask。
想法：直接调 `torch_npu.npu_fusion_attention(..., sparse_mode=3)`（右下对齐三角）绕开 mask 张量。

实测（`tools/probes/probe_fa_sparse_mode.py`，B512·H4·Lq480·Lk1203·D64·bf16）：

| 调用 | vs 底右 mask | vs 左上 mask | vs **无 mask** |
|---|---:|---:|---:|
| `sparse_mode=2` | 3.2e-1 | 3.2e+0 | **0.000** |
| `sparse_mode=3` | 3.2e-1 | 3.2e+0 | **0.000** |

`Lq==Lk` 时同样与「无 mask」逐位相同 ⇒ **`atten_mask=None` 时 `sparse_mode` 被忽略**，算子退化成全注意力。
所以「用内置 causal 省掉 `q<k` 层的掩码」这条路不通；G1 已经是这条思路能拿到的全部（`q==k` 层，−5.1%）。
（复现要点：关键字是 `atten_mask`，不是 `attn_mask`；后者会直接报 `Unknown keyword argument`。）

补充两条实测（同一探针）：① 只要给了 `atten_mask`，`sparse_mode=3` 直接报 `error code 161001`（该形状不合法），
只有 `sparse_mode=1` 能跑；② **FA 的 `atten_mask` 语义是「True = 屏蔽」，与 PyTorch SDPA 的「True = 保留」相反** ——
传反了输出 max|Δ| = 4.531（全错），改成屏蔽语义后 **max|Δ| = 0.000（逐位等价）**、`dk/dv` 反向也逐位相同。
这条语义反了，是 `BSH` 布局能落地的关键（见 §6 方向 1：fwd 2.215 → 1.513 ms，−31.7%）。

## 6 可优化方向（按 收益 ÷ 风险 排序）

| # | 方向 | 证据 | 预计收益 | 风险 |
|---|---|---|---:|---|
| ~~D1~~ | ~~融合残差加 + rms_norm~~ | `Add` 101.3 ms/471 次 | **实测不可用**：`npu_add_rms_norm` 无 autograd kernel，梯度为 None（E14） | 已否决 |
| **D2** | grad 桶/裁剪相关拷贝 | `InplaceAdd 67.8` 含梯度累加 | DDP bucket view 实测 −0.15%（E15a）；`ReduceSum/Muls` 经查**不是 grad-clip**（本仓无 `clip_grad_norm_`），来源未定位 | 低优先 |
| **D3** | **NS 权重预打包，去掉每步 `select`+`transpose`+copy** | `select 64.4 + TransposeBwd 36.9 + InplaceCopy 中的一部分 ≈ 100 ms` | **−50~70 ms（5.6~7.8%）** | 中（要改参数布局与 state_dict 兼容） |
| **D4** | **FA 反向效率 35.7% → 45%**（分块/减少 softmax 重算） | 158.7 ms/14 次，理论 56.6 ms | **−30~45 ms（3.4~5.0%）** | 高（算法级，要改注意力实现） |
| ~~D5~~ | ~~完整图模式~~ | — | **实测被卡**： inductor 无可用 Triton config（E16）、ACL capture 内部 stream 未 join + 捕获期 OOM（E17） | 需专项工程，非开关级 |
| **D6** | **结构对齐论文**（现在只剪 Q，不剪 K/V/FFN；每样本 31.53 GFLOP vs 论文折算 1.29） | FLOPs 差 24.5× | 量级级收益 | 极高（等于改写模型，且会改变与论文的对比口径） |

**建议顺序**：D1 → D2（两项都是低风险、合计约 5~6%，一两轮就能验完）→ D3（唯一能砍 layout 三分之一的中风险项）→ D5（唯一有两位数百分比空间的项，但必须先在 1 die 上验图捕获与精度）→ D4/D6 单独立项。

## 7 复现

```bash
# 档 B 全步 profile（8 die，3 稳态步）
ONETRANS_NS_BMM=1 ONETRANS_FUSED_NORM=1 ONETRANS_CAT_INPLACE=1 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 scripts/run_onetrans_s.py \
  --device npu --batch-size 1024 --grad-accum 2 --steps 6 --warmup 2 \
  --flops-per-sample 31529400000 --profile-dir /tmp/pB --profile-steps 3

# 仅前向（no_grad；用于把 fwd / bwd 摊开）
... --no-backward                      # 若不加 --no-backward 里的 no_grad，2 个 micro-batch 的图不释放会 OOM

# 分析
python3 tools/probes/analyze_profile_deep.py results/prof_8die_trackB_kernel_details.csv \
  --operator results/prof_8die_trackB_operator_details.csv --step-ms 892.5 --per-card-batch 2048
```


---

## 7 追加（2026-11-17）：在 G1+G2+G3 配置下重抓的 profile —— 「切片+散播」链已基本消失

16 die、当前最优配置（`SDPA_CAUSAL + FULL_PROJ + FA_BSH`）、3 稳态步：**墙钟 730.3 ms，kernel 并集 725.8 ms，4,254 kernel/步**。
与只用 G1 时（墙钟 848.9 ms）逐项对比：

| kernel | G1 | G1+G2+G3 | 变化 |
|---|---:|---:|---:|
| `InplaceCopy_Slice` | 46.8 (190) | **5.4 (106)** | **−41.4 ms** |
| `InplaceCopy_ViewCopy` | 45.4 (318) | **4.9 (250)** | **−40.5 ms** |
| FA 前向/反向附属 `Transpose` | 14.1 + 17.9 | 6.4 + 8.4 | −17.2 ms |
| `InplaceCopy_Transpose` | 23.3 (334) | 13.3 (314) | −10.0 ms |
| `InplaceZero` | 30.7 (588) | 22.8 (588) | −7.9 ms |
| `InplaceCopy_TensorMove` | 24.9 (232) | 28.7 (244) | +3.8 ms |
| `InplaceCopy_Cast` | 57.8 (924) | 58.0 (948) | +0.2 ms |
| `Add`（InplaceAdd 85.1 + Add 16.7） | 101.4 (471) | 101.8 (471) | +0.4 ms |

⇒ 原先估的「切片 + 散播 ≈110 ms」中约 **−100 ms 已由 G2（全量投影）吃掉**，剩下的分摊在别处。

**新 profile 的构成（ms/步）**：FA 反向 131.4 ｜ Mm 109.9 ｜ **Add 101.8（InplaceAdd 85.1/373 + Add 16.7/98）** ｜ Cast 58.0（948）｜
Addmm 52.7 ｜ FA 前向 41.0 ｜ RmsNorm 36.1+14.9 ｜ Silu 33.9+22.8 ｜ TensorMove 28.7 ｜ Zero 22.8（588）｜
ReduceSum 20.6（158）｜ Transpose 13.3 ｜ FA 附属转置 14.8 ｜ Slice 5.4 ｜ ViewCopy 4.9。

**归因（三支探针的结论，不是推测）**

1. **Python 层零来源**：`tools/probes/probe_op_callers.py` 显示每步 Python API 只发 `to` 14 次（RMSNorm 每次把 fp32 权重 cast 成 bf16）、
   `narrow` 6、`contiguous` 3 → **`zeros / zero_ / add_ / sum / slice_backward` 全部来自 C++ autograd 内部**（CopySlices、
   slice_backward、梯度累加、DDP 桶），改 Python 调用点无效，只能改图结构。
2. **形状归因**（`tools/probes/probe_cpu_shapes.py`，CPU 侧 aten + `record_shapes`）：
   - `slice_backward` 102 次里 **70 次是 12 行形状**（`[12,3,256,256]`、`[12,1536,256]`、`[12,256,1536]`、`[12,128,256]` 等）
     ⇒ 切片成本主要来自 **NS（12 token）权重/激活**，不是 q 的大切片（大切片每种只有 1 次）。
   - `copy_` 832 次里 `[128,12,256]`（133）、`[12,128,256]`（84）、`[12,256,256]`（63）同样以 NS 形状为主。
   - `_to_copy` 448 次里 **属于 NS 形状的约 220 次**（`[12,128,256]` 84、`[12,256,256]` 42、`[12,128,1536]` 21、`[12,256,1536]` 14、
     `[12,1536,256]` 14 …），其余是 `[256]` 权重 70、`[256,256]` 56 与 seq 激活（`[128,1203,256]` 24 等）。
3. 结论：**剩余开销分两类** ——（a）NS 路径的几百个小 kernel（每次几 µs，量大）；（b）seq 路径的大激活 cast（每次数百 µs，约 145 次、合计 ≈58 ms）。

**下一批候选（按 机制确定度 × 收益 排序）**

| 编号 | 改动 | 机制 | 预估 |
|---|---|---|---|
| O1 | 除 norm 外全线显式 bf16（干路激活不再走 autocast 边界），权重另存 fp32 master 供优化器 | 直接消灭 ~145 次大激活 cast（58 ms 中的大头） | −40~55 ms（5~7%），精度需门禁 |
| O2 | NS 权重按组拆成独立 Parameter（13×3 与 FFN 两组），彻底不切参数 | 消灭 12 行形状的 slice_backward/zeros/copy（~70+ 次） | −15~25 ms（2~3%），改 state_dict 命名 |
| O3 | 减少梯度累加节点（accum 2→1） | InplaceAdd 85 ms 里 162 次来自 2 个 micro-batch 的累加 | 需 per-GPU 2048 + 重算 = 档 A，整体更慢，否决 |


### 7.1 Cube 侧三个杠杆的实测（全部否决）

`tools/probes/probe_cube_levers.py`，单 die、当前配置（bf16 干路）：

| 杠杆 | 实测 | 判定 |
|---|---|---|
| **qkv 投影合并**：同输入上 3 次 `linear(256→256)` → 1 次 `linear(256→768)` | (B1024·L1203) 1.244 → 1.136 ms（**−8.7%**），输出 **逐位相同** | 单算子有效，但每步 qkv 总时长仅约 **5.0 ms**（fwd），折算每步只省 **≈0.44 ms（0.07%）** ⇒ 不值得改 |
| **FA `inner_precise=1`**（高性能模式） | fwd 2.999→2.995 ms、fwd+bwd 14.766→14.769 ms | **完全无差别** |
| **mask dtype bool → uint8** | fwd 2.996 ms、fwd+bwd 14.764 ms | **完全无差别** |

同时算出 FA 的实际效率：单次 fwd+bwd（B1024·H4·Lq480·Lk1203·D64）算力 ≈1.06 TFLOP / 14.77 ms
⇒ **71.8 TF/s = 432 TF 的 16.6%**。这是厂商融合算子在「带掩码 + q≠k」形状下的实际水平，
没有 API 旋钮能改；要提只能自写 AscendC kernel 或走图模式（后者仍卡在旁路 stream 未 join）。

⇒ **cube 侧在当前栈上已无便宜的杠杆**。剩余空间只在这三类里：① 图模式专项（定位算子内部旁路 stream）；
② AscendC 自写 attention/融合算子（工程量大）；③ 结构性减算力（更激进 pyramid / K-V 剪枝，**与论文冲突**）。
