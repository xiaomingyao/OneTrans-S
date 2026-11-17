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


## 8. 图模式排查（2026-11-17，结论：两条路都要改代码，且上限只有 ~0.8% 的 launch 收益）

### 8.1 手动 `torch.npu.graph` 捕获 —— attention 算子全部不可捕获

`tools/probes/probe_graph_op_bisect.py`（每个算子单独一个进程，避免失败后上下文损坏）：
同一进程、同一机器、同一 CANN 下，**非 attention 算子全 OK，attention 全 FAIL**。

| 算子 | capture 结果 |
|---|---|
| `F.linear` bf16 | **OK** |
| `npu_rms_norm`（融合） | **OK** |
| 手写 rmsnorm（cast 链） | **OK** |
| `npu_add_rms_norm`（融合） | **OK** |
| `silu` / `mean→fp32` / `zeros(分配)` | **OK** |
| `SDPA is_causal`（q==k） | **FAIL** `capture_end ... 107025` |
| `SDPA 显式掩码`（q≠k） | **FAIL** 同码 |
| `npu_fusion_attention` BSH 前向（带掩码） | **FAIL** 同码 |
| `npu_fusion_attention` 前向+反向 | **FAIL** 同码 |

`tools/probes/probe_graph_capture_modes.py` 把 `torch.npu.graph` 的另两个参数也试了：

| capture 参数 | 结果 |
|---|---|
| 默认 | `107025` |
| `auto_dispatch_capture=True` | `captures_underway.empty() INTERNAL ASSERT FAILED`（NPUCachingAllocator.cpp:2933） |
| `capture_error_mode="thread_local"` | 同上 assert |
| 两者同开 | 同上 assert |

⇒ **旁路 stream 在 attention 算子的 kernel 实现内部**（不是我们的代码、不是 DDP/HCCL），
本版本的 `torch.npu.graph` 无法把 `aclnnFlashAttentionScore*` 的旁路流 join 回来；
换 capture 模式只会撞到 allocator 的内部 assert。这条路线要通，只能不用融合 attention 算子。

### 8.2 TorchAir (`torch.compile(backend="npu")`) —— 编译期就失败，但**错在我们自己的写法**

`tools/probes/probe_torchair_bisect.py`（B64、静态 shape、粒度二分）：

| 粒度 | 结果 |
|---|---|
| `CausalMaskAttention` | **FAIL** — Dynamo: `failed to run FX node with fake tensors: getitem(FakeTensor(size=(1203,256)), (slice(None), slice(244, None)))` |
| `OneTransBlock` | **FAIL** — `EZ9999`，GE `[InferShape] The k-axis of a(1191) and b(256) must be the same` |
| `MultiOneTransBlock` | **FAIL** — 同上 |
| 整模 | **FAIL** — 同上 |

`1191 = 1203 − 12`，说明 GE 侧把「去掉 NS 12 行后的序列维」当成了某个 matmul 的 k 轴。
根因指向我们为 G2/G3 写的写法对图编译器不友好：`out[:, seq_len:, :] = torch.bmm(...).permute(1,0,2)`（原地写 + permute + 切片混合），
Dynamo 在 `x[:, seq_len:, :]` 的 fake-tensor getitem 上就失败。
⇒ 这条路**不是平台打不开**，而是需要把投影/掩码写法改成 trace 友好（eager 下等价、图下可推形状）的版本。

### 8.3 收益上限：先量气泡，再决定值不值得改

O1 配置、单 die、batch 1024、profile 2 个稳态步：

| 项 | 数值 |
|---|---|
| 稳态步时（墙钟） | **306.3 ms** |
| kernel 时长总和（2 步） | **607.7 ms** ⇒ 每步 **303.9 ms** |
| **气泡** | **≈2.4 ms / 步 = 0.8%** |

⇒ 图模式**单靠消 launch 开销最多只能拿 0.8%**；它的真正价值在 GE 融合（`Add/Silu/RmsNorm/Transpose`
这类小算子每步合计仍有 ~100 ms 级）。但要拿到融合，得先过 8.2 那道编译关。

**O1 后的 kernel 构成（单 die，2 步合计，注意与 §7 的 16 die 数据不同口径）**：
FA 反向 131.6（14 次）｜Mm 108.6（152）｜Add 56.5（398）｜Addmm 53.6（78）｜FA 前向 41.0（14）｜
SiluBwd 34.0（54）｜TensorMove 25.2（192）｜RmsNormGrad 24.3（28）｜Silu 22.6（54）｜ReduceSum 20.8（158）｜
RmsNorm 15.6（28）｜Zero 13.8（580）｜Transpose 12.4（276）+ FA 附属 8.4+6.4。
（`Cast` 已跌出前 15 ⇒ autocast 的 cast 链确实被 O1 消灭了。）

### 8.4 结论

| 路线 | 状态 | 需要的动作 | 上限 |
|---|---|---|---|
| 手动 `torch.npu.graph` + 融合 attention | **死**（算子不可捕获） | 换成非融合 attention（性能反而倒退） | — |
| TorchAir `torch.compile(backend="npu")` | **卡在编译期** | 把投影/掩码改成 trace 友好写法（开发任务） | 0.8%（launch）+ GE 融合的未知增量 |
| eager（现状） | 可用 | 继续做小算子级优化 | — |


## 9. 用 Triton 验证「自写 attention / 融合 kernel」有没有戏 —— **没有**（2026-11-17）

### 9.1 先确认 Triton-Ascend 的能力下限

`tools/probes/probe_triton_capability.py`（本机 `triton 3.2.0` + `triton_ascend 3.2.1.dev20260530`）：

| 能力 | 结果 |
|---|---|
| `tl.dot`（128³ bf16） | **OK**，max|Δ|=7.6e-6 vs torch |
| 标量参数（float） | **OK** |
| 循环 + `tl.trans` + online softmax 片段 | 能跑（数值未对齐，见下） |

⇒ 基础设施够用，可以做一次「同工作量」的前向对比。

### 9.2 前向对比：Triton 版 flash-attention vs 厂商 FA v4

`tools/probes/probe_triton_attn_fwd.py`，形状取本仓真实值（BH=batch×heads=1024，D=64，底右因果掩码）。

| 形状 | Triton fwd | FA v4 fwd | 比值 |
|---|---:|---:|---:|
| Lq=480, Lk=1203 | 9.14 ms | 2.97 ms | **3.08×**（+207%） |
| Lq=1203, Lk=1203 | 17.31 ms | 7.47 ms | **2.32×**（+132%） |

tile 调参（BH=512）：

| BM×BN | 480×1203 | 1203×1203 |
|---|---:|---:|
| 128×64 | 3.12× | 2.66× |
| 64×64 | 3.82× | 2.91× |
| 64×128 | 3.08× | 2.32× |
| 128×128 | 编译/执行失败 | 编译/执行失败 |

**最好的配置也慢 2.3–2.7×**，而且这只是**前向**；FA 的反向是前向的 3 倍耗时（单 die 每步 fwd 20.5 ms / bwd 65.8 ms），
自写还要再实现 dq/dk/dv 三个 kernel 并复现同样效率。

**过程中的一个自纠**：第一版 kernel 我把「底右因果」的 n_start 写成了 `pid_m·BM + (Lk−Lq)`，
等于把**每一行都能看到的前段列**当成掩码跳过了 —— 结果「快 29~51%」是假的（工作量少了），
修正成 `n_start=0`、只跳尾部后，真实结果是慢 2.3–3.1×。这类「跳错方向」的 bug 只会表现为「又快又错」，
以后再看到异常加速，先核对跳过的区间是不是掩码方向反了。

### 9.3 结论

| 候选 | 判定 | 依据 |
|---|---|---|
| 自写 attention（Triton 验证线） | **否决** | 前向就慢 2.3–3.1×；反向未写，且厂商反向占本步 65.8 ms/步 |
| 自写 attention（AscendC 硬写） | **不建议** | 要在 Cube 上追平厂商融合算子（带掩码 + q≠k + 反向三件套），工程量与风险都远超本项目目标 |
| 自写融合小算子（add+rmsnorm 已试） | **否决** | §5.2：fwd +5.3%、bwd −0.9%，已被 HBM 带宽打满 |

⇒ **「自写 kernel」这条路整体不推荐**。剩余可动的只有：结构性减算力（与论文冲突，需你决策）、
或把 TorchAir 那条图编译路打通（§8.2，要改 trace 友好写法，上限 0.8% launch + GE 融合增量）。


## 10. TOP-DOWN 耗时分解与利用率（当前最优配置：G1+G2+G3+O1，16 die，档 B，batch 2048/卡/步）

数据源：`results/prof16_o1_kernel_details.csv`、`results/prof16_o1_step_trace_time.csv`（3 个稳态步）；
前向口径另跑 `--no-backward`（164.9 ms/步墙钟，3 步）；家族级"前向/反向"由两次 profile **相减**得到。

> 方法说明（重要）：先用 `record_function` 标记 + "kernel 起始时间落在谁区间内"做归属，**结果是错的**
> （`P_loss` 被算成 337 ms）——因为 CPU 侧标记是**入队窗口**，设备 kernel 会在后面的
> `loss.item()` 同步时才真正执行，两者时间轴不对齐。改用**差分法**（全步 − 前向-only）才得到下面的可信数字。

### 10.1 第一层：步时总账（官方 step_trace_time，3 步均值）

| 项 | ms/步 | 占步时 |
|---|---:|---:|
| **Stage（步时）** | **614.27** | 100% |
| Computing（设备计算） | **605.47** | 98.6% |
| Communication（AllReduce 合计） | 19.47 | 3.2% |
| ├─ 其中被计算掩盖 | 14.80 | — |
| └─ 其中暴露（Not Overlapped） | 4.67 | 0.8% |
| Free（气泡） | 4.13 | 0.7% |

⇒ **这个步已经没有调度/气泡/通信可挖**：三者合计只占 1.5%。要快只能砍"计算"本身。

### 10.2 第二层：前向 vs 反向（差分，设备 kernel）

| 阶段 | ms/步 | 占设备 kernel | 说明 |
|---|---:|---:|---|
| 前向 | **164.2** | 25.8% | 含 2 个 micro-batch |
| 反向 + 优化器 + 通信 | **471.9** | 74.2% | 含激活重算（checkpoint 的那一遍前向） |

反向是前向的 **2.87×**，其中一部分是重算（重算量与一次前向同量级）——这也是为什么档 A（开重算）
比档 B 慢 22%。

### 10.3 第三层：算子家族分解（ms/步，%对 636.1 ms 的 kernel 总时长）

| 家族 | 全步 | 前向 | 反向 | 占全步 |
|---|---:|---:|---:|---:|
| **GEMM（Mm/Addmm/Matmul）** | **176.1** | 58.9 | 117.2 | **27.7%** |
| **FA 反向 + 附属 Transpose** | **139.7** | — | 139.7 | **22.0%** |
| Add/Sub | 57.7 | 12.4 | 45.3 | 9.1% |
| SiLU（前向 Swish + 反向 SiluGrad） | 56.5 | 22.5 | 34.0 | 8.9% |
| 搬运/重排（TensorMove/ViewCopy/Transpose/Slice/Cat） | 46.3 | 5.5 | 40.8 | 7.3% |
| FA 前向 + 附属 Transpose | 47.4 | 47.5 | — | 7.5% |
| RMSNorm（+Grad） | 40.1 | 15.6 | 24.5 | 6.3% |
| 归约（ReduceSum） | 20.8 | — | 20.8 | 3.3% |
| 通信 AllReduce | 19.5 | — | 19.5 | 3.1% |
| Zero（ZerosLike） | 14.0 | — | 14.0 | 2.2% |
| 其他 | 13.9 | 0.1 | 13.8 | 2.2% |
| Cast | 4.1 | 1.7 | 2.4 | 0.6% |

**前向 164.2 的内部构成**：GEMM 58.9（35.9%）｜FA 47.5（28.9%）｜SiLU 22.5（13.7%）｜RMSNorm 15.6（9.5%）｜
Add 12.4（7.6%）｜搬运 5.5（3.3%）｜Cat+Cast 3.4（2.1%）

**反向 471.9 的内部构成**：FA 反向 139.7（29.6%）｜GEMM 117.2（24.8%）｜Add 45.3（9.6%）｜
搬运 40.8（8.7%）｜SiLU 反向 34.0（7.2%）｜RMSNormGrad 24.5（5.2%）｜ReduceSum 20.8（4.4%）｜
通信 19.5（4.1%）｜Zero 14.0（3.0%）｜其他+Cast 16.2（3.4%）

### 10.4 第四层：TOP 单算子（ms/步）

| 算子 | ms/步 | 次数/步 | 平均/次 | 归类 |
|---|---:|---:|---:|---|
| `FlashAttentionScoreGradV4` | **131.3** | 14 | 9.4 ms | cube |
| `Mm_MatMulV3` | **108.6** | 152 | 714 µs | cube |
| `Add_AddAiCore` | 56.9 | 479 | 119 µs | vector |
| `Addmm_MatMulV3`（linear+bias） | 53.6 | 78 | 688 µs | cube |
| `FlashAttentionScoreV4` | 41.0 | 14 | 2.9 ms | cube |
| `SiluBackward_SiluGrad` | 33.9 | 54 | 628 µs | vector |
| `InplaceCopy_TensorMove` | 26.0 | 236 | 110 µs | vector/搬运 |
| `RmsNormGrad` | 24.5 | 28 | 874 µs | vector |
| `Silu_Swish` | 22.6 | 54 | 418 µs | vector |
| `ReduceSum` | 20.8 | 158 | 132 µs | vector |
| `RmsNorm` | 15.6 | 28 | 556 µs | vector |
| `InplaceZero_ZerosLike` | 14.0 | **580** | 24 µs | vector |
| `InplaceCopy_Transpose` | 11.6 | 241 | 48 µs | 重排 |
| `HcclLaunchAicpuKernel` | 11.2 | 26 | — | 通信 |

### 10.5 利用率（按各自 FLOP 公式算的实测效率，分母 432 TF/s）

| 部分 | 时间 | 该部分 FLOPs/卡/步 | 实测 | 效率 |
|---|---:|---:|---:|---:|
| FA 前向 | 41.0 ms | 7.94 TFLOP | **193.8 TF/s** | **44.8%** |
| FA 反向 | 131.3 ms | 19.86 TFLOP | **151.3 TF/s** | **35.0%** |
| GEMM（前+反） | 176.1 ms | 16.30 TFLOP | **92.6 TF/s** | **21.4%** |
| **cube 合计** | **348.4 ms（54.8%）** | **44.11 TFLOP** | **126.6 TF/s** | **29.3%** |
| 整步（记账口径 64.57 TFLOP） | 614.3 ms | 64.57 TFLOP | 105.1 TF/s | **24.33%（MFU，官方口径）** |

口径提示：上表"该部分 FLOPs"只算矩阵型算力（44.11 TFLOP），比记账口径 31.529 GFLOP/sample ×2048
= 64.57 TFLOP 小约 1.46×（记账把激活/归约等也算进去了）。**两个口径不能相除算收益**。

### 10.6 瓶颈排序（按"可动的期望收益"）

| 名次 | 瓶颈 | 证据 | 可动性 |
|---|---|---|---|
| **1** | **FA 反向 131.3 ms，效率仅 35.0%** | 单算子最大；API 旋钮（`inner_precise`/mask dtype/sparse_mode）已实测无效；Triton 自写慢 2.3–3.1× | **低**（只剩等厂商/换算子） |
| **2** | **GEMM 176.1 ms，效率仅 21.4%** | 152 次 `Mm` 平均 714 µs、78 次 `Addmm` 平均 688 µs；另有 210 次 **18 µs** 的 `BatchMatMul`（NS 12 行路径）+ 142 次 BatchMatMul 附属 Transpose | **中**：形状天生小（M=12 / N=256 / K=256），只能靠"合并成大批"（已在 qkv 合并上验证逐位相同但只省 0.07%） |
| **3** | **向量侧合计 ~240 ms（37.6%）**：Add 57.7｜SiLU 56.5｜搬运 46.3｜RMSNorm 40.1｜ReduceSum 20.8｜Zero 14.0｜Cast 4.1 | 次数极多：Zero 580、Cast 569、Add 479、ViewCopy 250、Transpose 241、TensorMove 236；这些是 autograd/DDP 内部产生的（`probe_op_callers` 已证 Python 层只发 14 次 `to`、6 次 `narrow`） | **中低**：要消只能改图结构（参数合并/去视图/自定义反向） |
| 4 | 通信 19.5 ms（暴露仅 4.67） | step_trace_time | 低（已基本掩盖） |
| 5 | 气泡 4.13 ms | step_trace_time | 无线可挖 |

**结论**：当前是**纯计算受限**（98.6%），其中 cube 侧 348 ms 只有 **29.3%** 效率、向量侧 240 ms 是
memory-bound 的小算子海。要往 427 ms 走（还需 −30.5% ≈ −187 ms），单靠调参没有路径；
要么提高 cube 效率（FA 反向/GEMM 形状），要么把向量侧的 100+ ms 小算子海压掉（改图结构）。


### 10.7 自纠 + 补充：GEMM 效率算错了；向量侧带宽实测；SIMT 评估

#### ① 自纠：10.5 表里 GEMM 的 21.4% 是错的

我在 §10.5 把「GEMM 前向+反向的总时间 176.1 ms」除以「**只算前向**的 FLOPs 16.30 TFLOP」，
得出 21.4% —— **分母少了一半**。线性层反向要算两份梯度（grad_input + grad_weight），FLOPs ≈ 2× 前向：

| 部分 | 时间 | 正确 FLOPs | 正确效率 | §10.5 错值 |
|---|---:|---:|---:|---:|
| GEMM（前+反） | 176.1 ms | 16.30 × 3 = **48.9 TFLOP** | **277.6 TF/s = 64.3%** | ~~21.4%~~ |
| FA 前向 | 41.0 ms | 7.94 TFLOP | 193.8 TF/s = 44.8% | 44.8%（对） |
| FA 反向 | 131.3 ms | 19.86 TFLOP（2.5×） | 151.3 TF/s = **35.0%** | 35.0%（对） |
| cube 合计 | 348.4 ms | **76.7 TFLOP** | **220.1 TF/s = 50.9%** | ~~29.3%~~ |

**修正后的瓶颈排序跟着变**：

| 名次 | 瓶颈 | 证据 | 结论 |
|---|---|---|---|
| **1** | **向量侧 240 ms（39% 步时）** | 不产生任何矩阵算力；把整步效率从 cube 的 50.9% 拉低到 28.9% | 最该动 |
| **2** | **FA 反向 131.3 ms @ 35.0%** | cube 侧最差的一环 | API 无解（已实测） |
| 3 | GEMM 176.1 ms @ **64.3%** | 已接近这个形状的合理水平（tall-skinny，N/K=256） | **不是瓶颈** |

#### ② 向量侧带宽实测（`tools/probes/probe_vec_bandwidth.py`，1024×1203×256 bf16，单张量 602 MiB）

| 算子 | 时间 | 有效带宽 | 判读 |
|---|---:|---:|---|
| SiLU（读1写1） | 0.397 ms | **3178 GB/s** | 已贴 HBM 上限 |
| SiLU inplace | 0.393 ms | 3208 GB/s | 同上 |
| Add 残差（读2写1） | 0.587 ms | **3222 GB/s** | 已贴上限 |
| `npu_rms_norm` 融合 | 0.873 ms | **1445 GB/s** | **只有峰值的 45%**（疑似读两遍：求和 + 归一化） |
| 手写 rmsnorm（fp32 中间量） | 3.959 ms | 319 GB/s | 千万别手写 |

⇒ 向量侧唯一有头的是 **RMSNorm**：全步 40.1 ms（fwd 15.6 + grad 24.5），若做到 SiLU 同级带宽
可压到 ~18 ms，**净收益 ≈ 22 ms（3.6%）**。其余（SiLU/Add）已经打满，只能靠融合减少遍数。

#### ③ 950 SIMT 多线程对这条路的帮助（评估，未实测）

按 SIMT 的定位（逐元素 / 条件分支 / 不规则访问；规则向量与 cube 仍是 SIMD/cube 的活）：

| 目标 | SIMT 有没有用 | 依据 |
|---|---|---|
| FA 反向 131.3 ms（35%） | **没用** | cube 侧；Triton 自写已实测慢 2.3–3.1× |
| GEMM 176.1 ms（64.3%） | **没用** | 形状受限（M=12/256, N=256），不是线程并行问题 |
| 向量侧 240 ms 里的 SiLU/Add | **没用** | 实测已 3.2 TB/s，带宽打满 |
| **RMSNorm 40.1 ms** | **可能有用**：SIMT + `__ubuf__` 缓存，一遍读入算平方和再归一化，省掉第二遍读 | 目标 ~22 ms |
| NS 12 行不规则小算子（210 次 BatchMatMul×18 µs + 58 次 Slice/AsStrided + 580 次 Zero×24 µs） | **可能有用**：小形状 + 零碎调度正是 SIMT 的强项 | 上限 ~10–15 ms（这批合计只有 ~20 ms，且 <100 µs 的 kernel 全步只占 30.4 ms/4.8%） |

**结论**：SIMT 有帮助，但上限 ≈ **30–35 ms（5%）**，且需要新开一条 AscendC SIMT kernel 开发线
（bisheng 编译 + 直调 + 精度门禁）。相对「还要 −187 ms」的目标，它是**必要但不充分**。
优先级：先吃 RMSNorm（收益确定、公式简单、可先 Triton/SIMT 双验证），再考虑 NS 不规则路径。


---

## 11. ReduceSum 21 ms 归因（2026-11-17）—— 结论：是 Linear 的 bias 梯度，且已贴峰值；**我原先的 −20 ms 估算撤回**

### 11.1 先看邻居：它是谁发出的

`results/prof16_o1_kernel_details.csv`（16 die、3 稳态步）里 `ReduceSum` **每步 200 次 / 21.05 ms**
（3 步分别是 21.07 / 21.07 / 21.05，极稳定）。按 `Start Time` 排同 stream 邻居：

```
SiluGrad(3.60) → Mm(2.27) → Mm(2.43) → ReduceSum(1.17) → Add(0.60) → Cast(0.002) → ZerosLike(0.003)
```

这是 `F.linear` 反向的标准形状：`dW = dyᵀx`（Mm）、`dx = dyW`（Mm）、**`db = dy.sum(0)`（ReduceSum）**、
再 `Add` 累加进梯度、`Cast` 把 bf16 梯度落到 fp32 参数。⇒ **ReduceSum 就是 Linear 的 bias 梯度**，
不是第三方来源，也不是 `clip_grad_norm`（全仓没有调用）。

时长谱是双峰：`0–20 µs` 120 次（1.18 ms）+ `20–100 µs` 20 次（1.48 ms）+ `100–300 µs` 48 次（8.82 ms）
+ `300–600 µs` 4 次（1.59 ms）+ `>600 µs` 8 次（7.98 ms）。

### 11.2 隔离探针：它到底慢不慢（`probe_bias_grad_reduce.py` / `…reduce2.py`）

| 形状 | dtype | 时间 | 有效带宽 |
|---|---|---:|---:|
| (1.23 M, 256) | bf16 | 0.198 ms | **3189 GB/s** |
| (1.23 M, 768) | bf16 | 0.590 ms | **3208 GB/s** |
| (1.23 M, 1024) | bf16 | 0.751 ms | **3358 GB/s** |
| (1.23 M, 1024) | fp32 | 1.288 ms | 3919 GB/s |
| (1.23 M, 1024) | bf16 **`sum(0, dtype=fp32)`** | **3.863 ms** | **653 GB/s（5.1× 慢）** |
| (1.23 M, 1024) | bf16 **非连续（转置）** | **12.989 ms** | **194 GB/s（17× 慢）** |
| (1.23 M, 1024) | bf16 | `ones@dy`（cube） | 0.782 ms | 3228 GB/s（**0.96×，无收益**） |

两个结论：

1. **contiguous + 同 dtype 时它已经是峰值带宽**，profile 里 1.17 ms 对应 (1.23 M, 1024) fp32 量级，
   ⇒ 21.05 ms ≈ 每步约 **70 GB** 的梯度读流量，是"每个 Linear 的 dy 被第三遍读"的必然代价。
2. 有两条**病态路径**（5.1× / 17×），但**不是**当前的主因（否则会出现 13 ms 级的单个 kernel）。

**所以「定位并消除 ReduceSum ≈ −20 ms」这个估算不成立，撤回。** 真实可动量 ≤8 ms（把峰值以下那部分补平），
且要么把 bias 折进 `dW` 的 GEMM（`[x | 1]` 增广，只对 `out > in` 的层净赚：up/qkv 赚 ~0.4–0.55 ms/层，
down/out 反而亏），要么写 AscendC 融合 kernel。

### 11.3 顺带量出向量侧的"地质结论"：**它已经在峰值，没空间**

| 算子 | 实测带宽 |
|---|---:|
| SiLU（2 单位） | 3178 GB/s |
| Add（3 单位） | 3222 GB/s |
| `sum(0)`（2 单位） | 3189–3358 GB/s |
| `sum(0, dtype=fp32)` | 653 GB/s |
| 非连续 sum | 194 GB/s |

向量侧 240 ms/步 ≈ **800 GB/步** 的必然结果（800 GB ÷ 3.2 TB/s ≈ 250 ms）。
⇒ **"把小算子写更快"这条路不存在**；只有"减少搬运字节"（融合，需要 AscendC 或厂商融合算子）
或"减少算力"（fp8 / 结构）两条。

### 11.4 试过但作废：Triton 元素级融合在本环境**静默不执行**

写 `probe_fuse_add_silu{,2,3}.py` 验证「Add(3 单位) + SiLU(2 单位) → 融合 3 单位，理论 −40%」。三次都被自己否掉：

| 版本 | 现象 | 判定 |
|---|---|---|
| v1 | 形状只有 32 MiB，算出 4455 GB/s（超 HBM 峰值） | 数据落在 cache，计时无效 |
| v2 | 真实 2.52 GB 张量，算出 **213 247 GB/s** | 异步下发未同步，计时无效 |
| v3 | 改成逐次 `synchronize()`，仍得 **127 781 GB/s**，且 `max|Δ| = 0.00e+00` | **kernel 静默未执行**：`torch.empty_like(a)` 复用了刚释放的 `silu(a+b)` 中间缓冲，内容"恰好"正确，误差 0 是假象 |

**判据写法（以后必须遵守）**：任何带宽 > 4000 GB/s 的结果直接判 INVALID；融合探针必须用一个**预填垃圾值的输出缓冲**做对拍，
否则 `empty_like` 复用会伪造"逐位相同"。本环境（triton 3.2.0 + triton_ascend 3.2.1.dev）元素级 kernel 也不可靠，
融合收益只能用 AscendC 或厂商融合算子验证。

### 11.5 对"剩余空间"表的修正

| 项 | 原估 | 修正后 | 原因 |
|---|---:|---:|---|
| ReduceSum 消除 | −20 | **−8（且需 AscendC）** | 已在峰值，只剩"第三遍读 dy"这一个结构性问题 |
| Zero（580 次 / 14 ms） | −14 | **≈ −6** | 都是 `zeros_like` 后紧跟 `TensorMove`/`Slice`（autograd 的 slice/expand 反向），改它要动 autograd |
| 搬运/重排（TensorMove 26 + Transpose 28.4） | −15 | **≈ −5** | 485 次 Transpose 的 p50 只有 9.6 µs，主要是 NS 12 行路径与 ViewCopy 链 |
| 向量侧小算子整体 | — | **0（已在峰值）** | 见 §11.3 |

⇒ 一档从 **−67 ms 下修到 ≈ −30 ms**（⇒ 584 ms），二档（cube 效率）不变 ⇒ **全确定性手段的下限是 ~554 ms**，
比 §10 的 493 ms 更接近现实，也进一步说明**427 ms 只能靠自定义融合反向 / FP8 / 减算力**。
