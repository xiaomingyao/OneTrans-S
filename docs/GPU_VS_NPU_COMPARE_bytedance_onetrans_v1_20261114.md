# GPU 参考实现 vs NPU 迁移实现：逐模块 / 逐张量对照

项目：`/root/bytedance-onetrans/`　版本：v1　日期：2026-11-14
参考侧：`GPU_Reference`（commit **8febd7e**，零改动），**在 CPU 上执行**（本机无 GPU）
迁移侧：`OneTrans_NPU`，`npu:0`（Ascend950DT_9582，CANN 9.1.0，torch_npu 2.7.1.post4）
复现：`python3 tests/compare_ref_vs_npu.py --all`、`python3 tests/compare_mask_semantics.py`
原始日志：`results/compare_gpu_vs_npu_950_20261114.txt`

参数规模说明：本对照用配置（`d_model=64 / ns_len=8 / seq_len=32 / pyramid=3`）为 **2,227,970** 个参数、747 个梯度张量；冒烟/精度脚本的默认配置约 6.0M，两者不是同一配置，数字不可互引。

对照口径：同一 `seed=1234` 构造权重（两侧同一份 `models/taac_onetrans.py`）→ 同一份落盘输入（B=8）→ 前向 logits → cross-entropy loss → 反向全参数梯度。**不换数据、不换超参、不换模型代码**，只换设备。

---

## 1. 一句话结论

| 维度 | 结论 |
|---|---|
| **代码结构** | 模型 / 损失 / 指标 / 数据管道 **4 个文件逐字一致（0 行）**；147 行改动全在装配层（2 个文件） |
| **参数** | 2,227,970 个参数、747 个梯度张量，`state_dict` 键序与键名**完全一致**，同 seed 下两侧参数 **max\|Δ\|=0（逐位相同）** |
| **fp32 数值** | logits `1.79e-07`、loss `5.96e-08`、全参数梯度 `1.21e-06`，cos 均 `1.00000000` |
| **bf16 数值** | logits `2.31e-03`、loss `2.80e-04`、梯度 `1.82e-02`，cos `0.99997882` 起 |
| **mask 语义** | 4 种 mask 类型 float 两侧 `max\|Δ\|=0`；bool 路径「允许集」与参考 float 逐元素一致 |

**一句话**：这不是"两份实现各写一遍"，而是**同一份模型代码跑在两种设备上**——差异集中在设备接线、精度能力、多卡装配、kernel 落点。

---

## 2. 逐模块对照

| 模块 | GPU 参考位置 | NPU 位置 | 是否一致 | 说明 |
|---|---|---|---|---|
| 数据读取 `load_train_split` | `utils/taac_data.py` | 同 | **逐字一致** | HF 数据集 / 本地 parquet，走同一函数 |
| 特征张量化 `build_tensors` | `utils/taac_data.py` | 同 | **逐字一致** | non_seq 183 维、seq 32×45 |
| 非序列 tokenizer | `models/taac_onetrans.py:61` | 同 | **逐字一致** | `Linear(183 → ns_len×d_model)` |
| 序列 tokenizer | `models/taac_onetrans.py:62` | 同 | **逐字一致** | `Linear(45 → d_model)` |
| sep token | `models/taac_onetrans.py:64` | 同 | **逐字一致** | 可学习参数 |
| 金字塔调度 `linear_pyramid_schedule` | `models/taac_onetrans.py:16` | 同 | **逐字一致** | 长度序列对齐 32 |
| `MultiOneTransBlock`（token 分组 QKV） | `main_pytorch.py:202` | 同 | **逐字一致** | 每 token 组一套 KQV |
| `OneTransBlock` / `FFNLayer` / `RMSNorm` | `main_pytorch.py:140/8/21` | 同 | **逐字一致** | — |
| **attention mask 生成** | `main_pytorch.py:65` | 同 + 9 行 | **语义等价，多一个 env 开关** | `ONETRANS_ATTN_MASK=bool` 才换 dtype，默认 float |
| SDPA 调用 | `main_pytorch.py` | 同 | 代码一致、**落 kernel 不同** | float 掩码不落 FA；bool 才落 `FlashAttentionScoreV4` |
| head（LN→Linear→SiLU→Linear） | `models/taac_onetrans.py:96` | 同 | **逐字一致** | — |
| loss（cross-entropy） | 训练脚本 | 同 | **逐字一致** | 只在 `.float()` 上算，两侧同 |
| 指标 AUC / ACC | `utils/metrics.py` | 同 | **逐字一致** | — |
| **混合精度** | 训练脚本 | +16 行 | **行为不同（受硬件限制）** | NPU 只放行 bf16；fp16 无 GradScaler ⇒ 回落 fp32 |
| **设备接线** | 训练脚本 | +5 行 | **必须不同** | `import torch_npu` + `set_device` |
| **多卡** | 参考实现没有 | 新增约 100 行 | **新增能力** | HCCL + DDP + 数据切分 |
| **进程收尾** | 无 | +8 行 | **新增** | NPU teardown 会 SIGSEGV/SIGBUS |

---

## 3. 数值对照（三层）

同权重、同输入、同 mask 类型（`paper_causal`），只换设备：

| 层 | NPU fp32 | NPU bf16 | 判据 |
|---|---|---|---|
| logits（B×C） | max\|Δ\|=**1.79e-07**，rel 5.92e-07，cos **1.00000000** | max\|Δ\|=**2.31e-03**，rel 7.66e-03，cos 0.99997882 | fp32 ≤1e-4 / bf16 ≤2e-2 |
| loss（标量） | Δ=**5.96e-08**（rel 8.96e-08） | Δ=**2.80e-04**（rel 4.20e-04） | fp32 ≤1e-5 / bf16 ≤1e-3 |
| 全参数梯度（747 个） | rel_max=**1.21e-06** | rel_max=**1.82e-02** | fp32 ≤1e-5 / bf16 ≤2e-2 |
| 参数本身 | **max\|Δ\|=0**（逐位相同） | **max\|Δ\|=0** | 同 seed 同代码必须为 0 |

梯度误差最大的 5 个张量（fp32 全部 ≤1.2e-6，cos 均 1.00000000；bf16 最大 1.82e-02）：

| 张量 | fp32 rel | bf16 rel |
|---|---:|---:|
| `stack_blocks.0.otb_list.1.cma.kqv_list.3.0.weight` | 1.20e-06 | — |
| `stack_blocks.0.otb_list.0.cma.kqv_list.7.0.weight` | — | 1.82e-02 |
| `stack_blocks.1.otb_list.0.cma.kqv_list.8.0.bias` | 1.18e-06 | — |
| `stack_blocks.1.otb_list.0.cma.kqv_list.7.1.bias` | — | 1.61e-02 |
| `base_block.otb_list.0.cma.kqv_list.2.0.bias` | — | 1.59e-02 |

### 一个必须写进口径的坑：梯度「数值零」

`stack_blocks.2.otb_list.1.cma.kqv_list.0.1.{weight,bias}` 两个参数：

| 侧 | 梯度 absmax | 非零元素 |
|---|---:|---:|
| 参考（CPU fp32） | 2.818e-10 | 4096 / 4096 |
| 迁移（NPU fp32） | **0.0** | 0 / 4096 |

这两个参数在该配置下被 mask 乘穿（贡献恒 0），参考侧留下 `2.8e-10` 的浮点残渣，NPU 侧精确为 0。如果直接用 `rel = max|Δ| / max|g_ref|` 计算，会得到 **rel=1.0、cos=0.0**，看上去像"梯度完全不一致"——**这是分母陷阱，不是实现差异**。

正确口径（本次采用）：两侧量级都低于 `max(1e-8, 全局|g|max×1e-10)` 时判为「数值零梯度」，单独计数、不参与 rel_max。本例 2 个，占 747 的 0.27%。

---

## 4. mask 语义对照（4 种 mask 类型）

| mask_type | float 两侧 max\|Δ\| | `ONETRANS_ATTN_MASK=bool` 时 dtype | 允许集与参考一致 | 语义分类 |
|---|---:|---|---|---|
| `paper_causal` | **0.0e+00** | `torch.bool` | ✅ | 硬掩码（True=参与） |
| `hard_mask` | **0.0e+00** | `torch.bool` | ✅ | 硬掩码（True=参与） |
| `origin` | **0.0e+00** | `torch.float32`（开关不生效） | ✅ | 软偏置（+1e-9） |
| `bimask_soft` | **0.0e+00** | `torch.float32`（开关不生效） | ✅ | 软偏置（0/1 加性） |

`paper_causal` 允许位抽查（前 3 行 × 前 6 列，1=True=允许）：

```
ref  float  [1, 0, 0, 0, 0, 0]
bool        [1, 0, 0, 0, 0, 0]

ref  float  [1, 1, 0, 0, 0, 0]
bool        [1, 1, 0, 0, 0, 0]

ref  float  [1, 1, 1, 0, 0, 0]
bool        [1, 1, 1, 0, 0, 0]
```

结论：`bool` 的 `True` 与 float 的「未填 `-inf`」是同一集合，**语义等价**（注意方向：float 用 `-inf` 表示禁止，bool 用 `True` 表示参与；写反会静默改训练结果）。

---

## 5. 设备行为差异（同一份代码，不同执行路径）

| 维度 | GPU 参考（float 掩码） | NPU（bool 掩码） | 差 |
|---|---|---|---|
| attention 段 kernel 数 | **45 个**（`InplaceCopy×15 + Muls×6 + Matmul×6 + Add×3 + Softmax×3 + IsNegInf×3 + All×3 + InplaceFillScalar×3`…） | **6 个**（`FlashAttentionScoreV4×3 + LogicalNot×3`） | 45 → 6 |
| 落 FA 的 kernel | **0 个** | 3 个 | — |
| attention 段总时长（3 个 block，msprof） | 140.3 µs | **38.3 µs** | **3.66×** |
| 端到端步时（B=64/S=64，4 次独立运行） | — | +1.1% / +3.6% / +4.0% / **−0.5%** | 中枢 **+2.4%**（中位） |
| loss | 0.809698 | 0.809698 | Δ=5.96e-08 |

口径提醒：**kernel 段 3.66× ≠ 模型 3.66×**。该实现成本主要在「每个 token 组一套独立 KQV/FFN 参数」，attention 只占小头；端到端 4 次里有 1 次是负值，说明 +1~4% 的收益处在噪声边缘，只能按"小幅、不稳定"记账。此前记录过的 `46.3 µs → 12.3 µs` 是**单次 attention op 探针**口径，与本次「模型内 3 个 block 的 attention 段总和」不可混比。

---

## 6. 差异清单（与 GPU 实现不同的全部地方）

| 编号 | 差异 | 文件/行 | 是否影响数值 |
|---|---|---|---|
| H1 | 设备初始化（`import torch_npu` + `set_device`） | 脚本 5 行 + `npu_compat.py` | 否（fp32 逐位同级） |
| H2 | 混合精度能力门（NPU 仅 bf16、无 GradScaler） | 脚本 16 行 | 用 bf16 时 1e-3 级 |
| H3 | 多卡 HCCL/DDP + 数据切分 + rank0 验证 | 约 100 行 | 单卡不生效 |
| H4 | mask dtype 开关（默认 float = 原行为） | `main_pytorch.py` 9 行 | 默认不变；bool 时允许集等价 |
| H5 | 进程收尾（`shutdown` + `os._exit`） | 8 行 | 否 |
| H6 | 计时口径（`npu.synchronize` 前后取墙钟） | 9 行 | 否（只影响日志） |
| — | HF 栈、`HF_ENDPOINT` 镜像 | 环境项，不改代码 | 否 |

---

## 7. 结论与边界

**成立的部分**：单卡 fp32 下，NPU 实现与 GPU 参考实现在 logits / loss / 全部 747 个梯度上分别做到 `1e-7 / 1e-8 / 1e-6` 级一致；bf16 下为 `1e-3 / 1e-4 / 1e-2` 级；4 种 mask 语义完全等价。

**边界**：
1. 参考侧在 **CPU** 上执行（本机无 GPU）。CPU fp32 与 GPU fp32 之间的差异未测——若 GPU 上用了 TF32/cuDNN 非确定性算法，差值可能大于 1e-6。
2. 只对照了 `paper_causal` 端的 forward/backward；`origin` / `bimask_soft` / `bimask_hard` 只对照了 mask 张量，未做端到端 loss/梯度对照。
3. 对照是单 die；多卡数值一致性由 `tests/ddp_equiv_npu.py` 单独验证（W=2 fp32 `worst rel|Δg|=1.2e-6`）。
4. 未做长训收敛对照（同数据同 seed 跑 N 步比 loss 曲线）。
