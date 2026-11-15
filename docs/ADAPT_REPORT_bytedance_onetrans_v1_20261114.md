# OneTrans 参考实现 → 昇腾 950：代码改动对照与适配报告

项目：`/root/bytedance-onetrans/`　版本：v1　日期：2026-11-14
参考实现：`https://github.com/WestbrookLong/OneTrans_Pytorch`，commit **8febd7e**（`git status --porcelain` 为空，零改动）
迁移实现：`OneTrans_NPU/`　运行环境：容器 `zeus-950-v2`（镜像 `zeus-950-v2-snap:20261102`）、CANN 9.1.0、torch 2.7.1+cpu / torch_npu 2.7.1.post4
对照原文：`docs/npu_changes.diff`（427 行，`diff -ru --exclude=.git --exclude=__pycache__ --exclude='*.bak'`）

---

## 1. 结论

| 项 | 数字 / 结论 |
|---|---|
| 参考实现规模 | Python **1,234 行**（6 个文件） |
| 实改行数 | **204 行**（新增 174 / 删除 49，`npu_changes.diff`） |
| 改动占比 | **16.5%**，且集中在装配层 |
| 模型文件改动 | `models/taac_onetrans.py` **0 行**、`utils/metrics.py` **0 行** |
| 新增文件 | `npu_compat.py`（160 行）+ `tests/` 11 个（冒烟/精度/多卡/mask/AMP 探针） |
| 平台相关改动 | 6 类：设备接线、混合精度、数据依赖、性能语义（mask dtype）、多卡装配、测试资产 |
| 不可回退改动 | **0 处**（mask 走 env 开关，默认与参考实现逐字等价；多卡靠 `WORLD_SIZE` 自动判定） |

**一句话**：模型结构、损失、优化器语义、指标计算一行没动；204 行全部落在"设备 / 精度 / 数据入口 / 多卡装配"这四处接线和新增测试上。

---

## 2. 改动分类总览

| 类 | 文件 | 行数 | 性质 | 是否可回退 |
|---|---|---:|---|---|
| A 设备接线 | `scripts/run_taac2026_sample.py` + `npu_compat.py` | 8 | `torch.device(...).type` → `npu_compat.init_device()`，`import torch_npu` + `set_device` | 传 `--device cpu/cuda` 即回原路径 |
| B 混合精度 | 同上 | 12 | AMP 放行规则（NPU 仅 bf16）、no-op scaler、autocast 委托 | `--no-amp` 或 `--amp-dtype fp16`（NPU 上自动禁用） |
| C 数据依赖 | `utils/taac_data.py`、`utils/common.py` | 15 | HF 栈延迟导入 + 去顶层类型注解 | 装上 HF 栈即走原路径 |
| D 性能语义 | `main_pytorch.py` | 9 | 可选 bool attention mask（落 `FlashAttentionScoreV4`） | `ONETRANS_ATTN_MASK` 默认 `float` = 原行为 |
| E 多卡装配 | `scripts/run_taac2026_sample.py` + `npu_compat.py` | 约 120 | HCCL 初始化、DDP 包壳、数据切分、rank0 侧打印/计时、收尾 | `WORLD_SIZE<=1` 时整条路径不生效 |
| F 新增资产 | `npu_compat.py`、`tests/*` | 160 + 1,035 | 兼容层 + 11 个探针 | 不参与运行时 |

---

## 3. 逐文件对照

### 3.1 `main_pytorch.py`（9 行，唯一带性能含义的模型侧改动）

改动只有三处，全部围绕 `create_attention_mask()` 的返回 dtype：

```python
# ① 顶部：env 开关，默认保持原行为
+ import os
  import torch
  ...
+ # [NPU] 可选：把"0 / -inf"型加性 mask 换成 bool mask。
+ # 950 实测：float 加性 mask 会让 SDPA 退回 Cast+Muls+BatchMatMul+Softmax 手工路径；
+ # bool mask 才落 FlashAttentionScoreV4。默认 float，保持与参考实现逐字等价。
+ _MASK_DTYPE = os.environ.get("ONETRANS_ATTN_MASK", "float").lower()

# ② paper_causal 分支
      allowed = col_idx <= q_abs
+     if _MASK_DTYPE == "bool":
+         return allowed                 # bool: True = 参与
      mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
      return mask.masked_fill(~allowed, torch.finfo(dtype).min)

# ③ hard_mask 分支（同样两行）
      if self.mask_type == "hard_mask":
+         if _MASK_DTYPE == "bool":
+             return origin_allowed
```

**为什么只改这两个分支**：`paper_causal` / `hard_mask` 是"允许 0 / 禁止 `finfo.min`"的硬掩码，语义上等价于 bool；`origin`（+1e-9 偏置）与 `bimask_soft`（0/1 加性偏置）是**软偏置**，转 bool 会改语义，所以保持 float。

**收益（kernel 明细，B=8/H=4/S=64/D=16，bf16，3 次迭代）**

| mask 形态 | kernel 构成 | 单次 attention |
|---|---|---:|
| float 加性（原实现） | `Cast×15 + Muls×6 + BatchMatMulV3×6 + SoftmaxV2×3 + SelectV2`… 共 45 个 | 46.3 µs |
| bool | `FlashAttentionScoreV4×3 + LogicalNot×3`，共 6 个 | **12.3 µs（3.8×）** |
| 无 mask | `FlashAttentionScoreV4×3` | 9.3 µs |

**端到端只有 +1.6% / +3.7%**（B=64/S=64、B=32/S=256，20 次中位），loss 一致（Δ≤1.2e-7）。原因：该实现的成本主要在"每个 token 组一套独立 KQV/FFN 参数"。**kernel 3.8× ≠ 模型 3.8×**。

### 3.2 `utils/taac_data.py`（12 行）与 `utils/common.py`（3 行）

```python
# taac_data.py：HF 栈改为函数内导入
- from datasets import Dataset, DownloadConfig, load_dataset
- from huggingface_hub import hf_hub_download, list_repo_files
+ def _hf_imports():
+     from datasets import Dataset, DownloadConfig, load_dataset
+     from huggingface_hub import hf_hub_download, list_repo_files
+     return Dataset, DownloadConfig, load_dataset, hf_hub_download, list_repo_files

- def load_train_split(...) -> Dataset:
+ def load_train_split(...):
+     Dataset, DownloadConfig, load_dataset, hf_hub_download, list_repo_files = _hf_imports()

# common.py：只为类型注解用，去掉
- from datasets import Dataset
- def take_rows(dataset: Dataset, max_rows: int | None) -> list[dict[str, Any]]:
+ def take_rows(dataset, max_rows: int | None) -> list[dict[str, Any]]:
```

原因：容器里原本没有 `datasets`/`pyarrow`，顶层 import 会让**任何**入口（包括 `--synthetic`）在 import 阶段就崩。v2 已把 HF 栈装上（清华源），真数据路径验证通过，但延迟导入保留 —— 它让"无数据依赖的冒烟/精度测试"成立。

### 3.3 `scripts/run_taac2026_sample.py`（180 行）—— 改动主体

**① 设备（3 行）**

```python
- def parse_device_type(device: str) -> str:
-     return torch.device(device).type
+ def parse_device_type(device: str) -> str:
+     """[NPU] 保留原签名；实际初始化（含 torch_npu 导入/set_device）在 npu_compat。"""
+     return npu_compat.init_device(device)
```

原实现 `--device` 默认 `cuda if torch.cuda.is_available() else cpu`，NPU 机器上会**静默落 cpu**（能跑、慢、不报错）。现在设备类型由兼容层返回，并在日志首行打印环境指纹。

**② 混合精度（9 行，全部改成委托）**

```python
- def should_enable_amp(device_type, amp, amp_dtype):
-     if not amp: return False
-     if device_type != "cuda": return False          # ← NPU 上 AMP 在此被静默关掉
-     if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported(): return False
-     return True
+ def should_enable_amp(device_type, amp, amp_dtype):
+     return npu_compat.should_enable_amp(device_type, amp, amp_dtype)

- def build_scaler(device_type, use_amp, amp_dtype):
-     enabled = use_amp and device_type == "cuda" and amp_dtype == torch.float16
-     return torch.amp.GradScaler(device=device_type, enabled=enabled)
+ def build_scaler(device_type, use_amp, amp_dtype):
+     return npu_compat.build_scaler(device_type, use_amp, amp_dtype)   # NPU → no-op scaler

- def autocast_context(...): return torch.autocast(device_type=device_type, ...)
+ def autocast_context(...): return npu_compat.autocast_context(...)
```

**③ 合成数据入口（24 行新增）**

```python
+ def build_synthetic_tensors(rows, seq_len, seed):
+     """无外部数据依赖的合成集：标签由 non-seq 特征的线性组合决定，保证优于随机猜测。"""
+     g = torch.Generator().manual_seed(seed)
+     ...  # non_seq(rows,64) / seq(rows,seq_len,32) / labels(rows)
+ 新增参数：--synthetic、--synthetic-rows（默认 2048）
```

**④ 多卡装配（约 90 行）**

| 改点 | 改前 | 改后 |
|---|---|---|
| 初始化 | 无 | `rank, world_size, local_rank = npu_compat.init_distributed()`；`WORLD_SIZE<=1` 时返回 `(0,1,0)`，行为不变 |
| 设备 | `--device npu` | 多卡时 `args.device = f"{device_type}:{local_rank}"` |
| 日志 | 所有 rank 都 print | `log = print if rank == 0 else no-op`，只 rank0 输出 |
| 数据 | 单份 index | 训练集 `train_idx[rank::world_size]`（不填充不重复）；验证集**只在 rank0 全量**，其余给空集 |
| 模型 | 直接 model | `DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=not args.ddp_strict)` |
| 指标 | 单进程本地统计 | **不做跨卡归约**（与生产 zeus-onetrans 一致）；rank0 口径即全集 |
| 计时 | 无 | 每 epoch `npu.synchronize()` 前后取墙钟，打印 `step_ms` / `global_samples_per_s` / `steady_epoch_wall_s` |
| 落盘 | 所有 rank 都写 | 仅 rank0 写 metadata/checkpoint；`model.module.state_dict()` 解包 |
| 收尾 | 无 | `npu_compat.shutdown()` + `os._exit(0)`（规避 teardown SIGSEGV/SIGBUS） |
| 新参数 | — | `--ddp-strict`（关掉 find_unused_parameters，仅当确认每步用满参数时用） |

**⑤ 调试开关（约 8 行，env 门控，默认不生效）**：`ONETRANS_DBG=1` 时按 step 打印 `data / fwd_done / bwd_done`，用于定位"卡在哪一步"。

### 3.4 新增 `npu_compat.py`（160 行）

职责五件事，全部与具体模型无关：

| 函数 | 作用 | 关键点 |
|---|---|---|
| `init_device` | 返回 device_type 并初始化 npu | `import torch_npu` 必须先于任何 npu 张量 |
| `should_enable_amp` | AMP 放行规则 | NPU 只放行 bf16；fp16 需 GradScaler，NPU 侧没有 |
| `build_scaler` | scaler 工厂 | NPU 返回 no-op 占位（`scale/step/update/unscale_` 空操作），训练循环调用形态不变 |
| `autocast_context` / `describe` | autocast + 环境指纹 | `describe` 打印 torch / torch_npu / soc |
| `init_distributed` / `shutdown` | HCCL 装配与收尾 | 单进程短路；收尾必须 sync + destroy |

### 3.5 新增 `tests/`（11 个文件，1,035 行，不参与运行）

| 文件 | 行数 | 作用 |
|---|---:|---|
| `smoke_npu.py` | 93 | 3 步前反向 + 参数更新/有限性断言 |
| `accuracy_npu.py` | 128 | 精度三段（fp32/bf16/训练轨迹） |
| `ddp_equiv_npu.py` | 142 | DDP 梯度等价性（vs 单进程全局 batch） |
| `probe_hccl_dtypes.py` | 70 | HCCL 各 dtype 的 all_gather / all_reduce 支持面 |
| `probe_ddp_collective.py` | 80 | "DDP 之后发集合通信"的最小复现 |
| `probe_amp_npu.py` | 80 | AMP 是否真生效（默认 dtype / 输出 dtype / scaler 形态） |
| `bench_amp_npu.py` | 112 | fp32 vs bf16 同进程 A/B（步时 + 峰值 HBM） |
| `bench_mask_mode.py` | 92 | float vs bool mask 端到端对照 |
| `probe_sdpa_kernel.py` | 42 | SDPA 落哪个 attention kernel |
| `run_all_950.sh` | 21 | 单卡一键（冒烟 + 精度 + 端到端） |
| `run_ddp_950.sh` | 47 | 多卡一键（等价性 + 1/2/4/8 die 扩展性） |

---

## 4. 零改动证据

| 证据 | 命令 | 结果 |
|---|---|---|
| 参考仓干净 | `git -C GPU_Reference status --porcelain` | 空 |
| commit 锁定 | `git -C GPU_Reference rev-parse HEAD` | `8febd7eff9b1f2f63fe1fa4e364af8a8755f51fb`（记录在 `REFERENCE_COMMIT.txt`） |
| 模型文件逐字一致 | `diff -u GPU_Reference/models/taac_onetrans.py OneTrans_NPU/models/taac_onetrans.py` | 无差异（0 行） |
| 指标文件逐字一致 | `diff -u GPU_Reference/utils/metrics.py OneTrans_NPU/utils/metrics.py` | 无差异（0 行） |
| 其余四个文件 | `diff -u` 行数 | `main_pytorch.py` 9 / `scripts/*` 180 / `taac_data.py` 12 / `common.py` 3 |

---

## 5. 改动后的复跑验证（全部在最终代码上重跑）

| 验证 | 实测 | 判据 | 结论 |
|---|---|---|---|
| 冒烟 | 3 步 loss 1.6105→0.7949、`param_delta=3.008e-03`、稳态 79–80 ms/步 | loss 有限 + 参数更新 + loss 下降 | PASS |
| 精度 A（fp32 vs CPU fp32） | max\|Δ\|=1.49e-7，cos 0.99999988 | ≤1e-4 / cos≥0.99999 | PASS |
| 精度 B（bf16 vs CPU fp32） | max\|Δ\|=2.59e-3，cos 0.99998170 | ≤2e-2 / cos≥0.9995 | PASS |
| 精度 C（10 步轨迹） | Δloss=5.96e-8、Δacc=0 | Δloss≤1e-2 / Δacc≤2pp | PASS |
| 多卡梯度等价 | W=2 fp32 `worst rel\|Δg\|=1.2e-6`、Δloss=0；W=4 fp32、W=2/4 bf16 均 PASS | ≤1e-5 / ≤2e-2 | PASS |
| 多卡扩展性（每卡 batch 64） | 1/2/4/8 die = 74.0/98.6/114.5/134.7 ms；867.6/1 296/2 235/3 799.7 samples/s；加速比 1.00/1.49/2.58/4.38× | 加速比随卡数单调上升 | PASS |
| AMP 生效性 | ctx 内 `is_autocast_enabled=True`、matmul 输出 bf16、默认 dtype=fp16、`npu/fp16` 被禁 | 生效标志 + 输出 dtype | PASS |
| AMP 划算性 | fp32 54.3 ms vs bf16 69.3 ms（**慢 27.5%**），HBM 109→99 MiB | 同进程 A/B | **负收益（默认 fp32）** |
| 真数据链路 | `TAAC2026/data_sample_1000`（1000×120）单卡/2 卡均跑通 | 能下载→张量化→前反向→出指标 | PASS（仅链路） |

---

## 6. 改动的风险与回退

| 改动 | 风险 | 回退方式 |
|---|---|---|
| `main_pytorch.py` bool mask | 语义若理解错会静默改训练结果（`True` 是"参与"，与 float 的 `-inf` 相反） | 不设 `ONETRANS_ATTN_MASK`（默认 float）= 原行为；`tests/bench_mask_mode.py` 可对照 loss 是否一致 |
| DDP 包壳 | 子模块漏包 DDP 会导致各 rank 参数漂移（生产踩过） | 本实现只有单一 model，无此风险；扩展时必须把有参子模块都包进 DDP |
| `find_unused_parameters=True` | 每步多一次图遍历，有少量开销 | 确认每步用满参数后可 `--ddp-strict` 关掉；关掉后若报 `Expected to have finished reduction` 属预期 |
| 训练集 stride 切分 | 每个 rank 的样本数差 ≤1（非填充），最后一步 batch 略不齐 | 影响只在 metric 的第 4 位小数；要严格齐批可改用 DistributedSampler + drop_last |
| `os._exit(0)` | 跳过解释器收尾（flush 已显式做） | 仅多卡路径生效；单进程保留正常退出 |
| HF 延迟导入 | 首次调用多一次 import 开销（毫秒级） | 无 |

---

## 7. 边界

1. **真数据只到"链路通"**：1000 行、2 epoch、二分类弱信号（单卡 AUC 0.5261→0.5429），不下任务效果结论。
2. **多卡只测单机**：122 上 8 die 已实测；跨机 16 卡（122+111，需 `ub_verbs --apply 8` 与 env 化 rendezvous）未做。
3. **未做长时间收敛验证**：2–3 epoch 只证明"能训、指标下降"。
4. **扩展性数字绑定模型规模**（6.0M 参数、每卡 batch 64）：换模型/换 batch 需重测。
5. **`origin` / `bimask_soft` / `bimask_hard` 三种 mask 未逐一验数值**（只保证 float 路径可运行）。
