# GPU 与 NPU 代码实现对比（OneTrans 参考实现 → 昇腾 950）

项目：`/root/bytedance-onetrans/`　版本：v3（本次仅改呈现：代码块语法高亮）　日期：2026-11-14
参考实现：`https://github.com/WestbrookLong/OneTrans_Pytorch`，commit **8febd7e**（零改动）
迁移实现：`OneTrans_NPU/`　环境：容器 `zeus-950-v2`（`zeus-950-v2-snap:20261102`）、CANN 9.1.0、torch 2.7.1+cpu / torch_npu 2.7.1.post4
逐行原文：`patches/npu_changes.diff`（458 行；含 `npu_compat.py` 新增文件全文）

> v2 相对 v1：① **删除合成数据集入口**，数据路径与 GPU 侧完全一致（HF `TAAC2026/data_sample_1000` / 本地 parquet）；② **HF 延迟导入回退**（栈已装好，不再需要改数据代码）；③ 调试打印删除；④ 本报告**只列与硬件相关的改动**，非硬件改动一律回退为参考实现。

---

## 1. 结论

| 项 | 数字 |
|---|---|
| 实改文件 | **2 个**：`main_pytorch.py`、`scripts/run_taac2026_sample.py` |
| 实改行数 | **147 行**（+113 / −34） |
| 未改文件 | `models/taac_onetrans.py`、`utils/metrics.py`、`utils/common.py`、`utils/taac_data.py`（**四个文件逐字一致**） |
| 数据集 | 与 GPU 侧**同一个**：`TAAC2026/data_sample_1000`（1,000 行 × 120 列），无合成数据替代 |
| 改动性质 | 全部是硬件适配：设备初始化、混合精度能力门、HCCL/DDP 装配、attention kernel 落点、设备收尾、设备侧计时 |
| 可回退性 | 全部可回退：不传 `--device npu` / 不设 `ONETRANS_ATTN_MASK` / `WORLD_SIZE=1` 即回到参考实现行为 |

**一句话**：147 行里没有一行是为了"让数据能读进来"或"让脚本能跑起来"的便利性改动，全部围绕「昇腾这颗芯片上设备怎么建、精度能力有什么、多卡怎么连、mask 落在哪个 kernel、进程怎么退」。

---

## 2. 硬件相关改动分类

| 编号 | 硬件相关点 | 文件 | 行数 | 触发条件 |
|---|---|---|---:|---|
| H1 | **设备初始化**：`import torch_npu` + `set_device` | 脚本 + `npu_compat.py` | 5 | `--device npu` |
| H2 | **混合精度能力门**：NPU 只支持 bf16 AMP、无 GradScaler | 脚本 + `npu_compat.py` | 16 | `--amp` |
| H3 | **多卡 HCCL/DDP 装配**（含数据切分、rank0 验证、不跨卡归约） | 脚本 + `npu_compat.py` | 约 100 | `WORLD_SIZE>1`（torchrun） |
| H4 | **attention kernel 落点**：加性 float mask 不落 FA，bool 才落 `FlashAttentionScoreV4` | `main_pytorch.py` | 9 | `ONETRANS_ATTN_MASK=bool` |
| H5 | **设备收尾**：teardown SIGSEGV/SIGBUS | 脚本 + `npu_compat.py` | 8 | 多卡 |
| H6 | **设备侧计时口径**：`torch.npu.synchronize()` 前后取墙钟 | 脚本 | 9 | 恒开（只影响日志） |

不在上表、且已**回退为参考实现**的项：HF 依赖延迟导入（0 行）、合成数据集（删除）、`ONETRANS_DBG` 调试打印（删除）。

---

## 3. 逐项代码对照

### H1 设备初始化（`init_device`）

```python
# 脚本：签名不变，实现委托
- def parse_device_type(device: str) -> str:
-     return torch.device(device).type
+ def parse_device_type(device: str) -> str:
+     """[NPU] 保留原签名；实际初始化（含 torch_npu 导入/set_device）在 npu_compat。"""
+     return npu_compat.init_device(device)

# npu_compat.py
+ def init_device(device: str) -> str:
+     device_type = torch.device(device).type
+     if device_type == "npu":
+         import torch_npu          # 必须先于任何 npu 张量
+         torch.npu.set_device(0)
+     return device_type
```

为什么是硬件相关：`npu` 不是 `cuda` 的别名，`torch_npu` 必须在建任何 npu 张量前导入，且设备要显式 `set_device`。原脚本 `--device` 默认 `cuda if torch.cuda.is_available() else cpu`，在昇腾机器上会**静默落 CPU**（能跑、慢一个量级、不报错）。判据：日志首行 `soc=Ascend950DT_9582`。

### H2 混合精度能力门（NPU 只放行 bf16）

```python
- def should_enable_amp(device_type, amp, amp_dtype):
-     if not amp: return False
-     if device_type != "cuda": return False            # ← NPU 上 AMP 在此被静默关掉
-     if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported(): return False
-     return True
- def build_scaler(device_type, use_amp, amp_dtype):
-     enabled = use_amp and device_type == "cuda" and amp_dtype == torch.float16
-     return torch.amp.GradScaler(device=device_type, enabled=enabled)
+ def should_enable_amp(device_type, amp, amp_dtype):
+     return npu_compat.should_enable_amp(device_type, amp, amp_dtype)
+ def build_scaler(device_type, use_amp, amp_dtype):
+     return npu_compat.build_scaler(device_type, use_amp, amp_dtype)   # NPU → no-op scaler
```

| 组合 | 昇腾上可用 | 处置 |
|---|---|---|
| npu + bf16 | ✅ | 放行；bf16 动态范围同 fp32，不需要 scaler |
| npu + fp16 | ❌ 无可用 GradScaler | 禁用 AMP、回落 fp32，日志写明 |
| cuda + fp16/bf16 | ✅ | 原逻辑不变 |

no-op scaler 保留 `scale/step/update/unscale_` 调用形态，训练循环不用改。判据：`amp=True amp_dtype=bf16 grad_scaler=False`；实测 `get_autocast_dtype("npu")` 默认是 **fp16**，所以 `bfloat16` 必须显式传。

### H3 多卡 HCCL/DDP 装配

```python
+ rank, world_size, local_rank = npu_compat.init_distributed()   # WORLD_SIZE<=1 → (0,1,0)，行为不变
+ if world_size > 1:
+     args.device = f"{device_type}:{local_rank}"
+ is_main = rank == 0
+ log = print if is_main else (lambda *a, **k: None)

+ if world_size > 1:
+     model = nn.parallel.DistributedDataParallel(
+         model, device_ids=[local_rank], find_unused_parameters=not args.ddp_strict,
+         broadcast_buffers=False)

# 数据：训练集按 rank 切分；验证集只在 rank0 全量
+ train_idx = train_idx[rank::world_size]
+ if rank != 0:
+     val_idx = val_idx[:0]
```

`npu_compat.init_distributed()`：

```python
+ world_size = int(os.environ.get("WORLD_SIZE", "1"))
+ if world_size <= 1:
+     return 0, 1, 0                    # 单进程路径完全不变
+ local_rank = int(os.environ.get("LOCAL_RANK", "0"))
+ torch.npu.set_device(local_rank)    # ASCEND_RT_VISIBLE_DEVICES 已重编号
+ dist.init_process_group(backend="hccl", timeout=datetime.timedelta(minutes=10))
```

三条与生产 `zeus-onetrans` 一致的硬件约定：

1. `device_ids=[local_rank]` + `broadcast_buffers=False` + `find_unused_parameters=True`（该模型最深金字塔块有 16 个参数不参与前向，不开会报 `Expected to have finished reduction`）；
2. **训练循环里不做任何跨 rank 归约**：训练集切分 + rank0 全量验证，指标天然是全集口径。实测在 epoch 末加 `all_gather` 会卡在 D2H 拷贝（py-spy 栈停在 `b.cpu()`）；`all_gather_object` 更早一步卡在 pickle；
3. HCCL 标量归约只吃 float32（fp64 报 `Unsupported data type at::kDouble`），`all_reduce(MAX)` 在 float 上会静默卡住。

### H4 attention kernel 落点（`main_pytorch.py`，9 行）

```python
+ _MASK_DTYPE = os.environ.get("ONETRANS_ATTN_MASK", "float").lower()   # 默认 float = 原行为
  ...
  if self.mask_type == "paper_causal":
      allowed = col_idx <= q_abs
+     if _MASK_DTYPE == "bool":
+         return allowed                 # bool: True = 参与
      return mask.masked_fill(~allowed, torch.finfo(dtype).min)
  ...
  if self.mask_type == "hard_mask":
+     if _MASK_DTYPE == "bool":
+         return origin_allowed
```

为什么是硬件相关：**mask 的 dtype 直接决定 SDPA 在 950 上走哪个 kernel**（实测 B=8/H=4/S=64/D=16，bf16）：

| mask 形态 | kernel 构成 | 单次 attention |
|---|---|---:|
| float 加性（原实现） | `Cast×15 + Muls×6 + BatchMatMulV3×6 + SoftmaxV2×3 + SelectV2`… 45 个 kernel | 46.3 µs |
| bool | `FlashAttentionScoreV4×3 + LogicalNot×3`，6 个 kernel | **12.3 µs（3.8×）** |
| 无 mask | `FlashAttentionScoreV4×3` | 9.3 µs |

只对 `paper_causal` / `hard_mask`（0/-inf 型）转换；`origin`（+1e-9 偏置）与 `bimask_soft`（0/1 加性偏置）是软偏置，转 bool 会改语义。端到端收益仅 +1.6%（S=64）/ +3.7%（S=256），因为该实现成本主要在多组小矩阵。

### H5 设备收尾

```python
+ npu_compat.shutdown()          # torch.npu.synchronize() + destroy_process_group()
+ if world_size > 1:
+     sys.stdout.flush(); sys.stderr.flush()
+     os._exit(0)                # 跳过解释器 teardown
```

实测：训练跑完后非 0 号 rank 在解释器退出阶段收到 **SIGSEGV(-11) / SIGBUS(-7)**，`torchrun` 因此报 `ChildFailedError` —— 失败发生在 teardown，训练本身已完成。另实测 HCCL 固定端口 16666 会被上一次挂死进程占住（`Bind_IP_Port(EI0020)`），起跑前须清残留进程。

### H6 设备侧计时口径

```python
+ if device_type == "npu":
+     torch.npu.synchronize()      # 取墙钟前后都同步，否则量到的是下发时间
+ t_epoch = time.perf_counter()
  ... 训练 / 验证 ...
+ if device_type == "npu":
+     torch.npu.synchronize()
+ wall = time.perf_counter() - t_epoch
+ log(f"[perf] step_ms={wall / max(len(train_loader),1)*1000:.1f} "
+     f"global_samples_per_s={args.batch_size * world_size / (wall / max(len(train_loader),1)):.1f} ...")
```

---

## 4. 数据路径与 GPU 完全一致（非硬件改动已回退）

| 项 | v1 状态 | v2 状态 |
|---|---|---|
| 数据集 | 新增 `--synthetic` 合成集（无 HF 栈时的替代） | **删除**；只走 `load_train_split()`（HF `TAAC2026/data_sample_1000` 或 `--local-parquet`），与参考实现同一入口 |
| `utils/taac_data.py` | 12 行改动（HF 延迟导入） | **0 行**（HF 栈 `datasets 5.2.0 / huggingface_hub 2.4.0 / pyarrow 26.0.1` 已装入容器） |
| `utils/common.py` | 3 行改动 | **0 行** |
| `ONETRANS_DBG` 调试打印 | 8 行 | **删除** |

环境要求（写在报告里而不是藏在代码里）：容器需装 HF 栈，并设 `HF_ENDPOINT=https://hf-mirror.com`（容器内 `huggingface.co` 不可达：curl `000`；`hf-mirror.com` `200`）。

---

## 5. 零改动证据与新增资产

| 证据 | 命令 | 结果 |
|---|---|---|
| 参考仓干净 | `git -C GPU_Reference status --porcelain` | 空 |
| commit 锁定 | `git -C GPU_Reference rev-parse HEAD` | `8febd7e…`（`REFERENCE_COMMIT.txt`） |
| 四个文件逐字一致 | `diff -u GPU_Reference/<f> OneTrans_NPU/<f>` | `models/taac_onetrans.py`、`utils/metrics.py`、`utils/common.py`、`utils/taac_data.py` 均 0 行 |

新增（不是"修改"）：`npu_compat.py`（160 行，硬件兼容层）、`tests/` 11 个探针（1,035 行，覆盖冒烟/精度/梯度等价/HCCL dtype/AMP/mask 落点）。

---

## 6. 改动后的实测（全部用与 GPU 相同的数据集）

| 验证 | 实测 | 判据 | 结论 |
|---|---|---|---|
| 冒烟 | 3 步 loss 1.6105→0.7949、`param_delta=3.008e-03`、稳态 79–80 ms/步 | 有限 + 有更新 + 下降 | PASS |
| 精度 A（NPU fp32 vs CPU fp32） | max\|Δ\|=1.49e-7，cos 0.99999988 | ≤1e-4 / cos≥0.99999 | PASS |
| 精度 B（NPU bf16 vs CPU fp32） | max\|Δ\|=2.59e-3，cos 0.99998170 | ≤2e-2 / cos≥0.9995 | PASS |
| 精度 C（10 步轨迹） | Δloss=5.96e-8、Δacc=0 | Δloss≤1e-2 / Δacc≤2pp | PASS |
| 真数据集单卡（1000 行，5 epoch，batch 32） | 25 步/epoch、78.5 ms/步、406 samples/s | 能训、指标产出 | PASS |
| 真数据集 2 卡（每卡 32） | 13 步/epoch、95.5–98.6 ms/步、636–670 samples/s | torchrun 无报错 | PASS |
| 多卡梯度等价（W=2，fp32） | `worst rel\|Δg\|=1.2e-6`、Δloss=0 | ≤1e-5 | PASS |
| AMP 生效性 | ctx 内 `is_autocast_enabled=True`、matmul 输出 bf16、默认 dtype=fp16 | 生效标志 + 输出 dtype | PASS |
| AMP 划算性 | fp32 54.3 ms vs bf16 69.3 ms（慢 27.5%），HBM 109→99 MiB | 同进程 A/B | **负收益，默认 fp32** |

### 扩展性（真数据集 `data_sample_1000`，每卡本地 batch 64，8 epoch）

| die | 每 epoch 步数 | 稳态 step | samples/s | 加速比 | 每卡效率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 77.2 ms | 829.5 | 1.00× | 100% |
| 2 | 7 | 98.2 ms | 1 303.6 | **1.57×** | 78.6% |
| 4 | 4 | 109.0 ms | 2 348.3 | **2.83×** | 70.8% |
| 8 | 2 | 138.9 ms | 3 686.2 | **4.44×** | 55.5% |

口径说明：**稳态 = 剔除首个 epoch 后 7 个 epoch 的均值**（与 `results/scaling_real_data_950_20261114.txt` 的 `steady_epoch_wall_s n=7` 一致）；数据集只有 1000 行（train 800），8 卡时每 epoch 仅 2 步 ⇒ 8 卡数字**仅供参考**（稳态 7 个 epoch = 14 步）。要更稳的扩展比需要更大数据集或更长 epoch 数。

---

## 7. 风险与回退

| 改动 | 风险 | 回退 |
|---|---|---|
| bool mask | `True` = 参与，与 float 的 `-inf` 相反，写反会静默改训练结果 | 不设 `ONETRANS_ATTN_MASK`（默认 float）= 原行为 |
| DDP 包壳 | 有参子模块漏包会导致各 rank 参数漂移 | 本实现单一 model；扩展时须把有参子模块都包进 DDP |
| `find_unused_parameters=True` | 每步多一次图遍历开销 | 确认每步用满参数后 `--ddp-strict` 关闭 |
| stride 切分 | 每 rank 样本数差 ≤1，末批 batch 略不齐 | 影响只在指标第 4 位小数 |
| `os._exit(0)` | 跳过解释器收尾（已显式 flush） | 仅多卡路径 |

---

## 8. 边界

1. 数据集只有 1000 行（官方 `data_sample_1000`），**不下任务效果结论**；8 卡扩展比样本量偏少。
2. 只测单机 8 die；跨机 16 卡（122+111，`ub_verbs --apply 8` + env 化 rendezvous）未做。
3. 未做长时间收敛验证（5–8 epoch 只证明能训、指标下降）。
4. 扩展比绑定模型规模（6.0M 参数）；换模型/换 batch 需重测。
5. `origin` / `bimask_soft` / `bimask_hard` 三种软偏置 mask 未逐一验数值。
