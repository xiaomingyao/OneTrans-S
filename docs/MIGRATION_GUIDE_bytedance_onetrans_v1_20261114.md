# OneTrans（PyTorch 参考实现）迁移昇腾 950 指导

项目：`/root/bytedance-onetrans/`　日期：2026-11-14
参考实现：`https://github.com/WestbrookLong/OneTrans_Pytorch`，commit **8febd7e**（工作区干净、零改动）
迁移目标：Ascend 950DT（`Ascend950DT_9582`）、CANN 9.1.0、torch 2.7.1+cpu / torch_npu 2.7.1.post4

---

## 0. 结论

| 项 | 结果 |
|---|---|
| 迁移面 | **82 行改动 / 1,233 行代码 = 6.6%**；模型代码（`models/`、`main_pytorch.py` 主干）**0 行改动** |
| 冒烟 | **PASS**：3 步 loss 1.6105→0.7949，参数更新 3.0e-3，稳态 **79–80 ms/步**（6.0M 参数） |
| 精度 | **PASS**：NPU fp32 vs CPU fp32 `max\|Δ\|=1.49e-7`（cos 0.99999988）；NPU bf16 `2.59e-3`（cos 0.99998170）；10 步训练 loss 与 CPU **逐位一致**（Δ=5.96e-8） |
| 端到端 | **PASS**：2 epoch（1,024 行合成集、bf16 AMP）val AUC 0.8787→**0.9372**、val acc 0.5833→**0.6912** |
| 唯一有性能含义的改动 | **加性 float mask 会让 SDPA 在 950 上退回手工路径**（Cast+Muls+BatchMatMul+Softmax，46.3 µs/次）；改 **bool mask** 才落 `FlashAttentionScoreV4`（12.3 µs/次，**快 3.8×**）；模型端到端 +1.6%（S=64）/ +3.7%（S=256） |

**一句话**：这个参考实现迁移到 950 的改动集中在"设备 + 混合精度 + 数据依赖"三处接线，模型结构和训练语义不需要动；唯一值得改的算法细节是 attention mask 的 dtype。

---

## 1. 项目结构

```
/root/bytedance-onetrans/
├── GPU_Reference/            # 原仓（git clone，未修改；commit 8febd7e）
├── OneTrans_NPU/             # 迁移后实现（与参考实现逐文件可比）
│   ├── main_pytorch.py       # 9 行改动：可选 bool mask（env 开关，默认保持原行为）
│   ├── models/taac_onetrans.py   # 0 行改动
│   ├── utils/
│   │   ├── common.py         # 3 行：去掉 datasets 顶层依赖
│   │   ├── metrics.py        # 0 行
│   │   └── taac_data.py      # 12 行：HF 依赖延迟导入
│   ├── scripts/run_taac2026_sample.py  # 58 行：设备/AMP 委托 + --synthetic
│   ├── npu_compat.py         # 新增 96 行：设备、AMP、scaler
│   └── tests/                # 新增：冒烟 / 精度 / mask 对照 / kernel 探针 / 一键脚本
├── docs/                     # 本指导（md + html）
├── results/                  # 实测日志、kernel_details 台账
└── REFERENCE_COMMIT.txt      # 参考仓 commit hash
```

**未改动证据**：`git -C GPU_Reference status --porcelain` 为空；`REFERENCE_COMMIT.txt` 记录 `8febd7eff9b1f2f63fe1fa4e364af8a8755f51fb`。

---

## 2. 改动清单（逐文件）

| 文件 | 改动 | 改点 | 原因 |
|---|---:|---|---|
| `npu_compat.py` | +96 | `init_device` / `should_enable_amp` / `build_scaler` / `autocast_context` / `describe` | 把"设备差异"收敛到一处，模型与训练逻辑零改动 |
| `scripts/run_taac2026_sample.py` | 58 | 四个设备/AMP 函数改为委托；新增 `--synthetic`/`--synthetic-rows` | 原实现 `device_type != "cuda"` 直接关 AMP，NPU 上跑不到 bf16 |
| `utils/taac_data.py` | 12 | `datasets`/`huggingface_hub` 改为函数内导入 | 容器无 `datasets`/`pyarrow`，顶层导入会让 `--synthetic` 也起不来 |
| `utils/common.py` | 3 | 去掉 `from datasets import Dataset`，注解改无类型 | 同上（仅类型注解用到） |
| `main_pytorch.py` | 9 | `ONETRANS_ATTN_MASK=bool` 时返回 bool 掩码 | 见 §3.4：float 掩码不落 FA |
| `models/taac_onetrans.py` | 0 | — | 模型结构与语义与设备无关 |
| `utils/metrics.py` | 0 | — | 纯指标计算 |

---

## 3. 逐项改动（代码级）

### 3.1 设备：`npu` 不是 `cuda` 的别名

```python
def init_device(device: str) -> str:
    device_type = torch.device(device).type
    if device_type == "npu":
        import torch_npu          # 必须在任何 npu 张量之前
        torch.npu.set_device(0)
    return device_type
```

- 原脚本 `--device` 默认是 `cuda if torch.cuda.is_available() else cpu`，NPU 机器上会落 `cpu`（能跑但慢 10× 以上，且不触发任何 NPU 相关错误，属于**静默降级**）。
- 判据：日志第一行必须打印 `device_type=npu` 且 `soc=Ascend950DT_9582`。

### 3.2 混合精度：NPU 只放行 bf16

| 组合 | 是否可用 | 处理 |
|---|---|---|
| `cuda + fp16` | ✅（GradScaler） | 原逻辑不变 |
| `cuda + bf16` | ✅ | 原逻辑不变 |
| **`npu + bf16`** | ✅ `torch.autocast("npu", bfloat16)` | 放行，不需要 scaler |
| **`npu + fp16`** | ❌ 无可用 GradScaler | **禁用**，回落 fp32 并在日志说明 |

```python
def should_enable_amp(device_type, amp, amp_dtype):
    if not amp or device_type not in ("cuda", "npu"):
        return False
    if amp_dtype == "fp16":
        return device_type == "cuda"
    ...
```

Scaler 用一个 no-op 实现占位（`scale/step/update/unscale_` 全空操作），这样训练循环里 `scaler.scale(loss).backward()` 的调用形态不用改，也避免 `torch.amp.GradScaler(device="npu")` 在 2.7.1 上不可用的风险。

判据：日志里 `amp=True amp_dtype=bf16 grad_scaler=False`。

### 3.3 可选依赖：HF 数据栈延迟导入

容器里 `datasets`、`pyarrow`、`huggingface_hub` 都没装（`huggingface.co` 可达但包缺失）。原实现顶层 `from datasets import ...` 会让**任何**运行入口在 import 阶段就崩。改法：只在 `load_train_split()` 内导入，`utils/common.py` 的类型注解去掉。

判据：`python3 scripts/run_taac2026_sample.py --device npu --synthetic ...` 在未装 HF 栈时能跑完 2 个 epoch。

### 3.4 attention mask 的 dtype 决定落不落 FA（本次最关键）

参考实现 `create_attention_mask()` 返回**加性 float 掩码**（允许位 0、禁止位 `finfo.min`），SDPA 拿到它之后在 950 上**不会**走 flash kernel。实测 kernel 明细（B=8, H=4, S=64, D=16，bf16，3 次迭代）：

| mask 形态 | kernel 构成 | 单次 attention 耗时 |
|---|---|---:|
| **float 加性 mask**（原实现） | `Cast×15` + `Muls×6` + `BatchMatMulV3×6` + `SoftmaxV2×3` + `SelectV2` …（45 个 kernel） | **46.3 µs** |
| **bool mask** | `FlashAttentionScoreV4×3` + `LogicalNot×3`（6 个 kernel） | **12.3 µs**（3.8× 快） |
| 无 mask | `FlashAttentionScoreV4×3` | 9.3 µs |

改法（`main_pytorch.py`，env 开关，默认保持原行为）：

```python
_MASK_DTYPE = os.environ.get("ONETRANS_ATTN_MASK", "float").lower()
...
if self.mask_type == "paper_causal":
    allowed = col_idx <= q_abs
    if _MASK_DTYPE == "bool":
        return allowed            # bool: True = 参与
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)
```

注意语义差异：**float 掩码用 `-inf` 表示禁止，bool 掩码用 `True` 表示参与**。只对 `paper_causal` / `hard_mask`（0/-inf 型）做转换；`origin`（+1e-9 偏置）与 `bimask_soft`（0/1 加性偏置）是软偏置，转 bool 会改语义，保持 float。

模型级收益（同一模型、同数据、20 次中位）：

| 配置 | float mask | bool mask | 差异 |
|---|---:|---:|---:|
| B=64, S=64 | 55.9 ms/步 | 55.0 ms/步 | **+1.6%** |
| B=32, S=256 | 55.8 ms/步 | 53.8 ms/步 | **+3.7%** |

loss 完全一致（Δ≤1.2e-7）。

![mask 形态代价](docs/fig_mask_cost.png)收益不大是因为这个参考实现的成本主要在"每个 token 组一套独立 KQV/FFN 参数"（`ns_len+1` 组小矩阵），attention 只占一小部分——这也是迁移时容易误判的地方：**kernel 层 3.8× 不等于模型层 3.8×**。

### 3.5 数据：`--synthetic` 模式

新增开关，用固定 seed 造 `non_seq(rows,64)`、`seq(rows,seq_len,32)`、`labels`（由 non-seq 线性组合决定，保证优于随机），使冒烟与精度测试不依赖 HF 下载。真实数据路径仍保留（`load_train_split`），装上 HF 栈即可用。

---

## 4. 测试结果

### 4.1 冒烟（`tests/smoke_npu.py`）

```
[env] {'device_type': 'npu', 'torch': '2.7.1+cpu', 'torch_npu': '2.7.1.post4', 'soc': 'Ascend950DT_9582'}
[model] params=6.00M  blocks=1+4  seq_len=64  ns_len=4
[step 0] loss=1.610520 grad_norm=0.9789 610.1 ms
[step 1] loss=1.199051 grad_norm=1.1797  79.4 ms
[step 2] loss=0.794862 grad_norm=1.1984  80.0 ms
[check] loss_finite=True param_delta=3.008e-03 loss_drop=0.815657
SMOKE PASS
```

第 0 步 610 ms 是首步图编译/预热，稳态 79–80 ms。

### 4.2 精度（`tests/accuracy_npu.py`，CPU fp32 作 golden）

| 判据 | 实测 | 门槛 | 结论 |
|---|---|---:|---|
| A. NPU fp32 vs CPU fp32 | max\|Δ\|=**1.49e-7**，cos=0.99999988 | ≤1e-4 / cos≥0.99999 | PASS |
| B. NPU bf16 vs CPU fp32 | max\|Δ\|=**2.59e-3**，cos=0.99998170，rel=5.4e-3 | ≤2e-2 / cos≥0.9995 | PASS |
| C. 10 步训练轨迹 | loss 1.594162→0.375242（两端完全相同，Δ=**5.96e-8**），acc 均 1.0000 | Δloss≤1e-2、Δacc≤2pp | PASS |

方法要点：NPU 模型由 CPU `state_dict` 深拷贝构造（保证同 init），数据用固定 `torch.Generator` 生成，CPU/NPU 各跑同一批数据 10 步。

### 4.3 端到端训练（`scripts/run_taac2026_sample.py`，bf16 AMP）

```
[run] device=npu samples=1024 train=820 val=204
[run] amp=True amp_dtype=bf16 grad_scaler=False mask_type=paper_causal pyramid_layers=3
[epoch 01] train_loss=1.4596 train_auc=0.7168 val_loss=1.2231 val_auc=0.8787 val_acc=0.5833
[epoch 02] train_loss=0.8922 train_auc=0.9557 val_loss=0.8480 val_auc=0.9372 val_acc=0.6912
```

### 4.4 mask 形态对照

见 §3.4 两张表（kernel 明细在 `results/kd_{float_mask,bool_mask,none}.csv`）。

---

## 5. 复现（三条命令）

```bash
# 0) 同步到 122 并放入容器
HOST=/mnt/paas/<工号>/bytedance-onetrans
tar czf - -C /root/bytedance-onetrans OneTrans_NPU | ssh 122 "tar xzf - -C $HOST"
ssh 122 "docker cp $HOST/OneTrans_NPU/. zeus-950-v2:/tmp/onetrans_npu/"

# 1) 冒烟 + 精度 + 端到端（一键）
bash /root/bytedance-onetrans/OneTrans_NPU/tests/run_all_950.sh

# 2) mask 形态对照（可选，两种模式各起一个子进程）
ssh 122 "docker exec zeus-950-v2 bash -lc 'cd /tmp/onetrans_npu && \
  ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /opt/buildtools/torch_v2_pt2.7.1/bin/python tests/bench_mask_mode.py --batch 64 --seq-len 64'"
```

环境：容器 `zeus-950-v2`，解释器 `/opt/buildtools/torch_v2_pt2.7.1/bin/python`（torch 2.7.1+cpu、torch_npu 2.7.1.post4、CANN 9.1.0）。

---

## 6. 踩坑与判据

| 现象 | 原因 | 处置 |
|---|---|---|
| `--device` 默认落 `cpu`，不报错 | 原脚本按 `cuda.is_available()` 选设备 | 显式 `--device npu`，并打印 `device_type` 判据 |
| AMP 在 NPU 上静默失效 | `should_enable_amp` 要求 `device_type == "cuda"` | 兼容层放行 npu+bf16，日志打印 `amp=True grad_scaler=False` |
| `ModuleNotFoundError: datasets` | 容器无 HF 栈 | 延迟导入 + `--synthetic` |
| `module 'torch_npu.npu' has no attribute 'profiler'` | 2.7.1 的 profiler 命名空间是 `torch_npu.profiler` | 用 `torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU])` |
| `UserWarning: Cannot create tensor with interal format...` | NPU 张量工厂对 `preserve_format` 的限制，出现在 autograd `ones_like` | 无害，可忽略 |
| attention 比预期慢 | float 加性 mask 未落 FA | 改 bool mask（§3.4），并用 profiler 断言 kernel 名 |
| 以为 kernel 3.8× 会变成模型 3.8× | 该实现成本主要在多组小 FFN/KQV | 用 `tests/bench_mask_mode.py` 量端到端，不拿 kernel 比值外推 |

---

## 7. 与 OneTrans 生产版（zeus-onetrans）的关系

| 维度 | 本参考实现 | 生产 `zeus-onetrans/rerun-production` |
|---|---|---|
| 任务 | TAAC 分类（HF 数据集） | 搜推排序训练（hkv 稀疏表 + torchrec 1.2.0+npu） |
| attention 入口 | 4D SDPA（`[B,H,S,D]`）+ 加性/bool mask | `npu_fusion_attention_v2`，**3D BSH** `[B,N,H*D]`，`sparse_mode=1` |
| 每层 token 数 | `seq_len + 1 + ns_len`（默认 16+1+4） | 192 / 128 / 96 / 80（pyramid [128,64,32,16,0] + NS 64） |
| 头数 | `num_heads=4`，D=32/头 | **H=1，D=256**（`Q.shape[-1]==d_model` 强制单头） |
| 精度 | bf16 AMP | bf16 + 数值门 2e-2 + 失败回退 SDPA |

可借鉴到生产的结论：① bool mask 才落 FA（生产已是 bool，方向正确）；② NPU 侧 AMP 只走 bf16；③ kernel 级收益不等于端到端收益，必须用端到端步时判定。

---

## 8. 边界（未做）

1. **未跑真实 HF 数据**：容器缺 `datasets`/`pyarrow`，`huggingface.co` 可达但未安装；冒烟与精度均用合成数据（对"迁移正确性"足够，对"任务效果"不作结论）。
2. **单 die 单进程**：未做 DDP/多卡，也未测 8 卡扩展性。
3. **未做长时间收敛验证**：2 epoch 只证明"能训、loss/指标下降"，不代表论文复现精度。
4. **未测其它 mask 模式**：`origin` / `bimask_soft` / `bimask_hard` 只保证可运行（float 路径），未逐一验证数值。
