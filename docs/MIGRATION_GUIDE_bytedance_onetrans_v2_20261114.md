# OneTrans（PyTorch 参考实现）迁移昇腾 950 指导

项目：`/root/bytedance-onetrans/`　版本：v2　日期：2026-11-14
参考实现：`https://github.com/WestbrookLong/OneTrans_Pytorch`，commit **8febd7e**（工作区干净、零改动）
迁移目标：Ascend 950DT（`Ascend950DT_9582`）、CANN 9.1.0（V100R001C11SPC001B243）、torch 2.7.1+cpu / torch_npu 2.7.1.post4

> v2 相对 v1 的变化：① 装上 HF 数据栈并**跑通真数据集**（v1 的"未跑真实 HF 数据"边界已撤）；② 新增**多卡 DDP/HCCL** 章节与 1/2/4/8 die 扩展性实测；③ 新增"用哪个镜像"的确认与判据；④ 踩坑表从 6 条补到 12 条（多卡 6 条全部带 py-spy/日志证据）。

---

## 0. 结论

| 项 | 结果 |
|---|---|
| 迁移面 | 参考实现 Python 共 **1,234 行**，其中改 **204 行（16.5%）**：`main_pytorch.py` 9 / `scripts/run_taac2026_sample.py` 180 / `utils/taac_data.py` 12 / `utils/common.py` 3 / `models/taac_onetrans.py` **0** / `utils/metrics.py` **0**；另新增 `npu_compat.py`（约 150 行）与 `tests/` 5 个探针 |
| 冒烟 | **PASS**：3 步 loss 1.6105→0.7949，参数更新 3.0e-3，稳态 **79–80 ms/步**（6.0M 参数） |
| 精度 | **PASS**：NPU fp32 vs CPU fp32 `max\|Δ\|=1.49e-7`（cos 0.99999988）；NPU bf16 `2.59e-3`（cos 0.99998170）；10 步训练 loss 与 CPU **逐位一致**（Δ=5.96e-8） |
| 端到端（合成） | **PASS**：2 epoch（1,024 行、bf16 AMP）val AUC 0.8787→**0.9372** |
| 端到端（**真数据**） | **PASS**：`TAAC2026/data_sample_1000`（1,000 行 × 120 列 → non_seq 183 维 / seq 45×32 / 2 类）2 epoch 跑完，val AUC 0.5261→**0.5429**。仅证明"链路通"，**不对任务效果下结论**（1000 行、2 epoch、弱标签信号） |
| **多卡** | **PASS**：DDP/HCCL 2/4/8 die 全部跑通；梯度等价性 2 卡 fp32 `worst rel\|Δg\|=1.2e-6`（门槛 1e-5）、`Δloss=0`；4 卡 fp32、2/4 卡 bf16 均 PASS |
| 扩展性 | 每 rank 本地 batch 锁 64：1/2/4/8 die = **867.6 / 1 296 / 2 235 / 3 799.7 samples/s**，加速比 **1.00 / 1.49 / 2.58 / 4.38×**（效率 74.5% / 64.4% / 54.7%） |
| 唯一有性能含义的模型改动 | **加性 float mask 会让 SDPA 在 950 上退回手工路径**（`Cast+Muls+BatchMatMul+Softmax`，46.3 µs/次）；改 **bool mask** 才落 `FlashAttentionScoreV4`（12.3 µs/次，**快 3.8×**）；模型端到端 +1.6%（S=64）/ +3.7%（S=256） |

**一句话**：模型结构和训练语义在 950 上一行不用改；改动集中在「设备 / 混合精度 / 数据依赖 / 多卡装配」四处接线，唯一值得动的算法细节是 attention mask 的 dtype。

---

## 1. 项目结构

```
/root/bytedance-onetrans/
├── GPU_Reference/            # 原仓（git clone，未修改；commit 8febd7e）
├── OneTrans_NPU/             # 迁移后实现（与参考实现逐文件可比）
│   ├── main_pytorch.py       # 9 行：可选 bool mask（env 开关，默认保持原行为）
│   ├── models/taac_onetrans.py   # 0 行
│   ├── utils/common.py       # 3 行：去掉 datasets 顶层依赖
│   ├── utils/metrics.py      # 0 行
│   ├── utils/taac_data.py    # 12 行：HF 依赖延迟导入
│   ├── scripts/run_taac2026_sample.py  # 设备/AMP 委托 + --synthetic + 多卡 DDP
│   ├── npu_compat.py         # 新增：设备、AMP、scaler、HCCL/DDP 装配
│   └── tests/                # 新增：冒烟 / 精度 / mask 对照 / kernel 探针 / DDP 等价性 / HCCL dtype 探针
├── docs/                     # 本指导（md + html）+ npu_changes.diff
├── results/                  # 实测日志（冒烟、精度、真数据、扩展性）
└── REFERENCE_COMMIT.txt      # 参考仓 commit hash
```

**未改动证据**：`git -C GPU_Reference status --porcelain` 为空；`REFERENCE_COMMIT.txt` 记录 `8febd7eff9b1f2f63fe1fa4e364af8a8755f51fb`。

---

## 2. 环境与镜像（先确认再动手）

### 2.1 本次实际使用的运行环境

| 项 | 值 | 判据 |
|---|---|---|
| 容器 | `zeus-950-v2`（122 上） | `docker ps` |
| 镜像 | `zeus-950-v2-snap:20261102` | `docker inspect zeus-950-v2 --format '{{.Config.Image}}'` |
| CANN | 9.1.0 / `V100R001C11SPC001B243` | `readlink -f /usr/local/Ascend/ascend-toolkit/latest` → `/usr/local/Ascend/cann-9.1.0`；`ascend_toolkit_install.info` |
| torch / torch_npu | 2.7.1+cpu / 2.7.1.post4 | 容器内 `python -c "import torch,torch_npu;print(...)"` |
| 多卡前提 | UB 用户态库（`liburma*/libumq*/libdlock*`）+ `/etc/hccl_rootinfo.json` 已挂进容器 | `docker inspect ... .Mounts`；8 die 实测可组网 |

**结论**：这镜像对本任务（纯 PyTorch + torch_npu + HCCL）**是正确的** —— CANN 9.1.0 + torch 2.7.1 正是 v1 记录的环境，8 die 实测能组网、能扩展。

**但它是别人工作区的 snapshot（commit）镜像，不是干净发布镜像**：换镜像会改掉所有数字。备选 GA 镜像 `onetrans-950-ga:v2-ub-20261024` 实测默认是 **torch 2.10.0+cpu / torch_npu 2.10.0.post4**（同时也带 `/opt/buildtools/torch_v2_pt2.7.1`）。所以：

- 要**复现本文数字** → 用 2.7.1 解释器：`/opt/buildtools/torch_v2_pt2.7.1/bin/python`；
- 要**干净可交付环境** → 用 GA 镜像新建容器 + 指定 2.7.1 解释器，**重跑全部测试并重出数字**，不要把 2.10 的结果和本文混在同一张表里。

判据：`[env]` 行必须打印 `torch=2.7.1+cpu torch_npu=2.7.1.post4 soc=Ascend950DT_9582`。

### 2.2 HF 数据栈（v1 的缺口，v2 已补）

容器内 `huggingface.co` **不可达**（`curl` 返回 000），`pypi.org` 时通时不通，`hf-mirror.com` 与 `pypi.tuna.tsinghua.edu.cn` 可达。做法：

```bash
# 1) 装包（清华源）
/opt/buildtools/torch_v2_pt2.7.1/bin/python -m pip install --no-input \
  -i https://pypi.tuna.tsinghua.edu.cn/simple datasets huggingface_hub pyarrow
# 实得：datasets 5.2.0 / huggingface_hub 2.4.0 / pyarrow 26.0.1

# 2) 下载走镜像（必须显式设，否则默认 endpoint 直接超时）
export HF_ENDPOINT=https://hf-mirror.com
```

判据：`load_train_split("TAAC2026/data_sample_1000", cache)` 返回 1000 行 / 120 列；训练脚本 `[run] device=npu samples=1000 train=800 val=200`。实测日志 `results/train_hf_real_950_20261114.txt`。

注意：代码里仍保留 v1 的"HF 依赖延迟导入 + `--synthetic`"，所以**没装 HF 栈也能跑冒烟/精度**，两条路径互不影响。

---

## 3. 改动清单（逐文件）

| 文件 | 改动 | 改点 | 原因 |
|---|---|---|---|
| `npu_compat.py` | +约 150 | `init_device` / `should_enable_amp` / `build_scaler` / `autocast_context` / `describe` / `init_distributed` / `shutdown` | 把设备与多卡差异收敛到一处，模型与训练逻辑零改动 |
| `scripts/run_taac2026_sample.py` | 180 | 设备/AMP 委托；`--synthetic`；DDP 装配、数据切分、rank0 侧打印与计时、收尾 | 原实现 `device_type != "cuda"` 直接关 AMP；无多卡；无合成数据入口 |
| `utils/taac_data.py` | 12 | `datasets`/`huggingface_hub` 改为函数内导入 | 容器可无 HF 栈 |
| `utils/common.py` | 3 | 去掉 `from datasets import Dataset` | 同上 |
| `main_pytorch.py` | 9 | `ONETRANS_ATTN_MASK=bool` 时返回 bool 掩码 | float 掩码不落 FA（§4.4） |
| `models/taac_onetrans.py` | **0** | — | 与设备无关 |
| `utils/metrics.py` | **0** | — | 纯指标计算 |

---

## 4. 逐项改动（代码级）

### 4.1 设备：`npu` 不是 `cuda` 的别名

```python
def init_device(device: str) -> str:
    device_type = torch.device(device).type
    if device_type == "npu":
        import torch_npu          # 必须在任何 npu 张量之前
        torch.npu.set_device(0)
    return device_type
```

原脚本 `--device` 默认 `cuda if torch.cuda.is_available() else cpu`，NPU 机器上会落 `cpu`（能跑但慢 10× 以上、且不报任何 NPU 相关错误，属于**静默降级**）。判据：日志首行 `device_type=npu` 且 `soc=Ascend950DT_9582`。

### 4.2 混合精度：NPU 只放行 bf16

| 组合 | 可用 | 处理 |
|---|---|---|
| `cuda + fp16` | ✅（GradScaler） | 原逻辑不变 |
| `cuda + bf16` | ✅ | 原逻辑不变 |
| **`npu + bf16`** | ✅ `torch.autocast("npu", bfloat16)` | 放行，不需要 scaler |
| **`npu + fp16`** | ❌ 无可用 GradScaler | **禁用**，回落 fp32 并在日志说明 |

Scaler 用 no-op 占位（`scale/step/update/unscale_` 空操作），训练循环调用形态不变。判据：`amp=True amp_dtype=bf16 grad_scaler=False`。

**验证混合精度要分成"生效"和"划算"两件事**（只看 `amp=True` 不够，它只是我自己函数的返回值）：

*生效性*（`tests/probe_amp_npu.py`，实测）：

| 检查 | 实测 | 含义 |
|---|---|---|
| `torch.get_autocast_dtype("npu")` | **float16** | 不显式给 dtype 就是 fp16；训练必须写 `dtype=bfloat16` |
| `torch.is_autocast_enabled("npu")`（ctx 内） | **True** | 不是"WARNING 后静默禁用" |
| `matmul` 输出 dtype（ctx 内） | **bfloat16**（ctx 外 float32、默认 ctx 内 **float16**） | 算子真的降了精度 |
| no-op scaler | `is_enabled()=False`、`state_dict()={}`、`scale()` 形状不变 | 调用形态与真 GradScaler 等价 |
| 放行规则 | `npu/bf16=True`、`npu/fp16=**False**`、`cuda/fp16=True`、`cpu/bf16=False` | fp16 在 NPU 被挡掉；实测 `--amp-dtype fp16` 时日志 `amp=False`，训练仍正常（回落 fp32） |

*划算性*（`tests/bench_amp_npu.py`，同一进程交替跑 fp32 / bf16，3 轮×10 步取中位，剥离前 2 步预热）：

| 档位 | 步时中位 | 峰值 HBM 增量 |
|---|---:|---:|
| fp32 | **54.3 ms** | 109 MiB |
| bf16 autocast | **69.3 ms** | 99 MiB |
| 比值 | **1.275（bf16 慢 27.5%）** | −9% |

**这个参考实现上 bf16 是负收益**：模型只有 6.0M 参数、每个 token 组一套小矩阵，GEMM 太小，吃不到 bf16 算力，反而多付 cast 开销；显存只省 9%。所以本实现默认 fp32，bf16 只在"显存吃紧"或"换成大 GEMM 模型"时开。**不要拿"开了 AMP"当优化项写进报告**，要用同进程 A/B 的步时判定。

### 4.3 可选依赖：HF 数据栈延迟导入

只在 `load_train_split()` / `fallback_download()` 内导入；`utils/common.py` 的类型注解去掉。判据：未装 HF 栈时 `--synthetic` 能跑完 2 epoch。

### 4.4 attention mask 的 dtype 决定落不落 FA（本次最关键的单点改动）

参考实现返回**加性 float 掩码**（允许位 0、禁止位 `finfo.min`），SDPA 在 950 上**不走** flash kernel。实测（B=8, H=4, S=64, D=16，bf16，3 次迭代）：

| mask 形态 | kernel 构成 | 单次 attention |
|---|---|---:|
| **float 加性 mask**（原实现） | `Cast×15 + Muls×6 + BatchMatMulV3×6 + SoftmaxV2×3 + SelectV2`…（45 个 kernel） | **46.3 µs** |
| **bool mask** | `FlashAttentionScoreV4×3 + LogicalNot×3`（6 个 kernel） | **12.3 µs**（3.8×） |
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

语义差异：**float 用 `-inf` 表示禁止，bool 用 `True` 表示参与**。只对 `paper_causal` / `hard_mask`（0/-inf 型）转换；`origin`（+1e-9 偏置）与 `bimask_soft`（0/1 加性偏置）是软偏置，转 bool 会改语义，保持 float。

模型级收益（同模型同数据，20 次中位）：B=64/S=64 55.9→55.0 ms（**+1.6%**）；B=32/S=256 55.8→53.8 ms（**+3.7%**）；loss 完全一致（Δ≤1.2e-7）。

收益不大是因为该实现成本主要在"每个 token 组一套独立 KQV/FFN 参数"（`ns_len+1` 组小矩阵）。**kernel 层 3.8× ≠ 模型层 3.8×**，迁移时最容易误判。

### 4.5 多卡：与生产 zeus-onetrans 对齐的三条约定

生产 `rerun-production/work/Onetrans/train_zeus_v2prof.py` 的写法（本次照抄）：

```python
# main_rank.py：rendezvous 与设备
dist.init_process_group("hccl", init_method=f"tcp://{addr}:{port}", rank=rank_id, world_size=world_size)
torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))

# train_zeus_v2prof.py：包壳参数
model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
if int(os.environ.get("ZEUS_TOWER_DDP", "1")) == 1:
    loss_head = DDP(loss_head, device_ids=[local_rank], broadcast_buffers=False)
```

三条约定：

1. **`device_ids=[local_rank]` + `broadcast_buffers=False` + `find_unused_parameters=True`** —— 本模型的 `find_unused_parameters` 不是可选项（见 §6 第 5 条）。
2. **训练循环里不做任何跨 rank 指标归约** —— 生产侧只记录各 rank 自己的 `loss.item()`（`rank==0` 或 `ZEUS_LOG_ALL_RANKS=1`）。本文照此实现：训练集按 rank 切分，**验证集在 rank0 上跑全量**，因此不变量（loss/AUC/acc）本身就是全集口径，不需要集合通信。
3. **有参数的子模块必须在 DDP 内** —— 生产代码有明确教训注释（`[rerun-v2 P1-7]`）：8 个任务塔原来在 DDP 之外，导致"各 rank 参数各自漂移、仅 rank0 被保存"。本实现只有一个 model，无此问题，但改组时必须记住。

落地代码（`npu_compat.init_distributed`）：

```python
world_size = int(os.environ.get("WORLD_SIZE", "1"))
if world_size <= 1:
    return 0, 1, 0                      # 单进程路径行为完全不变
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.npu.set_device(local_rank)        # ASCEND_RT_VISIBLE_DEVICES 已重编号，local_rank 即可见卡序号
dist.init_process_group(backend="hccl", timeout=datetime.timedelta(minutes=10))
return dist.get_rank(), world_size, local_rank
```

数据切分：`train_idx = train_idx[rank::world_size]`（stride 切分，不填充、不重复）；`rank != 0` 时验证集给空集。

启动：`torchrun --nproc_per_node=W --master_port=<port> scripts/run_taac2026_sample.py --device npu ... --batch-size <每卡 batch>`

---

## 5. 测试结果

### 5.1 冒烟（`tests/smoke_npu.py`）

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

### 5.2 精度（`tests/accuracy_npu.py`，CPU fp32 作 golden）

| 判据 | 实测 | 门槛 | 结论 |
|---|---|---|---|
| A. NPU fp32 vs CPU fp32 | max\|Δ\|=**1.49e-7**，cos=0.99999988 | ≤1e-4 / cos≥0.99999 | PASS |
| B. NPU bf16 vs CPU fp32 | max\|Δ\|=**2.59e-3**，cos=0.99998170 | ≤2e-2 / cos≥0.9995 | PASS |
| C. 10 步训练轨迹 | loss 1.594162→0.375242（两端相同，Δ=**5.96e-8**），acc 均 1.0000 | Δloss≤1e-2、Δacc≤2pp | PASS |

方法要点：NPU 模型由 CPU `state_dict` 深拷贝构造（保证同 init），数据用固定 `torch.Generator`，CPU/NPU 各跑同一批数据 10 步。

### 5.3 端到端（合成数据 / 真数据）

```
# 合成集（v1）
[run] device=npu samples=1024 train=820 val=204
[epoch 02] train_loss=0.8922 train_auc=0.9557 val_loss=0.8480 val_auc=0.9372 val_acc=0.6912

# 真数据 TAAC2026/data_sample_1000（v2，单 die）
[run] device=npu samples=1000 train=800 val=200
[run] non_seq=(1000, 183) seq=(1000, 32, 45) classes=2
[epoch 01] train_loss=0.3974 train_auc=0.4443 val_loss=0.4227 val_auc=0.5261 val_acc=0.8500
[epoch 02] train_loss=0.3684 train_auc=0.4554 val_loss=0.4246 val_auc=0.5429 val_acc=0.8500
```

真数据链路证明的是"能下载、能张量化、能前反向、能出指标"，**不是任务效果**：1000 行、2 epoch、二分类弱信号，AUC 0.53 属正常。

### 5.4 多卡梯度等价性（`tests/ddp_equiv_npu.py`）

| 规模 | dtype | worst rel\|Δg\|（门槛） | Δloss | 结论 |
|---|---|---|---:|---|
| W=2，每卡 8 | fp32 | **1.2e-6**（1e-5） | 0 | PASS |
| W=4，每卡 8 | fp32 | < 1e-5（1e-5） | 0 | PASS |
| W=2 / W=4 | bf16 | < 2e-2（2e-2） | 0 | PASS |

口径：DDP（W 卡 × 局部 batch）all-reduce 后的平均梯度 == 单进程 global-batch 梯度；loss 用各 rank 局部 loss 的均值。参考值在**同一设备**上重算（跨设备比较会位移小数点）。

### 5.5 扩展性（每 rank 本地 batch 固定 64；synthetic 16384 行；3 epoch；bf16 AMP）

| die 数 | 全 batch | 稳态 step | samples/s | 加速比 | 每卡效率 | step 相对 1 卡增量 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 74.0 ms | 867.6 | 1.00× | 100% | — |
| 2 | 128 | 98.6 ms | 1 296.0 | **1.49×** | 74.5% | +24.6 ms |
| 4 | 256 | 114.5 ms | 2 235.0 | **2.58×** | 64.4% | +40.5 ms |
| 8 | 512 | 134.7 ms | 3 799.7 | **4.38×** | 54.7% | +60.7 ms |

口径必须写清楚：**锁每卡本地 batch**（否则扩展比会 <1 或 >卡数，属口径错）。原始日志 `results/scaling_ddp_950_20261114.txt`。

另一个口径（**固定全局 batch = 64**，即每卡 batch = 64/W）实测**反而更慢**：1 卡 76.4 ms（838 samples/s）→ 2 卡 103 ms（621 samples/s）= **0.74×**。原因是该模型每步固定开销大（每个 token 组一套小矩阵），batch 减半并不减半步时，而 DDP 同步成本全暴露。**结论：这个实现要多卡提速，必须让每卡 batch 不掉下来（线性缩放 batch + 相应调 lr），不要固定全局 batch 加卡。**

---

## 6. 踩坑与判据（12 条，全部有本机证据）

| # | 现象 | 原因 | 处置 |
|---:|---|---|---|
| 1 | `--device` 默认落 `cpu`，不报错 | 原脚本按 `cuda.is_available()` 选设备 | 显式 `--device npu`，打印 `device_type` 判据 |
| 2 | AMP 在 NPU 上静默失效 | `should_enable_amp` 要求 `device_type == "cuda"` | 兼容层放行 npu+bf16，日志 `amp=True grad_scaler=False` |
| 3 | `ModuleNotFoundError: datasets` | 容器无 HF 栈 | 延迟导入 + `--synthetic`；或按 §2.2 安装 |
| 4 | `module 'torch_npu.npu' has no attribute 'profiler'` | 2.7.1 的 profiler 在 `torch_npu.profiler` | 用 `torch_npu.profiler.profile(activities=[...ProfilerActivity.NPU])` |
| 5 | `UserWarning: Cannot create tensor with interal format...` | NPU 张量工厂对 `preserve_format` 的限制（autograd `ones_like`） | 无害，忽略 |
| 6 | attention 比预期慢 | float 加性 mask 未落 FA | 改 bool mask（§4.4），profiler 断言 kernel 名 |
| 7 | **训练 epoch 结束后一汇总指标就卡死**（无报错无输出） | 在训练循环后紧接自己的集合通信：`all_gather_object` 走 pickle/`torch.save`、张量 `all_gather` 后直接 `.cpu()` 也会停 | **先 `torch.npu.synchronize()` 再 D2H**；但**最终方案是不在循环里做跨卡归约**（§4.5 约定 2）。py-spy 栈证据：停在 `b.cpu()` |
| 8 | `HCCL allreduce: Unsupported data type at::kDouble` | HCCL 标量归约不支持 fp64 | 归约张量用 **float32** |
| 9 | `all_reduce(op=MAX)` 在 float 上**卡住**（不是报错） | 该组合 HCCl 未支持 | 改用 `int64 all_gather` 各 rank 尺寸后取本地 max（`tests/probe_hccl_dtypes.py` 可复现：fp32/bf16/fp16/int32/int64/uint8 的 all_gather 全 OK，fp64 all_reduce 报错） |
| 10 | DDP 报 `Expected to have finished reduction in the prior iteration` | 模型最深金字塔块有 **16 个参数不参与前向**（`stack_blocks.2.otb_list.*` 的 token 组索引 4 超出该层实际组数） | `find_unused_parameters=True`（生产同），默认开；确认每步都用满参数才用 `--ddp-strict` 关 |
| 11 | 多卡跑完 `torchrun` 报 `ChildFailedError`，exitcode **-11 / -7** | 训练已完成，崩溃发生在解释器 teardown（SIGSEGV/SIGBUS） | `npu_compat.shutdown()`（sync + `destroy_process_group`）+ `os._exit(0)`；**不要把这个当成训练失败** |
| 12 | `Communication_Error_Bind_IP_Port(EI0020) ... port 16666 already bound` | 上一次挂死进程仍占 HCCL 固定端口 | 起跑前清残留进程（`pkill -f run_taac2026`；模式串会自匹配，按 `/proc/*/cmdline` 过滤更稳） |

---

## 7. 与 OneTrans 生产版（zeus-onetrans）的关系

| 维度 | 本参考实现 | 生产 `zeus-onetrans/rerun-production` |
|---|---|---|
| 任务 | TAAC 分类（HF 数据集） | 搜推排序训练（hkv 稀疏表 + torchrec 1.2.0+npu） |
| attention 入口 | 4D SDPA（`[B,H,S,D]`）+ 加性/bool mask | `npu_fusion_attention_v2`，**3D BSH**，`sparse_mode=1` |
| 每层 token 数 | `seq_len + 1 + ns_len`（默认 16+1+4） | 192 / 128 / 96 / 80（pyramid [128,64,32,16,0] + NS 64） |
| 头数 | 4 头，D=32/头 | **H=1，D=256**（`Q.shape[-1]==d_model` 强制单头） |
| 精度 | bf16 AMP | bf16 + 数值门 2e-2 + 失败回退 SDPA |
| 多卡 | DDP + HCCL，单机 8 die | DDP + HCCL + DMP/UB，**16 卡跨 122/111**（`ub_verbs --apply 8`，NNODES/NODE_RANK/MASTER_ADDR 走 env） |
| 指标归约 | 不做跨卡归约；rank0 全量验证 | 不做跨卡归约；各 rank 记录本地 loss |

可借鉴到生产的结论：① bool mask 才落 FA；② NPU 侧 AMP 只走 bf16；③ kernel 级收益不等于端到端收益；④ 训练循环内不加集合通信；⑤ 子模块必须都在 DDP 内。

---

## 8. 复现

```bash
# 0) 同步并放进容器（122）
cd /root/bytedance-onetrans && tar czf - OneTrans_NPU | ssh 122 "H=/mnt/paas/<工号>/bytedance-onetrans; \
  rm -rf \$H/OneTrans_NPU; mkdir -p \$H; tar xzf - -C \$H"
ssh 122 "docker exec zeus-950-v2 bash -lc 'rm -rf /tmp/onetrans_npu && mkdir -p /tmp/onetrans_npu' && \
  docker cp /mnt/paas/<工号>/bytedance-onetrans/OneTrans_NPU/. zeus-950-v2:/tmp/onetrans_npu/"

# 1) 单卡：冒烟 + 精度 + 端到端（一键）
bash /root/bytedance-onetrans/OneTrans_NPU/tests/run_all_950.sh

# 2) 真数据（需 §2.2 的 HF 栈 + 镜像 endpoint）
ssh 122 "docker exec zeus-950-v2 bash -lc 'cd /tmp/onetrans_npu && HF_ENDPOINT=https://hf-mirror.com \
  ASCEND_RT_VISIBLE_DEVICES=0 /opt/buildtools/torch_v2_pt2.7.1/bin/python scripts/run_taac2026_sample.py \
  --device npu --epochs 2 --batch-size 64 --seq-len 32 --d-model 64 --multi-num 2 --num-pyramid-layers 3 \
  --amp --amp-dtype bf16'"

# 3) 多卡：等价性 + 1/2/4/8 扩展性
bash /root/bytedance-onetrans/OneTrans_NPU/tests/run_ddp_950.sh 64 8192

# 4) mask 形态对照（可选）
ssh 122 "docker exec zeus-950-v2 bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=0 \
  /opt/buildtools/torch_v2_pt2.7.1/bin/python tests/bench_mask_mode.py --batch 64 --seq-len 64'"
```

解释器：`/opt/buildtools/torch_v2_pt2.7.1/bin/python`（torch 2.7.1+cpu、torch_npu 2.7.1.post4、CANN 9.1.0）。

---

## 9. 边界（未做 / 不作结论）

1. **真数据只到"链路通"**：1000 行、2 epoch、二分类弱信号（val AUC 0.5261→0.5429），不下任务效果结论；也未做全量 TAAC 训练。
2. **多卡只测单机**：122 上 8 die（`ASCEND_RT_VISIBLE_DEVICES=0..7`）已实测；**未做跨机 16 卡**（那需要 `ub_verbs --apply` 与 NNODES/NODE_RANK/MASTER_ADDR 走 env，见 `onetrans-training`）。
3. **未做长时间收敛验证**：2–3 epoch 只证明"能训、指标下降"。
4. **未测其它 mask 模式的数值**：`origin`/`bimask_soft`/`bimask_hard` 只保证 float 路径可运行。
5. **多卡扩展性数字绑定本模型规模**（6.0M 参数）：更大模型/更大 batch 下 DDP 同步占比会下降，加速比应重测。
