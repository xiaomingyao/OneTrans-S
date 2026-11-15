# onetrans 迁移昇腾 950：按通用五阶段流程的实现对照

**对象**：`https://github.com/WestbrookLong/OneTrans_Pytorch`（commit `8febd7e`，`GPU_Reference/` 零改动）→ `OneTrans_NPU/`
**环境**：Ascend950DT_9582 × 1 / CANN 9.1.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 / Python 3.11.0
**组织方式**：按通用流程五阶段展开（环境与设备 → 稀疏特征 → 注意力算子 → 混合精度 → 精度固化）；每阶段给出 **GPU 实现与 NPU 实现并排**，再给「为什么」与「判据」。

## 0. 结论摘要

| 阶段 | 本模型的改动 | 判据（实测） | 结论 |
|---|---|---:|---|
| 一 环境与设备 | 新增 `npu_compat.py` 160 行；脚本装配层 138 行 | `device_type=npu` + SoC 名正确 | 必改 |
| 二 稀疏特征与 Embedding | **0 行** | `grep -c "nn.Embedding\|EmbeddingBag\|torchrec"` = 0 | 不改 |
| 三 注意力算子 | `main_pytorch.py` 9 行（env 开关） | kernel 45→6、46.3→12.3 µs | 可选，收益 1.6–3.7% |
| 四 混合精度 | 0 行（走兼容层） | fp32 56.0 ms vs bf16 69.0 ms | **关掉更划算** |
| 五 精度固化 | 新增测试 6 个脚本 | A 1.49e-7 / B 2.591e-3 / C Δloss 5.96e-8 | 三关全 PASS |

模型文件（`models/taac_onetrans.py`）、损失、指标、数据管道 **全部 0 行改动**；改动集中在装配层，占参考实现 1,233 行的 **5.4%（基础）/ 11.9%（含多卡）**。

---

## 1. 场景画像

| 项 | 值 |
|---|---|
| 任务 | TAAC 数据集 5 类分类 → 实际标签 2 类（`label_mapping {1:0, 2:1}`） |
| 输入 | `[seq_tokens(16) , SEP , ns_tokens(4)]`，non_seq 183 维 + seq 45 特征 / seq_len 32 |
| 结构 | tokenizer（nn.Linear）→ base block × 2 → 金字塔压缩（linear schedule）→ 池化 → 分类头 |
| 特性 | 每个 token 组独立 KQV/FFN 参数（`ns_len+1` 组）；注意力 4D SDPA；5 种 mask 模式 |
| 参数 | 6.00M |
| 参考侧基线 | CPU fp32；GPU 侧未在本机测（无 GPU 卡），数值对照统一以 CPU fp32 作 golden |

---

## 2. 阶段一：环境与设备

### 2.1 设备选择：`npu` 不是 `cuda` 的别名

```python gpu GPU_Reference/scripts/run_taac2026_sample.py:26
def parse_device_type(device: str) -> str:
    return torch.device(device).type


# 调用侧：默认值来自"CUDA 可用则 cuda，否则 cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
```

```python npu OneTrans_NPU/npu_compat.py:22
def init_device(device: str) -> str:
    """初始化设备并返回 device_type（'cpu' / 'cuda' / 'npu'）。"""
    device_type = torch.device(device).type
    if device_type == "npu":
        import torch_npu  # noqa: F401  # 必须在任何 npu 张量之前导入

        torch.npu.set_device(0)
    return device_type
```

**为什么不能只 `import torch_npu`**：`import torch_npu` 只注册 PrivateUse1 后端，不替换任何 CUDA API——`cuda.is_available()` 仍是 False，原脚本的默认值会把设备判成 `cpu`，**不报错、只慢十几倍**，属于静默降级。实测三模式对照：只 `import torch` / 只 `import torch_npu` 时 `cuda.is_available()` 都是 False，只有 `transfer_to_npu` 才为 True，而它会抹掉"到底跑在哪颗芯片"这条判据。

**判据**：启动日志必须有 `device_type=npu` + SoC 名 + 版本指纹：

```bash
[env] {'device_type': 'npu', 'torch': '2.7.1+cpu',
       'torch_npu': '2.7.1.post4', 'soc': 'Ascend950DT_9582'}
```

### 2.2 环境指纹：报数字前先固定这四项

```python npu OneTrans_NPU/npu_compat.py:87
def describe(device_type: str) -> dict[str, str]:
    info = {"device_type": device_type, "torch": torch.__version__}
    if device_type == "npu":
        import torch_npu
        info["torch_npu"] = torch_npu.__version__
        info["soc"] = torch.npu.get_device_name(0).split("/")[-1]
    elif device_type == "cuda":
        info["cuda"] = torch.version.cuda or ""
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
```

容器里两套 torch 并存（3.12 + torch 2.10.0 / 3.11 + torch 2.7.1+post4），**必须先确认用的哪套**：本工程全部数据来自 `/opt/buildtools/torch_v2_pt2.7.1/bin/python`。同一例在两套栈上复跑差 0.16%，但版本号混着报等于没报。

### 2.3 多卡：DDP over HCCL

**先交代 GPU 侧的事实**：参考实现**只有单进程单卡**（`--device cuda`），全仓 `init_process_group` / `dist.` / `DistributedDataParallel` / `torchrun` 命中数 **= 0**（判据：`grep -rn 'dist\.\|DistributedDataParallel\|init_process_group' --include='*.py' GPU_Reference | wc -l` → `0`）。所以本节 GPU 栏给的是**等价写法**（明确标注：参考仓未提供），NPU 栏才是新增接线。

```python gpu GPU_Reference/scripts/run_taac2026_sample.py:109
# GPU 参考实现：单卡。设备只由 --device 决定，没有任何分布式初始化
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.set_defaults(amp=torch.cuda.is_available())
...
model = build_model(args, non_seq_x, seq_x, labels).to(args.device)   # :228
# 全仓无 init_process_group / DDP / torchrun ⇒ 单进程单卡
```

```python npu OneTrans_NPU/npu_compat.py:111
def init_distributed() -> tuple[int, int, int]:
    """按 torchrun 注入的 env 初始化 HCCL，返回 (rank, world_size, local_rank)。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0          # 单进程：行为与参考实现逐字等价（不多做一次通信初始化）
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl", timeout=datetime.timedelta(minutes=10))
    return dist.get_rank(), world_size, local_rank
```

```python gpu GPU_Reference/scripts/run_taac2026_sample.py（等价写法，参考仓未提供）
# 若要在 GPU 上做同样的多卡，等价形态如下（本仓没有这段代码，需自行补；
# 启动方式：torchrun --nproc_per_node=N scripts/run_taac2026_sample.py --device cuda）
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    find_unused_parameters=True,
    broadcast_buffers=False,
)
```

```python npu OneTrans_NPU/scripts/run_taac2026_sample.py:337
if world_size > 1:
    # DDP 只包外壳，模型定义零改动；梯度按 rank 平均 = 全局 batch 的均值梯度
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if device_type == "npu" else None,
        find_unused_parameters=not args.ddp_strict,   # 最深金字塔块有 16 个参数不参与前向
        broadcast_buffers=False,                      # 本模型无 BN，省一次广播
    )
```

```python npu OneTrans_NPU/npu_compat.py:101
# 与生产 zeus-onetrans 对齐的三条约定
#   ① DDP(model, device_ids=[local_rank], broadcast_buffers=False,
#          find_unused_parameters=True)
#   ② 训练循环里不做任何跨 rank 指标归约（生产侧只记录各 rank 本地 loss）
#   ③ 验证在 rank0 上跑全量，训练集按 rank 切分
# 反例（实测）：在训练 epoch 结束后紧接着发 all_gather/all_gather_object
# 汇总指标，进程会停在 D2H 拷贝上一动不动（py-spy 栈在 b.cpu()），无报错、无输出。
```

**GPU 侧 ↔ NPU 侧的三处差别**（同一段多卡逻辑，写法不同）

| # | GPU（等价写法） | NPU（本工程） | 为什么 |
|---|---|---|---|
| 1 | `init_process_group("nccl")` | `init_process_group("hccl")` | 通信后端；`nccl` 在 NPU 上前提就不存在 |
| 2 | 由 `torchrun` 自动 `set_device`，也可不显式设 | 必须显式 `torch.npu.set_device(local_rank)` | 不设则进程落到 `npu:0`，多进程抢同一 die |
| 3 | `find_unused_parameters=True` 可省（模型无分支） | **必开**：最深金字塔块 16 个参数不参与前向 | 省了会报 `Expected to have finished reduction in the prior iteration` |

**为什么 `find_unused_parameters=True` 不是可选项**：金字塔压缩让部分参数在浅层步不参与前向，缺了它会直接报 `Expected to have finished reduction in the prior iteration`。

**为什么训练循环里不许做集合通信**：本工程实测过——epoch 末尾 `all_gather_object` 卡在 pickle + D2H，进程静默挂死；改成"各 rank 写本地文件、rank0 事后聚合"后正常。另：跨卡归约的标量必须 float32，HCCL 不支持 fp64（`Unsupported data type at::kDouble`）。

**判据**：`[dist] world_size=N rank=r local_rank=l` 行 + `torchrun` 退出码 0（若见 exitcode -11，是 teardown SIGSEGV，训练其实已完成，用 `shutdown()`+`os._exit(0)` 收尾）；多卡梯度等价 `rel|Δg| ≤ 1e-5`（fp32，`tests/ddp_equiv_npu.py`）。

---

## 3. 阶段二：稀疏特征与 Embedding —— 本模型 0 改动

```python gpu GPU_Reference/models/taac_onetrans.py:61
# 特征表示：整数/浮点特征先拼成向量，再用 nn.Linear 投影成 token（稠密路径）
self.non_seq_tokenizer = nn.Linear(non_seq_dim, ns_len * d_model)   # 183 → ns_len*d_model
self.seq_tokenizer = nn.Linear(seq_feature_dim, d_model)            # 45 → d_model

# 全仓无 nn.Embedding / nn.EmbeddingBag / torchrec / fbgemm_gpu / sparse table
```

```python npu OneTrans_NPU/models/taac_onetrans.py:61
# NPU 侧：同一份文件，0 行改动
# 判据（实测）：grep -c "nn.Embedding|EmbeddingBag|torchrec|fbgemm" → 0
```

**结论**：`models/taac_onetrans.py` 与参考实现逐字节相同。稀疏侧（hkv 稀疏表、torchrec 分片、embedding cache）在本模型里不存在，所以 §2 阶段二的全部工作量为 0。

**什么情况下这个阶段才会变成 10%+ 的改动面**（对照生产 OneTrans 的经验）：

| 特征 | 现象 | 需要做的事 |
|---|---|---|
| 有 `nn.EmbeddingBag` / `torchrec.EmbeddingCollection` | NPU 无对应 kernel，报算子不支持 | 换 torchrec 1.2.0+npu 或改密集表（视规模） |
| 索引 `int64` 且基数 > 2^31 | NPU 侧 index_select 溢出 | 降 int32 并校验基数 |
| 带 `fbgemm_gpu` 依赖 | import 直接失败 | 该模块只能在 GPU 上跑，需改写为稠密或 CPU 侧预处理 |
| 稀疏表分布不均 | 多卡负载倾斜、embedding 通信放大 | 按频次重分片（生产侧做法） |

**判据**：先 grep 再谈改动面。0 命中就直接写"本阶段无改动"，不要为了凑工作量去改数据代码。

---

## 4. 阶段三：注意力与算子落点

### 4.1 调用点：同一个 SDPA，落不同 kernel

```python gpu GPU_Reference/main_pytorch.py:127
output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_mask,      # 加性 float 掩码（0 / finfo.min）
    dropout_p=0.0,
    is_causal=False,
)
```

```python npu OneTrans_NPU/main_pytorch.py:127
output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_mask,      # bool 掩码（True = 参与）时落 FlashAttentionScoreV4
    dropout_p=0.0,
    is_causal=False,
)
```

调用点**完全没变**——变的只是 mask 的 dtype。

### 4.2 mask 构造：这一处才是性能开关

```python gpu GPU_Reference/main_pytorch.py:65
def create_attention_mask(self, query_len, key_len, device, dtype):
    if self.mask_type == "paper_causal":
        row_idx = torch.arange(query_len, device=device).unsqueeze(1)
        col_idx = torch.arange(key_len, device=device).unsqueeze(0)
        q_abs = row_idx + (key_len - query_len)
        allowed = col_idx <= q_abs
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)
```

```python npu OneTrans_NPU/main_pytorch.py:70
_MASK_DTYPE = os.environ.get("ONETRANS_ATTN_MASK", "float").lower()

def create_attention_mask(self, query_len, key_len, device, dtype):
    if self.mask_type == "paper_causal":
        row_idx = torch.arange(query_len, device=device).unsqueeze(1)
        col_idx = torch.arange(key_len, device=device).unsqueeze(0)
        q_abs = row_idx + (key_len - query_len)
        allowed = col_idx <= q_abs
        if _MASK_DTYPE == "bool":
            return allowed                      # True = 参与 → 落 FA
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)
```

**为什么默认仍是 float**：语义等价优先。`ONETRANS_ATTN_MASK=float` 与参考实现逐字一致，是数值对照的基准；`=bool` 是**有性能含义的等价改写**，用 env 开关切换，两边都能跑同一套精度用例（实测 loss 差 ≤1.2e-7）。

**语义方向不能想当然**（三个 API 两套约定）：

| 接口 | 掩码类型 | 语义 |
|---|---|---|
| `F.scaled_dot_product_attention` | float 加性 | 0 = 允许，`finfo.min` = 禁止 |
| `F.scaled_dot_product_attention` | bool | **True = 参与** |
| `torch_npu.npu_fusion_attention` | bool (`atten_mask`) | **True = 丢弃**（与 SDPA 相反） |

软偏置型 mask（`origin` 的 `+1e-9`、`bimask_soft` 的 0/1）**不能转 bool**——它们表达的是偏置不是禁止，转了会改变数值。本工程只对 `paper_causal` / `hard_mask` 两个"纯禁止"分支提供 bool 版本。

### 4.3 kernel 证据（同一 attention，B8/H4/S64/D16，bf16）

| mask 形态 | kernel 构成 | 单次耗时 | kernel 数 |
|---|---|---:|---:|
| float 加性（参考默认） | `Cast×15 + Muls×6 + BatchMatMulV3×6 + SoftmaxV2×3 + SelectV2…` | 46.3 µs | 45 |
| **bool** | `FlashAttentionScoreV4×3 + LogicalNot×3` | **12.3 µs** | 6 |
| 无 mask | `FlashAttentionScoreV4×3` | 9.3 µs | 3 |

**为什么 float 会退化**：SDPA 的融合路径要求掩码能被内核直接消费；加性 float 掩码需要先做 dtype 转换与逐元素加法，框架于是回落到"矩阵乘 + 独立 softmax"的分解实现。bool 掩码可以进 kernel 的 mask 参数，整条链路才能融合。

**判据**：profiler 里必须出现 `FlashAttentionScoreV4`；若看到 `BatchMatMulV3 + SoftmaxV2 + Muls`，就是没落融合算子。

### 4.4 模型级收益：别把 kernel 比值当端到端收益

| 配置 | float mask | bool mask | 差异 |
|---|---:|---:|---:|
| B=64, S=64 | 55.9 ms/步 | 55.0 ms/步 | **+1.6%** |
| B=32, S=256 | 55.8 ms/步 | 53.8 ms/步 | **+3.7%** |

kernel 层 3.8×，模型层只有 1.6–3.7%：该实现的成本主要花在"每个 token 组一套 KQV/FFN"的小矩阵上，attention 只占一小段。**报收益必须用端到端步时**，kernel 比值只能用来解释"为什么落没落融合算子"。

---

## 5. 阶段四：混合精度与显存

### 5.1 AMP 门禁：NPU 只放行 bf16

```python gpu GPU_Reference/scripts/run_taac2026_sample.py:40
def should_enable_amp(device_type: str, amp: bool, amp_dtype: str) -> bool:
    if not amp:
        return False
    if device_type != "cuda":          # ← NPU 上这里直接判死
        return False
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        return False
    return True
```

```python npu OneTrans_NPU/npu_compat.py:32
def should_enable_amp(device_type: str, amp: bool, amp_dtype: str) -> bool:
    if not amp:
        return False
    if device_type not in ("cuda", "npu"):
        return False
    if amp_dtype == "fp16":
        # NPU 侧没有可用的 GradScaler 实现；fp16 需要 loss scaling 才能稳定，
        # 因此 NPU 只放行 bf16（bf16 动态范围与 fp32 相同，不需要 scaler）。
        return device_type == "cuda"
    if amp_dtype == "bf16" and device_type == "cuda":
        return torch.cuda.is_bf16_supported()
    return True
```

**为什么 fp16 必须拒**：fp16 的动态范围装不下 fp32 的梯度，需要 GradScaler 做 loss scaling；torch_npu 2.7.1 上没有可用实现，"开着 fp16 跑"实际会得到大量下溢/NaN。拒绝比静默出错好。

### 5.2 scaler：用 no-op 占位，保持调用形态

```python gpu GPU_Reference/scripts/run_taac2026_sample.py:50
def build_scaler(device_type: str, use_amp: bool, amp_dtype: torch.dtype):
    scaler_enabled = use_amp and device_type == "cuda" and amp_dtype == torch.float16
    return torch.amp.GradScaler(device=device_type, enabled=scaler_enabled)
```

```python npu OneTrans_NPU/npu_compat.py:74
def build_scaler(device_type: str, use_amp: bool, amp_dtype: torch.dtype) -> Any:
    if device_type == "npu":
        return _NoOpScaler(enabled=False)   # scale/step/update 均为空操作
    return torch.amp.GradScaler(
        device=device_type,
        enabled=use_amp and device_type == "cuda" and amp_dtype == torch.float16,
    )
```

`_NoOpScaler` 保留 `scale() / step() / update() / unscale_()` 四个方法，训练循环一个字都不用改（`scaler.scale(loss).backward()` 原样保留），日志里打印 `grad_scaler=False`。

### 5.3 autocast：`torch.autocast("npu")` 可用，但默认 dtype 是 fp16

```python npu OneTrans_NPU/npu_compat.py:81
def autocast_context(device_type: str, amp_dtype: torch.dtype, use_amp: bool):
    return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp)
```

**坑**：`torch.get_autocast_dtype("npu")` 默认返回 **fp16**，必须显式传 `bfloat16`，否则你以为在跑 bf16、其实在跑 fp16（且没有 scaler）。

### 5.4 划算性：这个小模型上 bf16 更慢

同进程交替 A/B，3 轮 × 10 步取中位（B=64、S=64）：

| 档位 | 步时中位 | 峰值 HBM 增量 |
|---|---:|---:|
| fp32 | **56.0 ms** | 109 MiB |
| bf16 | **69.0 ms** | 99 MiB |
| 比值 | **1.232（bf16 慢 23.2%）** | −9.2% |

原因：6M 参数全是小 GEMM（每 token 组一套 KQV/FFN），bf16 无法摊薄精度转换与 cast 开销；显存只省 10 MiB。**结论：本模型默认关 AMP**，`--amp` 只在更大规模或 HBM 紧张时开。

**判据**：`生效性`（ctx 内 `is_autocast_enabled` + matmul 输出 dtype）与 `划算性`（同进程交替步时比 <1）必须分开测。只测前者就宣布"AMP 已启用"，是本类迁移最常见的过度声明。

---

## 6. 阶段五：精度对齐与固化

### 6.1 三段判据（A/B/C）

```python npu OneTrans_NPU/tests/accuracy_npu.py:1
# A) NPU fp32 vs CPU fp32  —— 证明"同一份实现，只差硬件"
# B) NPU bf16 vs CPU fp32  —— 证明混合精度可接受
# C) 同 init 同数据 10 步    —— 证明"训练轨迹不发散"
```

| 判据 | 实测 | 门槛 | 结果 |
|---|---|---:|---|
| A. logits | max\|Δ\|=**1.49e-7**，cos=0.99999988 | ≤1e-4 / cos≥0.99999 | PASS |
| B. logits (bf16) | max\|Δ\|=**2.591e-3**，cos=0.99998170，rel=5.431e-3 | ≤2e-2 / cos≥0.9995 | PASS |
| C. 10 步 loss | CPU 1.594162→0.375242 / NPU 同值，Δ=**5.96e-8** | Δloss≤1e-2 | PASS |
| C. acc | CPU 1.0000 / NPU 1.0000 | Δacc≤2pp | PASS |

### 6.2 与 GPU 参考实现做"实现对照"的要求

三层证据，缺一层客户就会继续问：

| 层 | 比什么 | 本工程做法 |
|---|---|---|
| 代码层 | `diff -u` 行数 + 哪些文件 0 行 | 模型/损失/指标/数据管道 0 行 → 才敢说"同一份代码" |
| 结构层 | 同 seed 构造，`state_dict` 键序键名一致、参数 `max\|Δ\|=0` | `tests/compare_ref_vs_npu.py` 各起子进程（`models.*` 同名会串味） |
| 数值层 | 同权重 + 同输入：logits / loss / 全部参数梯度 | fp32 侧逐参数比梯度，见下 |

**梯度比较的两个陷阱**（都会让你误判"实现不一致"）：
1. **分母陷阱**：被 mask 乘穿的参数，CPU 侧留 `2.8e-10` 浮点残渣、NPU 侧精确 `0`，直接算 `Δ/|g_ref|` 会得到 rel=1.0、cos=0.0。阈值取 `max(1e-8, 全局|g|max×1e-10)`，单独计数。
2. **mask 语义比对"允许集"而不是比张量**：float 用 `-inf` 禁止、bool 用 `True` 参与，直接比张量一定不等。

### 6.3 多卡梯度等价

```bash
torchrun --nproc_per_node=2 tests/ddp_equiv_npu.py --dtype fp32 --local-batch 8
```

```
[A] 梯度等价 world=2 local_batch=8 global_batch=16 dtype=fp32
    单进程 |g|max=1.543596e-01 worst rel|Δg|=1.211e-06 门槛=1.0e-05
[B] loss 单进程=1.59416199 DDP均值=1.59416199 |Δ|=0.000e+00 门槛=1.0e-06
[verdict] A=True B=True  → DDP-EQUIV PASS
```

**扩展性口径**：锁每卡本地 batch（8），报 samples/s 与加速比；固定全局 batch 加卡在小模型上会**变慢**（每步固定开销不随 batch 缩小而降）。

### 6.4 固化：一条命令跑完五阶段

```bash
bash tests/run_all_950.sh        # 冒烟 → 精度 → 端到端（2 epoch）
bash tests/run_ddp_950.sh 2      # 多卡：梯度等价 + 吞吐
```

