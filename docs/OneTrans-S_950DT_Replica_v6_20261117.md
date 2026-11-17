# OneTrans-S 昇腾 950DT 复刻报告（v6 初稿）

> 数据基准：16 die（2 节点 × 8）· CANN 9.1.0 · torch 2.10.0 / torch_npu 2.10.0 · 2026-11-17
> 代码：`github.com/xiaomingyao/OneTrans-S`（上游 GPU 参考锁定 `8febd7e`）

## 0 摘要

| # | 结论 |
|---|---|
| 1 | **复刻等效**：本仓与上游 GPU 参考在同配置下 **949 个张量 key+shape 全等、前向输出 md5 相同、参数量 91,026,946 一致** |
| 2 | **步时**：档 A（论文口径，2048/卡 + 重算）**752.0 ms**；档 B（全局对齐，1024×2）**614.7 ms** |
| 3 | **判据**（同精度档 273 ms × 1.05 = **287 ms**）：差距 **2.62× / 2.14×**；MFU **25.8% / 31.6%** |
| 4 | **根因**：纯计算受限（98.6%），cube 348.4 ms 仅 **50.9%** 效率；vector 侧 240 ms **已贴 HBM 峰值**；cube∥vector 无法重叠（15%） |
| 5 | **天花板**：cube 跑满峰值且 vector 不动 = **426 ms = 273 ms 的 1.56×** ⇒ **287 ms 在现行算法 + 可用算子下不可达** |

## 1 目的与判据

### 1.1 目的

在昇腾 950DT（16 die）上复刻论文 arXiv:2510.26104v3 的 **OneTrans-S 训练**，实测步时 / 吞吐 / 显存 / MFU，与论文 **16×H100** 并列比较。范围：**只训练**，不含推理。

### 1.2 判据三档

论文 Table 4 为**单点消融**（四行百分比均相对 407 ms），步时最低是 203 ms（+FA 单点），
但 **273 ms（+mixed precision + 重算）才是与本仓同精度档、同 batch 口径的一行** ⇒ 主判据取 273。

| 档 | 论文基准 | 判据（×1.05） | 本仓档 A 752.0 | 本仓档 B 614.7 |
|---|---:|---:|---:|---:|
| **主判据（同精度档）** | **273 ms** | **287 ms** | **2.62×** | 2.14× |
| 参考（未优化 fp32 基线） | 407 ms | 427 ms | 1.85× | 1.51× |
| 参考（论文最强单点） | 203 ms | 213 ms | 3.70× | 3.03× |

### 1.3 口径纪律

1. **跨平台只并列不相除**：407/273/203 是 H100 上的论文数字，本仓是 950DT 实测，比值只表示步时之比。
2. **档 A / 档 B 不混读**：档 A 每卡 2048 + 激活重算（与论文一致）；档 B 每卡 1024×2、无重算（只对齐全局 batch 32,768）。
3. **精度分两级**：L1 布局类（ns-layout / 去 cat / 全量投影 / FA BSH / SDPA causal）数值等价；**L2 数值档变更**（融合 rms_norm、显式 bf16 干路）单独记账。

## 2 OneTrans-S 模型架构（论文）

### 2.1 数据流

```
非序列特征 ──Auto-Split tokenizer──┐
                                   ├─ 12 个 NS token ─┐
序列特征 ──timestamp-aware 融合─────┘                  ├─ [SEP] + concat → 1203 个位置
                                   └─ 1190 个 S token ┘
                                                                  ↓
                              pyramid stacked OneTrans blocks（逐层尾部截断 1203 → 12）
                                                                  ↓
                                          mean pooling → LayerNorm → Linear → sigmoid
```

### 2.2 两个关键设计

| 设计 | 内容 |
|---|---|
| **参数共享** | 序列 token（S）共享**一套** QKV 与 FFN；**每个 NS token 各自一套**（`ns_len + 1` 组，共 12 + 1） |
| **金字塔尾部截断** | 每个 block 只让**尾部 L 个位置**做 query（K/V 仍为层输入全长），逐层 1190→12，取 32 的倍数，顶层对齐 NS 数 ⇒ 掩码为**底右对齐因果** |
| 其余 | Pre-Norm、RMSNorm、SiLU×2、无 dropout、无 bias 的 LayerNorm head |

### 2.3 关键超参（论文 §4.1.4 / Table 2）

| 项 | 值 | 项 | 值 |
|---|---|---|---|
| 宽度 d / 头数 H | 256 / 4 | 层数 | "6 stacked"（实测口径对应 7 次 block-pass） |
| 序列长度 | 1190 → 12（32 对齐） | NS token | 12（+[SEP] 1 ⇒ 总 1203） |
| 参数量 | **91 M**（含稀疏 embedding） | 训练算力 | 2.64 TFLOPs（nominal，见附 A） |
| per-GPU batch | 2048（全局 32,768） | 优化器 | dense RMSProp(5e-3)、sparse Adagrad，无 weight decay |
| 梯度裁剪 | global grad-norm，dense 90 / sparse 120 | 精度 | 基线 fp32；优化档 mixed precision |

### 2.4 论文性能（Table 4，单点消融）

| 形态 | 步时 | 显存 |
|---|---:|---:|
| 未优化基线 | 407 ms | 53.13 GB |
| +pyramid | 290 ms | 30.50 GB |
| +cross-request KV cache | 284 ms | 22.10 GB |
| +FlashAttention | **203 ms** | 21.84 GB |
| **+mixed precision + 重算** | **273 ms** | 27.10 GB |

> OneTrans-S 的 **MFU 论文未给**（Table 5 的 30.8% 属于 OneTransL + bf16），故本报告不列 H100 侧 MFU。

## 3 迁移适配

### 3.1 训练环境

| 项 | 值 |
|---|---|
| 加速器 | Ascend950DT_9582，**16 die = 2 节点 × 8**（122 + 111），1 die = 1 卡 |
| 软件栈 | CANN 9.1.0 · torch 2.10.0+cpu · torch_npu 2.10.0 · triton 3.2.0 |
| 单卡算力 | bf16 Cube **432 TF/s**（32 AI core × 1650 MHz；真机 matmul 校核 429.8 TF = 99.5%） |
| 单卡 HBM | 容器可见 **91.75 GiB/die**（标称 144 GB，差异待核）；实测带宽 3.22–3.92 TB/s |
| 对照 H100 SXM | bf16 989.4 TF/s · 80 GB HBM3 · 3.35 TB/s · 机器平衡点 295（本机 108–135） |

### 3.2 代码适配（基于上游 `WestbrookLong/OneTrans_Pytorch` @ `8febd7e`）

| 类别 | 改动 | 说明 |
|---|---|---|
| **等价（L1）** | NS 专有 QKV/FFN 由逐 token Linear 合成 `bmm`（`NS_BMM`） | 单算子 −8.7%，输出逐位相同 |
| | 去 `cat`：预分配 + 原地写（`CAT_INPLACE`） | Cat 51.4→7.8 ms/步 |
| | 全量投影 + 原地覆盖 NS 行（`FULL_PROJ`） | 省一次整块 alloc + 一次全量写 |
| | FA `BSH` 布局（`FA_BSH`） | 单算子 −31.7%，q 1 ULP |
| | SDPA 内置 causal（`SDPA_CAUSAL`） | 仅 `q_len == k_len` 的层等价 |
| **数值档变更（L2）** | 融合 `npu_rms_norm` 替代手写 norm 链（`FUSED_NORM`） | 对 fp64 参考反而更准（3.891e-3 vs 7.704e-3） |
| | **显式 bf16 干路**（`EXPLICIT_BF16`） | 关 autocast；`mean rel|Δg| = 0.61%`，属精度档变更 |
| **工程** | torchrun + DDP、合成数据常驻 device、profiler 钩子、阶段标记 | 步时口径不含取数与 H2D |

### 3.3 与上游逐位等价核验（`tools/probes/compare_upstream_model.py`）

| 校验项 | 结果 |
|---|---|
| 参数量 | 91,026,946（两边相同） |
| state_dict | **949 个张量，key 集合与形状全等，0 处不同** |
| 前向输出 md5（同种子权重 + 同输入） | **`c4392b1c57956629…` 完全相同** |
| pyramid schedule | `[1203, 960, 736, 480, 256, 12]`（相同） |

⇒ 本仓在优化开关全关时与上游**同义**；所有偏差都来自上表的显式开关。

### 3.4 与论文对齐评估

| 项 | 论文 | 本仓 | 判定 |
|---|---|---|---|
| 结构（d/H/序列/NS/SEP/共享/PreNorm/掩码） | 256 / 4 / 1190→12 / 12 / 有 / 是 / 底右因果 | 同 | ✅ |
| 参数量 | 91 M（含稀疏表） | **91,026,946**（纯稠密） | ✅ 数值 / ⚠️ 构成不同 |
| per-GPU batch | 2048 | 档 A = 2048 | ✅ |
| **计算量** | 2.64 TFLOPs（nominal 6ND） | **2.649 TFLOPs（同口径，差 0.03%）**；实际分组计算 **40.91 GFLOP/样本** | ✅ 同口径一致（附 A） |
| **梯度裁剪** | global grad-norm，dense 90 | **未实现** | ⚠️ 缺失 |
| **优化器** | RMSProp(5e-3) | AdamW(5e-3) | ⚠️ 不同（不影响步时口径） |
| **FFN 宽度** | 论文未给 | 1536（=6×d） | ⚠️ 推定值（靠它把参数对齐到 91 M） |
| 层数表述 | "6 stacked" | 1 base + 6 pyramid = 7 个 block | ⚠️ 歧义（论文 nominal 算力按 7 次 pass 才对得上） |

## 4 性能优化

### 4.1 冒烟与开箱性能

| 配置 | 步时 | MFU | HBM |
|---|---:|---:|---:|
| **开箱（无任何优化，档 A）** | **1,942.1 ms** | 7.70% | 52.30 GB |
| 开箱（1 die，batch 256） | 233.7 ms | 8.00% | 20.79 GB |
| 开箱（16 die，档 A，全局 32,768） | 1,942.1 ms | 7.70% | 52.30 GB |

### 4.2 Profiling TOP-DOWN（当前最优配置，3 个稳态步）

| 层 | 数字 | 结论 |
|---|---|---|
| 调度 | Computing **605.5 ms = 98.6%**；通信 19.5（暴露 4.7）；气泡 4.1 | **纯计算受限**，调度/通信/气泡合计 1.5% |
| 阶段 | 前向 164.2 ms / 反向+优化+通信 471.9 ms（2.87×） | 反向是主战场 |
| 算子族 | cube **348.4 ms @50.9%**：FA 前 41.0（44.8%）、**FA 反 131.3（35.0%）**、GEMM 176.1（64.3%）；vector ~240 ms：Add 57.7 / SiLU 56.5 / 搬运 46.3 / RMSNorm 40.1 / ReduceSum 21.1 / Zero 14.0 | cube 一半算力在空转；vector 已贴峰值 |
| 关键约束 | ① vector 带宽 3178–3358 GB/s（≈峰值，不可再快）② cube∥vector 双流重叠仅 **15%** ③ 注意力**无任何量化入口**（`aclnnQuantFlashAttentionScore` 不在 libopapi.so） | 决定可达上限 |

### 4.3 优化执行与收益表（16 die · n=8 中位）

| # | 优化 | 开关 | 步时 | 相对起点 | MFU | 类别 |
|---|---|---|---:|---:|---:|---|
| — | 起点 | — | 1,942.1 ms | — | 7.70% | — |
| 1 | NS 专有投影合成 bmm | `NS_BMM` | 1,442.9 ms | −25.70% | 10.36% | L1 |
| 2 | 融合 rms_norm | `FUSED_NORM` | 1,228.6 ms | −36.74% | 12.17% | L2 |
| 3 | 去 cat（原地写） | `CAT_INPLACE` | 1,112.1 ms | −42.74% | 13.44% | L1 |
| 4 | 换档 B（1024×2、无重算） | — | 959.2 ms | −50.61% | 15.58% | 口径 |
| 5 | SDPA 内置 causal | `SDPA_CAUSAL` | 847.3 ms | −56.37% | 17.64% | L1 |
| 6 | FA `BSH` + 全量投影 | `FA_BSH` + `FULL_PROJ` | 730.0 ms | −62.41% | 20.48% | L1 |
| 7 | **显式 bf16 干路** | `EXPLICIT_BF16` | **614.7 ms** | **−68.35%** | **31.6%** | L2 |

> 档 A（论文口径，同配置 + 激活重算）：**752.0 ms / MFU 25.8% / HBM 39.17 GB**。
> MFU 为"主口径"（Linear/mm + attention 因果半三角 = 83.8 TFLOP/卡/步）；只算 Linear/mm 的下界为 24.3% / 19.9%。

### 4.4 否决清单（同口径实测，避免重复投入）

| 项 | 一句话原因 |
|---|---|
| 图模式（capture / TorchAir） | attention 算子全部不可捕获（107025）；且气泡上限仅 0.8% |
| 自写 attention（Triton） | 前向慢 2.32–3.12×；理论上限 0.6%（算术强度 481，算力受限） |
| FP8 训练 | `torch.mm`/`npu_quant_matmul` 报 161002；`_scaled_mm` 0.3 TF/s（回退路径） |
| 量化版 attention | 底层 `aclnnQuantFlashAttentionScore` 缺失，bf16 输入都报错 |
| int8 训练 | GEMM 真有 2×（802 TF/s），但量化误差 2.5–4.6%（bf16 0.61%） |
| add+silu / add+norm 融合 | 前者无配对位置、Triton 静默不执行；后者上限 0.7–1.35% |
| RMSNorm 自定义反向 | 真模型零收益（autograd 本就用 rstd） |
| cube 三杠杆（qkv 合并 / `inner_precise` / mask dtype） | 0.07%/步 或完全无差别 |
| Embedding 异构缓存、cross-request KV cache | 本仓无稀疏表、无 request/历史维度 |
| 每卡 4096 / 2048 免重算 / `torch.compile` / DDP 各开关 | 算子失败、OOM 或 ±0.1% 噪声 |

## 5 结论

### 5.1 复刻等效性

与上游 GPU 参考**逐位等价**（949 张量、前向 md5 同、参数 91,026,946、schedule 同）；
与论文主干超参对齐，**4 处差异已标注**（梯度裁剪缺失、优化器不同、FFN 宽度为推定、层数表述歧义）。
计算量在同一 nominal 口径下与论文一致（2.649 T vs 2.648 T/样本）。

### 5.2 性能结论

| 口径 | 步时 | 全局 sps | MFU（主口径） | 显存 | vs 273 ms |
|---|---:|---:|---:|---:|---:|
| 档 A（论文口径） | **752.0 ms** | 43,575.4 | **25.8%** | 39.17 GB | **2.62×** |
| 档 B（全局对齐） | **614.7 ms** | 53,307.4 | **31.6%** | 58.48 GB | 2.14× |

### 5.3 为什么 287 ms 不可达（三条硬约束）

1. **vector 已贴峰值**：240 ms，带宽 3178–3358 GB/s；只能靠融合减少字节。
2. **attention 无法加速**：FA 27.8 TFLOP 跑满 bf16 峰值也要 64 ms ⇒ **240 + 64 = 304 ms > 287 ms**；且无量化入口。
3. **硬件峰值只有一半**：950DT bf16 432 TF = H100 989.4 TF 的 **43.7%** ⇒ cube 即使 100% 也要 177.5 ms，加 vector 240 = **426 ms（1.56×）**。

### 5.4 可达路径与待办

| 阶段 | 动作 | 档 B 步时 | MFU |
|---|---|---:|---:|
| 现状 | — | 615 ms | 31.6% |
| A | 模型内 GEMM 277 → 400 TF/s（隔离同形状实测可达 376–427） | 561 ms | 34.6% |
| +C | 零碎确定性（ReduceSum/Zero/搬运/add+norm/NS GEMM/图模式/通信） | 531 ms | 36.5% |
| +D | int8 只换 GEMM（精度档变更） | 470 ms | 41.3% |
| +E | 减算力 −10%（与论文结构冲突） | 408 ms | 47.5% |

**待办**：① 补 global grad-norm clipping（dense 90）；② GEMM 277→400 的逐层归因；③ MFU 收敛为单一算子级口径。

## 附 A 计算量口径（nominal vs actual）

| 口径 | 公式 | 本仓值 | 论文 |
|---|---|---|---:|
| **nominal（论文同口径，6ND）** | `6 × N × Σ(block-pass token)` = 6 × 91,026,946 × 4850 | **2.6489 TFLOP/样本** | 2.6481 TFLOP |
| **actual（按分组形状）** | 逐层 QKV/FFN/attention 实际矩阵乘 | **40.91 GFLOP/样本**（83.8 TFLOP/卡/步） | 论文未给 |

**关系**：nominal 把全部 91 M 参数乘到每个位置，而实际只用到 13 组中的 1 组 ⇒ **nominal = actual × 64.7**。
⇒ nominal 只能用于与论文同口径对齐，**不能当 MFU 分子**（否则会得出"峰值 1%"的假数）。

## 附 B 证据索引

| 结论 | 文件 / 提交 |
|---|---|
| 与上游逐位等价 | `tools/probes/compare_upstream_model.py`、`docs/MODEL_ALIGNMENT_AUDIT.md` |
| 论文数字与口径（唯一真源） | `docs/TOPIC_ONETRANS_S.md` §1、§4.1 |
| 论文 4 项优化在昇腾的适用性 | `docs/ASSESS_PAPER_4_OPTS_ON_ASCEND.md` |
| TOP-DOWN 分解与 ReduceSum 归因 | `docs/PROFILE_ROUND2_AFTER_OPT.md` §10–§11 |
| 剩余空间与路径 | `docs/OPT_REMAINING_ROADMAP.md` |
| 否决项证据 | `docs/OneTrans-S_950DT_Optimization_v5_20261117.md` §5、`docs/OPT_TRITON_MIXED_MHA_ASSESSMENT.md` |
| 全部原始日志 | `results/`（A/B、profile、fp8/int8 探针） |

## 附 C 复现命令（3 条）

```bash
# 单 die 冒烟（约 3 分钟）
ASCEND_RT_VISIBLE_DEVICES=0 python3 scripts/run_onetrans_s.py --device npu \
  --batch-size 1024 --steps 30 --warmup 5 --checkpoint-act --flops-per-sample 31529400000

# 16 die 双档（档 B）
torchrun --nproc_per_node 8 --nnodes 2 scripts/run_onetrans_s.py --device npu \
  --batch-size 1024 --grad-accum 2 --steps 30 --warmup 5

# 16 die profile（3 稳态步）
torchrun --nproc_per_node 8 --nnodes 2 scripts/run_onetrans_s.py --device npu \
  --batch-size 1024 --grad-accum 2 --steps 25 --warmup 20 --profile-steps 3 --profile-dir /tmp/pf
```
