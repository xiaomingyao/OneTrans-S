# OneTrans-S

**OneTrans-S 在昇腾 950DT 上的训练复核** —— 按论文（arXiv:2510.26104v3）OneTrans-S 的形状把模型拉起来，量真实训练步的**步时 / 吞吐 / 显存 / MFU**，并与论文的 **H100** 数字并列。

> 范围：**只做训练**。推理、cross-request KV cache、p99 延迟那套一律不做。
> 状态：形状已对齐（参数量 `91,026,946` vs 论文 91 M，+0.03%），1 die / 8 die 已实测；**结构等价性未闭合**（§5）。
> 论文归档 `papers/OneTrans_2510.26104v3.pdf`；话题与证据链 `docs/TOPIC_ONETRANS_S.md`；原始日志 `results/`。

## 1. 这个仓做什么

1. **复刻形状**：论文 OneTrans-S = 6 个堆叠 OneTrans block、d=256、H=4、序列 token 1190 → 12（每层对齐 32）、NS-token 12、per-GPU batch 2048。
2. **复核训练**：在 950DT 上按这个形状跑训练步，量四个数（稳态步时 / 吞吐 / HBM 峰值 / MFU），留可复跑探针与原始日志。
3. **与 H100 并列**：只并列绝对值与各自口径的百分比，**不相除算收益**（跨平台纪律）。

上游 GPU 实现 `WestbrookLong/OneTrans_Pytorch` 以 **git submodule** 锁定 commit，**上游源码零改动**：

```bash
git clone --recurse-submodules https://github.com/xiaomingyao/OneTrans-S
git -C GPU_Reference rev-parse HEAD        # 8febd7e…
git -C GPU_Reference status --porcelain    # 空 = 未改动
```

## 2. 形状对账（论文 vs 本仓实测）

| 项 | 论文 OneTrans-S | 本仓实测 | 判定 |
|---|---|---|---|
| 参数量（dense，不含稀疏表） | 91 M（Table 2） | **91,026,946 = 91.03 M** | +0.03%，PASS |
| 层数 | 6 stacked blocks（§4.1.4） | 7 = 6 个 pyramid 段 + 1 个全长期 base block | 实现多 1 个 base block，结构对账里要说明 |
| 宽度 / heads | d=256 / H=4 | 256 / 4 | PASS |
| pyramid | 1190 → 12，每层取 32 的整数倍（§4.1.4） | `[1203, 960, 736, 480, 256, 12]`（起点 1203 = 1190 + 12 + 1 个 `[SEP]`） | 形状一致，起点含 SEP |
| per-GPU batch | 2048（§4.1.2） | `per_card_batch=2048`，8 die 时全局 16,384 | PASS |
| 每样本训练算力 | 2.64 TFLOPs @batch 2048（口径未说明） | **31.53 GFLOP/sample**（CPU `FlopCounterMode` 实测） | ❌ 差 24.5×，见 §5 |

参数只是必要条件：能落到 91 M 的形状**至少三组解**，所以本仓不拿参数量当结构等价的证据。

## 3. 950DT 实测（2026-11-16）

环境（本轮 S 复核）：`Ascend950DT_9582` / CANN 9.1.0 / **torch 2.10.0+cpu + torch_npu 2.10.0** / 容器 `zeus-950-v2`。
口径：bf16 autocast + **bool mask**（落 `FlashAttentionScoreV4`）+ AdamW；**token 常驻 device ⇒ 步时不含取数与 H2D**（纯计算口径）；MFU 分母 = 单 die Cube bf16 **432 TF**（32 AI core × 1650 MHz）。

| 配置 | 每卡 batch | 全局 batch | 稳态步时 | 每卡 sps | 全局 sps | HBM 峰值 | MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 die | 256 | 256 | 233.7 ms | 1,095.6 | 1,095.6 | 20.79 GB | 8.00% |
| 1 die | 512 | 512 | 434.6 ms | 1,178.2 | 1,178.2 | 40.39 GB | 8.60% |
| 1 die | 1024 | 1024 | 837.0 ms | 1,223.4 | 1,223.4 | 79.55 GB | 8.93% |
| 1 die | 1024（+激活重算） | 1024 | 966.4 ms | 1,059.6 | 1,059.6 | 26.64 GB | 7.73% |
| **1 die** | **2048（+重算）** | 2,048 | **1,901.8 ms** | 1,076.9 | 1,076.9 | 51.96 GB | 7.86% |
| **8 die** | **2048（+重算）** | 16,384 | **1,934.2 ms** | 1,058.9 | **8,470.9** | 52.30 GB | 7.73% |
| **8 die** | 2048 × accum 2（+重算） | **32,768** | **3,870.8 ms** | 1,058.2 | **8,465.4** | 52.95 GB | 7.72% |

三条实测结论：

1. **显存是硬约束**：1 die 每卡 2048 不开激活重算直接 OOM（已分配 86.63 GiB、还需 2.34 GiB）；**每卡 4096 任何组合都上不去**（见 §3.1）⇒ 论文 per-GPU 2048 是这台机器的实际可达上沿，全局 32,768 只能靠 `--grad-accum 2`。
2. **吞吐在 batch ≥ 512 后进入平台**（每卡 1.06k–1.22k sps）：batch 256 时固定开销占比高；把全局 batch 从 16,384 抬到 32,768 只把步时线性拉长 2.001×，吞吐 8,470.9 → 8,465.4（−0.06%）⇒ 该形状每卡算力已饱和。
3. **8 卡扩展 7.87×（效率 98.3%）**，每卡 sps 仅降 1.7% ⇒ 该形状下 DDP 梯度同步**不是**瓶颈；瓶颈在单卡 kernel 效率（MFU 7.7–8.9%）。

### 3.1 每卡 batch=4096 为什么上不去（2026-11-16 排查）

论文全局 32,768 ÷ 8 die = 每卡 4096 是**算术值**，但物理上装不下。逐级压显存，缺口从 11.11 GiB 收到 18 MiB，最后卡在 **FlashAttention 反向的 workspace** 而不是激活量：

| 步骤 | 手段 | 结果 |
|---|---|---|
| 1 | b4096 + 激活重算（基线） | OOM，需再 11.11 GiB（已分配 80.01 / 91.75） |
| 2 | `PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512` | OOM（80.01 / 需 11.11）——不是碎片问题 |
| 3 | `expandable_segments:True` | OOM（79.91 / 需 11.11） |
| 4 | `ONETRANS_FFN_CHUNK=131072`（分块 FFN） | OOM，需再 386 MiB（89.66） |
| 5 | 预分配写分片（去掉 `torch.cat` 的整张量复制），chunk=32768 | OOM，需再 22 MiB（90.20） |
| 6 | 输入 token 存 bf16（`--input-dtype bf16`） | OOM，需再 18 MiB（89.93） |
| 7 | `garbage_collection_threshold:0.5` | 显存过了，算子失败：`aclnnFlashAttentionScoreGradV4` / `ERR00100 PTA call acl api failed` |

两个可复用的实验旋钮（默认关闭，不影响已记录结果）：

- `ONETRANS_FFN_CHUNK=<rows>`：FFN 按行分块，**数学逐元素等价**，但 bf16 下**非逐位一致**（b256 两步 loss `0.68401/1.74561` vs `0.68401/1.74525`，相对差 2.1e-4）。
- `--input-dtype bf16`：常驻 device 的输入 token 改 bf16（4096 batch 下省 0.6 GiB）。

⇒ 结论：**96 GiB/die 上 OneTrans-S（seq 1190）单卡上沿 = 2048**；要 per-card 4096 必须走 16 die（122+111，全局 32,768 一次前向）或改造注意力分块（未做，标待办）。

原始日志：`results/s_train_950_1die_8die.txt`（含每步 ms / loss / OOM 原始报错）。

## 4. 与论文 H100 对比（按**全局 batch = 32,768** 对齐）

### 4.1 对齐轴（哪些对齐、哪些必须并列说明）

| 轴 | 论文 H100 | 本仓 950DT | 是否对齐 |
|---|---|---|---|
| **全局 batch** | 16 卡 × 2048 = **32,768** | 8 die × micro 2048 × accum 2 = **32,768** | ✅ **本对比的对齐轴** |
| per-GPU batch | 2048 | 2048（micro） | ✅ 对齐（每卡每次前向 2048 样本） |
| 卡数 | 16 | 8 | ❌ 不一致，禁换算 |
| 并行方式 | 16 卡数据并行，一次前向覆盖 32,768 | 8 die 数据并行 × **2 个微批串行累积**，一次优化器更新覆盖 32,768 | 全局 batch 等价（均值梯度），但**步时含 2 次前反向** |
| 精度 / 优化档 | **fp32、未优化基线**（Table 4 首行） | **bf16 + bool mask(FA) + 激活重算** | ❌ 不一致，只能并列不能比快慢 |
| 数据 | 工业自有数据（29.1B impressions） | 合成 token 常驻 device（不含取数与 H2D） | ❌ 不一致，算力不可对表 |
| 优化器 | Adagrad(sparse) + RMSProp(dense) | AdamW（密度参数统一） | ❌ 不一致，步时含优化器开销 |

⇒ 全局 batch 与 per-GPU batch 两条对齐后，**剩下的差异是平台、精度档、数据来源**，所以下表只并列绝对值 + 口径百分比，**不做收益%**。

### 4.2 主对比表（全局 batch 32,768）

| 指标 | 论文 16 × H100 | 本仓 8 die × 950DT | 差（仅描述，非收益） |
|---|---:|---:|---|
| 全局 batch | 32,768 | 32,768 | 对齐 |
| per-GPU batch | 2,048 | 2,048（micro）× 2 accum | 对齐 |
| **稳态步时** | **407 ms** | **3,870.8 ms** | 9.51×（口径不同，见 §4.1） |
| **每卡吞吐** | 5,032 sps（2048 ÷ 0.407 s，**推算**） | **1,058.2 sps**（实测） | 4.76× |
| 全局吞吐 | 80,511 sps（**推算**，论文未给） | **8,465.4 sps**（实测） | 9.51×（含卡数差 2×） |
| 训练显存峰值 | 53.13 GB（fp32 未优化档） | 52.95 GB（bf16+重算） | 并列，数值接近但精度档不同，不作因果 |
| MFU | **S 未给**；OneTransL 30.8%（bf16）；DCNv2+DIN 13.4% | **7.72%**（分母 432 TF；分子 31.53 GFLOP/sample） | 无可比项（S 侧论文未给） |
| 训练算力/样本 | 2.64 TFLOPs @batch 2048（口径未说明） | 31.53 GFLOP/sample（CPU 探针实测） | ❌ 24.5×，见 §5 |

两条读法纪律：

1. **步时 9.51× 不能读成「950DT 比 H100 慢 9.5 倍」**：论文那档是 fp32 未优化基线（相当于每步还有 fp32 物化 attention 的开销），本仓是 bf16+FA+重算档；要判断平台差距，只能等论文给出 S 的 bf16 档数字，**目前论文未给**。
2. **每卡吞吐 4.76× 是同一 per-GPU batch 下的实测/推算并列值**，分子分母都标了来源；换算成「收益%」属禁用操作。

### 4.3 若按 per-GPU 对齐（不做累积）

| 全局 batch | 论文（16 卡） | 本仓（8 die） | 说明 |
|---|---|---|---|
| 16,384 | 论文无此档 | **1,934.2 ms / 8,470.9 sps** | 本仓实测；与 32,768 档吞吐相同（−0.06%） |
| 32,768 | **407 ms / 80,511 sps（推算）** | **3,870.8 ms / 8,465.4 sps** | 论文原始口径 |
| 32,768（若上 16 die，122+111） | — | 未实测；按每卡 2048 推算 ≈1.93 s/步 | 标「待测」，不作结论 |

三条引用纪律（详见 `docs/TOPIC_ONETRANS_S.md` §5）：① 跨平台不相除；② 论文 H100 那档是 fp32 未优化基线，不能与本仓 bf16 优化档直接比快慢，只能并列；③ 论文没给 S 的 bf16/MFU，「按 OneTransL 的 30.8% 推 S」属口径混用（本仓曾算错过一次，已在话题文档自纠）。

## 5. 已知差距：结构对账未闭合

| 判据 | 论文 | 本仓 | 结论 |
|---|---|---|---|
| 参数量 | 91 M | 91.03 M | ✅ 但属**多解命中**（ffn/ns 至少三组组合都能落到 91 M） |
| pyramid 开/关的算力比值 | 8.08 / 2.64 = **3.06×**（Table 3 vs Table 2） | **1.68×** | ❌ 本仓 pyramid 只剪 query、不剪 K/V/FFN |
| 每样本训练算力 | 2.64 TFLOPs（口径未说明；摊到 batch 2048 ≈ 1.29 GFLOP/sample） | 31.53 GFLOP/sample | ❌ 24.5× |

⇒ 现阶段只能宣称「**同形状**的 OneTrans 训练」，不能宣称「= 论文 OneTrans-S」。复现脚本：`tools/probes/`（参数量 / 每样本 FLOPs / pyramid 比值，CPU 可跑）。

## 6. 训练入口

```bash
cd OneTrans_NPU

# 1 die 冒烟（形状对齐、跑通即可）
python3 scripts/run_onetrans_s.py --device npu --batch-size 64 --steps 10 --warmup 3

# 1 die 论文 per-GPU batch（必须开激活重算，否则 OOM）
python3 scripts/run_onetrans_s.py --device npu --batch-size 2048 --steps 20 --warmup 5 \
  --checkpoint-act --flops-per-sample 31529400000

# 8 die（参数面不变，只换拉起方式）
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  scripts/run_onetrans_s.py --device npu --batch-size 2048 --steps 20 --warmup 5 \
  --checkpoint-act --flops-per-sample 31529400000 --out-json ../results/s_8die.json

# 8 die 且全局 batch = 32,768（论文口径）：micro 2048 × accum 2
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  scripts/run_onetrans_s.py --device npu --batch-size 2048 --grad-accum 2 --steps 20 --warmup 5 \
  --checkpoint-act --flops-per-sample 31529400000 --out-json ../results/s_8die_g32768.json
```

常用旋钮只有 `--batch-size / --grad-accum / --steps / --checkpoint-act / --flops-per-sample` 五个；形状默认值就是论文 S，不要手改模型代码。
判据：`[shape] params=91,026,946` ∧ `[shape] pyramid_schedule=[1203, 960, 736, 480, 256, 12]` ∧ `[perf] steady_median_step_ms` 有值 ∧ 退出码 0。

## 7. 迁移与精度验证（历史资产，仍然有效）

接线改动只在 `npu_compat.py` + 两个入口脚本，`models/*`、`utils/*` 零改动（上游 submodule 亦零改动）；触碰上游 2 文件 147 行，占 1,233 行活跃 Python 的 11.9%。逐项对照 →
[`docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md`](docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md)、[`docs/MIGRATION.md`](docs/MIGRATION.md)、[`patches/npu_changes.diff`](patches/npu_changes.diff)。

| # | 修改点 | 修改说明 |
|---|---|---|
| 1 | 新增兼容层 `npu_compat.py`（160 行） | 设备初始化 / AMP 能力门 / no-op scaler / autocast / HCCL / 收尾集中一处；`models`、`utils` 保持 0 行 |
| 2 | attention mask 形态开关 | `ONETRANS_ATTN_MASK=float\|bool`（+9 行）。float 加性 mask 让 SDPA 退回 Cast+Muls+BatchMatMul+Softmax（45 kernel / 139 µs）；bool mask 落 `FlashAttentionScoreV4`（6 kernel / 36.9 µs，3.77×） |
| 3 | AMP 能力门 | NPU 只放行 bf16（平台 autocast 默认 fp16），GradScaler 改 no-op |
| 4 | 多卡装配 | HCCL + `device_ids=[local_rank]` + `broadcast_buffers=False` + `find_unused_parameters=True`（最深 pyramid 块有 16 个参数本步不参与前向）；循环内不做跨卡归约 |
| 5 | 计时与收尾 | 计时前 `torch.npu.synchronize()`；收尾 `shutdown()` + `os._exit(0)`，避免 teardown 抖动被误报成失败 |

精度三段（NPU vs CPU fp32 golden，全部 PASS，日志 `results/accuracy_950_rerun.txt`）：fp32 前向 `max|Δ|=1.49e-07`（cos 0.99999988）；bf16 前向 `max|Δ|=2.59e-03`（cos 0.99998170）；10 步训练轨迹 `Δloss=5.96e-08`、`Δacc=0.00 pp`。

> ⚠️ 环境差异：以上历史验证跑在 **torch 2.7.1+cpu / torch_npu 2.7.1.post4 / Python 3.11**（`/opt/buildtools/torch_v2_pt2.7.1`，SoC `Ascend950DT` V120 板卡 `IT23HMDB_D2`、CANN 9.1.0）下；§3 的 S 复核跑在同一节点另一容器（**torch 2.10.0+cpu / torch_npu 2.10.0 / Python 3.12**）。两套数字各自绑定自己的环境，**不跨环境比较**。toy 配置下的功能/多卡历史结果（冒烟 76.9 ms、mask kernel 3.77×、DDP `worst rel|Δg|=1.21e-06`、1/2/4/8 die 扩展 4.44×）见 `results/` 与 `docs/MIGRATION.md`。

## 8. 目录

| 目录 | 内容 |
|---|---|
| `GPU_Reference/` | submodule → 上游 GPU 实现 @ `8febd7e`（零改动，diff 与数值基准） |
| `OneTrans_NPU/` | NPU 实现：`npu_compat.py` 兼容层 + `scripts/run_onetrans_s.py`（本仓主入口）+ 数据版入口 + `tests/` 探针 |
| `papers/` | 论文 PDF + 文本（v3） |
| `docs/` | 话题与证据链（`TOPIC_ONETRANS_S.md`）、代码实现对比、五阶段迁移对照 |
| `tools/probes/` | 参数量 / 每样本 FLOPs / pyramid 比值三个 CPU 探针 |
| `results/` | 实测日志（每行可追溯到命令与环境） |
| `patches/` | 完整改动 diff |

## 9. 快速开始（历史探针）

```bash
# 冒烟（前向 + 反向 + 3 步）
python OneTrans_NPU/tests/smoke_npu.py --device npu

# 精度三段（A: fp32 / B: bf16 / C: 10 步训练轨迹）
python OneTrans_NPU/tests/accuracy_npu.py --steps 10

# 与参考实现对照（两侧各起子进程，比结构 + 数值）
python OneTrans_NPU/tests/compare_ref_vs_npu.py

# 真实数据集端到端训练（toy 形状；bool mask 变体加 ONETRANS_ATTN_MASK=bool）
HF_ENDPOINT=https://hf-mirror.com python OneTrans_NPU/scripts/run_taac2026_sample.py \
  --device npu --batch-size 32 --epochs 2 --seq-len 32 --d-model 64 --multi-num 2 \
  --num-pyramid-layers 3 --amp --amp-dtype bf16 --output-dir /tmp/onetrans_out
```

多卡（DDP/HCCL）、mask 与 AMP 单项 A/B、HCCL dtype 探针、一键批量脚本 → [`docs/MIGRATION.md`](docs/MIGRATION.md) §6.4。
