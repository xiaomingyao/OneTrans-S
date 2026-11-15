# onetrans-d

ByteDance OneTrans（TAAC2026 推荐模型）在昇腾 **950DT** 的开源实现。
整个移植工程包括昇腾适配代码、冒烟测试、精度验证、迁移指导。

## 1. 项目介绍

上游 `WestbrookLong/OneTrans_Pytorch`（序列建模 + 金字塔多尺度注意力 + 多任务塔）以 **git submodule** 引用并锁定 commit，**上游源码零改动**：

```bash
git clone --recurse-submodules https://github.com/xiaomingyao/onetrans-d
git -C GPU_Reference rev-parse HEAD        # 8febd7e…；status --porcelain 为空 = 未改动
```

本仓做的事：

1. **迁移**：把这份 GPU 代码迁到 950DT，改动只在接线层（见 §2）。
2. **取证**：冒烟、三段精度、DDP 梯度等价、1–8 die 扩展性，全部留可复跑探针与日志（见 §3）。
3. **复现**：4 条命令重跑核心结论（见 §4），逐项细节见 `docs/`。

| 目录 | 内容 |
|---|---|
| `GPU_Reference/` | submodule → 上游原仓 @ `8febd7e`（零改动，diff 与数值对照的基准） |
| `OneTrans_NPU/` | 移植实现：`npu_compat.py`（兼容层）+ `main_pytorch.py` + `scripts/run_taac2026_sample.py` + `tests/`（探针） |
| `docs/` | `GPU_VS_NPU_CODE_IMPLEMENTATION.md`（代码实现对比）· `MIGRATION.md`（五阶段迁移对照） |
| `patches/` | `npu_changes.diff`（完整改动） |
| `results/` | 各类实测日志（§3 每行标了对应文件） |
| `tools/` | 文档与配图生成器 |

### 1.1 验证环境

| 项 | 值 |
|---|---|
| SoC | `Ascend950DT` V120（板卡 `IT23HMDB_D2`，NPU Name `9582`） |
| 规模 | 单机 8 die；单 die HBM 96 GiB |
| CANN | toolkit **9.1.0** |
| 驱动 / 固件 | 宿主机 **25.1.rc2**（Firmware `7.0.9.100.203`） |
| 镜像 / OS | `zeus-950-v2-snap:20261102` / Ubuntu 22.04.5（glibc 2.35） |
| Python | 3.11.0（`/opt/buildtools/torch_v2_pt2.7.1`） |
| PyTorch | **2.7.1+cpu** + **torch_npu 2.7.1.post4** |
| 真实数据可选栈 | `datasets` 5.2.0 · `huggingface_hub` 2.4.0 · `pyarrow` 26.0.1 |

环境判据：

1. 运行首行必须打印 `{'torch': '2.7.1+cpu', 'torch_npu': '2.7.1.post4', 'soc': 'Ascend950DT_9582'}`，对不上不能与本仓数字对比。
2. 950 上 AMP 只放行 bf16（平台 autocast 默认 dtype 是 fp16；本仓按实测只用 bf16），FP8 无直调入口。
3. 数字绑定本表环境，换机型或换 CANN / torch 版本需重测。

## 2. 从 GPU 迁移到 NPU

改动集中在「接线层」，模型定义与数据管道**零改动**：触碰上游 2 个文件 **147 行**（`main_pytorch.py` +9/−0、`scripts/run_taac2026_sample.py` +104/−34，占 1,233 行活跃 Python 的 11.9%），另新增兼容层 `npu_compat.py` 160 行；`models/*`、`utils/*` **0 行**。

| # | 修改点 | 修改说明 |
|---|---|---|
| 1 | 新增兼容层 `npu_compat.py` | 把设备初始化、AMP 能力门、no-op scaler、autocast 上下文、HCCL 分布式、收尾统一收进一个文件（160 行），业务代码只调这层接口；因此 `models`、`utils` 保持 0 行，换机型只改这一处 |
| 2 | attention mask 形态开关 | `main_pytorch.py` 新增 `ONETRANS_ATTN_MASK=float\|bool`（+9 行），默认 float 与上游逐字等价。950 上 float 加性 mask 会让 SDPA 退回 Cast+Muls+BatchMatMul+Softmax 手工路径（45 kernel / 139 µs），bool mask 才落 `FlashAttentionScoreV4`（6 kernel / 36.9 µs，3.77×） |
| 3 | AMP 能力门 | `should_enable_amp()` 在 NPU 只放行 bf16（平台 autocast 默认 dtype 是 fp16），GradScaler 改为 no-op；本模型 bf16 实测更慢，故默认关闭，只有显式 `--amp --amp-dtype bf16` 才开 |
| 4 | 多卡装配 | HCCL + `device_ids=[local_rank]` + `broadcast_buffers=False` + `find_unused_parameters=True`（最深金字塔块有 16 个参数本步不参与前向）；训练集按 rank 切分、验证只在 rank0 跑全量，**循环内不做跨卡归约**——实测 epoch 末 `all_gather` + `.cpu()` 会卡在 D2H |
| 5 | 计时与收尾 | 计时前加 `torch.npu.synchronize()`（否则步时偏小）；收尾 `shutdown()` + `os._exit(0)`，避免平台 teardown 抖动被误报成任务失败 |

细节（每项的「改前 vs 改后 + 判据」）：

- 逐项说明 → [`docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md`](docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md)
- 按五阶段并排对照（GPU 左 / NPU 右，标出哪段零改动） → [`docs/MIGRATION.md`](docs/MIGRATION.md)
- 逐行 diff → [`patches/npu_changes.diff`](patches/npu_changes.diff)（对上游副本 `patch -p1` 可精确复原 `OneTrans_NPU/`）

## 3. 测试验证

### 3.1 精度验证（NPU 输出 vs CPU fp32 golden）

| 用例 | 判据 | 实测 |
|---|---|---|
| A：NPU **fp32** 前向 vs CPU fp32 golden | `max|Δ| ≤ 1e-4` 且 `cos ≥ 0.99999` | `max|Δ| = 1.490e-07`、`cos = 0.99999988`、`rel = 3.43e-07` |
| B：NPU **bf16** 前向（autocast）vs CPU fp32 golden | `max|Δ| ≤ 2e-2` 且 `cos ≥ 0.9995` | `max|Δ| = 2.591e-03`、`cos = 0.99998170`、`rel = 5.43e-03` |
| C：10 步训练轨迹（loss / acc 逐步比对） | `Δloss ≤ 1e-2` | `Δloss = 5.960e-08`、`Δacc = 0.00 pp` |

指标含义：

- `max|Δ|`：把 NPU 输出与 CPU golden 逐元素相减，取绝对值的**最大值**（最大绝对误差，单位同输出量纲）。
- `rel`：`mean(|Δ|) / mean(|golden|)`，平均绝对误差相对 golden 平均幅值的比例。
- `cos`：两输出展平后的余弦相似度，1 表示方向完全一致。

**验收结论：OK —— 三段判据全部满足**（脚本 `[verdict] A=True B=True C=True` → `ACCURACY PASS`）：fp32 最大绝对误差 `1.5e-07`（判据 1e-4）、bf16 `2.6e-03`（判据 2e-2），余弦相似度 `≥0.99998`（判据 0.9995/0.99999），10 步训练的 loss 差 `6e-08`、accuracy 差 `0.00 pp`。

日志：`results/accuracy_950_20261114_final.txt`

### 3.2 功能、性能与多卡

| 项 | 结果 | 判定 | 日志 |
|---|---|---|---|
| 冒烟（前向 + 反向 + 3 步优化） | loss 有限且下降（1.610520 → 0.794862）、参数已更新（`param_delta = 3.008e-03`）；首步 ~610–645 ms（编译预热）→ 稳态 **76.9 / 77.3 ms** | **PASS** | `results/smoke_950_20261114_final.txt` |
| mask 形态（kernel 级） | float 加性 139.0 µs / 45 kernel → bool 36.9 µs / 6 kernel（`FlashAttentionScoreV4`），**3.77×**；端到端中位 **+2.4%** | **PASS**（kernel 落点验证成立）→ 作为可选开关保留，不默认开 | `results/kd_*.csv`、`results/bench_mask_mode_950_20261114.txt` |
| 多卡扩展（每 rank 本地 batch 64） | 1/2/4/8 die 稳态 **829.5 / 1,303.6 / 2,348.3 / 3,686.2** samples/s = **1.00 / 1.57 / 2.83 / 4.44×**（效率 78.6% / 70.8% / 55.5%） | **PASS**（吞吐随 die 数单调上升，扩展比符合预期） | `results/scaling_real_data_950_20261114.txt` |
| DDP 梯度等价 | 2 die DDP 与单进程同全局 batch：梯度 `worst rel|Δg| = 1.211e-06`（门槛 1e-5），loss 差 `Δ = 0.000e+00` | **PASS — 等价** | `results/verify_950_20261115_rerun.txt` |
| 端到端训练（真实数据） | 1,000 行 2 epoch 跑通：单 die 13 步/epoch、**78.9 ms/步 / 811.2 samples/s**、val AUC 0.5429；2 die **99.1 ms/步 / 646.0 samples/s**（样本极小，不代表任务精度） | **PASS**（能训完、指标正常产出） | `results/train_hf_real_950_20261114.txt`、`results/train_hf_real_2card_950_20261114.txt` |

指标含义（DDP 行）：

- `worst rel|Δg|`：逐个参数张量算 `max|g_DDP − g_单进程| / max|g_单进程|`（分母取该张量自身最大绝对值，下限 `1e-12`），再在全部 747 个张量上取**最大值** —— 即「最坏那一个梯度张量的相对偏差」，比平均相对误差更严格。

**汇总：§3 全部验证项通过** —— 精度 A/B/C 判据满足（`ACCURACY PASS`）、冒烟 PASS、mask 落点验证成立、多卡扩展 4.44×、DDP 梯度等价、端到端训练跑通；无失败项。

口径：kernel 级加速比 ≠ 端到端收益；多卡数字绑定「每卡本地 batch」，换口径不可比。2026-11-15 的逐条复跑记录见 `results/verify_950_20261115_rerun.txt`。

## 4. 快速开始

```bash
# 冒烟（前向 + 反向 + 3 步）
python OneTrans_NPU/tests/smoke_npu.py --device npu

# 精度三段（A: fp32 / B: bf16 / C: 10 步训练轨迹）
python OneTrans_NPU/tests/accuracy_npu.py --steps 10

# 与参考实现对照（两侧各起子进程，比结构 + 数值）
python OneTrans_NPU/tests/compare_ref_vs_npu.py

# 单卡端到端训练（真实数据集；bool mask 变体加 ONETRANS_ATTN_MASK=bool）
HF_ENDPOINT=https://hf-mirror.com python OneTrans_NPU/scripts/run_taac2026_sample.py \
  --device npu --batch-size 32 --epochs 2 --seq-len 32 --d-model 64 --multi-num 2 \
  --num-pyramid-layers 3 --amp --amp-dtype bf16 --output-dir /tmp/onetrans_out
```

多卡（DDP/HCCL 扩展性）、mask 与 AMP 单项 A/B、HCCL dtype 探针、一键批量脚本的完整命令与判据 → [`docs/MIGRATION.md`](docs/MIGRATION.md) §6.4「固化：一条命令跑完五阶段」；各探针用途见 `OneTrans_NPU/tests/`。

两个 `run_*.sh` 是本机 ssh + docker 封装（含 `<工号>` 占位路径），换机器需改顶部 5 个变量；本机直跑用上面 4 条即可。
