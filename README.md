# onetrans-d

ByteDance OneTrans（TAAC2026 推荐模型）在昇腾 **950DT** 的开源实现。
整个移植工程包括昇腾适配代码、冒烟测试、精度验证、迁移指导。

## 1. 项目介绍

上游 `WestbrookLong/OneTrans_Pytorch`（序列建模 + 金字塔多尺度注意力 + 多任务塔）以 **git submodule** 引用并锁定 commit，**上游源码零改动**：

```bash
git clone --recurse-submodules https://github.com/xiaomingyao/onetrans-d
git -C GPU_Reference rev-parse HEAD        # 8febd7e…；status --porcelain 为空 = 未改动
```

本仓把这份 GPU 代码迁到 950DT，并证明迁移是对的：冒烟、三段精度对齐、与参考实现的代码/结构/数值对照、AMP 划算性、DDP 梯度等价与扩展性。

| 目录 | 内容 |
|---|---|
| `GPU_Reference/` | submodule → 上游原仓 @ `8febd7e`（零改动，diff 与数值对照的基准） |
| `OneTrans_NPU/` | 移植实现：`npu_compat.py`（兼容层）+ `main_pytorch.py` + `scripts/run_taac2026_sample.py` + `tests/`（探针） |
| `docs/` | `GPU_VS_NPU_CODE_IMPLEMENTATION.md`（代码实现对比）· `MIGRATION.md`（五阶段迁移对照） |
| `patches/` | `npu_changes.diff`（完整改动） |
| `results/` | 各类实测日志（§3 每行标了对应文件） |
| `tools/` | 文档与配图生成器 |

**验证环境**：`Ascend950DT` V120（板卡 `IT23HMDB_D2`，NPU `9582`）单机 8 die、单 die HBM 96 GiB；CANN toolkit **9.1.0** + 宿主机驱动/固件 **25.1.rc2**（Firmware `7.0.9.100.203`）；镜像 `zeus-950-v2-snap:20261102`（Ubuntu 22.04.5 / glibc 2.35）；Python 3.11.0 + **torch 2.7.1+cpu** + **torch_npu 2.7.1.post4**（`/opt/buildtools/torch_v2_pt2.7.1`）；真实数据链路另需 `datasets` 5.2.0 / `huggingface_hub` 2.4.0 / `pyarrow` 26.0.1。

判据：运行首行必须打印 `{'torch': '2.7.1+cpu', 'torch_npu': '2.7.1.post4', 'soc': 'Ascend950DT_9582'}`，对不上就不能与本仓数字对比；950 上 AMP 只放行 bf16。

## 2. 从 GPU 迁移到 NPU：关键修改

改动集中在「接线层」，模型定义与数据管道**零改动**：触碰上游 2 个文件 **147 行**（`main_pytorch.py` +9/−0、`scripts/run_taac2026_sample.py` +104/−34，占 1,233 行活跃 Python 的 11.9%），另新增兼容层 `npu_compat.py` 160 行；`models/*`、`utils/*` **0 行**。

| # | 关键修改 | 为什么这么改 |
|---|---|---|
| 1 | 平台差异收敛到 `npu_compat.py` 一个文件（设备初始化 / AMP 能力门 / no-op scaler / autocast / 分布式 / 收尾） | 其余代码只调这层接口，`models`、`utils` 才能保持 0 行；换机型只改一个文件 |
| 2 | 新增 `ONETRANS_ATTN_MASK=float\|bool` 开关，默认 float（与上游逐字等价） | 950 上 float 加性 mask 使 SDPA 退回 Cast+Muls+BatchMatMul+Softmax 手工路径（45 kernel / 139 µs）；bool mask 才落 `FlashAttentionScoreV4`（6 kernel / 36.9 µs） |
| 3 | `should_enable_amp()` 在 NPU 只放行 bf16，GradScaler 用 no-op | 平台 autocast 默认 dtype 是 fp16 而非 bf16；且本模型 bf16 实测慢 27.5%，默认关闭 |
| 4 | HCCL + `device_ids=[local_rank]` + `broadcast_buffers=False` + `find_unused_parameters=True`；训练集按 rank 切分、验证只在 rank0 跑全量，**循环内不做跨卡归约** | epoch 末 `all_gather` + `.cpu()` 实测会卡在 D2H；AUC/acc 本来不需要集合通信 |
| 5 | 计时加 `torch.npu.synchronize()`；收尾 `shutdown()` + `os._exit(0)` | 不同步的计时偏小；平台 teardown 抖动会被误报成任务失败 |

细节（每项的「改前 vs 改后 + 判据」）：

- 逐项说明 → [`docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md`](docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md)
- 按五阶段并排对照（GPU 左 / NPU 右，标出哪段零改动） → [`docs/MIGRATION.md`](docs/MIGRATION.md)
- 逐行 diff → [`patches/npu_changes.diff`](patches/npu_changes.diff)（对上游副本 `patch -p1` 可精确复原 `OneTrans_NPU/`）

## 3. 测试验证

| 项 | 结果 | 日志 |
|---|---|---|
| 精度（NPU vs CPU fp32） | fp32 `max|Δ|=1.490e-07` / cos `0.99999988`；bf16 `2.591e-03` / `0.99998170`；10 步 Δloss `5.96e-08` | `results/accuracy_950_20261114_final.txt` |
| 与参考实现对照 | 参数 `max|Δ|=0`；logits `1.788e-07`、loss `5.960e-08`、747 个梯度 `rel_max=1.205e-06` | `results/compare_gpu_vs_npu_950_20261114.txt` |
| 冒烟 / 单卡步时 | PASS；首步 ~610–645 ms（编译预热）→ 稳态 **75–78 ms** | `results/smoke_950_20261114_final.txt` |
| AMP 划算性 | fp32 **54.3 ms** vs bf16 **69.3 ms**（比 1.275，慢 27.5%），HBM 只省 9.2% ⇒ **默认关** | `results/amp_bench_fp32_vs_bf16_950_20261114.txt` |
| mask 形态 | kernel 级 float 139.0 µs / bool 36.9 µs（**3.77×**）；端到端中位仅 **+2.4%** | `results/kd_*.csv`、`results/bench_mask_mode_950_20261114.txt` |
| 多卡扩展（每 rank batch 64） | 1/2/4/8 die 稳态 **829.5 / 1,303.6 / 2,348.3 / 3,686.2** samples/s = **1.00/1.57/2.83/4.44×** | `results/scaling_real_data_950_20261114.txt` |
| DDP 梯度等价 | `worst rel|Δg| = 1.211e-06`（门槛 1e-5）、`Δloss = 0` | `results/verify_950_20261115_rerun.txt` |
| 端到端训练（真实数据） | 1,000 行 2 epoch：78.9 ms/步、811.2 samples/s（样本极小，不代表任务精度） | `results/train_hf_real_950_20261114.txt` |

口径：kernel 级加速比 ≠ 端到端收益；本模型 AMP 是负收益；多卡数字绑定「每卡本地 batch」，换口径不可比。2026-11-15 的逐条复跑记录见 `results/verify_950_20261115_rerun.txt`。

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
