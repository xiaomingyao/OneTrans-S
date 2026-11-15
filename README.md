# onetrans-d

ByteDance **OneTrans**（TAAC2026 推荐模型）参考实现 → 昇腾 **950DT** 的移植工程、验证脚本与交付文档。

参考实现**不 vendored**，以 git submodule 指向原仓并锁定 commit（零改动）；移植实现全部在 `OneTrans_NPU/`。

```bash
git clone --recurse-submodules https://github.com/xiaomingyao/onetrans-d
# 已 clone 过：git submodule update --init GPU_Reference
```

上游：`https://github.com/WestbrookLong/OneTrans_Pytorch` @ `8febd7eff9b1f2f63fe1fa4e364af8a8755f51fb`（见 `REFERENCE_COMMIT.txt` / `.gitmodules`）。

## 目录

| 目录 | 内容 |
|---|---|
| `GPU_Reference/` | **submodule** → `WestbrookLong/OneTrans_Pytorch` @ `8febd7e`（未改一行，用于 diff 与数值对照） |
| `OneTrans_NPU/` | 移植实现：`npu_compat.py`（设备/AMP/scaler/分布式收尾）+ 装配脚本 + `tests/` 探针 |
| `docs/` | 迁移指导 v1–v4、适配报告 v1–v3、按阶段并排实现（GPU \| NPU）、`npu_changes.diff`、出图脚本 |
| `results/` | 全部实测日志（冒烟 / 精度三段 / AMP / mask / DDP 扩展 / 真实数据 / 合成数据） |

## 改动面

| 口径 | 行数 | 分母 | 占比 |
|---|---:|---:|---:|
| 单卡基础适配 | **67** | 1,233（活跃 Python 行） | **5.4%** |
| 含多卡（DDP/HCCL 装配） | **147** | 1,233 | **11.9%** |

模型文件 `GPU_Reference/models/*` 与 `utils/*` **diff = 0 行**；改动集中在 `npu_compat.py` 与装配脚本。

## 实测结论（950DT 单机 8 die）

| 项 | 结果 | 日志 |
|---|---|---|
| 冒烟 | PASS，首步 601 ms（编译预热）→ 稳态 **76.4 / 76.5 ms** | `results/smoke_950_*_final.txt` |
| 精度 A（NPU fp32 vs CPU fp32） | `max|Δ| = 1.49e-7`，cos `0.99999988` | `results/accuracy_950_*_final.txt` |
| 精度 B（NPU bf16 vs CPU fp32） | `max|Δ| = 2.591e-3`，cos `0.99998170` | 同上 |
| 精度 C（10 步训练轨迹） | `Δloss = 5.96e-08`，`Δacc = 0.00 pp` | 同上 |
| AMP 划算性（同进程交替） | fp32 **56.0 ms / 109 MiB** vs bf16 **69.0 ms / 99 MiB** ⇒ bf16 慢 **23.2%** | `results/amp_bench_*.txt` |
| DDP 梯度等价（world=2） | `worst rel|Δg| = 1.211e-06`（门槛 1e-5）、`Δloss = 0` | `results/scaling_ddp_*.txt` |
| mask 形态 | float 加性 45 kernel / **46.3 µs**；bool **12.3 µs（3.8× kernel 级）** | `results/bench_mask_mode_*.txt`、`kd_*.csv` |
| mask 端到端 | B64/S64 **+1.6%**、B32/S256 **+3.7%**（模型级，非 kernel 级） | 同上 |
| 端到端（合成数据 2 epoch） | val AUC 0.8787 → **0.9372**，acc 0.5833 → 0.6912 | `results/train_synthetic_*.txt` |
| 真实 HF 数据链路 | `data_sample_1000`（1,024 行）单卡 / 2 卡各 2 epoch 跑通，指标见日志（样本极小，不代表任务精度） | `results/train_hf_real_*.txt` |

结论口径：kernel 级加速比 ≠ 端到端收益；本模型 AMP **默认关闭**（bf16 更慢）；bool mask 作为可选优化，默认保持 float 加性 mask 与原实现逐字等价。

## 复现

```bash
# 依赖：CANN 9.x + torch 2.7.1 + torch_npu 2.7.1.post4（Python 3.11）
# 1) 冒烟
python tests/smoke_npu.py --device npu
# 2) 精度三段
python tests/accuracy_npu.py --device npu
# 3) 与 GPU 参考实现做结构/数值对照
python tests/compare_ref_vs_npu.py
# 4) 单卡 / 多卡训练
bash tests/run_all_950.sh
bash tests/run_ddp_950.sh
```

数据：默认走上游的 `--synthetic`；真实数据用 HF `TAAC2026/data_sample_1000`（需 `datasets` + `huggingface_hub` + `pyarrow`，下载端配 `HF_ENDPOINT`）。

## 文档

| 文档 | 用途 |
|---|---|
| `docs/MIGRATION_GUIDE_bytedance_onetrans_v4_20261114.{md,html}` | 迁移指导（最新版：改什么、为什么、判据） |
| `docs/ADAPT_REPORT_bytedance_onetrans_v3_20261114.{md,html}` | 适配报告（按「设备 / 精度能力门 / 集合通信 / kernel 落点 / 收尾 / 计时」分类） |
| `docs/MIGRATION_BY_STAGE_onetrans_v2_20261114.{md,html}` | 五阶段并排实现（GPU 左 / NPU 右，代码高亮） |
| `docs/GPU_VS_NPU_COMPARE_bytedance_onetrans_v1_20261114.{md,html}` | 与参考实现的三层证据链（代码 / 结构 / 数值） |
| `docs/npu_changes.diff` | 完整改动 diff |

> `docs/` 下同时保留了 v1–v3 历史版本，其中 v1/v2 的「合成数据集」口径已废弃，以 v3/v4 为准。

## 脱敏与边界

- 导出到本仓库前已做内部信息替换：工号路径（`/mnt/paas/<工号>/…`）、内网 IP、内部域名、主机名统一替换为 `<工号>` / `<内网IP>` / `<内部域名>` / `<内部主机>`；仓库为 **private**。
- 本仓库**不含** checkpoint、profiling 原始目录、数据集本体。
- 数字均来自 950DT 单机 + 上述软件栈，换机型或换 CANN/torch 版本需重测，不可直接外推。
