# onetrans-d

ByteDance OneTrans（TAAC2026 推荐模型）在昇腾 **950DT** 的开源实现。
整个移植工程包括昇腾适配代码、冒烟测试、精度验证、迁移指导。

## 1. 项目介绍

**上游**：`WestbrookLong/OneTrans_Pytorch` —— OneTrans 的 PyTorch 参考实现（序列建模 + 金字塔多尺度注意力 + 多任务塔，TAAC2026 推荐任务）。
本仓以 **git submodule** 引用原仓并锁定 commit，**上游源码零改动**：

```bash
git clone --recurse-submodules https://github.com/xiaomingyao/onetrans-d
# 已 clone 过：git submodule update --init GPU_Reference
git -C GPU_Reference rev-parse HEAD    # 8febd7eff9b1f2f63fe1fa4e364af8a8755f51fb
git -C GPU_Reference status --porcelain # 空 = 未被改动
```

**本仓做的事**：把这份 GPU 代码迁到昇腾 950DT，并证明迁移是对的——冒烟、三段精度对齐、与参考实现的代码/结构/数值三层对照、AMP 划算性、DDP 梯度等价与扩展性、改动面与判据，全部留下可复跑的探针与日志。

| 目录 | 内容 |
|---|---|
| `GPU_Reference/` | **submodule** → 上游原仓 @ `8febd7e`（零改动，用于 diff 与数值对照） |
| `OneTrans_NPU/` | 移植实现：`npu_compat.py`（兼容层）+ `main_pytorch.py`（+9 行）+ `scripts/run_taac2026_sample.py` + `tests/`（13 个探针 + 2 个批量脚本） |
| `docs/` | 迁移指导 v4、适配报告 v3、分阶段并排实现 v2、GPU/NPU 对照 v1、`npu_changes.diff`、配图与 HTML 生成器 |
| `results/` | 全部实测日志（冒烟 / 精度三段 / AMP / mask / DDP / 真实数据 / 合成数据） |

**验证环境**：Ascend 950DT 单机 8 die；CANN 9.x；Python 3.11；torch 2.7.1+cpu、torch_npu 2.7.1.post4。

## 2. 迁移到昇腾 950 的修改说明

改动全部落在「接线层」：模型与数据管道零改动，平台差异收敛到一个新文件。

| 文件 | 改动 | 具体内容 |
|---|---|---|
| `OneTrans_NPU/npu_compat.py` | **新增 160 行** | 唯一平台差异出口：`init_device()`（导入 torch_npu + `set_device`）、`should_enable_amp()`（能力门，NPU 只放行 bf16）、`build_scaler()`（NPU 用 no-op scaler）、`autocast_context()`、`describe()`（打印环境指纹）、`init_distributed()`（HCCL + RANK/WORLD_SIZE/LOCAL_RANK）、`barrier()`、`shutdown()`（sync + destroy + `os._exit(0)`） |
| `OneTrans_NPU/main_pytorch.py` | **+9 行 / −0** | 只加一个开关 `ONETRANS_ATTN_MASK=float\|bool`。float 是 `0 / -inf` 加性 mask（与原实现逐字等价，**默认**）；bool 直接把「允许位」返给 SDPA。原因：950 上 float 加性 mask 会让 SDPA 退回 Cast+Muls+BatchMatMul+Softmax 手工路径，bool mask 才落 `FlashAttentionScoreV4` |
| `OneTrans_NPU/scripts/run_taac2026_sample.py` | **+104 / −34** | ① 设备 / AMP / scaler 三处改为调兼容层；② 训练集按 rank stride 切分、验证集只在 rank0 跑全量（**循环内不做跨卡归约**）；③ DDP 装配 `device_ids=[local_rank]` / `broadcast_buffers=False` / `find_unused_parameters=True`（最深金字塔块有 16 个参数本步不参与前向）；④ 计时加 `torch.npu.synchronize()`；⑤ 收尾 `shutdown()` + `os._exit(0)`，避免平台 teardown 抖动被误报为任务失败；⑥ `print` → `log`，只在 rank0 输出 |
| `models/*`、`utils/*` | **0 行** | `diff -rq` 无差异：模型定义、损失、指标、数据管道全未改 |

**改动面**（分母 = 1,233 行活跃 Python）：

| 口径 | 行数 | 占比 |
|---|---:|---:|
| 触碰上游文件（`main_pytorch.py` 9 + 脚本 138 = +113/−34） | **147** | **11.9%** |
| ├ 其中单卡基线（设备/AMP/scaler/计时/mask 开关） | 67 | 5.4% |
| └ 其中多卡装配（DDP + rank 切分 + 收尾） | 80 | 6.5% |
| 新增兼容层 `npu_compat.py`（不计入分母，新文件） | 160 | — |


1. **模型文件零改动**——要改模型才能跑，说明改错了层；
2. **训练循环里不做跨卡归约**——实测 epoch 末 `all_gather` + `.cpu()` 会卡在 D2H；改由「训练集按 rank 切分 + 验证在 rank0 全量」，指标不需要集合通信；
3. **AMP 默认关闭**——本模型 bf16 反而更慢（见 §3），`should_enable_amp` 只在显式 `--amp --amp-dtype bf16` 时放行，且 GradScaler 走 no-op。

## 3. 测试结论

| 项 | 结果 | 日志 |
|---|---|---|
| 冒烟 3 步 | PASS；首步 614.3 ms（编译预热）→ 稳态 **76.9 / 77.3 ms** | `results/smoke_950_20261114_final.txt` |
| 精度 A：NPU fp32 vs CPU fp32 | `max\|Δ\| = 1.490e-07`、cos `0.99999988`、rel `3.43e-07` | `results/accuracy_950_20261114_final.txt` |
| 精度 B：NPU bf16 vs CPU fp32 | `max\|Δ\| = 2.591e-03`、cos `0.99998170`、rel `5.43e-03` | 同上 |
| 精度 C：10 步训练轨迹 | `Δloss = 5.960e-08`、`Δacc = 0.00 pp` | 同上 |
| 结构 + 数值对照（logits/loss/梯度） | 键名键序一致、参数 `max\|Δ\| = 0`；logits `1.788e-07`、loss `5.960e-08`、梯度 `rel_max = 1.205e-06`（n=747，2 个数值零梯度，阈值 1e-8）；bf16 为 `2.313e-03 / 2.797e-04 / 1.817e-02` | `results/compare_gpu_vs_npu_950_20261114.txt` |
| AMP 划算性（同进程交替 3 轮×10 步取中位） | fp32 **54.3 ms / 109 MiB** vs bf16 **69.3 ms / 99 MiB** ⇒ 步时比 **1.275**（bf16 慢 **27.5%**），HBM 省 9.2% | `results/amp_bench_fp32_vs_bf16_950_20261114.txt` |
| 多卡扩展（真实数据集 `data_sample_1000`，每 rank 本地 batch 64，8 epoch，稳态 n=7） | 1/2/4/8 die = **848.4 / 1,303.5 / 2,367.8 / 3,755.6** samples/s ⇒ **1.00 / 1.54 / 2.79 / 4.43×**，效率 76.8% / 69.8% / 55.3% | `results/scaling_real_data_950_20261114.txt` |
| mask 形态（kernel 级，3 次 attention 调用合计） | float 加性 **45 kernel / 139.0 µs**；bool **6 kernel / 36.9 µs**（`FlashAttentionScoreV4`×3 + `LogicalNot`×3）；单次 **46.3 µs vs 12.3 µs = 3.77×** | `results/kd_float_mask.csv`、`kd_bool_mask.csv`、`kd_none.csv`（无 mask 3 kernel / 28.0 µs） |
| mask 形态（模型级，B64/S64，4 次重复） | `−0.5% / +1.1% / +3.6% / +4.0%`，中位 **+2.4%**——收益在噪声边缘 | `results/bench_mask_mode_950_20261114.txt` |
| 端到端训练（真实数据 `TAAC2026/data_sample_1000`，1,000 行，2 epoch） | 单 die：13 步/epoch、**78.9 ms/步 / 811.2 samples/s**、val AUC 0.5429；2 die：13 步/epoch、**99.1 ms/步 / 646.0 samples/s**、val AUC 0.4610（样本极小，**不代表任务精度**，只证明能训且指标产出） | `results/train_hf_real_950_20261114.txt`、`results/train_hf_real_2card_950_20261114.txt` |

**三条结论**：

1. **迁移正确性可证**：模型零改动前提下，fp32 误差 1.5e-7 量级、bf16 2.6e-3 量级、10 步训练 loss 差 6e-8。
2. **不要把 kernel 加速比当收益**：bool mask kernel 级 3.77×，端到端中位只有 +2.4%；AMP 在本模型是负收益（步时 +27.5%）。
3. **多卡按生产口径接线**：循环内不做集合通信；1→8 die 扩展比 4.43×（效率 55.3%），口径绑定「每卡本地 batch 64」，换口径不可比。

## 4. 复刻命令

```bash
# 0) 取代码（含参考实现）
git clone --recurse-submodules https://github.com/xiaomingyao/onetrans-d && cd onetrans-d

# 1) 冒烟：3 步前反向 + 优化器步
python OneTrans_NPU/tests/smoke_npu.py --device npu
#    判据：loss 有限且下降；step_ms 第 0 步是编译预热，勿当稳态

# 2) 精度三段（A/B/C）
python OneTrans_NPU/tests/accuracy_npu.py --device npu
#    判据：A ≤1e-4 且 cos≥0.99999；B ≤2e-2 且 cos≥0.9995；C Δloss≤1e-2

# 3) 与参考实现对照（结构 + 数值；两侧各起子进程）
python OneTrans_NPU/tests/compare_ref_vs_npu.py
#    判据：键名/键序一致、参数 max|Δ|=0、747 个梯度 rel_max ≤1e-5

# 4) mask 形态 A/B（kernel 级）
python OneTrans_NPU/tests/bench_mask_mode.py --child --mode float
python OneTrans_NPU/tests/bench_mask_mode.py --child --mode bool

# 5) AMP 划算性（同进程交替，取中位）
python OneTrans_NPU/tests/bench_amp_npu.py

# 6) 单卡训练（合成数据）
python OneTrans_NPU/scripts/run_taac2026_sample.py --device npu --synthetic --epochs 2
#    bool mask 变体：命令前加 ONETRANS_ATTN_MASK=bool

# 7) 多卡（DDP/HCCL）
bash OneTrans_NPU/tests/run_ddp_950.sh      # 默认 2 卡
#    判据：worst rel|Δg| ≤1e-5、Δloss=0；起跑前确认无残留进程占 HCCL 端口

# 8) 真实数据（需 HF 栈：datasets + huggingface_hub + pyarrow，配 HF_ENDPOINT）
python OneTrans_NPU/tests/probe_hccl_dtypes.py   # 先确认集合通信 dtype 支持面
python OneTrans_NPU/scripts/run_taac2026_sample.py --device npu --epochs 2
```

一键全量：`bash OneTrans_NPU/tests/run_all_950.sh`。

## 5. 文档与边界

| 文档 | 用途 |
|---|---|
| `docs/MIGRATION_GUIDE_bytedance_onetrans_v4_20261114.md` | 迁移指导（改什么、为什么、判据） |
| `docs/ADAPT_REPORT_bytedance_onetrans_v3_20261114.md` | 适配报告（按「设备 / 精度能力门 / 集合通信 / kernel 落点 / 收尾 / 计时」分类） |
| `docs/MIGRATION_BY_STAGE_onetrans_v2_20261114.md` | 五阶段并排实现（GPU 左 / NPU 右） |
| `docs/GPU_VS_NPU_COMPARE_bytedance_onetrans_v1_20261114.md` | 三层证据链（代码 / 结构 / 数值） |
| `docs/npu_changes.diff` | 完整改动 diff |

- `docs/` 只保留每个文档的最新 `.md`；历史版本与 `.html` 已移出，需要 HTML 用 `docs/gen_html*.py` 现场生成。
- 脱敏：导出前已把工号路径（`/mnt/paas/<工号>/…`）、内网 IP、内部域名、主机名替换为占位符；仓库为 **private**。
- 本仓**不含** checkpoint、profiling 原始目录、数据集本体。
- 所有数字来自 950DT 单机 + 上述软件栈，换机型或换 CANN/torch 版本需重测，不可直接外推。
