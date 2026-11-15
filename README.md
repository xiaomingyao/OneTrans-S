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
| `docs/` | `GPU_VS_NPU_CODE_IMPLEMENTATION.md`（GPU 与 NPU 代码实现对比）、`MIGRATION.md`（按五阶段的迁移实现对照）、`figures/` |
| `patches/` | `npu_changes.diff`（完整改动 diff） |
| `tools/` | 文档与配图生成器（`gen_html.py` / `gen_html_pair.py` / `gen_fig_mask_cost.py` / `build_merged_v3.py`） |
| `results/` | 全部实测日志（冒烟 / 精度三段 / AMP / mask / DDP / 真实数据 / 合成数据） |

**验证环境**：Ascend 950DT 单机 8 die；CANN 9.x；Python 3.11；torch 2.7.1+cpu、torch_npu 2.7.1.post4。

## 2. 工程结构与模块说明

### 2.1 目录职责

```
onetrans-d/
├── GPU_Reference/            # submodule：上游参考实现，零改动（diff 与数值对照的唯一基准）
├── OneTrans_NPU/             # 移植实现（改动只在这里）
│   ├── main_pytorch.py       # 模型 + 训练主程序（325 行；对上游客 +9 行）
│   ├── npu_compat.py         # 平台兼容层，新增 160 行，唯一平台差异出口
│   ├── models/               # TAACOneTransClassifier + 金字塔调度（116 行，0 改动）
│   ├── utils/                # 数据解析 331 行 / 指标 47 行 / 通用 43 行（0 改动）
│   ├── scripts/              # 训练装配入口（452 行，+104/−34）
│   └── tests/                # 13 个探针 + 2 个批量脚本，全部可独立跑
├── docs/                     # GPU 与 NPU 代码实现对比 / 按五阶段的迁移对照
├── patches/npu_changes.diff   # 完整改动 diff（458 行：2 个改动文件 + npu_compat.py 全文）
├── results/                  # 15 项：11 份日志 + 3 份 kernel 明细 + 运行元数据
└── tools/                    # 文档与配图生成器（不改代码，只产 HTML/图）
```

### 2.2 四个关键文件的分工

| 文件 | 行数 | 职责 | 改动 |
|---|---:|---|---|
| `OneTrans_NPU/npu_compat.py` | 160 | **唯一平台差异出口**：设备初始化（`init_device`）、AMP 能力门（`should_enable_amp`）、scaler（`build_scaler`，NPU 用 no-op）、autocast（`autocast_context`）、环境指纹（`describe`）、分布式（`init_distributed` / `barrier` / `shutdown`） | 新增 |
| `OneTrans_NPU/scripts/run_taac2026_sample.py` | 452 | **训练装配入口**：argparse（数据集 / 序列长度 / 金字塔层数 / batch / amp / DDP 开关）→ 数据加载 → 模型构建 → 训练与验证循环 → 逐 epoch 日志与 `[perf]` 统计 | +104/−34 |
| `OneTrans_NPU/main_pytorch.py` | 325 | **模型与训练主干**：`FFNLayer` / `RMSNorm` / `CausalMaskAttention`（注意力 + mask 构造）/ `OneTransBlock` / `MultiOneTransBlock` / `main()` | +9（仅 mask dtype 开关） |
| `OneTrans_NPU/models/taac_onetrans.py` | 116 | `TAACOneTransClassifier` 分类头与 `linear_pyramid_schedule`（金字塔序列长度调度） | 0 |

数据侧 `utils/taac_data.py`（331 行）负责把 HF 数据行解析成 `non_seq` / `seq` 两组张量（兼容 flat schema 与嵌套 schema）；指标在 `utils/metrics.py`（AUC / acc）。

### 2.3 测试与探针（`OneTrans_NPU/tests/`）

| 脚本 | 回答什么问题 |
|---|---|
| `smoke_npu.py` | 能否跑起来：前向 → 反向 → 3 步优化，断言 loss 有限、参数有更新 |
| `accuracy_npu.py` | 精度是否对齐：A) NPU fp32 vs CPU fp32、B) NPU bf16 vs CPU fp32、C) 10 步训练轨迹 |
| `compare_ref_vs_npu.py` | 与上游是否同一份实现：同 seed 建模型比 state_dict 键序/参数，再比 logits / loss / 全部梯度（两侧各起子进程） |
| `compare_mask_semantics.py` | 4 种 mask 类型的「允许集」是否逐元素一致（比语义不比张量） |
| `probe_sdpa_kernel.py` | SDPA 到底落到哪个 kernel：float 加性 mask vs bool mask |
| `bench_mask_mode.py` | mask 形态的端到端步时 A/B（同进程交替） |
| `bench_amp_npu.py` | AMP 划不划算：同进程交替 fp32 / bf16，取中位步时与峰值 HBM |
| `probe_amp_npu.py` | 「以为开了 AMP」与「真的开了」是否一致：autocast 生效性、默认 dtype、输出 dtype、scaler 形态 |
| `probe_hccl_dtypes.py` | HCCL 的 `all_gather` / `all_reduce` 支持哪些 dtype（fp64 归约不支持等） |
| `probe_ddp_collective.py` | 最小复现：DDP 反向后紧接自己的集合通信会不会卡住 |
| `ddp_equiv_npu.py` | 多卡梯度等价：DDP(W 卡 × 本地 batch) 的梯度是否等于单进程全局 batch |
| `probe_transfer_to_npu.py` | `import torch_npu` 与 `transfer_to_npu` 分别替换了多少 CUDA API（三模式各起子进程） |
| `run_all_950.sh` / `run_ddp_950.sh` | 单机批量：冒烟 + 精度 + 端到端；多卡：1/2/4/8 die 扩展性 |

设计约束：探针都是**独立可跑的单文件**，失败时只影响自己；训练入口只做装配，不含模型或平台细节。

## 3. 迁移到昇腾 950 的修改说明

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
3. **AMP 默认关闭**——本模型 bf16 反而更慢（见 §4），`should_enable_amp` 只在显式 `--amp --amp-dtype bf16` 时放行，且 GradScaler 走 no-op。

## 4. 测试结论

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

### 4.1 现场复跑校验（2026-11-15，Ascend 950DT / CANN 9.1.0 / torch 2.7.1 + torch_npu 2.7.1.post4）

按 §5 的命令逐条在 950 上重跑，用于确认「仓库里的代码 + 文档」自洽；原始输出见 `results/verify_950_20261115_rerun.txt`。

| # | 检查 | 结果 | 与 `results/` 的一致性 |
|---|---|---|---|
| 1 | `patch -p1` 还原三个文件 | `main_pytorch.py` / `scripts/run_taac2026_sample.py` / `npu_compat.py` 与仓库逐字节一致 | — |
| 2 | 上游零改动 | `models/*`、`utils/*` 与 submodule 逐字节相同；仅 2 个目标文件有 diff（+9/−0、+104/−34） | 与 §3 改动面一致 |
| 3 | `py_compile` + 两个入口 `--help` | 0 报错 | — |
| 4 | `smoke_npu.py --device npu` | **SMOKE PASS**；首步 645.3 ms → 稳态 75.5 / 75.0 ms；loss 1.610520 → 1.199051 → 0.794862 | loss 三次逐位相同；步时 75.0–75.5 vs 日志 76.9–77.3（同批抖动） |
| 5 | `accuracy_npu.py --steps 10` | **ACCURACY PASS**；A `1.490e-07` / cos `0.99999988`；B `2.591e-03`；C Δloss `5.960e-08` | 与日志**逐位相同** |
| 6 | `compare_ref_vs_npu.py` | **COMPARE DONE**；参数 `max|Δ|=0`；bf16 logits `2.313e-03`、loss `2.797e-04`、梯度 `1.817e-02`；fp32 梯度 `1.2048e-06`（n=747） | 与日志一致 |
| 7 | `torchrun -n2 probe_hccl_dtypes.py --dtype float32` | PASS（`sum=96.0`） | — |
| 8 | `torchrun -n2 ddp_equiv_npu.py --local-batch 8` | **DDP-EQUIV PASS**；`worst rel|Δg| = 1.211e-06`（门槛 1e-5）、`Δloss = 0` | 补齐了原先未落盘的等价性数字 |
| 9 | `bench_amp_npu.py` | fp32 **54.8 ms / 109 MiB** vs bf16 **68.9 ms / 99 MiB**，比 **1.257** | 日志 54.3/69.3（比 1.275）；同批抖动 |
| 10 | `bench_mask_mode.py` | bool vs float **+1.6%** | 日志 4 次重复 −0.5%…+4.0%，本次落在区间内 |
| 11 | 单卡端到端训练（400 行 / 2 epoch） | 跑通；78.3 ms/步、408.9 samples/s、val AUC 0.4038 | 数据子集不同，不做数值对比 |

**本轮发现并已修的三处文档/代码问题**：

1. `accuracy_npu.py` 不接受 `--device`（只有 `--steps`）——README 原命令会直接报 `unrecognized arguments`；
2. `bench_mask_mode.py` 没有 `--mode` 参数（内部自己起 float/bool 两个子进程）——原命令同样报错；
3. 三个多 rank 探针（`probe_hccl_dtypes` / `probe_ddp_collective` / `ddp_equiv_npu`）缺 `LOCAL_RANK` 时抛 `KeyError` 堆栈——已改为打印 `torchrun` 用法并退出。

## 5. 复刻命令

```bash
# 0) 取代码（含参考实现）
git clone --recurse-submodules https://github.com/xiaomingyao/onetrans-d && cd onetrans-d

# 1) 冒烟：3 步前反向 + 优化器步
python OneTrans_NPU/tests/smoke_npu.py --device npu
#    判据：loss 有限且下降；step_ms 第 0 步是编译预热，勿当稳态

# 2) 精度三段（A/B/C）
python OneTrans_NPU/tests/accuracy_npu.py --steps 10
#    判据：A ≤1e-4 且 cos≥0.99999；B ≤2e-2 且 cos≥0.9995；C Δloss≤1e-2

# 3) 与参考实现对照（结构 + 数值；两侧各起子进程）
python OneTrans_NPU/tests/compare_ref_vs_npu.py
#    判据：键名/键序一致、参数 max|Δ|=0、747 个梯度 rel_max ≤1e-5

# 4) mask 形态 A/B（端到端步时；脚本内部自己起两个子进程跑 float / bool）
python OneTrans_NPU/tests/bench_mask_mode.py
#    单侧跑法：ONETRANS_ATTN_MASK=float python OneTrans_NPU/tests/bench_mask_mode.py --child

# 5) AMP 划算性（同进程交替，取中位）
python OneTrans_NPU/tests/bench_amp_npu.py

# 6) 单卡端到端训练（真实数据集；脚本没有 --synthetic，数据来自 HF）
HF_ENDPOINT=https://hf-mirror.com python OneTrans_NPU/scripts/run_taac2026_sample.py \
  --device npu --batch-size 32 --epochs 2 --seq-len 32 --d-model 64 \
  --multi-num 2 --num-pyramid-layers 3 --amp --amp-dtype bf16 --output-dir /tmp/onetrans_out
#    bool mask 变体：命令前加 ONETRANS_ATTN_MASK=bool

# 7) 多卡（DDP/HCCL）：梯度等价 + 1/2/4/8 die 扩展性
bash OneTrans_NPU/tests/run_ddp_950.sh 64   # 参数 = 全局 batch（默认 64，按卡数均分到每 rank）
#    只验梯度等价（不跑扩展性）：torchrun --nproc_per_node=2 OneTrans_NPU/tests/ddp_equiv_npu.py --local-batch 8
#    判据：worst rel|Δg| ≤1e-5、Δloss=0；起跑前确认无残留进程占 HCCL 端口

# 8) 集合通信 dtype 支持面（多卡前先确认；探针都是多 rank 脚本，必须 torchrun 起）
torchrun --nproc_per_node=2 OneTrans_NPU/tests/probe_hccl_dtypes.py --dtype float32
```

一键全量：`bash OneTrans_NPU/tests/run_all_950.sh`（冒烟 → 精度 → 端到端）。

> 这两个 `.sh` 是**本机封装**：内部写死了 `ssh 122` + `docker exec` + `/mnt/paas/<工号>/…` 中转路径，换机器需改顶部 5 个变量（`SRC` / `HOST` / `CT` / `PY` / 数据集）。只想在本机直接跑时，用上面 1–8 的逐条命令即可，那些命令不带任何宿主路径假设。

## 6. 文档与边界

| 文档 | 用途 |
|---|---|
| `docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md` | **GPU 与 NPU 代码实现对比**：按「设备 / 精度能力门 / 集合通信 / kernel 落点 / 收尾 / 计时」分类，逐项给「改前 vs 改后 + 判据」 |
| `docs/MIGRATION.md` | **按五阶段的迁移实现对照**：逐段并排（GPU 左 / NPU 右），标出哪段零改动、哪段必须改 |
| `patches/npu_changes.diff` | 完整改动 diff：`main_pytorch.py`（+9/−0）、`scripts/run_taac2026_sample.py`（+104/−34）、`npu_compat.py`（新增全文）；`cd` 到 `OneTrans_NPU/` 上层的同名副本 `patch -p1` 可精确复原当前 `OneTrans_NPU/` |

- `docs/` 只放这两篇最新 `.md` 与 `figures/`；迁移指导（v4）、GPU/NPU 数值对照（v1）、历史版本与全部 `.html` 已移出仓库，脚本统一在 `tools/`。
- 脱敏：导出前已把工号路径（`/mnt/paas/<工号>/…`）、内网 IP、内部域名、主机名替换为占位符；仓库为 **private**。
- 本仓**不含** checkpoint、profiling 原始目录、数据集本体。
- 所有数字来自 950DT 单机 + 上述软件栈，换机型或换 CANN/torch 版本需重测，不可直接外推。
