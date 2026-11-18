# 真实数据（TaobaoAd_x1）训练结果汇总 — 2026-11-18

**范围**：真实 CTR 数据（21,929,911 行、CTR 5.15%），16 die（122 rank0-7 + 111 rank8-15），全局 batch 32,768，两档序列长度各跑一次。
**两条都不能当"论文 OneTrans-S 复现"**：序列要么是真实 50 长度、要么是拼接出来的 1190（见 §3）。

## 1 结果（数字由 `results/real_data_16die_results_20261118.json` 机器生成，勿手改）

| 项 | 档 1：seq=50（真实窗口） | 档 2：seq=1190（拼接，对齐论文序列长度） |
|---|---:|---:|
| 稳态步时（n=60 中位） | **123.5 ms** | **715.1 ms** |
| 均值 / 最小 / 最大 | 124.3 / 121.2 / 139.1 ms | 716.0 / 711.3 / 763.1 ms |
| 全局吞吐 | **265,307 samples/s** | **45,822 samples/s** |
| 每 die 吞吐 | 16,582 samples/s | 2,864 samples/s |
| HBM 峰值 | **5.87 GB/die** | **58.77 GB/die** |
| 参数量 | 91,919,894 | 91,919,894 |
| loss（末 10 步均值） | 0.206 | 0.207 |
| MFU | 未采（未传 `--flops-per-sample`） | 未采 |
| 形状 | d=256 · h=4 · ffn=1536 · 6 层 · ns=12 | 同左，`pyramid_schedule=[1203,960,736,480,256,12]` |
| 每 die batch | micro 1024 × accum 2 = 2,048 | 同左 |

两档用的是**同一份数据的两种序列构造**，不是两个数据集：档 1 直接用原 50 长度窗口，档 2 把同一用户各曝光的 50 窗顺次拼到 1190。
**两档之间不相除算"加速比"**（序列长度差 23.8 倍，属口径不同）。

## 2 溯源表（数字 → 文件 → 复现命令）

| 数字 | 文件 | 复现命令 |
|---|---|---|
| 两档汇总 | `results/real_data_16die_results_20261118.json` | 解析下面两份日志的 `meas ms=` 与 `[perf]` 行 |
| 档 1 原始日志 | `results/real_taobao_16die_20261118_node122.log`（+ `_node111.log`） | 见 `docs/REAL_DATA_TAOBAO_16DIE_RUN.md` §2 |
| 档 2 原始日志 | `results/real_taobao_16die_seq1190_20261118_node122.log`（+ `_node111.log`） | 见 `docs/REAL_DATA_SEQ1190_16DIE_RUN.md` §2 |
| 50 窗数据集 | 122/111 容器 `/tmp/taobao/prep`（各 14 GB） | `tools/prep_taobao_ad.py --csv train.csv --out prep --seq-len 50` |
| 1190 数据集 | 122/111 容器 `/tmp/taobao/prep1190`（各 35 GB，2.6 M 样本） | `tools/prep_taobao_seq1190.py --src prep --out prep1190 --rows 2600000 --seq-len 1190` |
| 档 2 单 die 冒烟 | `results/` 无（日志在节点 `/tmp/taobao/smoke1190.log`） | 同 §2 命令但 `--batch-size 128 --steps 2`，判据 `EXIT=0` ∧ `pyramid_schedule=[1203,…]` |

## 3 判据与边界（引用这两行数时必须一起引用）

| 项 | 判据 / 实测 |
|---|---|
| 真 16 卡 | `[env] dist=world_size=16 rank=0`；111 侧 rank8-15 均有 reducer 初始化 |
| 代码双节点一致 | 5 个源文件 md5 三处一致（本地仓 / 122 容器 / 111 容器） |
| 数据双节点一致 | 两档数据集各自 7 个文件 `md5sum` 122 与 111 `diff` 为空（`MD5_ALL_MATCH`） |
| 步数完整 | 两档均 `meas ms` 计数 60（5 warmup + 60 计时），无 Traceback / NaN / OOM |
| **不是论文长历史** | 档 2 的 1190 是「同一用户 50 窗顺次拼接」（每用户平均 ~19 个曝光行，窗口高度重叠），**不是**论文的真实 1190 长历史 |
| 无 MFU | `flops_per_sample=0` ⇒ `mfu=nan%`；40.91 GFLOP/sample 是论文形状·12 域口径，不可套到 17 域真实数据 |

## 4 与已有两档并列（只并列，不相除）

| 口径 | 数据 | seq | 稳态步时 | 备注 |
|---|---|---:|---:|---|
| 本次档 2 | 真实 TaobaoAd_x1（17 NS 域 + 3 字段序列表） | 1190 | **715.1 ms** | HBM 58.77 GB/die |
| v5 报告（论文口径成绩） | 合成 token（12 NS 域、稠密 64d 序列特征） | 1190 | **614.7 ms** | MFU 24.32%；HBM 未采 |
| 本次档 1 | 真实 TaobaoAd_x1 | 50 | **123.5 ms** | HBM 5.87 GB/die |

档 2 比合成档慢 100.4 ms，**构成未拆**，候选三项：① 17 个 NS 域里 userid 表 1.14 M 行的查表；② 序列 3 字段各自查表（每步 3×B×L 次 gather）；③ 每步 29 MB/die 的 mmap 取数。要拆需同形状消融。

## 5 未做

1. **收敛验收**：两档各只跑 60 步（档 1 = 0.09 epoch，档 2 = 0.76 epoch），未跑满 epoch、未算 AUC。
2. **MFU / 算子级 profile**：未传 `--flops-per-sample`，也未抓 profiling；真实数据的 kernel 构成（查表 vs 计算占比）未测。
3. **与 H100 407 ms 的判定**：仍以 `docs/OneTrans-S_950DT_Optimization_v5_20261117.md` 的 614.7 ms 为准，本次两档不参与该判定。
