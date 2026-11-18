# 真实数据 seq=1190 · 16 die 训练实测 — 2026-11-18

**一句话**：真实 TaobaoAd_x1 数据拼到论文序列长度（seq=1190）、16 die、全局 batch 32,768 跑通 60 步；
稳态 **715.2 ms/步**、全局 **45,822 samples/s**（每 die 2,864）、HBM 峰值 **58.77 GB/die**，无错误无 NaN。

## 1 数据怎么来的（必须一起引用）

TaobaoAd_x1 每行只带**定长 50** 的行为窗口，没有 1190 长历史。本数据集由国家脚本
`tools/prep_taobao_seq1190.py` 构造：

- 样本 i 保留原行 i 的 NS 特征（16 域 + price 分桶 = 17）与 `clk` 标签；
- 序列 = **同一用户**各曝光行的 50 长度窗口顺次拼接（每用户平均 ~19 行，最多取 24 窗 = 1200），截到 1190，不足补 0；
- 2,600,000 样本（全局 32,768 ⇒ 79 步/epoch），逐文件 35 GB，122/111 各一份；
- 自检：抽 3 个样本按“该用户窗口顺次拼接”重算，与落盘值逐位相同。

⚠️ **语义边界**：这是“同一用户 50 窗拼接”，**不是**论文的真实 1190 长历史；用户内相邻窗口高度重叠（每个窗口是该用户当时的最近 50 次行为）。
因此本跑只用于**论文形状下的真实数据吞吐/显存**，不能宣称“论文数据规模复现”。

## 2 配置

| 项 | 值 |
|---|---|
| 节点 / 卡 | 122（rank 0-7）+ 111（rank 8-15），`world_size=16` |
| 形状 | d=256 · heads=4 · ffn=1536 · 6 pyramid 层 · seq=1190 · ns=12 token · sep → total 1203，`pyramid_schedule=[1203, 960, 736, 480, 256, 12]` |
| 参数 | **91,919,894**（91.92 M；比论文形状的 91.02 M 多，因 NS 是真实 17 个域） |
| batch | micro 1024 × grad-accum 2 = 2,048/die ⇒ 全局 32,768（与论文口径一致） |
| 后端 | `--source taobao --ns-backend dynamic_emb`（HKV），ns-edim 32 / seq-edim 32（序列 3 字段各自查表） |
| 开关 | NS_BMM / FUSED_NORM / CAT_INPLACE / SDPA_CAUSAL / FA_BSH / FULL_PROJ / EXPLICIT_BF16 全开 |
| 步数 | 65 步（5 warmup + 60 计时） |

命令（两节点各一次，仅 `--node_rank` 与日志名不同）：

```bash
ENVS='ONETRANS_NS_BMM=1 ONETRANS_FUSED_NORM=1 ONETRANS_CAT_INPLACE=1 ONETRANS_SDPA_CAUSAL=1 \
      ONETRANS_FA_BSH=1 ONETRANS_FULL_PROJ=1 ONETRANS_EXPLICIT_BF16=1'
ARGS='--device npu --source taobao --data-dir /tmp/taobao/prep1190 --d-model 256 --num-heads 4 \
      --ffn-hidden 1536 --num-pyramid-layers 6 --seq-len 1190 --ns-len 12 --batch-size 1024 \
      --grad-accum 2 --steps 60 --warmup 5 --lr 0.005 --ns-backend dynamic_emb \
      --ns-edim 32 --seq-edim 32'
docker exec -d zeus-950-v2 bash -lc 'cd /tmp/o16/OneTrans_NPU && HCCL_SOCKET_IFNAME=enp39s0f3 \
  HCCL_IF_BASE_PORT=50541 '"$ENVS"' nohup /opt/buildtools/torch_v2_pt2.7.1/bin/torchrun \
  --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=192.168.0.82 --master_port=29541 \
  scripts/run_onetrans_s.py '"$ARGS"' > /tmp/taobao/real1190.log 2>&1 &'
```

原始日志：`results/real_taobao_16die_seq1190_20261118_node{122,111}.log`

## 3 判据

| 判据 | 实测 |
|---|---|
| 真 16 卡 | `[env] dist=world_size=16 rank=0` |
| 形状 = 论文 S | `pyramid_schedule=[1203, 960, 736, 480, 256, 12]` |
| 两节点代码一致 | 5 个源文件 md5 三处一致（本地仓 / 122 / 111），`taobao_ad_data.py=a7f57a0f…` |
| 两节点数据一致 | `prep1190` 7 个文件 md5 122 与 111 逐一致 |
| 步数完整 | `meas ms` 计数 60，max 763.1 / min 711.3 |
| 无异常 | 无 Traceback、无 NaN、无 OOM（`mfu=nan%` 仅因未传 `--flops-per-sample`） |
| loss | 末 10 步均值 **0.207**（ln2=0.693、CTR 基准熵≈0.31） |

## 4 与另两档的并列（禁相除算收益）

| 口径 | 数据 | seq | 每 die batch | 稳态步时 | 全局吞吐 | HBM 峰值 |
|---|---|---:|---:|---:|---:|---:|
| 本次 | 真实 TaobaoAd_x1（17 NS 域、3 字段序列表） | 1190 | 2,048 | **715.2 ms** | 45,822/s | 58.77 GB |
| 对照 A（v5 报告） | 合成 token（12 NS 域、稠密 64d 序列特征） | 1190 | 2,048 | **614.7 ms** | 53,307/s（按 32,768/0.6147 折算，非实测） | 未采 |
| 对照 B（本文档同日） | 真实 TaobaoAd_x1 | 50 | 2,048 | **123.5 ms** | 265,307/s | 5.87 GB |

三行**数据形态/算子构成不同**，差值不能当加速比：A 与本次差在「合成稠密序列 vs 真实 3 字段查表 + 17 域 NS 表」
（+100.5 ms 的构成未拆），B 与本次差在序列长度（50 vs 1190）。要拆 A/B 需同池同形状的消融跑。

## 5 未做 / 待定

1. 未传 `--flops-per-sample` ⇒ **无 MFU**；真实形状的每样本 FLOPs 需先由探针算（40.91 GFLOP/sample 是论文形状、12 域口径，不可直接套）。
2. 只跑 60 步（0.06 epoch），**未做收敛验收**（要判收敛需跑满 epoch 并给 AUC）。
3. **与论文 H100 407 ms 的对比仍未成立**：本次是“真实数据 + 论文形状”，但数据是拼接构造，且缺 MFU；
   论文口径成绩仍引用 `docs/OneTrans-S_950DT_Optimization_v5_20261117.md`（同形状合成数据 614.7 ms、MFU 24.32%）。
4. A 与本次 +100.5 ms 的构成未拆：候选是“17 NS 域大表查表（userid 1.14 M 行）+ 3 字段序列查表 + 每步 29 MB/die mmap 取数”，三项都未单独 A/B。
