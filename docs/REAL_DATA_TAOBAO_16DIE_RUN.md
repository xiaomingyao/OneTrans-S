# 真实数据（TaobaoAd_x1）16 die 训练实测 — 2026-11-18

**一句话**：真实 CTR 数据在 16 die 上一次跑通（world_size=16），稳态 **123.5 ms/步**、全局 **265,307 samples/s**（每 die 16,582）、HBM 峰值 **5.87 GB/die**，无错误无 NaN。

> ⚠️ 口径：本次 `seq-len=50`（真实数据序列长度），**不是论文 S 的 1190**，因此**不可与论文 H100 407 ms 对比**。
> 论文口径性能仍以 `docs/OneTrans-S_950DT_Optimization_v5_20261117.md`（16 die、档 B 614.7 ms）为准。

## 1 环境与配置

| 项 | 值 |
|---|---|
| 节点 | 122（rank 0-7）+ 111（rank 8-15），`world_size=16` |
| 容器 | 122 `zeus-950-v2`、111 `zeus-950`；代码树 `/tmp/o16/OneTrans_NPU` |
| 栈 | torch 2.7.1 / torch_npu 2.7.1.post4 / SoC `Ascend950DT_9582` |
| 数据 | `/tmp/taobao/prep`（两节点各自容器内），21,929,911 行，CTR 5.15%，669 步/epoch @ global 32,768 |
| 模型 | d=256 · heads=4 · ffn=1536 · 6 pyramid 层 · seq=50 · ns=12 token（17 个 NS 域）→ **params 91.92 M** |
| batch | 每 die micro 1024 × grad-accum 2 = 2,048/die ⇒ 全局 32,768 |
| 后端 | `--ns-backend dynamic_emb`（HKV），ns-edim 32 / seq-edim 32 |
| 开关 | `ONETRANS_NS_BMM FUSED_NORM CAT_INPLACE SDPA_CAUSAL FA_BSH FULL_PROJ EXPLICIT_BF16` 全开 |

## 2 复现命令（两节点各一次，`node_rank` 不同）

```bash
ENVS='ONETRANS_NS_BMM=1 ONETRANS_FUSED_NORM=1 ONETRANS_CAT_INPLACE=1 ONETRANS_SDPA_CAUSAL=1 \
      ONETRANS_FA_BSH=1 ONETRANS_FULL_PROJ=1 ONETRANS_EXPLICIT_BF16=1'
ARGS='--device npu --source taobao --data-dir /tmp/taobao/prep --d-model 256 --num-heads 4 \
      --ffn-hidden 1536 --num-pyramid-layers 6 --seq-len 50 --ns-len 12 --batch-size 1024 \
      --grad-accum 2 --steps 60 --warmup 5 --lr 0.005 --ns-backend dynamic_emb \
      --ns-edim 32 --seq-edim 32'

# 122（node_rank 0）
docker exec -d zeus-950-v2 bash -lc 'cd /tmp/o16/OneTrans_NPU && HCCL_SOCKET_IFNAME=enp39s0f3 \
  HCCL_IF_BASE_PORT=50531 '"$ENVS"' nohup /opt/buildtools/torch_v2_pt2.7.1/bin/torchrun \
  --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=192.168.0.82 --master_port=29531 \
  scripts/run_onetrans_s.py '"$ARGS"' > /tmp/taobao/real16.log 2>&1 &'

# 111（node_rank 1，容器名 zeus-950，日志 real16_n1.log）
```

## 3 判据（全部实测通过）

| 判据 | 实测 |
|---|---|
| 真 16 卡 | `[env] dist=world_size=16 rank=0` |
| 两节点代码一致 | 5 个源文件 md5 与本地仓逐一致（`models/taac_onetrans.py` `3705fa8c…`） |
| 两节点数据一致 | `prep/` 7 个文件 md5 与 122 逐一致 |
| 步数完整 | 8 rank × 65 步（5 warmup + 60 计时），`meas ms` 计数 60 |
| 无异常 | 无 Traceback / 无 NaN / 无 OOM（唯一 `nan` 是 `mfu=nan%`，因未传 `--flops-per-sample`） |
| loss 有信号 | 稳定后均值 **0.20**（ln2=0.693、基准熵≈0.31 ⇒ 低于基准，模型在学） |

原始日志：`results/real_taobao_16die_20261118_node{122,111}.log`。

## 4 下一步（未做，按需）

1. 论文口径对比仍需 **seq=1190 + 全局 32,768** 的合成/真实形状跑，现有 614.7 ms 即该口径。
2. 本次只跑 60 步（0.09 epoch），**未做收敛验收**；要判收敛须跑满 epoch 并给 AUC。
3. `--flops-per-sample` 未传 ⇒ 无 MFU；真实数据形状的 FLOPs 需先由探针算出（当前 40.91 GFLOP/sample 是论文形状口径，不能套用）。
