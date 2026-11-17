# torchrec.EC vs RecSDK v2 DynamicEmbedding：昇腾实测对比

> 环境：950DT（1 die）· `/opt/buildtools/torch_v2_pt2.7.1`（torch 2.7.1+cpu / torch_npu 2.7.1）· **torchrec 1.2.0+npu** · **dynamic_emb v25.09**
> 探针：`tools/probes/bench_ec_vs_dynamicemb.py`（双臂）、`bench_dynamicemb_min.py`（B 臂最小可测）、`diag_dynamicemb.py`（诊断）
> 形状：12 张表 × 1,000,000 行 × 64 维（表数据 2.9 GiB f32），每步 12 × 2048 = **24,576 次查表**（≈本模型 NS token 规模）

## 0 结果

| 臂 | 每步 | 吞吐 | 梯度形态 | 优化器 |
|---|---:|---:|---|---|
| **A `torchrec.EmbeddingCollection`（原始 EC）** | **18.09 ms** | 1.36 Mlookup/s | **dense**：12 张表全量物化 = **2,930 MiB/步** | `torch.optim.Adagrad`（dense 梯度） |
| **B `BatchedDynamicEmbeddingTablesV2`（HKV 动态表，`use_index_dedup=True`）** | **10.62 ms** | **2.32 Mlookup/s** | 无 dense 梯度（优化器融进反向） | `EXACT_ROWWISE_ADAGRAD` |
| B′ 同上但 `use_index_dedup=False` | 1,463.85 ms | 0.02 Mlookup/s | 同上 | 同上 |

**结论**：同一形状下 **dynamic_emb 比原始 EC 快 1.70×**（18.09 / 10.62），
而且**省掉 2.9 GiB/步的 dense 梯度**——后者在 16 卡 data-parallel 下还要额外 all-reduce 2.9 GiB/步。
B′ 的 100× 退化需复核（疑似非 dedup 路径在本版本退化），**不作为结论引用**。

## 1 A 臂细节（torchrec.EC）

```python
cfg = [EmbeddingConfig(num_embeddings=1_000_000, embedding_dim=64, name=f"t{i}") for i in range(12)]
ec = EmbeddingCollection(tables=cfg, device=torch.device("npu:0"))
kjt = KeyedJaggedTensor(keys=[...12...], values=idx.reshape(-1), lengths=full((12,), 2048))
ec(kjt) → backward → torch.optim.Adagrad
```

- 输出是 `KeyedTensor`，其元素为 `JaggedTensor`（取 `.values()` 才是 dense 张量）；
- **梯度是 dense 的**：`is_sparse={False}`，12 张表的 `grad` 合计 **2,930 MiB**（= 全表大小）；
- 单步 18.09 ms 里，绝大多数是与「表大小」成正比的梯度物化，而不是 24,576 次查表本身。

## 2 B 臂细节（dynamic_emb / HKV）

```python
opt = DynamicEmbTableOptions(training=True, init_capacity=1_000_000, max_capacity=1_000_000,
        dim=64, embedding_dtype=torch.float32, index_type=torch.int64,
        bucket_capacity=128, block_size=128, io_block_size=1024,
        global_hbm_for_values=1_000_000*64*4, local_hbm_for_values=1_000_000*64*4)
mod = BatchedDynamicEmbeddingTablesV2(table_options=opt_lists, table_names=[...12...],
        feature_table_map=list(range(12)), pooling_mode=DynamicEmbPoolingMode.NONE,
        device=torch.device("npu:0"), use_index_dedup=True,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD, learning_rate=0.1)
mod(indices_flat, offsets)        # 优化器已在反向内完成，无需 optimizer.step()
```

用法要点（踩过的坑）：
1. `DynamicEmbTableOptions` 的 **`dim` / `max_capacity` 必须显式给**，否则传给 C++ 的是 `None`（`TypeError: incompatible constructor arguments`）；
2. `output_dtype` **不是** `DynamicEmbTableOptions` 的字段（是模块参数），传进 options 会 `TypeError`；
3. HKV 的表内存**不计入** `torch_npu.npu.max_memory_allocated`（分配在 torch 分配器之外），显存要按 `全局容量 × dim × 4 B` 自己算；
4. 同一进程内多次构建大表后，`_update_score()` 里会报 `aclrtSynchronizeStream failed` ⇒ 逐配置独立进程更稳（与 perf 探针的既有纪律一致）。

## 3 对本仓的意义

1. 本仓 OneTrans-S 复刻**不用任何表后端**（非序列特征走稠密 `nn.Linear`），所以这个对比用于回答「如果按论文形态接表，该选哪条」；
2. 若要接表：**dynamic_emb 明显优于原始 EC**（快 1.7×、无 dense 梯度），且它支持 rowwise Adagrad / SGD / Adam 等稀疏优化器；
3. 但**跨卡稀疏梯度同步仍是鸿沟**（`c10d::allreduce_` 在 NPU 不支持），dynamic_emb 需要走它自己的分片 + planner 路径，不能直接套 DDP。

## 4 复现

```bash
# 双臂（A 会写 dense 梯度 2.9 GiB，注意显存）
/opt/buildtools/torch_v2_pt2.7.1/bin/python3 tools/probes/bench_ec_vs_dynamicemb.py
# B 臂最小可测（推荐，数字最稳）
/opt/buildtools/torch_v2_pt2.7.1/bin/python3 tools/probes/bench_dynamicemb_min.py
```

## 5 16 die 数据并行实测（同形状）

探针 `tools/probes/bench_dynamicemb_16.py`：**2 节点 × 8 die = 16 die**（122 + 111），
每 die 持有**完整表副本**（数据并行真实形态；NPU 不支持 sparse all_reduce，故无跨卡稀疏同步），
形状与单 die 探针一致：12 表 × 1,000,000 行 × 64 维、每 die 每步 24,576 次查表。
启动：`torchrun --nnodes=2 --nprocs_per_node=8 --master_addr=192.168.0.82`，`HCCL_SOCKET_IFNAME=enp39s0f3`。

| 指标 | 1 die | 16 die | 比值 |
|---|---:|---:|---|
| 单步时延（中位） | 10.62 ms | **14.28 ms** | **1.34×**（同跑时每 die 变慢） |
| 单步时延（最慢 rank） | — | 15.91 ms | 最慢/中位 = **1.114** |
| 每 die 吞吐 | 2.32 Mlookup/s | 1.72 Mlookup/s | 0.74× |
| **聚合吞吐** | 2.32 Mlookup/s | **27.53 Mlookup/s** | **11.90×** |
| 扩展效率（对 16 die 线性） | — | **74.4%** | — |

逐 rank 单步（ms，rank0–15）：
`12.77 / 14.23 / 15.91 / 13.95 / 13.90 / 15.58 / 14.08 / 14.32 ‖ 14.51 / 14.25 / 13.94 / 14.11 / 15.57 / 15.59 / 14.33 / 14.32`

结论：
1. **16 die 同时跑，单 die 慢 34%**（10.62 → 14.28 ms），聚合仍拿到 **11.90×**（效率 74.4%）；
2. 慢 rank 分布**同时出现在两个节点**（node0 的 rank2/5、node1 的 rank4/5），
   ⇒ 抖动来自 **die 级差异**，不是跨节点网络或节点差异；
3. 每 die 变慢但**没有通信项**（表是副本、无 all-reduce）⇒ 34% 的代价来自
   HBM/PCIe/下发在 8 die 并发时的争用，属**同机多 die 竞争**，不随卡数继续恶化到不可用；
4. 与 EC 对比的结论不变：EC 在 1 M 行/表（d=256）单步就要 187.6 ms，HKV 16 die 最慢 rank 也只有 15.91 ms。

> HBM 峰值读数 0.0 GiB 是已知现象：HKV 表分配在 torch 分配器之外，`max_memory_allocated` 不覆盖。
> 按构造，每 die 表数据 = 12 × 1,000,000 × 64 × 4 B = **2.86 GiB**，加优化器状态约 2× ⇒ 约 5.7 GiB/die。
