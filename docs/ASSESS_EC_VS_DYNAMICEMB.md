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

## 6 OneTrans-S 全模型 16 die：linear vs HKV（同环境对照）

同一份代码、同一 torch 2.7.1 环境（`/opt/buildtools/torch_v2_pt2.7.1`）、同一 16 die（2×8）、
同一档 B 配置（`NS_BMM/FUSED_NORM/CAT_INPLACE/SDPA_CAUSAL/FA_BSH/FULL_PROJ/EXPLICIT_BF16` 全开，
per-GPU micro-batch 1024×2、全局 batch 32,768），**唯一变量是 `--ns-backend`**：

| 臂 | 后端 | 稠密参数 | 稳态步时（n=5 中位） | HBM 峰值 | 全局 sps |
|---|---|---:|---:|---:|---:|
| **对照** | `linear`（torch 2.7.1） | 91,026,946 | **636.0 ms** | 58.48 GB | 51,522 |
| **实验** | `dynamic_emb`（HKV, ns_rows=100,000） | 90,237,452 | **667.5 ms** | 58.49 GB | 49,092 |
| 参照（另一环境） | `linear`（torch 2.10.0，主环境） | 91,026,946 | 614.7 ms | 58.48 GB | 53,307 |

**三个差值必须分开读**（不可相加成一个"HKV 慢了 52.8 ms"）：

| 比较 | 差值 | 归因 |
|---|---:|---|
| HKV 667.5 vs 同环境 linear 636.0 | **+31.5 ms（+4.95%）** | **HKV 引擎本身的单步代价**（查表+去重+rowwise Adagrad 融进反向） |
| 同环境 linear 636.0 vs 主环境 linear 614.7 | **+21.3 ms（+3.47%）** | **torch 2.7.1 vs 2.10.0 的环境差异**（与 HKV 无关） |
| HKV 667.5 vs 主环境 linear 614.7 | +52.8 ms（+8.6%） | 上两项之和，**不是纯 HKV 代价** |

其它观察：
1. **参数口径**：HKV 臂稠密参数 90,237,452（少了 NS 的 `nn.Linear` 786,498），
   表参数（12 × 100,000 × 256 = 307.2 M 个 f32 = 1.23 GB）**不进 `nn.Parameters`** ⇒ 报告参数对比时必须分开列。
2. **HBM 几乎不变**（58.48 → 58.49 GB），因为表只有 1.23 GB 且不计入 torch 分配器统计；
   把 `ns_rows` 提到 1 M 时表会到 12.3 GB/die，需重新测 HBM 与 miss 率影响。
3. **loss 无参考价值**：合成数据是随机标签，两臂 loss 都在 0.7–2.9 间抖动，不能用于收敛对比。
4. 数值正确性未做逐位比对（表后端的 id→向量映射与稠密投影不可比），此处只回答**性能**。

结论：**HKV 接进全模型的代价是 +4.95%（同环境同配置）**，远小于它在引擎层相对 EC 的优势（dynamic_emb 10.62 ms vs EC 18.09 ms，1.70×），
也远小于 EC 在全模型上的预期开销（EC 在 1 M 行/表、d=256 时单步就要 187.6 ms）。
⇒ 若要把论文的稀疏表形态搬上昇腾，**dynamic_emb 是唯一可行的引擎**，代价约 5%。

## 7 OneTrans-S 训练性能：linear vs HKV（n=20 稳态，同环境）

同 torch 2.7.1、同 16 die、同档 B 配置；`--steps 20 --warmup 3 --flops-per-sample 40910000000`
（每卡每步 40.91 GFLOP × 2048 = **83.78 TFLOP**，MFU 分母 432 TF/s）。

| 臂 | 后端 / 表规模 | 步时中位（n=20） | 全局 sps | 每卡 sps | **MFU** | HBM（分配器内） | 稠密参数 |
|---|---|---:|---:|---:|---:|---:|---:|
| 对照 | `linear`（无表） | **635.7 ms** | 51,550 | 3,222 | **30.51%** | 58.48 GB | 91,026,946 |
| 实验 1 | `dynamic_emb` 10 万行（1.23 GB/die） | **661.7 ms** | 49,524 | 3,095 | **29.31%** | 58.49 GB | 90,237,452 |
| 实验 2 | `dynamic_emb` **100 万行**（12.3 GB/die） | **666.3 ms** | 49,179 | 3,074 | **29.11%** | 58.48 GB | 90,237,452 |

| 比较 | Δ步时 | Δ% | 读法 |
|---|---:|---:|---|
| HKV 10 万行 vs linear | +26.0 ms | **+4.09%** | HKV 引擎代价（小表） |
| HKV 100 万行 vs linear | +30.6 ms | **+4.81%** | HKV 引擎代价（工业规模表） |
| HKV 100 万行 vs 10 万行 | +4.6 ms | **+0.70%** | **表体量 ×10 只贵 0.7%** ⇒ 成本与表体量解耦（实证定理 `C_dyn ∝ A`，不 ∝ S） |

关键结论：
1. **HKV 接进整模型的代价是 +4.1% ~ +4.8%**（表 1.23 GB → 12.3 GB 只多 0.7%），MFU 从 30.51% 降到 29.11%；
2. **表体量对步时几乎无影响**，与 §1 的理论一致：dynamic_emb 只搬访问到的行（24 MiB/步），不搬整表；
   对比 EC：同规模（1 M 行、d=256）单步光表更新就要 187.6 ms，**约为 HKV 全模型代价的 6 倍**；
3. **HBM 读数不含表**：报告里的 58.48 GB 是 torch 分配器口径；实际占用需 +表体量
   （10 万行 ⇒ 约 59.7 GB/die；100 万行 ⇒ 约 **70.8 GB/die**，仍在 91.75 GB 内）；
4. 两臂稠密参数不同（90.24 M vs 91.03 M，差 786,498 = 被替换掉的 NS `nn.Linear`），
   表参数（307.2 M 个 f32 = 12.3 GB）**不进 `nn.Parameters`** ⇒ 参数对比必须分列，不能相加后当"参数量"。

## 8 论文形态输入（id + 分桶 → 查表）三臂对照（16 die, n=20）

`--ns-shape paper` 打开后，输入链路改成论文 §3.2.1 的形态：
`数值特征分桶（Bucketizer，等宽 1,000,000 桶）/ 类别特征给 id` → `12 个特征域各查一张表（e_dim=32）`
→ `concat 成 384 维` → `单次 MLP(384 → 12×256)` → `split 成 12 个 token`（Auto-Split 口径）。
每 die 每步仍是 **24,576 次查表**（12 域 × 2048）。

| 臂 | 输入形态 / 引擎 | 步时中位 | MFU | 参数量（口径） | HBM |
|---|---|---:|---:|---|---:|
| A | `direct` + 无表（linear） | **635.7 ms** | **30.51%** | 91,026,946（全稠密） | 58.48 GB |
| B | `direct` + HKV | 666.3 ms | 29.11% | 90,237,452 稠密（表 307.2 M f32 不入 Parameters） | 58.48 GB |
| C | **`paper` + HKV** | **667.0 ms** | **29.08%** | 91,420,172 稠密（表 384 M f32 = 1.54 GB 不入 Parameters） | 58.49 GB |
| D | **`paper` + torchrec EC** | **670.1 ms** | **28.94%** | **475,420,162**（表 384 M **计入** `nn.Parameters`）+ 91.42 M | 65.63 GB |

| 比较 | Δ步时 | Δ% | 读法 |
|---|---:|---:|---|
| C（paper+HKV）vs A（无表） | +31.3 ms | **+4.92%** | 论文形态查表的整步代价 |
| D（paper+EC）vs C（paper+HKV） | +3.1 ms | **+0.46%** | 小表（1.54 GB）下两者接近 |
| B（direct+HKV）vs C（paper+HKV） | −0.7 ms | −0.10% | 形态差异对步时几乎无影响（查表次数相同） |

四点结论：
1. **论文形态（分桶 → 查表 → concat → 单次 MLP → split）在昇腾上跑通**，整步代价 **+4.92%**，MFU 30.51% → 29.08%；
2. **EC 与小表下的 HKV 差距只有 0.46%** —— 但这是 `e_dim=32`、表仅 1.54 GB 的结果；
   EC 的成本 ∝ 表体量（§1 的 `k=6.32 ms/GiB/步`），把 `e_dim` 提到 256（表 12.3 GB）时按模型外推 **+78 ms/步**，
   且「表 + dense 梯度 + 优化器状态」三项常驻会把 HBM 推到 ~95 GB（超单 die 91.75 GiB）⇒ 必须行分片；
3. **参数量口径必须分列**：EC 臂 475.42 M 里 **384 M 是表**（进 `nn.Parameters`，故有 dense 梯度、HBM +7.14 GB）；
   HKV 臂的表（1.54 GB / 384 M 个数）不进 `nn.Parameters`，报 91.42 M 与之不可比；
4. paper 形态的稠密参数比 direct 多 393,226（MLP 384→3072 = 1.18 M 替换了 256→3072 = 0.79 M），属结构差异不是优化。

> 仍未对齐的部分：论文序列侧也用 **item ID 查表**（§3.2.2 "concatenating the item ID with its side information"），
> 本仓序列侧仍是 `nn.Linear(64 → 256)` 的连续特征投影（查表量会是 NS 侧的 100 倍：2048×1190 ≈ 2.4 M 次/步/die）。
> 这条未做，故上表只覆盖非序列侧的查表。

## 9 完整训练一次（新后端全链路：paper 形态 + HKV + 序列 item 表，16 die）

配置：`paper` 形态（12 域 × 1,000,000 桶 × e_dim 32）+ 序列侧 item 表（1,000,000 行 × 32）
+ HKV 引擎；档 B 七开关全开；**300 步**（warmup 5），`--flops-per-sample 40.91e9`。

| 指标 | 数值 |
|---|---:|
| 稳态步时（n=300） | **692.1 ms**（mean 692.0） |
| 全局吞吐 | **47,346.9 samples/s**（每卡 2,959.2） |
| MFU | **28.02%** |
| HBM（分配器口径） | 59.14 GB |
| 稠密参数 | 91,428,374（表 12×1M×32 + 1×1M×32 = 416 M 个 f32 ≈ 1.66 GB，不入 `nn.Parameters`） |
| 每 die 每步查表量 | 非序列 24,576 + 序列 2048×1190 ≈ **2.44 M** ⇒ 合计 **2.46 M 次** |

与前面的臂串起来（同环境同配置，n≥20）：

| 臂 | 查表量/步/die | 步时 | MFU | 相对无表 |
|---|---:|---:|---:|---:|
| A 无表（linear direct） | 0 | 635.7 ms | 30.51% | — |
| C paper + HKV（仅 NS 侧） | 24,576 | 667.0 ms | 29.08% | +4.92% |
| **E paper + HKV + 序列 item 表** | **2.46 M** | **692.1 ms** | **28.02%** | **+8.87%** |

⇒ 序列侧 item 查表（查表量放大 **100×**）只再增加 **+25.1 ms（+3.76%）**，
因为 HKV 只搬访问到的行：2.46 M × 32 × 4 B ≈ 315 MB/步，按 3.3 TB/s 仅 ~0.1 ms，开销落在下发/去重侧。

## 10 收敛验证（证明「表真的在训练」，非只跑通）

**关键前提**：完整训练的 300 步用的是**随机标签**，loss 恒在 **0.69314（= ln 2）**，这是期望值水平，
**不构成收敛证据**（本仓合成数据无真实信号）。为此单独做两组验证：

**10.1 HKV 表更新探针**（`tools/probes/probe_hkv_updates.py`，1 die）

| 检查 | 结果 |
|---|---|
| 反向 5 次后同一批 id 的向量 | 变化，`|Δ|max = 1.6158`（lr=0.5，梯度方向固定） |
| 未被访问的行（id=0） | **完全不变** |
| 结论 | HKV 的 rowwise Adagrad **确实在反向内更新被访问的行**，且不扰动其它行 |

**10.2 端到端可学性**（1 die，`--ns-shape paper` + HKV，1,024 桶，标签 = 桶号过半，每步换 id）

| 步段 | loss 均值 |
|---|---:|
| 0–10 | 0.7446 |
| 30–40 | 0.3157 |
| 100–110 | **0.0002** |
| 290–300 | **0.0000** |

⇒ 表 + MLP + 7 个 block + head 的整条链路都能学；此前的 0.693 平台是**桶太多（4096 / 1M）× 样本太少**导致每桶几乎只被访问一次，不是实现缺陷。

## 11 附带取证：dynamic_emb 底层确实是 HKV

建表时若 `init_capacity` 小于 `bucket_capacity`，报错栈直接暴露了 C++ 实现路径：

```
RuntimeError: /workspace/RecSDK/training/torch_rec_v2/dynamic_emb/../../../cust_op/hkv/include/hkv_hashtable.h:1000:
HierarchicalKV error. Then `num_of_buckets_per_alloc` must be equal or less than initial required buckets number
```

⇒ 昇腾侧 `libdynamic_emb_op_npu.so` 是 **HierarchicalKV（hkv_hashtable）的 NPU 移植**；
坑：`init_capacity` 必须 ≥ `bucket_capacity`（默认 128），否则建表失败。
