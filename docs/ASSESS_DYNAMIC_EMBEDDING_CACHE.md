# 方案评估：Device 侧动态 Embedding 缓存 + 流水线并行（缓解"推荐系统显存墙"）

- 日期：2026-11-17
- 结论：**该方案针对的瓶颈在本仓（OneTrans-S 稠密复刻）不存在**，不建议在当前仓库实施；
  它对应的是论文原始工业模型的**稀疏 ID 特征**那条线，属于另一个技术域（RecSDK/HKV/TBE）

## 1 提案的前提 vs 本仓实测

| 提案的判断 | 本仓实测证据 | 判定 |
|---|---|---|
| "超大尺寸的稀疏特征 Embedding 表" | `grep -rn "nn.Embedding\|EmbeddingBag\|sparse=True\|Hetu\|HKV\|TBE\|embedding_table"` 在 `OneTrans_NPU/` **零命中**；模型 **91,026,946 参数全稠密、共 81 个参数张量**，最大单张量 `kqv_w (13,3,256,256)` = 2.56 M ≈ **5 MB (bf16)**。整模型 182 MB (fp32) = HBM 的 **0.2%** | ❌ 不成立 |
| "Host↔Device PCIe 频繁同步" | 官方 `step_trace_time.csv` 有 **Preparing 列：0.42 / 0.72 / 0.42 ms**（3 个稳态步），占步时 **0.07~0.12%**；84 种 kernel 里 **gather/embed/lookup/memcpy/H2D 命中数 = 0**；训练数据是常驻 device 的合成 token，**步时不含取数/H2D**（口径已在 README §2.5 声明） | ❌ 不成立 |
| "撑爆 NPU 的 HBM" | 实测峰值：档 A **39.17 GB**、档 B **58.48 GB**，卡上限 **91.75 GB**。唯一一次 OOM 是「每卡 2048 免重算」（E5，FFN 激活需再 1.43 GiB），**是稠密激活不是表** | ❌ 方向错 |
| "AICore 等数据产生大量气泡" | `Computing 605.5 ms = 98.6%`；**Free（气泡）4.13~5.03 ms = 0.7%**；通信暴露 4.4~4.9 ms = 0.8%；`Bubble` 列恒为 0 | ❌ 相反：**纯计算受限** |

数字来源：`results/prof16_o1_step_trace_time.csv`、`results/prof16_o1_kernel_details.csv`（16 die、G1+G2+G3+O1、档 B、3 稳态步）。

## 2 逐条子措施在本仓的可行性

| 子措施 | 本仓可行性 |
|---|---|
| **异构缓存**：热点表常驻 HBM/L2、中低频放 Host、冷数据异步预取 | **无对象**。没有表可缓存；950DT 的 L2 是 MB 级，做 embedding cache 需要 HKV/RecSDK 那套专用栈，与本仓（纯 PyTorch 稠密 transformer）不在同一技术域 |
| **Embedding lookup 与上一步反向流水重叠** | 即使有 lookup 也不划算：**实测 cube∥vector 双流重叠效率只有 15%**（`probe_stream_overlap.py`：串行 2.192 ms → 并发 2.102 ms，理想下界 1.604 ms），重叠掉的时间只占理论值的 15% |
| 伪代码 `device_embedding_table.prefetch_async(ids)` | **本栈无此 API**。torch_npu 提供的是 `torch_npu.npu_prefetch`（权重/激活预取），用在 dense 权重上收益为 0 —— 权重每步都在 HBM 里，且已被「bf16 影子权重」消掉了 cast |
| `torch_npu.npu.stream(prefetch_stream/pcompute_stream)` 写法 | API 形态可用（`torch.npu.Stream` + `Event`），但见上：**没有可重叠的异构流量**，且重叠效率本身只有 15% |

## 3 本仓真正的"显存/数据"问题是什么

1. **显存墙确实存在，但内容是激活不是表**：档 B 58.48/91.75 GB；档 A（2048/卡 + 激活重算）39.17 GB；
   每卡 4096 或 2048 免重算都撞墙（E5 OOM、`FlashAttentionScoreGradV4` 失败）。已用「重算 + 显式 bf16 + 全量投影」压下来。
2. **再省显存的手段**（都是 dense 激活侧）：分块重算（chunked checkpointing）、序列分块、
   activation offload 到 host —— **最后这一条才会真正用上 PCIe**，代价是每步几十 GB 的 H2D/D2H，会明显伤性能。
3. **步时不含取数**是当前口径的前提（合成 token 常驻 device）。真接入数据时，需要单独量
   dataloader → pinned → H2D 的流水，那属于工程集成，不是本方案。

## 4 如果目标确实是论文的"完整 OneTrans"（含稀疏 ID 特征）

那是另一条线，需要单独评估：
- 真正的对象是 RecSDK/HKV/TBE 的**分片稀疏表**（GB~TB 级）、all-to-all、缓存命中率；
- 关键指标会变成 PCIe/RDMA 带宽、HKV 命中率、embedding 与 dense 的反向重叠窗口；
- 环境里有 `tf950-test`（TF1.15 + mx_rec）镜像可做对照实验，但它与当前的稠密 OneTrans-S 复刻**不是同一个模型**。

## 5 判定

- 对当前仓库：**不实施**。给不存在的问题做优化，会把工程复杂度加上去而收益为 0。
- 若要推进，先在纸面上定三个数：**表有多大 / 热点命中率多少 / PCIe(或 RDMA) 可用带宽多少**，
  三者任一不成立，这套架构就没有存在的必要。
- 与本仓已确认的瓶颈（cube 利用率 50.9%、vector 侧 240 ms 已贴峰值、GEMM 模型内只有 277 TF/s 而隔离能到 400+）相比，
  本方案的收益上限是 **0**。
