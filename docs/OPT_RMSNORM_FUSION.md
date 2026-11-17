# RMSNorm 融合优化说明（设计稿）

> 目标：把 `residual add + RMSNorm` 两遍 HBM 往返压成一遍，吃掉实测出来的 45% 带宽缺口。
> 预期 **−20~25 ms/步（3.5~4%）**；数学等价（不碰模型结构），但**属于 L2（算子边界）**，必须过精度门禁。

## 1. 现状：它有多慢

来自 3 稳态步 profile（16 die，档 B，batch 2048/卡/步，当前最优配置 G1+G2+G3+O1）：

| 项 | ms/步 | 次数/步 | 平均/次 |
|---|---:|---:|---:|
| `RmsNorm`（前向） | 15.6 | 28（7 block × 2 microbatch × rms_0/rms_1） | 556 µs |
| `RmsNormGrad`（反向） | 24.5 | 28 | 874 µs |
| **合计** | **40.1** | | **占步时 6.5%** |

单算子带宽实测（`tools/probes/probe_vec_bandwidth.py`，1024×1203×256 bf16，单张量 602 MiB）：

| 算子 | 时间 | 有效带宽 | 说明 |
|---|---:|---:|---|
| SiLU（读1写1） | 0.397 ms | **3178 GB/s** | 该机器的实际 HBM 上限 |
| Add 残差（读2写1） | 0.587 ms | **3222 GB/s** | 同上 |
| **`npu_rms_norm` 融合** | **0.873 ms** | **1445 GB/s** | **只有峰值的 45%** |

⇒ 不是"RMSNorm 天生慢"，而是**厂商 kernel 在 HBM 上多走了一趟**：算 `mean(x²)` 需要读一遍，
归一化输出还要再读一遍 x（若中间量落 fp32 则更差）。1.2 GB÷0.873 ms=1445 GB/s 对应"读两遍"的量级。

## 2. 为什么这两步可以融合

模型里两者**在图上是紧邻的**，只是分属两个 block：

```
block N:   ... → attn/ffn 输出 sub_out
           h = residual + sub_out          ← 一次 Add（读2写1）
block N+1: y = rmsnorm(h) * gamma          ← 一次 RMSNorm（读1写1，内部再读一遍）
           ... 用 y 算 attention/FFN
```

`h` 除了喂给下一层的 norm 之外**没有别的消费者**（残差流本身要继续往下传，所以要留一份）。
所以完全可以合成一个 kernel：**一遍读进 `residual` 和 `sub_out`，在 UB 里算出 `h`，用 UB 副本算平方和，
再归一化写出 `y`，同时把 `h` 写回残差流**。

## 3. 收益怎么算

单次（1 个 block × 1 个 microbatch，602 MiB 张量）：

| 方案 | HBM 流量 | 理论时间（3.2 TB/s） | 实测 |
|---|---:|---:|---:|
| 现状：Add + RMSNorm 分开 | 1.8 + 1.8(+0.6) = **3.6~4.2 GB** | 1.1~1.3 ms | **1.46 ms**（0.587 + 0.873） |
| 融合（一遍 UB 驻留） | 读 1.2 + 写 1.2 = **2.4 GB** | **0.75 ms** | 待测（目标 ≤0.9 ms） |

反向同理：现在 `RmsNormGrad`（读 h、dy、gamma，写 dx）后面还跟着 Add 的梯度分发（两次写）；
融合后一次读 `residual`、`sub_out`、`dy`，一次写 `d_residual`（两个分支的梯度在核内相加）。

**全步折算**：前向 28 次 × ~0.6 ms ≈ 10~13 ms；反向 28 次 × ~0.4 ms ≈ 8~10 ms ⇒ **合计 −20~25 ms（3.5~4%）**。

## 4. 不能用现成的 `npu_add_rms_norm`

E14 已经实测过：这个融合算子**没有 autograd kernel**，三个输入的梯度全是 `None`
（`grad = None` 静默发生，训练会"正常跑完"但权重不更新）⇒ **训练侧绝对不能用**。
只能自己写前向 + 自己写反向（自定义 `torch.autograd.Function`）。

## 5. 为什么 Triton 版本打平、这次要靠 SIMT/AscendC

上一轮 `probe_triton_add_rmsnorm.py` 的结果是 fwd 1.450 → 1.527 ms（+5.3%）、bwd 2.915 → 2.888（−0.9%），
即 Triton 生成的融合 kernel **没有吃到带宽差** —— 它同样把中间量落到了 HBM。
要吃到 1445 → 3178 GB/s 这段差，必须做到：**一个 tile 的 `h` 常驻 UB、平方和在 UB 内完成、只落一趟**。

950 支持 SIMT 多线程 + `__ubuf__` 直接寻址（见 `ascendc-simt-best-practices`）：
- 逐元素 + 行内归约，形状规整（行 = d = 256），适合 SIMT 线程网格；
- 每线程负责若干行，行内 `sum(x²)` 用 warp 协作归约，`h` 留在 UB 里第二次使用；
- 一次 `DataCopy` 进、一次出。

## 6. 实施步骤（每步有判据）

| 步 | 动作 | 判据（不过就停） |
|---|---|---|
| **S1** | 单算子右对照：AscendC SIMT kernel（前向）vs `add + npu_rms_norm`，同形状 1024×1203×256 | 融合 kernel **≤0.9 ms**（现状 1.46 ms），精度 max\|Δ\| ≤ 1 bf16 ULP |
| **S2** | 反向 kernel（dx、dgamma、d_residual 一次出）vs `RmsNormGrad + add_bwd` | **≤0.6 ms**（现状 ~0.87 + Add 分摊），`dgamma` 累加顺序变化须记 1-ULP 差异 |
| **S3** | 包成 `torch.autograd.Function`，接进 `RMSNorm`/`OneTransBlock`（开关 `ONETRANS_FUSED_ADD_NORM=1`，默认关） | 单 die batch 256 对拍：loss rel ≤1e-5、`max\|Δg\|` ≤1 ULP（同 O1/G2 的门禁口径） |
| **S4** | 16 die A/B（同节点同时段顺序跑，n≥8） | 稳态 ≥1% 且大于同臂三次极差；否则回滚 |

## 7. 风险与边界

- **会改数值档**：`h` 的加法顺序与 norm 内部累加顺序变了 ⇒ 1 ULP 级；按 L2 记账，不与 L1 的逐位结论混谈。
- 需要新增构建链（bisheng + 直调 kernel），是本项目第一次引入 AscendC 自写算子 ⇒ 单独目录 `tools/simt/`，不污染 `OneTrans_NPU/`。
- 若 S1 达不到 0.9 ms，说明该机型 UB 容量/带宽不支持单遍，方案作废（不硬凑）。

## 8. 与整体目标的关系

当前 614.3 ms，距「持平」判据 427 ms 还差 **−187 ms**。
RMSNorm 融合能拿 **20~25 ms（约 13% 的缺口）**，是确定性能拿到的一块；
剩下的大头仍在 FA 反向（131 ms @35%）与向量侧其它小算子（SiLU/Add 已打满带宽）。
