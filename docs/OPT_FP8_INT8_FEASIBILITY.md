# FP8 / INT8 训练可行性评估（950DT · torch_npu 2.10）

- 日期：2026-11-17
- 动机：cube 侧 348.4 ms 占步时 **54.8%**，若量化把 cube 峰值翻倍，是唯一能大幅逼近 427 ms 判据的单点手段
- 结论提前：**FP8 在本栈上用不了**（三条入口全封死）；**INT8 GEMM 真有 2×**（802 TF/s），但精度代价 5–8× bf16；
  且**即使全用 int8，GEMM 只占步时 28.7%，单靠它仍到不了 427 ms**（见 §5 预算表）

## 1 环境与入口

`torch 2.10.0+cpu` / `torch_npu 2.10.0` / `Ascend950DT_9582` / CANN 9.1.0。

fp8 dtype 基础能力（`probe_fp8_capability.py`）：

| dtype | 结果 |
|---|---|
| `torch.float8_e4m3fn` | ✅ 可创建/运算，`element_size=1` |
| `torch.float8_e5m2` | ✅ 同上 |
| `float8_e4m3fnuz` / `float8_e5m2fnuz` | ❌ `RuntimeError: ... has not been supported` |

## 2 FP8 的三个入口全部不可用（都有实测证据）

| 入口 | 结果 | 判定 |
|---|---|---|
| `torch.mm(fp8, fp8)` | `MmKernelNpuOpApi.cpp:43 ... error code is 161002` | ❌ 算子不支持 |
| `torch._scaled_mm(fp8, fp8, scale_a=…, scale_b=…)` | 能跑但 **2166 ms（0.3 TF/s）**，比 bf16 慢 **1350×** | ❌ 走了极慢的回退路径（逐元素模拟） |
| `torch_npu.npu_quant_matmul(fp8, fp8, scale, x1_dtype=float8_e4m3fn, …)` | `QuantMatmulKernelNpuOpApi.cpp:289 ... 161002` | ❌ 该 dtype 组合不支持 |
| `torch_npu.npu_quant_fusion_attention(fp8…)` | 输入准备阶段就失败 | ❌ 注意力无 fp8 |

`torch_npu` 里确实有一整套量化栈（`npu_quant_matmul` / `npu_add_quant_matmul` / `npu_dynamic_quant` /
`npu_anti_quant` / `npu_quant_fusion_attention(+backward)` / `npu_rms_norm_quant` / `npu_swiglu_quant` 等 40+ 个），
schema 里也有 `x1_dtype/x2_dtype/scale_dtype`，但**可用的数值类型是 int8，不是 fp8/HIF8**。

两个坑记一下：① `_scaled_mm` 的 `scale_a/scale_b` 必须传 **Tensor**（传 float 直接 `TypeError`）；
② 它的 mat2 语义是 `(K,N)`（不是 `(N,K)`）。

## 3 INT8 GEMM：**实测 802 TF/s = bf16 的 1.99×**

形状 `(1231872, 256) @ (256, 1024)`，单 die：

| 精度 | 时间 | 吞吐 |
|---|---:|---:|
| bf16 `torch.mm` | 1.604 ms | **403 TF/s** |
| int8 `npu_quant_matmul` | **0.805 ms** | **802 TF/s（1.99×）** |

顺带测出另一个重要事实（§5 要用）：**隔离状态下 bf16 GEMM 本身就能跑 376–427 TF/s（432 峰值分母的 87–99%）**，
而且**正反两个方向都行**（weight-grad 方向 `M=256, K=1.23M` 也有 425.7 TF/s）。
即模型里 GEMM 只有 277 TF/s（64.3%）**不是形状造成的**，是串行执行/交织造成的。

## 4 精度代价：int8 比 bf16 差 5–8 倍

`probe_int8_quant_precision.py`，bf16 → int8 → 反量化，指标 `mean(|Δ|/max(|x|,1e-3))`：

| 量化粒度 | 高斯 N(0,1) | 带 2 个 10σ 离群 |
|---|---:|---:|
| per-tensor（全张量一个 scale） | **4.63%** | **8.36%** |
| per-token（每行一个 scale） | 3.10% | 3.11% |
| per-32-block（mx 风格） | 2.46% | 2.47% |

端到端 matmul（4096×256 激活 × 256×1024 权重，权重 per-tensor）：

| 激活量化 | mean rel |
|---|---:|
| per-tensor | **6.01%** |
| per-token | **5.19%** |

对照本仓已记账的精度档：O1 显式 bf16 干路是 `mean rel|Δg| 0.61%`（已被定性为"训练精度档变更"）。
**int8 激活把这个数字再放大 5–10 倍** ⇒ 这不是"优化"，是训练算法的替换（需要 smoothquant / 误差反馈 / 分块 scale 等 recipe，
且要重新验证收敛）。离群值会把 per-tensor 从 4.6% 推到 8.4%，这一项在真实激活上通常更糟。

## 5 cube ∥ vector 不能重叠（双流实测 15% 效率）—— 决定了预算表长什么样

`probe_stream_overlap.py`：cube 核（mm 1.604 ms）与 vector 核（add 0.586 ms，已在 3228 GB/s 峰值）双流并发：

| 执行方式 | 时间 |
|---|---:|
| 同流串行 | 2.192 ms |
| **双流并发** | **2.102 ms** |
| 理想下界 `max(1.604, 0.586)` | 1.604 ms |

并发只重叠掉 0.090 ms（理论 0.587 ms），**效率 15%**。这与 16 die profile 的账一致：kernel 时长之和 636.1 ms
而设备忙的并集 605.5 ms ⇒ 真实重叠只有 4.8%。
**⇒ 步时 ≈ cube + vector + 通信，不能指望"用 vector 填 cube 的空"。**

### 达标预算表（每卡 64.57 TFLOP 记账口径，vector 下界 240 ms 已贴峰值、通信 4.7、气泡 4.1）

| 方案 | cube | 步时 | MFU | vs 427 ms |
|---|---:|---:|---:|---:|
| 现状（cube 50.9%） | 348.4 | 597（实测 614） | 25.0% | 差 170 |
| **bf16 全跑满峰值（100%）** | 177.5 | **426** | 35.1% | **恰好达标** |
| int8 只换 GEMM（FA 保持现状） | 233.3 | 482 | 31.0% | 差 55 |
| int8 GEMM + FA 反向提到 45% | 204.2 | 453 | 33.0% | 差 26 |
| 全 int8（FA 不支持，仅作上界） | 95.6 | 344 | 43.4% | 达标 |

**读法**：
1. **bf16 下 427 ms 就是物理下界**——要求 cube 从 50.9% 提到 **100%**，且 vector 侧一点不动。
2. int8 只解决 GEMM（48.9 TFLOP），FA 的 27.8 TFLOP 仍走 bf16 且只到 35%，所以 **≥453 ms**，仍差 26 ms。
3. 全 int8 才能到 344 ms，但 **FA 没有 int8/fp8 入口**，这条路不存在。

## 6 判定

- **FP8：不可用**（不是"效果一般"，是没有可用算子）。
- **INT8：速度真有 2×，但只对 GEMM，且精度 5–8× 恶化**。作为"性能档实验"可以做（不追求可训练），作为论文口径的训练复刻不可接受。
- **真正的瓶颈判断被这条实验改写了**：不是"精度不够低"，而是 **cube 利用率（50.9%）+ vector 不可压（240 ms）**。
  在 bf16 下把 GEMM 从 277 TF/s 提到接近隔离测出的 400 TF/s（+45%），GEMM 176.1 → ~122 ms，步时约 **560 ms**；
  再叠加 FA 反向 35%→45%（−29 ms）约 **531 ms**。**这才是当前最值得做的一档。**

### 下一步的两个可执行问题（按性价比排序）

1. **为什么模型内 GEMM 只有 277 TF/s，而同样形状隔离跑得到 400+？**（形状已排除）
   候选：小 M 层（Lq=480/256/12）占比、`Addmm`(带 bias) 走不同 kernel、每次 GEMM 前后的 LayoutCopy/ViewCopy 打断、
   NS 路径的 210 次 18 µs BatchMatMul。**这是能兑现 50 ms 级的唯一入口**，且不需要改数值。
2. FA 反向 35% → 45%：API 旋钮已全灭，需厂商或 AscendC，暂不投入。
