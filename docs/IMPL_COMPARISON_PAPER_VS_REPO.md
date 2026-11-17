# 实现层对比：论文规定 vs 上游 GPU 参考 vs 本仓昇腾

> 对照三方：**论文文字**（`arXiv:2510.26104v3`，无开源代码）/ **上游 GPU 参考**（`WestbrookLong/OneTrans_Pytorch@8febd7e`）/ **本仓**（`OneTrans_NPU`，950DT 16 die）
> 代码锚点均指 `OneTrans_NPU/main_pytorch.py`（行号为 `8febd7e` 基线上本仓版本）
> 性能数字：16 die、全局 batch 32,768、n=8 中位，逐项 A/B 单独开

## 0 结论

| # | 结论 |
|---|---|
| 1 | **算法层**（块结构、参数共享、金字塔、掩码、Pre-Norm）三方一致，本仓与上游**逐位等价** |
| 2 | **算子层**是本仓唯一实质改写的地方：把 7 类"逐 token 小算子"换成批量/融合算子 —— 这是 −68.35% 的全部来源 |
| 3 | **训练配置层**本仓落后论文两项：双优化器（用 AdamW 代替）与 grad-norm clip（**未实现**） |
| 4 | **分布式层**论文只说 data-parallel + all-reduce；上游是单卡；本仓用 torchrun + DDP over HCCL |
| 5 | 所有算子替换都能给"精度取证"：6 项 L1 逐位/1 ULP，2 项 L2 数值档变更（相对 fp64 更准或 0.61% 梯度偏差） |

## 1 对比边界（必须先声明）

- **论文没有代码**（`grep github/release` 零命中）⇒ "论文的实现"只能取**文字规定的算法**，无法核对真实码；
- **上游是第三方二手重实现**（TF demo → PyTorch），只提供"算法如何落成 PyTorch"的参考，不代表论文作者写法；
- 因此下表"论文"列写"文字是否规定"，不写"论文怎么实现"。

## 2 逐模块实现对照

| 模块 | 论文（文字） | 上游 GPU 参考 | 本仓昇腾 | 差异性质 |
|---|---|---|---|---|
| Block 结构 | Pre-Norm：RMSNorm→MixAttn→残差→RMSNorm→MixFFN→残差 | 同 | 同（`rms_0/attn/rms_1/ffn`），另提供 `_attn_part_impl`/`_ffn_part_impl` 两个可独立调用的分段入口（L474/L489） | 等价 |
| **QKV 组织** | S 共享一套、每个 NS 各一套 | `kqv_list[13]`，**逐 token 调 Linear**（每步 12×3 次 1-token matmul） | 默认同一写法；`NS_BMM=1` 时把 13 组拼成 **1 次 bmm**（L195/L400） | **实现改写（L1，逐位相同）** |
| Attention 内核 | FlashAttention-2 | `F.scaled_dot_product_attention` | `npu_fusion_attention`（aclnn **FA v4**，L342）；`SDPA_CAUSAL=1` 时少数层用内置 causal（L353） | **算子替换（L1）** |
| 布局 | 未规定 | BHSD 连续 | `FA_BSH=1` 走 **BSH 布局**，省两次 transpose+contiguous（L337） | 实现改写（L1，q 1 ULP） |
| Mask | 底右对齐因果（金字塔尾部截断） | 显式建 mask 张量 | 同（缓存 `_MASK_CACHE`，L234）；注意 **FA 的 `atten_mask` 语义是 True=屏蔽**，与 SDPA 相反 | 语义对齐，极性易错 |
| **FFN** | 每 token 一套、SiLU×2 | `ffn_list[13]`，逐 token | 同，但 `FULL_PROJ=1` 时改为**全量投影 + 原地覆盖 NS 行**（L288/L435） | 实现改写（L1） |
| **Norm** | RMSNorm | 手写 `x * rsqrt(mean(x²)+eps) * w` | `FUSED_NORM=1` 走厂商 `torch_npu.npu_rms_norm`（L158）；反向用 `npu_rms_norm_backward` 复用 rstd（L139） | 算子融合（L2，相对 fp64 更准） |
| **序列拼接** | [SEP] + concat | `torch.cat`（每步 51.4 ms） | `CAT_INPLACE=1` 预分配 + 原地写（L312） | 实现改写（L1，逐位相同） |
| **精度路径** | BF16/FP16 + 激活重算 | `autocast(fp16)` | 默认 autocast bf16；`EXPLICIT_BF16=1` 关 autocast + 权重影子缓存（L26/L64），loss 前 logits 回 fp32 | **精度档变更（L2）** |
| 反向 | 未规定 | autograd | autograd（FA/FFN/QKV 全走厂商 backward），无自写 kernel | 等价 |
| **分布式** | data-parallel，16 GPU all-reduce | **单卡**（无 init_process_group） | `torchrun` + DDP over HCCL，`broadcast_buffers=False`、`find_unused_parameters=True` | 实现补齐 |
| **优化器** | 双优化器、无 wd：sparse Adagrad(0.1,1.0) / dense RMSProp(0.005,0.99999,0) | `AdamW(lr=1e-3, wd=1e-4)` | `AdamW(lr=5e-3, wd=0)` | ⚠️ 与论文不同 |
| **梯度裁剪** | global grad-norm，dense 90 / sparse 120 | 无 | **未实现** | ❌ 缺口 |
| 数据管道 | 工业全量特征 | HF `datasets` + `DataLoader`，每步取数 | 合成 token **常驻 device**（`--source synth`），步时不含取数与 H2D | 口径选择 |
| Profiling | 未提 | 无 | `--profile-steps` 设备侧 profile + 阶段标记 | 工具补齐 |

## 3 算子改写的代价与收益（16 die 实测）

| 改写 | 开关 | 单步 Δ | 精度取证 | 归类 |
|---|---|---:|---|---|
| 13 组逐 token QKV/FFN → bmm | `NS_BMM` | **−25.70%** | 输出逐位相同 | L1 |
| 手写 RMSNorm 链 → `npu_rms_norm` | `FUSED_NORM` | **−14.85%** | 对 fp64：3.891e-3（手写 7.704e-3）⇒ 更准 | L2（更优） |
| `cat` → 预分配 + 原地写 | `CAT_INPLACE` | **−9.48%** | 逐位相同 | L1 |
| 显式 mask → 内置 causal（q==k 的层） | `SDPA_CAUSAL` | **−5.06%** | logits 逐位 0.0 | L1 |
| BHSD → BSH 布局 | `FA_BSH` | **−2.81%**（单算子 −31.7%） | q 1 ULP（9.766e-4） | L1 |
| 逐 token 投影 → 全量投影 + 原地覆盖 | `FULL_PROJ` | **−11.13%** | dk/dv 逐位 | L1 |
| autocast → 显式 bf16 干路 | `EXPLICIT_BF16` | **−15.79%** | `mean rel(Δg)=0.61%`、无 NaN | L2（档变更） |
| **累计** | — | **−68.35%**（1,942.1 → 614.7 ms） | — | — |

## 4 kernel 落点证据（每步计数，batch 1024×2、7 block）

| 算子 | 每步次数 | 说明 |
|---|---:|---|
| `FlashAttentionScoreV4` / `…GradV4` | 14 / 14 | = 7 block × 2 micro-batch ⇒ 布局与调用对齐 |
| `RmsNorm` / `RmsNormGrad` | 28 / 28 | = 7 × 2 × 2（rms_0+rms_1） |
| `Silu_Swish` / `SiluBackward` | 54 / 54 | FFN 两处 SiLU × 13 组 × 2 mb 摊平后的落点 |
| `ReduceSum` | 200 | = `F.linear` 的 bias 梯度 `dy.sum(0)` |
| `InplaceZero_ZerosLike` | 580 | 24 µs/次，下发受限 |
| `InplaceCopy_TensorMove` / `…Transpose` | 236 / 241 | 布局拷贝，插在 GEMM 之间 |

判据：**每类 kernel 的次数必须能与「block 数 × micro-batch × 组数」对上**，否则说明改写改错了层（本仓 14/28/54 三项都精确对上）。

## 5 数值证据（L1 与 L2 分级，不可混谈）

| 项 | 判据 | 结果 |
|---|---|---|
| logits（SDPA causal vs 显式 mask） | 逐位 | **0.0** |
| QKV bmm vs 逐 token Linear | 逐位 | **0.0** |
| FA BSH vs BHSD | dq / dk / dv | dk、dv **逐位**；dq **1 ULP（9.766e-4）** |
| 全量投影 + 原地覆盖 | dk/dv | **逐位** |
| 融合 RMSNorm（前向） | 对 fp64 相对误差 | **3.891e-3** vs 手写 7.704e-3 |
| 显式 bf16 干路 | loss rel / 梯度 | loss **9.8e-5**；`max\|Δg\| = 5.33e-4`；`mean rel(Δg, \|g\|≥1e-2) = 6.1e-3` |
| 全模型三关 | 张量 / loss / 梯度 | 1.49e-7 / 5.96e-8 / 4.88e-4 ⇒ 全 PASS |

## 6 未闭合与实现自由度

| 项 | 现状 | 能否闭合 |
|---|---|---|
| 梯度裁剪 | 未实现 | 能（一行 `clip_grad_norm_`），论文明确要求 |
| 双优化器 | AdamW 单优化器 | 能（本仓无稀疏表，dense 换 RMSProp 即可），但会改收敛轨迹 |
| tokenizer 结构 | 0.81 M vs 论文变体差异 13 M | **不能**：论文未披露 tokenizer 内部投影方式 |
| 位置编码 | 本仓无可训练位置编码参数 | **不能**：论文只画了 "Positional Information" 方框 |
| 层数口径 | 7 次 block-pass（1 base + 6 pyramid） | 推断（论文文字 6，但其 TFLOPs 按 7 次才对得上 2.64 T） |
| 数据管道 | 合成 token 常驻，不含取数 | 刻意：保证"纯计算口径"，与论文不可比时间部分已隔离 |

## 7 一句话总结

**算法照论文、算子照昇腾、训练配置半个论文**：
结构与参数可在代码级逐项对到论文，算子层做了 7 处替换（全部有逐位/ULP 取证），
但双优化器与 grad-norm clip 两项论文明确规定的训练配置尚未补齐 —— 补齐不影响步时口径，只影响"复刻完整性"。
