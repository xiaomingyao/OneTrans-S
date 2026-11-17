# 模型结构对账：论文 vs 上游 GPU 参考 vs 本仓昇腾（2026-11-17）

对象：
- 论文 `papers/OneTrans_2510.26104v3.txt` §4.1.4 / Table 2
- 上游 GPU 参考 `WestbrookLong/OneTrans_Pytorch`（submodule `GPU_Reference`，commit `8febd7e`）
- 本仓 `OneTrans_NPU/`（main_pytorch.py + models/taac_onetrans.py）

## 0 结论

1. **本仓与上游 GPU 参考：结构逐位等价**（不是"大致相同"，是同一份定义 + 一层可关的 NPU 优化开关）。
   证据：同一配置下 **949 个张量、state_dict 的 key 集合与形状全等、前向输出 md5 完全相同**、参数量同为 **91,026,946**、
   pyramid schedule 同为 `[1203, 960, 736, 480, 256, 12]`。
2. **本仓与论文：主干超参对齐，但有 4 处需要标注的差异/推定**（见 §2 表的「⚠️」行）：
   梯度裁剪缺失、优化器不同、FFN 宽度未知（以参数量反推）、「6 stacked blocks」与我们的 7 个 block 的表述歧义。
3. **论文的 2.64 TFLOPs 无法与本仓对账**，且该数字在论文自身口径下不自洽（见 §3）⇒ 只能当"论文自报"引用。

## 1 与上游的机器对账（`tools/probes/compare_upstream_model.py`）

两边模块同名（`main_pytorch` / `models.taac_onetrans`），故各起一个进程 dump 到 json 后比对：

| 校验项 | 上游 GPU_Reference | 本仓 OneTrans_NPU | 一致 |
|---|---|---|---|
| 参数量 | 91,026,946 | 91,026,946 | ✅ |
| state_dict 张量数 | 949 | 949 | ✅ |
| key 集合 | — | — | ✅ 全等（双方独有均为 0） |
| 张量形状 | — | — | ✅ 0 处不同 |
| pyramid schedule | `[1203,960,736,480,256,12]` | 同 | ✅ |
| 前向输出 md5（同 seed 权重 + 同输入，CPU） | `c4392b1c57956629…` | `c4392b1c57956629…` | ✅ **逐位相同** |
| 输出形状 / absmax | `[2,2]` / 0.29727 | 同 | ✅ |

配置：`d_model=256, heads=4, ffn_hidden=1536, ns_len=12, seq_len=1190, pyramid=6, align=32, sep=1,
non_seq_dim=256, seq_feat_dim=64, num_classes=2, multi_num=1, mask=paper_causal`。

唯一的代码差异是**可关闭的 NPU 优化开关**（`ONETRANS_NS_BMM` / `FUSED_NORM` / `CAT_INPLACE` / `SDPA_CAUSAL` /
`FA_BSH` / `FULL_PROJ` / `EXPLICIT_BF16` / `RMSNORM_FN` / `PHASE_MARKERS`）+ 阶段标记，默认全关时与上游同义。

## 2 与论文逐项对账

| 项 | 论文 | 本仓 | 判定 |
|---|---|---|---|
| 层数 | "6 stacked OneTrans blocks" | **1 base + 6 pyramid = 7 个 block** | ⚠️ 表述歧义（见 §2.1），参数量却与论文 91 M 吻合 |
| 宽度 d_model | 256 | 256 | ✅ |
| 头数 H | 4 | 4 | ✅ |
| 参数目标 | ≈100 M（Table 2 记 **91 M**，**不含** sparse embedding） | **91,026,946**（**纯稠密**，无稀疏表） | ✅ **同口径一致** |
| 序列 token | 1190 → 12，逐层线性收缩，取 32 的倍数，顶层对齐 NS 数 | `linear_pyramid_schedule(1203,12,6,32)` = `[1203,960,736,480,256,12]` | ✅ |
| NS token | 12（= 顶层序列长度） | 12 | ✅ |
| [SEP] token | 有（Table 3 Row 3 专门消融"去掉 [SEP]"） | 有（`use_sep_token=True`，总长 1203 = 1190+12+1） | ✅ |
| 参数共享 | S token 共享一套 QKV/FFN；每个 NS token 独立一套 | `kqv_list[ns_len+1]` + `ffn_list[ns_len+1]`，seq 用第 `ns_len` 组 | ✅ |
| Pre-Norm | 是 | `rms_0 → attn → +res → rms_1 → ffn → +res` | ✅ |
| 因果/金字塔掩码 | 金字塔尾部截断 ⇒ 底右对齐因果 | `paper_causal`：`col ≤ row + (Lk − Lq)` | ✅ |
| per-GPU batch | 2048 | 档 A = 2048（档 B = 1024×2） | ✅（档 A） |
| **梯度裁剪** | **global grad-norm clipping，dense 90 / sparse 120** | **未实现**（`grep clip_grad` 只在 `tests/` 里出现） | ⚠️ **训练配置缺失** |
| **优化器** | 双优化器、无 weight decay：sparse→Adagrad，dense→**RMSProp(lr=0.005)** | 统一 **AdamW(lr=5e-3)** | ⚠️ 不同（不影响步时口径，影响收敛） |
| **FFN 宽度** | **论文未给** | `ffn_hidden=1536`（= 6×d） | ⚠️ **推定值**：靠它把参数量对齐到 91 M（上游默认 256 只有 ~31 M） |
| 精度档 | 基线 fp32；优化后 "mixed precision" | bf16（档 A 含激活重算；O1 显式 bf16 为精度档变更） | ✅ 对应"mixed precision"那一档 |
| DCNv2+DIN 对照 | Table 2 给了 13.4% MFU、1.43 TFLOPs | — | 不需对齐 |

### 2.1 两个需要写进报告的口径说明

1. **层数口径**：论文正文说 "6 stacked OneTrans blocks"，但我们的实现是 **base block + 6 个 pyramid block**（上游亦如此），
   且只有在 `ffn_hidden=1536` 时参数量才刚好落到 **91,026,946 ≈ 论文 91 M**。
   若论文的 6 是"总层数"，则我们的 block 数多 1、且 FFN 宽度需另调才能到 91 M —— 两个自由度互相补偿，
   **无法从论文文字唯一确定**，故标注为**推定**，不声称"完全对齐"。
2. **参数量口径**：论文明确 "Params (model parameters **excluding** sparse embeddings)"；本仓是纯稠密实现（无稀疏表），
   两者**同口径**（都只数稠密参数），91 M 与 91,026,946 相差 0.03% ⇒ 构成可比的等价性证据（不是数值巧合）。

## 3 计算量对账（2.64 TFLOPs）

论文原文（§4.1.2 脚注段）："We report Params (model parameters **excluding** sparse embeddings) and
**TFLOPs (training compute in TFLOPs at batch size 2048)**"。

| 读法 | 论文 | 本仓（主口径） | 倍数 |
|---|---:|---:|---:|
| 每步（batch 2048，单卡） | 2.64 TFLOP | **83.8 TFLOP** | **31.7×** |
| 每样本 | 1.29 GFLOP | 40.91 GFLOP | **31.7×**（同比例） |
| 下界口径（本仓只算 Linear/mm） | — | 64.6 TFLOP/步 | 24.5× |

**但论文这一列自身不自洽**：一次前向就要 `2 × 91 M × 1190 = 216.6 GFLOP/sample`，
batch 2048 摊下来是 **443 TFLOP**，已经是 2.64 TFLOP 的 **168 倍**——还没算反向。
⇒ 2.64 TFLOPs **不可能**是"batch 2048 的训练计算量"，只能是某种归一化/笔误；
它的唯一可用之处是 Table 2 内部的相对比较（2.64T vs RankMixer+Transformer 2.51T）。

**本仓的计算量是自洽的**：91 M 参数里绝大部分是 **13 组 NS/seq 分组的 QKV 与 FFN**
（序列 token 只用其中 1 组），所以每 token 的有效算力远低于"全部参数都作用到每个 token"的朴素估算：
40.91 GFLOP/sample ÷ 1203 token = 34.0 MFLOP/token。

## 4 待办

1. **补 global grad-norm clipping**（dense 90）—— 论文明确要求，本仓缺；只影响收敛行为，不影响步时口径，
   但作为"复刻"应当补上，并在报告 §2 口径里注明"性能档未开裁剪"。
2. 把 §2.1 的两条推定（层数口径、FFN 宽度）写进报告 v5 的"未闭合项"。
3. （可选）把优化器换成论文的 RMSProp 做一次 loss 轨迹对照，验证收敛无系统性偏差。
