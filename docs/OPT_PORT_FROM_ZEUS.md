# 从 zeus-onetrans 移植优化措施：逐项判定与实测

> 输入：zeus-onetrans 的 opt_1109 调优包（8 个补丁）+ 混合精度/cast 相关两轮实验的结论。
> 判定标准用本仓既定准入四条（`onetrans-training` 技能 §7）：① 数学等价；② loss/梯度逐点一致；
> ③ 不引入新依赖/硬编码；④ 不改 dtype / 不改 mask 语义 / 不关边界检查。
> 结论先行：**8 项里只有「ns-layout（NS-token 专有参数合并成 bmm）」能干净移植，实测单步 −24.7%、吞吐 +32.8%**；
> FA 融合与混合精度本仓已具备且更干净；其余按准入准则不合。

## 1. 逐项判定表

| # | zeus 优化项 | 在 zeus 的实测 | 本仓判定 | 理由/本仓证据 |
|---|---|---|---|---|
| 1 | **ns-layout**：`einsum("bnd,ndk->bnk")` → `bmm`，NS-token 专有 QKV/FFN 不再逐 token 发 Linear | 单卡合体 −27.1% 的主要贡献之一 | ✅ **已移植（`ONETRANS_NS_BMM=1`）** | 数学逐元素等价（每个 token 仍用自己的权重，只是合成一次 bmm）；UT 见 §2；实测 §3 |
| 2 | **FA 融合**：SDPA → `npu_fusion_attention_v2(head_num=1, sparse_mode=3)` | 单卡 −55 ms | ⚠️ **不需要移植（本仓已更强）** | 本仓用 **bool mask** 就落到 `aclnnFlashAttentionScoreV4`：前向/反向各 7 次/步、无 `BatchMatMul+Softmax` 回退链（`results/fa_kernel_probe_950.txt`）。zeus 那条路丢 `attn_mask_2d`、`head_num` 写死、序列 >2048 语义不等价 ⇒ 按准入 ③④ 不合 |
| 3 | **混合精度 bf16 autocast** | 生产主档 | ⚠️ **已具备** | 本仓所有数（1/8/16 die）都是 bf16 autocast 档；`should_enable_amp` 在 NPU 只放行 bf16 |
| 4 | **P18：E2E bf16 + fp32 master（关 autocast）** | 633→**893 ms（负收益）**，Cast 降但 matmul 退回 fp32 | ❌ 不移植 | zeus 实测已证伪：关 autocast 让白名单算子的输入变 fp32，拖垮 matmul；本仓 `--input-dtype` 只做输入存储，不动 autocast |
| 5 | **cast-align / input-residual-bf16**（补丁 3/4） | −27 ms / −65 ms | ❌ 不合 | 违反准入 ④：norm 由 fp32 计算变 bf16、整条残差流降精度 |
| 6 | **ns-bias bf16**（补丁 2） | −27 ms | ❌ 不合 | 违反准入 ④：参数 dtype 变 bf16，ckpt 不兼容 |
| 7 | **TBE / fbgemm_ec / DMP**（补丁 7/8） | 单卡 −90 ms | ❌ 不适用 | 本仓**没有稀疏嵌入表**（零 `nn.Embedding`/`torchrec`/`hkv`），那条路服务稀疏表的 EC 分片 |
| 8 | **heterogeneous off**（补丁 6） | −14.6% | ❌ 永不合 | 语义变更（参数 −201.8M） |
| 9 | 取数侧优化（绑核 / 预取 / persistent_workers / pin_memory） | 生产有 | ⏸ 暂不适用 | 本仓主入口是**纯计算口径**（token 常驻 device，无 dataloader）；接真实数据后再谈 |

## 2. 已落地实现（ns-layout）

- 开关：`ONETRANS_NS_BMM=1`（默认 0 = 逐 token Linear 原路径，逐字不变）。
- 落点：`OneTrans_NPU/main_pytorch.py`
  - `CausalMaskAttention`：`kqv_list`（`(ns+1)×3` 个 `nn.Linear`）→ 合并参数 `kqv_w (ns+1, 3, d, d)` / `kqv_b`；NS 段 `(B, ns, d) →(ns,B,d)→ bmm(Wᵀ) → permute`。
  - `OneTransBlock`：`ffn_list`（`(ns+1)` 个 FFNLayer）→ `ffn_w1/w2/b1/b2`（Linear 布局 `(out,in)`）；NS 段两次 bmm。
  - 初始化与 `nn.Linear` 同分布（`kaiming_uniform_(a=√5)` + `U(±1/√fan_in)`）。
- CPU UT：`tools/probes/ut_ns_bmm_equiv.py`（不需要卡）
  - 参数量 **471,938 == 471,938**；
  - 前向 `max|Δout| = 5.96e-08`；
  - 一步反向后 `worst rel|Δg| = 1.34e-06`（判据 ≤1e-5，与仓库 DDP 梯度等价同标准）。
  - ⇒ **EQUIVALENT_PASS**（bmm 与 Linear 的 fp32 累加顺序不同，1e-6 级差异属数值噪声）。

## 3. 实测（950DT，S 形状，bf16 + bool mask + 激活重算，`--batch-size 2048`）

### 3.1 单 die（12 步中位，warmup 4）

| 臂 | 稳态步时 | 每卡 sps | MFU（÷432 TF） | HBM | 参数量 |
|---|---:|---:|---:|---:|---:|
| A：默认（逐 token Linear） | 1,908.9 ms | 1,072.9 | 7.83% | 51.96 GB | 91,026,946 |
| B：`ONETRANS_NS_BMM=1` | **1,437.8 ms** | **1,424.4** | **10.40%** | 51.93 GB | 91,026,946 |
| Δ | **−24.7%** | **+32.8%** | +2.57 pp | 持平 | 不变 |

### 3.2 16 die（122+111，全局 batch 32,768，一次前向）

| 臂 | 稳态步时 | 全局 sps | 每卡 sps | MFU |
|---|---:|---:|---:|---:|
| 16 die 基线（上一轮） | 1,942.0 ms | 16,873.0 | 1,054.6 | 7.70% |
| **16 die + ns_bmm** | **1,442.9 ms** | **22,710.2** | 1,419.4 | **10.36%** |
| Δ | **−25.7%** | **+34.6%** | +34.6% | +2.66 pp |

⇒ 与论文 H100（407 ms、每卡 5,032 sps 推算）的差距从 **4.77×** 收窄到 **3.55×**（仍不同精度档，只并列）。

### 3.3 顺带发现（下一轮靶子）

- DDP 在 bmm 模式下报警「`find_unused_parameters=True` 但本步无未用参数」⇒ 原实现最深 pyramid 块那 16 个未用参数在合并 bmm 后变成全用上了。**下一步可把 `find_unused_parameters=False`**（省一次 autograd 图遍历），未做、未计收益。
- profile 里 `Cast` 仍占 17%（`aclnnInplaceCopy_CastAiCore`），按 §1 的第 4/5 项判定，本仓不做 dtype 侧改动去吃它。

## 4. 复现命令

```bash
# 等价性 UT（CPU，无需卡）
python3 tools/probes/ut_ns_bmm_equiv.py

# 单 die A/B
python3 scripts/run_onetrans_s.py --device npu --batch-size 2048 --steps 12 --warmup 4 \
  --checkpoint-act --flops-per-sample 31529400000                    # A
ONETRANS_NS_BMM=1 python3 scripts/run_onetrans_s.py --device npu --batch-size 2048 --steps 12 --warmup 4 \
  --checkpoint-act --flops-per-sample 31529400000                    # B

# 16 die（两端各一条，注意 ONETRANS_NS_BMM 要在 docker exec 的 bash -lc 里给）
ONETRANS_NS_BMM=1 torchrun --nnodes=2 --node_rank=N --nproc_per_node=8 \
  --master_addr=192.168.0.82 --master_port=29531 scripts/run_onetrans_s.py \
  --device npu --batch-size 2048 --steps 4 --warmup 2 --checkpoint-act --flops-per-sample 31529400000
```
