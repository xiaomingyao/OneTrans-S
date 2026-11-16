# 实验方案：16 卡 OneTrans-S 性能优化（Round 1）

- 立项依据：`docs/PROFILE_16DIE_BOTTLENECK.md`（16 die rank0 profile，3 个稳态步）
- 基线：**16 die（122+111）· 每卡 2048 · 全局 32,768 · bf16 + bool mask(FAv4) + 激活重算 + `ONETRANS_NS_BMM=1`**
  → `steady_median_step_ms = 1443.2`（n=8 中位 1443.4，两轮复现差 <0.05%）、`22,702.5 sps`、`MFU 10.36%`、`HBM 52.27 GB`
- 基线 profile：3,083 kernel/步、compute 并集 99.8%、通信净暴露 0.33%、cube 内效率 30.9%、非 cube 占 66.9%

## 0 总原则（不可协商）

1. **单变量**：每个实验只改一处，其余档位固定（2048 / ckpt / ns_bmm=1 / 16 die）；不同时叠加两个改动。
2. **同节点同时段对照**：A/B 两臂在同一批卡上顺序跑，中间无他人作业；跨轮比较只用于复现性，不算收益。
3. **收益门槛**：16 die 稳态步时改善 **≥1%** 且大于同臂三次重复的极差，才记「有效」；否则记「无效」并保留数据。
4. **精度门禁**（先过才计时）：
   - 逐位等价类改动（ns-layout 型）→ `max|Δout| ≤ 1e-6`、`worst rel|Δg| ≤ 1e-5`；
   - 舍入语义变化类（融合算子、dtype 边界）→ 2 步 loss 相对差 `≤ 1e-5`，且梯度 `worst rel|Δg| ≤ 1e-5`；
   - 超差即判失败，回滚，不改容差迁就。
5. **可回滚**：改码前 `cp x.py x.py.bak`；每个实验独立 commit；失败项在 `results/` 留原始日志与失败签名。
6. **口径纪律**：profile 窗口 ≠ 稳态（可互证不可互算）；不与论文 407 ms 相除；收益一律用同口径 A/B。

## 1 实验总表

| 编号 | 假设（证据） | 改动 | 开关 | 预估收益 | 风险 | 精度类别 |
|---|---|---|---:|---:|---|---|
| **E1a** | RMSNorm 输出被 `weight(fp32)` 提升成 fp32，`x.float()` 再补一刀 ⇒ 每次 norm 3 次 cast，随后 linear 又 cast 回 bf16（Cast 677 次/步、185.4 ms 的直接来源） | `RMSNorm.forward` 的 `weight` 以输入 dtype 缓存（master 仍 fp32）；norm 内部 fp32 计算不变 | `ONETRANS_RMS_DTYPE=1` | 30~60 ms（2~4%） | 低 | 舍入语义（tol） |
| **E1b** | 手写 norm 链 = `Square 27.9 + Pow 14.4 + Rsqrt 0.3 + ReduceSum/Mul 份额`（≈42.6 ms/步，n=28） | 前向 `torch_npu.npu_add_rms_norm`（残差+norm 融合）/`npu_rms_norm`；反向 `npu_rms_norm_backward` | `ONETRANS_FUSED_NORM=1` | 40~80 ms（3~5%） | 中（反向接口需对拍） | 舍入语义（tol） |
| **E4** | profile 已证 bmm 模式下本步无未用参数，DDP 却仍开 `find_unused_parameters=True` | `DistributedDataParallel(..., find_unused_parameters=False)` | `--ddp-strict` | 3~10 ms（host 侧，需实测） | 低 | 逐位等价 |
| **E3** | pyramid 每层「切子序列 → 算 → 拼回」产生 Slice 84.2 + ViewCopy 32.9 + Cat 52.1 + Transpose 77.0（≈246 ms/步，17%） | 切片改 view/原地写，避免 materialize；进 FA 前布局按算子要求一次成型 | `ONETRANS_PYR_VIEW=1` | 上限 100~200 ms（7~14%） | **高**（非连续输入支持、隐性同步） | 逐位等价 |
| **E5** | 重算把前向算两遍（FA 前向 14 次/步 = 7×2）；1 die b1024 实测 +15.5% | 只重算 FFN 不重算 attention（或反之），找出「显存刚好够」的最小重算集 | `--ckpt-scope {all,ffn,attn}` | 视显存，未计 | 中（OOM） | 逐位等价 |
| **E2** | 权重每个 matmul 前被自动 cast（Cast 大头）；zeus「E2E bf16 + fp32 master」实测 **负收益**（633→893 ms） | 权重以 bf16 参与前向、master fp32 只给优化器 | 需**用户拍板** | 未知（历史证据为负） | 高 | 舍入语义 |

> 排序理由：E1a/E1b 是「不动结构、纯本地改写」，风险最低且直击 Cast/norm 两块 12.7% + 3%；E4 零风险顺手；
> E3 收益最大但要先解决非连续输入；E5 依赖显存；E2 只有历史反例，需先做小规模判定再谈。

## 2 逐实验设计

### E1a · RMSNorm 输出 dtype 归位（先做）

- **现状**（`OneTrans_NPU/main_pytorch.py:42`）：`x_fp32 = x.float()` → `rms` → `(x_fp32*rms).to(input_dtype) * self.weight`，
  而 `self.weight` 默认 **fp32** ⇒ 乘积按类型提升变 **fp32**，下游 `F.linear` 在 autocast 下再 cast 回 bf16。
  每次 norm = 1 次 `float()` + 1 次 `.to(bf16)` + 1 次 `bf16×fp32→fp32` + 下游 1 次 cast；28 次/步。
  **已就地验证**（容器 CPU 探针）：`bf16 张量 × fp32 张量 → torch.float32`；即 `norm 链输出 bf16 × weight(fp32) → fp32`。
- **改动**：`weight` 保留 fp32 主副本（优化器不变），前向用 `self.weight.to(x.dtype)` 的缓存副本（key=dtype，惰性建、`_version` 变化时失效）。
- **判据**：`ut` 用 b1 形状对拍 fwd/bwd，`worst rel|Δg| ≤ 1e-5`；然后 16 die A/B，步时改善 ≥1%。
- **失败即回滚**：`main_pytorch.py.bak`。

### E1b · 融合 norm 算子

- **改动**：`RMSNorm` 走 `torch_npu.npu_rms_norm(x, gamma_bf16, eps)`（返回 `(y, rstd)`，rstd 供反向）；
  残差分支若仍需要 `x + residual` 再 norm，则用 `npu_add_rms_norm(x, residual, gamma, eps)` 一步到位（省一次 Add，Add 现在 130.8 ms/步）。
- **风险点**：自定义 autograd（`npu_rms_norm_backward` 需要 rstd/dy/gamma），要先写 CPU/单卡 UT 对齐现有实现；
  现有实现是 fp32 归一化后再乘 fp32 weight，融合算子是 bf16 权重 ⇒ 属「舍入语义变化」，按 §0-4 的 tol 判据。
- **判据**：b128 形状 2 步 loss `rel ≤ 1e-5`；16 die 步时 ≥1%。

### E4 · DDP 关未用参数探测

- **判据**：跑 2 步不报 `Expected to have finished reduction`；步时改善（含 host 侧）≥0.5% 才留。

### E3 · pyramid 切片/拼接去 materialize

- **先探针后改码**：先写 `tools/probes/probe_pyr_view.py`，验证 ① `Slice` 后的 sub-tensor 直接喂 `F.linear`/FA 是否被接受（非连续）；
  ② 拼回能否用 `out[:, base:base+L] = ...` 原地写。两项都成立再改主路径。
- **判据**：逐位等价（layout 只影响内存排布，不影响数值）⇒ `max|Δout| ≤ 1e-6`，**不做 tol 放宽**；16 die ≥3% 才留。

### E5 · 最小重算集

- **扫描**：`ckpt_scope ∈ {all, ffn, attn}` × batch ∈ {2048}，记录是否 OOM + 步时；OOM 一次即跳过该组合，不试第二次。
- **判据**：不 OOM 且在 ≥1% 改善的组合里取最小显存方案。

### E6 · 复核 profile（每个实验通过后必做）

重抓 rank0 3 步，用 `tools/probes/analyze_profile.py` 出表，核对：Cast 次数、kernel 总数、cube 内效率三项是否按预期移动；
**kernel 数不降而步时改善 ⇒ 先怀疑测量口径（同节点/同档位/arm 顺序）而不是结论成立。**

## 3 执行顺序与门禁

```
E1a ──(UT tol 过 + 16die ≥1%)──> E1b ──(同上)──> E4 ──(≥0.5%)──> E3 探针 ──(两项支持)──> E3 主体 ──> E5 扫描
                     │
                     └─ 任一项不到门槛：保留数据、标「无效」、进 results/ 不进主表
```

- 每步产出：`results/exp_<编号>_{arm}.json` + 一行结论（步时/吞吐/MFU/Δ%）+ commit。
- 累积规则：只有**通过门禁**的改动才允许作为下一实验的新基线；否则基线不动。

## 4 采集协议（所有实验共用）

```bash
# 臂 A：当前基线       臂 B：加开关
ONETRANS_NS_BMM=1 <switches_A> torchrun --nnodes=2 --node_rank=N --nproc_per_node=8 \
  --master_addr=192.168.0.82 --master_port=<不同端口> scripts/run_onetrans_s.py \
  --device npu --batch-size 2048 --steps 8 --warmup 3 --checkpoint-act --flops-per-sample 31529400000
```
- 端口每臂换一个（295xx），避免 HCCL 16666/socket 残留；两端 `md5sum` 一致后才跑。
- 记录：`steady_median_step_ms`（n=8）、`global_samples_per_s`、`hbm_peak_alloc_gb`、`mfu_pct`。
- 三次重复的极差作为抖动基线；改善必须大于抖动。
- 卡占用前置检查：`npu-smi info -t proc-mem -i 0..7` 全为 "No process" 才开跑（122 曾有 DLRM 作业占用，勿杀他人作业）。

## 5 时间预算

| 实验 | 单臂耗时 | 轮次 | 合计 |
|---|---:|---:|---:|
| E1a | ~2 min（UT + 16die 2 臂） | 1 | ~5 min |
| E1b | ~10 min（UT 对拍 + 2 臂） | 1 | ~15 min |
| E4 | ~3 min | 1 | ~5 min |
| E3 | 探针 10 min + 主体 20 min | 2 | ~40 min |
| E5 | 3 组合 × 3 min | 1 | ~15 min |
| E6 | 每项 ~4 min | 4 | ~20 min |

## 6 待用户拍板

1. **E2（权重 bf16 前向 + fp32 master）是否允许试验**：zeus 侧历史为负收益（633→893 ms），但那是 910B 全栈；
   在本仓它等价于「Cast 大规模消失」，值得用 1 次 16 die 实验判定，**不默认执行**。
2. **E5 若最小重算集仍 OOM**：是否接受把 per-card batch 降到 1024 × accum 2（全局仍 32,768，但与会话内已验证的
   「8 die×2048×accum2 = 3870.8 ms」同口径）——**这属于改变主对比轴的定义，需你确认**。
