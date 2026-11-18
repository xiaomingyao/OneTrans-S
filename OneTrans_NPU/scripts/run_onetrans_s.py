#!/usr/bin/env python3
# coding=utf-8
"""OneTrans-S 训练复核入口（昇腾 950DT，只做训练，不含推理）。

目的：把论文 OneTrans-S 的**形状**（arXiv:2510.26104v3 §4.1.4 / Table 2）
拉到板上跑训练步，量出「稳态步时 / 吞吐 / 训练显存 / MFU」，供与论文 H100 数字并列。

形状默认值 = 论文 S：
  6 个堆叠 OneTrans block + 1 个全长期 base block、d=256、H=4、seq 1190 → 12（每层对齐 32）
  ns=12、[SEP]=1 ⇒ 首层输入 1203 token、per-GPU batch 2048

⚠️ 数据口径：论文工业数据不可得。本脚本用**常驻 device 的合成 token**，因此步时**不含取数与 H2D**
   （纯计算口径）；不要拿它跟含取数的端到端步时相除。

用法（单 die 冒烟）：
  python3 scripts/run_onetrans_s.py --device npu --steps 30 --warmup 5 --batch-size 64
用法（8 die）：
  torchrun --nproc_per_node=8 scripts/run_onetrans_s.py --device npu --steps 30 --batch-size 2048
"""
from __future__ import annotations

import argparse
import contextlib

from torch.autograd.profiler import record_function
import json
import os
import statistics
import sys
import time
from pathlib import Path

# bool mask 必须在导入 main_pytorch 之前设定（该模块在 import 时读 env）：
# 950 实测 float 加性 mask 会让 SDPA 退回 Cast+Muls+BatchMatMul+Softmax 手工路径，
# bool mask 才落 FlashAttentionScoreV4（见 docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md）。
os.environ.setdefault("ONETRANS_ATTN_MASK", "bool")
# [O1] 显式 bf16 需要权重影子（main_pytorch 在 import 时读这些 env，必须早于下面的 import）
if os.environ.get("ONETRANS_EXPLICIT_BF16", "0") == "1":
    os.environ.setdefault("ONETRANS_WEIGHT_SHADOW", "1")

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_onetrans import TAACOneTransClassifier, linear_pyramid_schedule
import main_pytorch
import npu_compat

# 论文 OneTrans-S 默认形状（§4.1.4 + Table 2）
PAPER_S = dict(d_model=256, num_heads=4, num_pyramid_layers=6, seq_len=1190, ns_len=12, ffn_hidden=1536)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OneTrans-S 训练复核（合成 token，纯计算口径）")
    # 形状
    p.add_argument("--d-model", type=int, default=PAPER_S["d_model"])
    p.add_argument("--num-heads", type=int, default=PAPER_S["num_heads"])
    p.add_argument("--ffn-hidden", type=int, default=PAPER_S["ffn_hidden"])
    p.add_argument("--num-pyramid-layers", type=int, default=PAPER_S["num_pyramid_layers"])
    p.add_argument("--multi-num", type=int, default=1, help="每个 pyramid 段内堆几个 block（论文=1）")
    p.add_argument("--seq-len", type=int, default=PAPER_S["seq_len"])
    p.add_argument("--ns-len", type=int, default=PAPER_S["ns_len"])
    p.add_argument("--pyramid-align", type=int, default=32)
    p.add_argument("--mask-type", default="paper_causal")
    p.add_argument("--no-sep-token", dest="use_sep_token", action="store_false")
    p.set_defaults(use_sep_token=True)
    # 数据（合成）
    p.add_argument("--seq-feat-dim", type=int, default=64)
    p.add_argument("--non-seq-dim", type=int, default=256)
    p.add_argument("--ns-backend", choices=("linear", "ec", "dynamic_emb"), default="linear",
                   help="非序列特征 tokenizer 后端：linear(默认,连续特征) / ec(torchrec EC) / dynamic_emb(RecSDK v2)")
    p.add_argument("--ns-rows", type=int, default=0,
                   help="ec / dynamic_emb 后端的表行数（表参数 = ns-len × ns-rows × d-model）")
    p.add_argument("--ns-shape", choices=("direct", "paper"), default="direct",
                   help="direct=连续特征直接投影(本仓默认)；paper=论文形态(数值分桶/类别 id → 查表 → concat → 单次 MLP → split)")
    p.add_argument("--ns-features", type=int, default=0,
                   help="paper 形态的特征域个数 F（默认=ns-len）")
    p.add_argument("--ns-edim", type=int, default=32,
                   help="paper 形态每域 embedding 维度（concat 后 F×edim 进 MLP）")
    p.add_argument("--seq-id-rows", type=int, default=0,
                   help="序列侧 item 表行数；>0 时序列走论文形态（item ID 查表 + side info → 共享 MLP）")
    p.add_argument("--seq-edim", type=int, default=32, help="序列侧 item embedding 维度")
    p.add_argument("--source", choices=("synth", "taobao"), default="synth",
                   help="synth=合成 token（只测性能）；taobao=真实 TaobaoAd_x1 数据（需 --data-dir）")
    p.add_argument("--data-dir", default="", help="TaobaoAd_x1 预处理目录（tools/prep_taobao_ad.py 产出）")
    p.add_argument("--fresh-ids", action="store_true",
                   help="每个 step 重新采样 id（完整训练口径；关闭时所有步复用同一批，只测性能）")
    p.add_argument("--task", choices=("random", "parity", "threshold"), default="random",
                   help="标签生成：random=随机(无可学信号，只测性能)；parity=id 奇偶(可学，用于完整训练收敛验证)")
    p.add_argument("--num-classes", type=int, default=2)
    # 训练
    p.add_argument("--device", default="npu" if hasattr(torch, "npu") else "cpu")
    p.add_argument("--batch-size", type=int, default=64, help="**每卡每个 micro-batch** 的 batch")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="梯度累积步数：全局 batch(每次优化器更新) = batch_size × 卡数 × grad_accum")
    p.add_argument("--steps", type=int, default=30, help="计时步数（不含 warmup）")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-3, help="论文 dense 用 RMSProp(5e-3)；这里统一 AdamW")
    p.add_argument("--amp", dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--input-dtype", choices=("fp32", "bf16"), default="fp32",
                   help="常驻 device 的输入 token 用哪种 dtype 存（bf16 省一半输入显存，仅用于压显存的对照实验）")
    p.add_argument("--checkpoint-act", action="store_true", help="激活重算（省显存、换时间）")
    p.add_argument("--seed", type=int, default=42)
    # 口径
    p.add_argument("--flops-per-sample", type=float, default=0.0,
                   help="每样本训练 FLOPs（来自 tools/probes/flops_paper_shape.py，0 表示不算 MFU）")
    p.add_argument("--peak-tflops", type=float, default=432.0, help="单 die Cube bf16 峰值（950DT 真机 32 核×1650MHz）")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--profile-dir", default=None, help="给了就在 rank0 上抓 profile（只在计时段，跳过 warmup）")
    p.add_argument("--profile-steps", type=int, default=2, help="抓几个稳态步")
    p.add_argument("--profile-with-stack", action="store_true", help="记录调用栈（可按代码位置归因算子）")
    p.add_argument("--ddp-strict", action="store_true", help="find_unused_parameters=False（bmm 模式下全部参数都参与）")
    p.add_argument("--no-backward", action="store_true", help="只跑前向（诊断 cast 归属用，不做训练）")
    return p.parse_args()


def build_model(args) -> nn.Module:
    return TAACOneTransClassifier(
        non_seq_dim=args.non_seq_dim,
        seq_feature_dim=args.seq_feat_dim,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        ns_len=args.ns_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        multi_num=args.multi_num,
        mask_type=args.mask_type,
        num_pyramid_layers=args.num_pyramid_layers,
        pyramid_align=args.pyramid_align,
        use_sep_token=args.use_sep_token,
        use_checkpoint=args.checkpoint_act,
    )


def _make_labels(task: str, non_seq: torch.Tensor, rows: int) -> torch.Tensor:
    """合成可学标签：parity=单桶奇偶；threshold=桶号过半（单调、易学，用于验证收敛）"""
    if task == "parity":
        return (non_seq[:, 0] % 2).to(torch.long)
    if task == "threshold":
        return (non_seq[:, 0] >= rows // 2).to(torch.long)
    raise ValueError(task)


def main() -> None:
    args = parse_args()
    os.environ["ONETRANS_NS_BACKEND"] = args.ns_backend
    if args.ns_rows:
        os.environ["ONETRANS_NS_ROWS"] = str(args.ns_rows)
    torch.manual_seed(args.seed)
    device_type = npu_compat.init_device(args.device)
    rank, world_size, local_rank = npu_compat.init_distributed()
    if world_size > 1:
        args.device = f"{device_type}:{local_rank}"
    # 表后端（ec / dynamic_emb）必须在**本 rank 的 die** 上建表，否则 gather 跨设备报错
    os.environ["ONETRANS_NS_DEVICE"] = str(args.device)
    os.environ["ONETRANS_NS_SHAPE"] = args.ns_shape
    if args.ns_features:
        os.environ["ONETRANS_NS_FEATURES"] = str(args.ns_features)
    os.environ["ONETRANS_NS_EDIM"] = str(args.ns_edim)
    os.environ["ONETRANS_SEQ_ID"] = "1" if args.seq_id_rows > 0 else "0"
    if args.seq_id_rows > 0:
        os.environ["ONETRANS_SEQ_ROWS"] = str(args.seq_id_rows)
        os.environ["ONETRANS_SEQ_EDIM"] = str(args.seq_edim)
    is_main = rank == 0
    log = print if is_main else (lambda *a, **k: None)

    # [O1] 显式 bf16：关掉 autocast，激活/输入 /权重全 bf16，只有 norm 内部与 loss 保持 fp32。
    # 目的：消灭 autocast 在「bf16 matmul ↔ fp32 逐元素」之间反复插入的激活 cast（16 die 上约 58 ms/步）。
    explicit_bf16 = os.environ.get("ONETRANS_EXPLICIT_BF16", "0") == "1"
    if explicit_bf16:
        args.amp = False                      # 不再让 autocast 插手
        args.input_dtype = "bf16"
        os.environ.setdefault("ONETRANS_WEIGHT_SHADOW", "1")   # 权重用每步一次的 bf16 影子，图仍接在 fp32 参数上
    phase_mark = os.environ.get("ONETRANS_PHASE_MARKERS", "0") == "1"
    use_amp = npu_compat.should_enable_amp(device_type, args.amp, args.amp_dtype)
    # 输入按训练精度落地（bf16 时省一半输入显存：4096 batch 下 1190×64 token 张量 1.25 GiB → 0.62 GiB）
    in_dtype = (torch.bfloat16 if ((args.input_dtype == "bf16" and use_amp and args.amp_dtype == "bf16") or explicit_bf16)
                else torch.float32)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = npu_compat.build_scaler(device_type, use_amp, amp_dtype)

    # ---- 真实数据源（TaobaoAd_x1）：须在 build_model 前把词表/形态写进 env ----
    src = None
    if args.source == "taobao":
        if not args.data_dir:
            raise SystemExit("--source taobao 需要 --data-dir")
        from taobao_ad_data import TaobaoAdSource
        src = TaobaoAdSource(args.data_dir, seq_len=args.seq_len, rank=rank, world_size=world_size,
                            global_batch=args.batch_size * args.grad_accum * world_size,
                            device=args.device)
        os.environ["ONETRANS_NS_SHAPE"] = "paper"
        os.environ["ONETRANS_NS_CARD"] = ",".join(str(c) for c in src.ns_cardinalities)
        os.environ["ONETRANS_NS_FEATURES"] = str(len(src.ns_cardinalities))
        os.environ["ONETRANS_NS_ROWS"] = str(max(src.ns_cardinalities))
        os.environ["ONETRANS_SEQ_FIELDS"] = ",".join(str(c) for c in src.seq_cardinalities)
        os.environ["ONETRANS_SEQ_ID"] = "0"
        args.ns_shape = "paper"
        if rank == 0:
            print(f"[data] taobao rows={len(src):,} steps/epoch={src.steps_per_epoch}"
                  f" ns_card={src.ns_cardinalities} seq_card={src.seq_cardinalities}"
                  f" ctr={src.meta['ctr']:.4f}", flush=True)

    model = build_model(args).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    total_tokens = args.ns_len + args.seq_len + (1 if args.use_sep_token else 0)
    schedule = linear_pyramid_schedule(total_tokens, args.ns_len, args.num_pyramid_layers, args.pyramid_align)

    log(f"[mode] explicit_bf16={int(explicit_bf16)} weight_shadow={os.environ.get('ONETRANS_WEIGHT_SHADOW','0')}")
    log(f"[env] {npu_compat.describe(device_type)} dist=world_size={world_size} rank={rank} local_rank={local_rank}")
    log(f"[shape] d={args.d_model} heads={args.num_heads} ffn={args.ffn_hidden} blocks={1 + args.num_pyramid_layers}"
        f" (multi={args.multi_num}) seq={args.seq_len} ns={args.ns_len} sep={args.use_sep_token} total={total_tokens}")
    log(f"[shape] pyramid_schedule={schedule}")
    log(f"[shape] params={n_params:,} ({n_params/1e6:.2f} M)")
    log(f"[run] amp={use_amp} input_dtype={in_dtype} dtype={args.amp_dtype} mask={args.mask_type} attn_mask_env={os.environ.get('ONETRANS_ATTN_MASK')} weight_shadow={main_pytorch._WEIGHT_SHADOW}"
        f" per_card_micro_batch={args.batch_size} grad_accum={args.grad_accum}"
        f" per_card_batch={args.batch_size * args.grad_accum} global_batch={args.batch_size * world_size * args.grad_accum}"
        f" ckpt_act={args.checkpoint_act}")

    # 合成 token 常驻 device：步时不含量化搬运（纯计算口径）
    # 一次生成 grad_accum 个 micro-batch 的数据，累积时按片取用
    n_local = args.batch_size * args.grad_accum
    gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    if src is not None:
        # 真实数据：每步在循环里从 mmap 取（这里只占位，避免多余分配）
        non_seq = seq = labels = seq_ids = None
    elif args.ns_shape == "paper":
        # 论文形态：连续原始值 → bucketize 成 id（每域 ns_rows 个桶）；类别特征等价于直接给 id
        from paper_input import Bucketizer
        _f = args.ns_features or args.ns_len
        _raw = torch.randn(n_local, _f, generator=gen).to(args.device)
        non_seq = Bucketizer(_f, args.ns_rows, device=args.device)(_raw).to(torch.int64)
    elif args.ns_backend == "linear":
        non_seq = torch.randn(n_local, args.non_seq_dim, generator=gen).to(args.device, dtype=in_dtype)
    else:
        if args.ns_rows <= 0:
            raise SystemExit(f"--ns-backend {args.ns_backend} 需要 --ns-rows > 0（表行数）")
        # 表后端吃 id：每个 NS 位置一个 id
        non_seq = torch.randint(0, args.ns_rows, (n_local, args.ns_len),
                                generator=gen).to(args.device)
    seq = torch.randn(n_local, args.seq_len, args.seq_feat_dim, generator=gen).to(args.device, dtype=in_dtype)
    labels = torch.randint(0, args.num_classes, (n_local,), generator=gen).to(args.device)
    seq_ids = None
    if args.seq_id_rows > 0:
        seq_ids = torch.randint(0, args.seq_id_rows, (n_local, args.seq_len),
                                generator=gen).to(args.device)
    if args.task == "parity":
        if non_seq.dtype not in (torch.int64, torch.int32):
            raise SystemExit("--task parity 需要 id 形态输入（--ns-shape paper 或 --ns-backend ec/dynamic_emb）")
        labels = _make_labels(args.task, non_seq, args.ns_rows)

    # [D2a] 梯度桶直视图：省掉 DDP 把 bucket 再拷回 param.grad 的那一步（profile 里 InplaceAdd/copy 的一部分）
    _bucket_view = os.environ.get("ONETRANS_DDP_BUCKET_VIEW", "0") == "1"
    # [D5-lite] NPU JIT 图编译模式：让后端把相邻逐元素算子融合、摊薄 launch
    if os.environ.get("ONETRANS_JIT_COMPILE", "0") == "1" and device_type == "npu":
        torch.npu.set_compile_mode(jit_compile=True)
        log("[run] jit_compile=True（NPU 图编译融合）")
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if device_type == "npu" else None,
        find_unused_parameters=not args.ddp_strict, broadcast_buffers=False,
        gradient_as_bucket_view=_bucket_view,
    ) if world_size > 1 else model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    step_ms: list[float] = []
    loss_item = 0.0

    prof = None
    if args.profile_dir and rank == 0:
        import torch_npu  # noqa: PLC0415  仅 NPU 侧需要
        prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
            schedule=torch_npu.profiler.schedule(wait=args.warmup, warmup=0, active=args.profile_steps, repeat=1),
            with_stack=bool(args.profile_with_stack),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.profile_dir)),
        )
        prof.start()
        log(f"[prof] rank0 profile_dir={args.profile_dir} active_steps={args.profile_steps}")

    for i in range(args.warmup + args.steps):
        if src is not None:
            _b = src.batch(i)
            non_seq, seq, seq_ids, labels = _b["ns"], _b["seq_x"], _b["seq_ids"], _b["label"]
        if args.fresh_ids:
            # 完整训练口径：每步换一批 id（在 device 上采样，避免 H2D 干扰）
            non_seq = torch.randint(0, args.ns_rows, non_seq.shape, device=args.device, dtype=torch.int64)
            if seq_ids is not None:
                seq_ids = torch.randint(0, args.seq_id_rows, seq_ids.shape,
                                        device=args.device, dtype=torch.int64)
            if args.task == "parity":
                labels = _make_labels(args.task, non_seq, args.ns_rows)
        if device_type == "npu":
            torch.npu.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for k in range(args.grad_accum):
            lo, hi = k * args.batch_size, (k + 1) * args.batch_size
            _nogr = torch.no_grad() if args.no_backward else contextlib.nullcontext()
            with _nogr, npu_compat.autocast_context(device_type=device_type, amp_dtype=amp_dtype, use_amp=use_amp):
                with record_function("P_fwd") if phase_mark else contextlib.nullcontext():
                    logits = model(non_seq[lo:hi], seq[lo:hi],
                                   seq_ids[lo:hi] if seq_ids is not None else None)
                    if explicit_bf16:
                        logits = logits.float()      # 只有这一步回到 fp32，一共 1 次/step
                with record_function("P_loss") if phase_mark else contextlib.nullcontext():
                    loss = crit(logits, labels[lo:hi]) / args.grad_accum
                    loss_item = loss.item() * args.grad_accum
            if not args.no_backward:
                with record_function("P_bwd") if phase_mark else contextlib.nullcontext():
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
        if not args.no_backward:
            with record_function("P_opt") if phase_mark else contextlib.nullcontext():
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
        main_pytorch.clear_weight_shadow_cache()   # 权重已更新 / 前向图已用完 ⇒ 下一步重新 cast
        if device_type == "npu":
            torch.npu.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= args.warmup:
            step_ms.append(dt)
        log(f"[step {i:03d}] {'warm' if i < args.warmup else 'meas'} ms={dt:8.1f} loss={loss_item:.5f}")
        if prof is not None:
            prof.step()

    if prof is not None:
        prof.stop()
        log("[prof] stop")

    med = statistics.median(step_ms)
    gbs = args.batch_size * world_size * args.grad_accum
    gflops = gbs * args.flops_per_sample / 1e9 if args.flops_per_sample > 0 else 0.0
    mfu = (gflops / 1e3) / (med / 1e3) / (world_size * args.peak_tflops) * 100 if gflops > 0 else float("nan")
    hbm_gb = (torch.npu.max_memory_allocated() / 1024**3) if device_type == "npu" else float("nan")
    log(f"[perf] steady_median_step_ms={med:.1f} mean={statistics.mean(step_ms):.1f} n={len(step_ms)}"
        f" global_samples_per_s={gbs / (med / 1e3):.1f} per_card_samples_per_s={args.batch_size * args.grad_accum / (med / 1e3):.1f}")
    log(f"[perf] hbm_peak_alloc_gb={hbm_gb:.2f} flops_per_sample={args.flops_per_sample/1e9:.3f} GFLOP"
        f" gflops_per_step(global)={gflops:.1f} mfu_vs_{args.peak_tflops:.0f}TF={mfu:.2f}%")

    if is_main and args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "shape": {"d_model": args.d_model, "heads": args.num_heads, "ffn": args.ffn_hidden,
                      "blocks": 1 + args.num_pyramid_layers, "seq": args.seq_len, "ns": args.ns_len,
                      "schedule": schedule, "params": n_params},
            "run": {"world_size": world_size, "per_card_micro_batch": args.batch_size,
                    "grad_accum": args.grad_accum, "per_card_batch": args.batch_size * args.grad_accum, "global_batch": gbs,
                    "amp": use_amp, "dtype": args.amp_dtype, "mask": args.mask_type},
            "perf": {"steady_median_step_ms": med, "mean_step_ms": statistics.mean(step_ms), "n": len(step_ms),
                     "global_samples_per_s": gbs / (med / 1e3), "hbm_peak_alloc_gb": hbm_gb,
                     "flops_per_sample": args.flops_per_sample, "mfu_pct": mfu, "peak_tflops": args.peak_tflops},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[out] {args.out_json}")

    npu_compat.shutdown()
    if world_size > 1:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
