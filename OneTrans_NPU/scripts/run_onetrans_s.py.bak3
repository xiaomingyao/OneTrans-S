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

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_onetrans import TAACOneTransClassifier, linear_pyramid_schedule
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device_type = npu_compat.init_device(args.device)
    rank, world_size, local_rank = npu_compat.init_distributed()
    if world_size > 1:
        args.device = f"{device_type}:{local_rank}"
    is_main = rank == 0
    log = print if is_main else (lambda *a, **k: None)

    use_amp = npu_compat.should_enable_amp(device_type, args.amp, args.amp_dtype)
    # 输入按训练精度落地（bf16 时省一半输入显存：4096 batch 下 1190×64 token 张量 1.25 GiB → 0.62 GiB）
    in_dtype = torch.bfloat16 if (args.input_dtype == "bf16" and use_amp and args.amp_dtype == "bf16") else torch.float32
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = npu_compat.build_scaler(device_type, use_amp, amp_dtype)

    model = build_model(args).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    total_tokens = args.ns_len + args.seq_len + (1 if args.use_sep_token else 0)
    schedule = linear_pyramid_schedule(total_tokens, args.ns_len, args.num_pyramid_layers, args.pyramid_align)

    log(f"[env] {npu_compat.describe(device_type)} dist=world_size={world_size} rank={rank} local_rank={local_rank}")
    log(f"[shape] d={args.d_model} heads={args.num_heads} ffn={args.ffn_hidden} blocks={1 + args.num_pyramid_layers}"
        f" (multi={args.multi_num}) seq={args.seq_len} ns={args.ns_len} sep={args.use_sep_token} total={total_tokens}")
    log(f"[shape] pyramid_schedule={schedule}")
    log(f"[shape] params={n_params:,} ({n_params/1e6:.2f} M)")
    log(f"[run] amp={use_amp} input_dtype={in_dtype} dtype={args.amp_dtype} mask={args.mask_type} attn_mask_env={os.environ.get('ONETRANS_ATTN_MASK')}"
        f" per_card_micro_batch={args.batch_size} grad_accum={args.grad_accum}"
        f" per_card_batch={args.batch_size * args.grad_accum} global_batch={args.batch_size * world_size * args.grad_accum}"
        f" ckpt_act={args.checkpoint_act}")

    # 合成 token 常驻 device：步时不含量化搬运（纯计算口径）
    # 一次生成 grad_accum 个 micro-batch 的数据，累积时按片取用
    n_local = args.batch_size * args.grad_accum
    gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    non_seq = torch.randn(n_local, args.non_seq_dim, generator=gen).to(args.device, dtype=in_dtype)
    seq = torch.randn(n_local, args.seq_len, args.seq_feat_dim, generator=gen).to(args.device, dtype=in_dtype)
    labels = torch.randint(0, args.num_classes, (n_local,), generator=gen).to(args.device)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if device_type == "npu" else None,
        find_unused_parameters=not args.ddp_strict, broadcast_buffers=False,
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
        if device_type == "npu":
            torch.npu.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for k in range(args.grad_accum):
            lo, hi = k * args.batch_size, (k + 1) * args.batch_size
            with npu_compat.autocast_context(device_type=device_type, amp_dtype=amp_dtype, use_amp=use_amp):
                logits = model(non_seq[lo:hi], seq[lo:hi])
                loss = crit(logits, labels[lo:hi]) / args.grad_accum
                loss_item = loss.item() * args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
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
