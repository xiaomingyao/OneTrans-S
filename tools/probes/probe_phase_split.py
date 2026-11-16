#!/usr/bin/env python3
# coding=utf-8
"""阶段拆分探针：把一个训练步拆成 fwd / bwd(DDP allreduce 在里) / opt.step / zero_grad 四段计时。

用途：kernel 级 profile 显示「66.9% 时间不在矩阵乘上」，但没有栈归因（651 个 Cast 类 ops/步 无用户栈），
      本探针用「分段计时」把时间落到阶段上，避免靠猜。

用法（与主入口同参数，只在分布式下有意义）：
  ONETRANS_NS_BMM=1 torchrun --nproc_per_node=8 scripts/../tools/probes/probe_phase_split.py \
      --device npu --batch-size 2048 --steps 5 --warmup 2 --checkpoint-act
"""
from __future__ import annotations

import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NPU_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "OneTrans_NPU"))
sys.path.insert(0, NPU_DIR)
sys.path.insert(0, os.path.join(NPU_DIR, "scripts"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import npu_compat  # noqa: E402
import run_onetrans_s as R  # noqa: E402


def sync(device_type: str) -> None:
    if device_type == "npu":
        torch.npu.synchronize()


def timed(fn, device_type: str) -> float:
    sync(device_type)
    t0 = time.perf_counter()
    fn()
    sync(device_type)
    return (time.perf_counter() - t0) * 1000


def main() -> None:
    args = R.parse_args()
    torch.manual_seed(args.seed)
    device_type = npu_compat.init_device(args.device)
    rank, world_size, local_rank = npu_compat.init_distributed()
    if world_size > 1:
        args.device = f"{device_type}:{local_rank}"
    is_main = rank == 0

    use_amp = npu_compat.should_enable_amp(device_type, args.amp, args.amp_dtype)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = npu_compat.build_scaler(device_type, use_amp, amp_dtype)

    model = R.build_model(args).to(args.device)
    n_local = args.batch_size * args.grad_accum
    gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    non_seq = torch.randn(n_local, args.non_seq_dim, generator=gen).to(args.device)
    seq = torch.randn(n_local, args.seq_len, args.seq_feat_dim, generator=gen).to(args.device)
    labels = torch.randint(0, args.num_classes, (n_local,), generator=gen).to(args.device)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if device_type == "npu" else None,
        find_unused_parameters=True, broadcast_buffers=False,
    ) if world_size > 1 else model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    acc = {k: [] for k in ("fwd", "bwd", "opt", "zero", "total")}

    for i in range(args.warmup + args.steps):
        sync(device_type)
        t0 = time.perf_counter()

        t_zero = timed(lambda: opt.zero_grad(set_to_none=True), device_type)

        holder = {}

        def _fwd():
            with npu_compat.autocast_context(device_type=device_type, amp_dtype=amp_dtype, use_amp=use_amp):
                holder["logits"] = model(non_seq, seq)
                holder["loss"] = crit(holder["logits"], labels)

        t_fwd = timed(_fwd, device_type)

        def _bwd():
            if scaler.is_enabled():
                scaler.scale(holder["loss"]).backward()
            else:
                holder["loss"].backward()

        t_bwd = timed(_bwd, device_type)

        def _opt():
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        t_opt = timed(_opt, device_type)
        sync(device_type)
        t_total = (time.perf_counter() - t0) * 1000

        if i >= args.warmup:
            for k, v in (("fwd", t_fwd), ("bwd", t_bwd), ("opt", t_opt), ("zero", t_zero), ("total", t_total)):
                acc[k].append(v)
        if is_main:
            print(f"[phase {i:02d}] {'warm' if i < args.warmup else 'meas'} "
                  f"zero={t_zero:7.1f} fwd={t_fwd:8.1f} bwd={t_bwd:8.1f} opt={t_opt:7.1f} total={t_total:8.1f} ms",
                  flush=True)

    if is_main:
        med = {k: statistics.median(v) for k, v in acc.items()}
        print("\n[phase] median ms: " + "  ".join(f"{k}={v:.1f}" for k, v in med.items()))
        s = med["fwd"] + med["bwd"] + med["opt"] + med["zero"]
        print(f"[phase] 分段和={s:.1f} vs total={med['total']:.1f}（差 {med['total']-s:+.1f} ms 为同步/测量开销）")
        print(f"[phase] 占比: fwd={100*med['fwd']/med['total']:.1f}% bwd={100*med['bwd']/med['total']:.1f}% "
              f"opt={100*med['opt']/med['total']:.1f}% zero={100*med['zero']/med['total']:.1f}%")

    npu_compat.shutdown()


if __name__ == "__main__":
    main()
