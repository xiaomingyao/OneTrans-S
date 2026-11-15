#!/usr/bin/env python3
# coding=utf-8
"""最小复现：DDP 反向之后紧接着发自己的集合通信，会不会卡住？

背景：3 卡以内的训练脚本里，train epoch 跑完最后一个 step 后，脚本自己做一次
all_gather 汇总指标 —— 实测直接停在 HcclAllgather（无异常、无输出，只能 kill）。
这里把该场景压到最小：DDP 包一个两层 MLP，跑 steps 步，然后做一次 int64 all_gather。

用法（torchrun 拉起）：
  timeout 90 torchrun --nproc_per_node=2 tests/probe_ddp_collective.py --steps 2 --sync none
  timeout 90 torchrun --nproc_per_node=2 tests/probe_ddp_collective.py --steps 2 --sync both
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_npu  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--sync", default="none", choices=["none", "sync", "barrier", "both"])
    ap.add_argument("--find-unused", action="store_true")
    a = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = f"npu:{local_rank}"

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4)
    ).to(dev)
    model = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=None, find_unused_parameters=a.find_unused, broadcast_buffers=False
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lf = torch.nn.CrossEntropyLoss()

    for s in range(a.steps):
        g = torch.Generator().manual_seed(100 + s)
        x = torch.randn(16, 32, generator=g).to(dev)
        y = torch.randint(0, 4, (16,), generator=g).to(dev)
        opt.zero_grad(set_to_none=True)
        loss = lf(model(x), y)
        loss.backward()
        opt.step()
        print(f"[repro] rank{rank} step{s} loss={loss.item():.4f}", flush=True)

    if a.sync in ("sync", "both"):
        torch.npu.synchronize()
    if a.sync in ("barrier", "both"):
        dist.barrier()
    print(f"[repro] rank{rank} before_gather sync={a.sync}", flush=True)

    t = torch.tensor([rank + 1], dtype=torch.int64, device=dev)
    bucket = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(bucket, t)
    torch.npu.synchronize()
    print(f"[repro] rank{rank} AFTER_GATHER ok values={[int(b.cpu().item()) for b in bucket]}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
