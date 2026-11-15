#!/usr/bin/env python3
# coding=utf-8
"""HCCL 集合通信 dtype 探针：all_gather / all_reduce 在昇腾上支持哪些 dtype。

背景：HCCl 对 dtype 的支持面窄于 NCCL。实测踩到两起：
  - all_reduce(float64) → 直接报 `HCCL allreduce: Unsupported data type at::kDouble`
  - all_gather(int64)   → 无报错、无输出，进程停在集合通信里（最危险的一种）
因此跨卡汇总前必须在目标机器上确认 dtype，而不是照 NCCL 经验写代码。

用法（torchrun 拉起，每 dtype 单独一次调用，外面套 timeout）：
  timeout 60 torchrun --nproc_per_node=2 tests/probe_hccl_dtypes.py --dtype float32 --op all_gather
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

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--op", default="all_gather", choices=["all_gather", "all_reduce"])
    ap.add_argument("--reduce", default="sum", choices=["sum", "max", "min", "prod"])
    a = ap.parse_args()

    if "LOCAL_RANK" not in os.environ:
        raise SystemExit(
            "[probe] 该脚本是多 rank 探针，需用 torchrun 启动，例如：\n"
            "  torchrun --nproc_per_node=2 %s" % os.path.basename(__file__)
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dt = DTYPES[a.dtype]

    t = torch.ones(4, 8, dtype=dt, device="npu") * (rank + 1)
    print(f"[probe] rank={rank} world={world} dtype={a.dtype} op={a.op} start", flush=True)
    if a.op == "all_gather":
        bucket = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(bucket, t)
        got = torch.cat(bucket, 0).float()
    else:
        dist.all_reduce(t, op=getattr(dist.ReduceOp, a.reduce.upper()))
        got = t.float()
    torch.npu.synchronize()
    if rank == 0:
        print(f"[probe] OK dtype={a.dtype} op={a.op}/{a.reduce} sum={float(got.sum()):.1f}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
