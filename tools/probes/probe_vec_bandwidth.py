#!/usr/bin/env python3
# coding=utf-8
"""向量侧带宽实测：确认 RMSNorm 是否真的只有 SiLU 的 1/4 带宽（决定下一步优化对象）。

做法：同一形状 (B, L, 256) bf16，分别测
  - SiLU（单读单写，纯 memory-bound 基准）
  - npu_rms_norm（融合，输入 bf16 + gamma bf16）
  - 手写 rmsnorm（fp32 中间量）
  - Add（残差，单读单写双输入）
算出每个算子的有效带宽 GB/s = (读字节 + 写字节) / 时间。
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seq", type=int, default=1203)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
DEV = "npu"
B, L, D = a.batch, a.seq, a.dim
torch.manual_seed(0)

x = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
y = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
g = torch.randn(D, device=DEV, dtype=torch.float32)
gb = g.to(torch.bfloat16)
nbytes = x.numel() * 2


def bench(fn, tag, rw):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / a.reps
    gbs = rw * nbytes / (ms / 1000) / 1e9
    print(f"  {tag:34s} {ms:7.3f} ms   {gbs:7.0f} GB/s（{rw}× {nbytes / 2**20:.0f} MiB）")
    return ms


print(f"=== 形状 {B}×{L}×{D} bf16（单张量 {nbytes / 2**20:.0f} MiB）")
bench(lambda: torch.nn.functional.silu(x), "SiLU (读1写1)", 2)
bench(lambda: torch_npu.npu_rms_norm(x, gb, 1e-6)[0], "npu_rms_norm 融合 (读1写1)", 2)


def manual():
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * g).to(torch.bfloat16)


bench(manual, "手写 rmsnorm (读1写1, fp32中间)", 2)
bench(lambda: x + y, "Add 残差 (读2写1)", 3)
bench(lambda: torch.nn.functional.silu(x, inplace=True), "SiLU inplace (读1写1)", 2)
