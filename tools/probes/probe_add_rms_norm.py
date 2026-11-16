#!/usr/bin/env python3
# coding=utf-8
"""探针：`torch_npu.npu_add_rms_norm` 能否替代「残差加 + npu_rms_norm」两步（D1 可行性）。

检查：① API 是否存在、返回结构；② 前向与反向相对「手写 add + npu_rms_norm」的误差；
      ③ 单次耗时对比（同形状，prefill 规模 B×L×d）。
判据：API 可用；max rel|Δout| ≤ 1e-5；反向 worst rel|Δg| ≤ 1e-5。
"""
import argparse
import os
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seq", type=int, default=1203)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()
DEV = a.device
if DEV == "npu":
    import torch_npu  # noqa: E402,F401

print(f"torch={torch.__version__} device={DEV} shape=({a.batch},{a.seq},{a.dim})")
print("npu_add_rms_norm 存在:", hasattr(torch_npu, "npu_add_rms_norm"))

if not hasattr(torch_npu, "npu_add_rms_norm"):
    print("VERDICT: API_MISSING")
    sys.exit(2)

eps = 1e-6
torch.manual_seed(0)
w = torch.randn(a.dim, device=DEV)
a1 = torch.randn(a.batch, a.seq, a.dim, device=DEV, dtype=torch.bfloat16) * 0.1
a2 = torch.randn(a.batch, a.seq, a.dim, device=DEV, dtype=torch.bfloat16) * 0.1


def ref():
    """当前线上写法：先加，再调融合 npu_rms_norm（E1b 档）。"""
    return torch_npu.npu_rms_norm(a1 + a2, w.to(a1.dtype), eps)[0]


def fused():
    return torch_npu.npu_add_rms_norm(a1, a2, w.to(a1.dtype), eps)


out_ref = ref()
r = fused()
outs = r if isinstance(r, (tuple, list)) else (r,)
print(f"融合返回 {type(r).__name__}，元素 {len(outs)}，shape {[tuple(t.shape) for t in outs if torch.is_tensor(t)]}")
y = outs[0]
d = (y.float() - out_ref.float()).abs()
rel = (d / out_ref.float().abs().clamp_min(1e-6)).max().item()
print(f"前向: max|Δ| {d.max().item():.3e}  max rel {rel:.3e}  out dtype {y.dtype}")

# 反向：两条路径都要求 d(out)/d(w) 与 d/da1
w1 = w.to(a1.dtype).clone().requires_grad_(True)
w2 = w.to(a1.dtype).clone().requires_grad_(True)
b1 = a1.clone().requires_grad_(True)
b2 = a1.clone().requires_grad_(True)
g = torch.randn_like(out_ref, dtype=torch.float32)

ref_out = torch_npu.npu_rms_norm(b1 + a2, w1, eps)[0]
ref_out.backward(g)

f_out = torch_npu.npu_add_rms_norm(b2, a2, w2, eps)[0]
f_out.backward(g)

for nm, x1, x2 in (("dw", w1.grad, w2.grad), ("da", b1.grad, b2.grad)):
    dd = (x1 - x2).abs()
    denom = x1.abs().clamp_min(1e-3 * x1.abs().max().clamp_min(1e-12))
    print(f"反向 {nm}: max|Δ| {dd.max().item():.3e}  worst rel {(dd / denom).max().item():.3e}")


def bench(fn, tag):
    for _ in range(3):
        fn()
    if DEV == "npu":
        torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        fn()
    if DEV == "npu":
        torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / a.reps
    print(f"{tag}: {ms:.3f} ms/call")
    return ms


print("\n--- 用时（前向，含 alloc）")
m1 = bench(ref, "手动 add + rms")
m2 = bench(lambda: fused()[0], "npu_add_rms_norm")
print(f"节省 {m1 - m2:.3f} ms/call（{(m1 - m2) / m1 * 100:.1f}%）")
