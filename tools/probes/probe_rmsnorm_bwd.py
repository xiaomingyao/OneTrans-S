#!/usr/bin/env python3
# coding=utf-8
"""S2 前置探针：`npu_rms_norm_backward(dy, x, gamma, rstd)` 直接调用 vs autograd 路径。

关键点：`npu_rms_norm` 前向就把 `rstd` 返回了（不需要反向时重算平方和）。
如果直接用 `npu_rms_norm_backward` 喂 rstd 能更快，就可以用自定义 autograd Function
把 rstd 存下来，在**不改模型结构**的前提下把 RmsNormGrad（24.5 ms/步）压下来。
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
eps = 1e-6

x = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
dy = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
gb = torch.randn(D, device=DEV, dtype=torch.float32).to(torch.bfloat16)
nbytes = x.numel() * 2


def bench(fn, tag, rw, note=""):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / a.reps
    print(f"  {tag:40s} {ms:7.3f} ms  {rw * nbytes / (ms / 1000) / 1e9:6.0f} GB/s  {note}")
    return ms


with torch.no_grad():
    y_f, rstd = torch_npu.npu_rms_norm(x, gb, eps)
print(f"前向返回: y{tuple(y_f.shape)}/{y_f.dtype}, rstd{tuple(rstd.shape)}/{rstd.dtype}")


# ---- 路径 A：autograd（现状）----
def bwd_autograd():
    xx = x.detach().clone().requires_grad_(True)
    gg = gb.detach().clone().requires_grad_(True)
    yy = torch_npu.npu_rms_norm(xx, gg, eps)[0]
    yy.backward(dy, retain_graph=False)


# ---- 路径 B：直接调 npu_rms_norm_backward（rstd 已知）----
def bwd_direct():
    return torch.ops.npu.npu_rms_norm_backward(dy, x, gb, rstd)


print("\n=== 反向对比（不含前向）")
t_a = bench(bwd_autograd, "A: autograd 反向（重算平方和）", 4)
t_b = bench(bwd_direct, "B: npu_rms_norm_backward(rstd)直接", 4)
print(f"  B/A = {t_b / t_a:.2f}x（{(t_b - t_a) / t_a * 100:+.1f}%）")

# ---- 数值一致性 ----
xx = x.detach().clone().requires_grad_(True)
gg = gb.detach().clone().requires_grad_(True)
yy = torch_npu.npu_rms_norm(xx, gg, eps)[0]
yy.backward(dy)
dx_a, dg_a = xx.grad, gg.grad
dx_b, dg_b = torch.ops.npu.npu_rms_norm_backward(dy, x, gb, rstd)
print("\n=== 数值对比（B 相对 A）")
print(f"  dx     : max|Δ|={(dx_a.float() - dx_b.float()).abs().max():.3e}  "
      f"mean rel={((dx_a.float() - dx_b.float()).abs() / (dx_a.float().abs() + 1e-6)).mean():.3e}")
print(f"  dgamma : max|Δ|={(dg_a.float() - dg_b.float()).abs().max():.3e}  "
      f"mean rel={((dg_a.float() - dg_b.float()).abs() / (dg_a.float().abs() + 1e-6)).mean():.3e}")

# ---- 前向 + 反向 合起来的两种写法 ----
def full_autograd():
    bwd_autograd()


def full_fn():
    with torch.enable_grad():
        xx = x.detach().clone().requires_grad_(True)
        yy, rr = torch_npu.npu_rms_norm(xx, gb, eps)
        torch.ops.npu.npu_rms_norm_backward(dy, x, gb, rr)


print("\n=== 组合（自定义 Function 的完整前后反向代价）")
print("  说明：自定义 Function 的 forward 会用 npu_rms_norm 拿 rstd，backward 用 B —— 见下方净收益")
t_full_a = bench(bwd_autograd, "A: autograd 整段(前向图+反向)", 4)
t_full_b = bench(lambda: torch_npu.npu_rms_norm(x, gb, eps), "B 的前向部分（只取 rstd/y）", 4)
print(f"  若前向本来就要做（现状已做），则反向净收益 = {t_a - t_b:.3f} ms/次")
