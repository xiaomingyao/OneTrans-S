#!/usr/bin/env python3
# coding=utf-8
"""h 项（Triton 版）：把「残差加 + RMSNorm」融成一个 kernel，前向/反向都与现有两步写法对比。

现有路径（线上）：`add` (aclnn Add) + `torch_npu.npu_rms_norm`；本文件用 Triton 写融合版：
  fwd: x = a + b;  rstd = 1/sqrt(mean(x^2)+eps);  y = x * rstd * w
  bwd: dx_i = rstd*(w_i*dy_i) - rstd^3 * x_i / N * Σ_j(w_j*dy_j*x_j)
       dw_i += Σ_rows dy_i * x_i * rstd        （用 atomic 累加）

判据：① 前向 max|Δ| ≤ 2e-2（bf16 域参考）、反向非零且相对误差 ≤ 2e-2；② 单次耗时比两步写法低 ≥20%。
"""
import argparse
import time

import torch
import triton
import triton.language as tl

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seq", type=int, default=1203)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--reps", type=int, default=20)
a = ap.parse_args()
DEV = a.device
if DEV == "npu":
    import torch_npu  # noqa: F401,E402

eps = 1e-6
B, L, D = a.batch, a.seq, a.dim
ROWS = B * L


@triton.jit
def _add_rmsnorm_fwd(A, Bp, W, Y, RSTD, ROWS, N: tl.constexpr, EPS: tl.constexpr,
                     BLOCK: tl.constexpr, RPP: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RPP + tl.arange(0, RPP)                     # (RPP,)
    cols = tl.arange(0, BLOCK)                               # (BLOCK,)
    rm = rows < ROWS
    cm = cols < N
    mask = rm[:, None] & cm[None, :]
    off = rows.to(tl.int64)[:, None] * N + cols[None, :]     # (RPP, BLOCK) 连续
    av = tl.load(A + off, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(Bp + off, mask=mask, other=0.0).to(tl.float32)
    x = av + bv
    var = tl.sum(tl.where(mask, x * x, 0.0), 1) / N
    rstd = 1.0 / tl.sqrt(var + EPS)                          # (RPP,)
    w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    tl.store(Y + off, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + rows, rstd, mask=rm)


@triton.jit
def _add_rmsnorm_bwd(DY, X, W, DX, DW, RSTD, ROWS, N: tl.constexpr,
                     BLOCK: tl.constexpr, RPP: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * RPP + tl.arange(0, RPP)
    cols = tl.arange(0, BLOCK)
    rm = rows < ROWS
    cm = cols < N
    mask = rm[:, None] & cm[None, :]
    off = rows.to(tl.int64)[:, None] * N + cols[None, :]
    dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=cm, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + rows, mask=rm, other=0.0).to(tl.float32)
    g = dy * w[None, :]
    s = tl.sum(tl.where(mask, g * x, 0.0), 1)                # (RPP,)
    rstd3 = (rstd * rstd * rstd)[:, None]
    dx = rstd[:, None] * g - rstd3 * x * s[:, None] / N
    tl.store(DX + off, dx.to(DX.dtype.element_ty), mask=mask)
    dw = tl.sum(tl.where(mask, dy * x * rstd[:, None], 0.0), 0)
    tl.atomic_add(DW + cols, dw.to(DW.dtype.element_ty), mask=cm)


def triton_fwd(a1, b1, w):
    y = torch.empty_like(a1)
    rstd = torch.empty(ROWS, device=a1.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(D)
    RPP = 32                                   # NPU grid 上限 65535 ⇒ 每个 program 处理 32 行
    grid = (triton.cdiv(ROWS, RPP),)
    _add_rmsnorm_fwd[grid](a1, b1, w, y, rstd, ROWS, N=D, EPS=eps, BLOCK=BLOCK, RPP=RPP)
    return y, rstd


def triton_bwd(dy, x, w, rstd, dw):
    dx = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(D)
    RPP = 32
    grid = (triton.cdiv(ROWS, RPP),)
    dw.zero_()
    _add_rmsnorm_bwd[grid](dy, x, w, dx, dw, rstd, ROWS, N=D, BLOCK=BLOCK, RPP=RPP)
    return dx


torch.manual_seed(0)
a1 = (torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16) * 0.1).contiguous()
b1 = (torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16) * 0.1).contiguous()
w = torch.randn(D, device=DEV, dtype=torch.bfloat16)
dw = torch.zeros(D, device=DEV, dtype=torch.bfloat16)

print(f"torch={torch.__version__} triton={triton.__version__} shape=({B},{L},{D}) rows={ROWS}")

# 参考：现有两步写法
x_ref = a1 + b1
y_ref = torch_npu.npu_rms_norm(x_ref, w, eps)[0]
rstd_ref = torch.rsqrt(x_ref.float().pow(2).mean(-1, keepdim=True) + eps)

try:
    y_tri, rstd_tri = triton_fwd(a1, b1, w)
    fwd_ok = True
except Exception as e:
    print("TRITON_FWD_FAILED:", type(e).__name__, str(e)[:400])
    fwd_ok = False

if fwd_ok:
    d = (y_tri.float() - y_ref.float()).abs()
    rel = (d / y_ref.float().abs().clamp_min(1e-3)).mean().item()
    print(f"[fwd] max|Δ| = {d.max().item():.3e}   mean rel = {rel:.3e}")
    dr = (rstd_tri - rstd_ref.reshape(-1)).abs().max().item()
    print(f"[fwd] rstd max|Δ| = {dr:.3e}")

    g = torch.randn_like(y_ref)
    try:
        dx_tri = triton_bwd(g, x_ref, w, rstd_tri, dw)
        bwd_ok = True
    except Exception as e:
        print("TRITON_BWD_FAILED:", type(e).__name__, str(e)[:300])
        bwd_ok = False

    if bwd_ok:
        # 参考反向（手写公式，fp32）
        xf = x_ref.float()
        rs = rstd_ref
        gf = g.float() * w.float()
        s = (gf * xf).sum(-1, keepdim=True)
        dx_ref = rs * gf - rs ** 3 * xf * s / D
        db = (dx_tri.float() - dx_ref).abs()
        print(f"[bwd] dx max|Δ| = {db.max().item():.3e}   mean rel = {(db / dx_ref.abs().clamp_min(1e-3)).mean().item():.3e}")


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
    print(f"{tag:38s} {ms:.3f} ms/call")
    return ms


m_two = bench(lambda: torch_npu.npu_rms_norm(a1 + b1, w, eps)[0], "现有: add + npu_rms_norm")
if fwd_ok:
    m_tri = bench(lambda: triton_fwd(a1, b1, w)[0], "Triton: 融合 add+rmsnorm(fwd)")
    print(f"fwd 差 {m_tri - m_two:+.3f} ms（{(m_tri - m_two) / m_two * 100:+.1f}%）")

# 反向对照：npu_rms_norm 的反向 + add 的反向
def ref_bwd():
    xx = (a1 + b1).requires_grad_(True)
    yy = torch_npu.npu_rms_norm(xx, w, eps)[0]
    yy.backward(g)
    return xx.grad


m_two_b = bench(ref_bwd, "现有: rmsnorm bwd + add bwd")
if fwd_ok and bwd_ok:
    m_tri_b = bench(lambda: triton_bwd(g, x_ref, w, rstd_tri, dw), "Triton: 融合 bwd")
    print(f"bwd 差 {m_tri_b - m_two_b:+.3f} ms（{(m_tri_b - m_two_b) / m_two_b * 100:+.1f}%）")
