#!/usr/bin/env python3
# coding=utf-8
"""候选①：attention 用 `BSH` 布局直接产出 `(B, L, d)`，省掉 transpose+contiguous 那一次整张量拷贝。

现状：`SDPA(q,k,v)` 得到 `(B,H,Lq,Dh)` → `.transpose(1,2).contiguous()` → `.view(B,Lq,d)`；
profile 里这部分是 `ViewCopy 45.3 + Transpose 32.4 ms/步`。
如果让 FA 直接以 `BSH`（`(B,L,H,D)`）布局输出，`view(B,Lq,d)` 就是零拷贝。

本探针对拍：① 数值是否与显式 mask 路径一致；② fwd / fwd+bwd（含拷贝）耗时。
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lq", type=int, default=480)
ap.add_argument("--lk", type=int, default=1203)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
DEV = "npu"
import torch_npu  # noqa: E402,F401

B, H, LQ, LK, D = a.batch, a.heads, a.lq, a.lk, a.dim
DM = H * D
torch.manual_seed(0)

# 以 (B, L, d) 形式给出 q/k/v（这正是 block 里 _project_one 的输出）
q3 = torch.randn(B, LQ, DM, device=DEV, dtype=torch.bfloat16)
k3 = torch.randn(B, LK, DM, device=DEV, dtype=torch.bfloat16)
v3 = torch.randn(B, LK, DM, device=DEV, dtype=torch.bfloat16)

row = torch.arange(LQ, device=DEV).unsqueeze(1)
col = torch.arange(LK, device=DEV).unsqueeze(0)
mask = (col <= row + (LK - LQ))                       # (LQ, LK) 底右对齐
print(f"B{B} H{H} Lq{LQ} Lk{LK} D{D}  mask {tuple(mask.shape)}")
print("FA 可用:", hasattr(torch_npu, "npu_fusion_attention"))


def to_bnsd(x3):                                       # 现状：view + transpose（view 免费）
    return x3.view(x3.size(0), -1, H, D).transpose(1, 2)


def path_sdpa():
    q, k, v = to_bnsd(q3), to_bnsd(k3), to_bnsd(v3)
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[None, None], dropout_p=0.0)
    y = y.transpose(1, 2).contiguous()                 # ← 这一步就是被 profiler 记成 ViewCopy/Transpose 的
    return y.view(y.size(0), -1, DM)


def path_fa_bsh(sm=1):
    q = q3.view(B, LQ, H, D)
    k = k3.view(B, LK, H, D)
    v = v3.view(B, LK, H, D)
    out = torch_npu.npu_fusion_attention(
        q, k, v, H, "BSH", atten_mask=mask, scale=D ** -0.5, keep_prob=1.0, sparse_mode=sm)
    y = out[0]                                         # (B, Lq, H, D) 连续 ⇒ view 零拷贝
    return y.view(B, -1, DM)


y_ref = path_sdpa()
for sm in (1, 3):
    try:
        y = path_fa_bsh(sm)
        d = (y.float() - y_ref.float()).abs().max().item()
        rel = ((y.float() - y_ref.float()).abs() / y_ref.float().abs().clamp_min(1e-2)).mean().item()
        print(f"FA BSH sparse_mode={sm}: fwd max|Δ| {d:.3e}  mean rel {rel:.3e}  {'<== 等价' if d < 1e-2 else ''}")
    except Exception as e:
        print(f"FA BSH sparse_mode={sm} FAILED: {type(e).__name__} {str(e)[:160]}")


def bench(fn, tag):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / a.reps
    print(f"{tag:36s} {ms:.3f} ms")
    return ms


m_sdpa = bench(path_sdpa, "现状: SDPA + transpose/contiguous")
try:
    m_fa = bench(lambda: path_fa_bsh(1), "候选: FA BSH(直接出 (B,L,d))")
    print(f"fwd 差 {m_fa - m_sdpa:+.3f} ms（{(m_fa - m_sdpa) / m_sdpa * 100:+.1f}%）")
except Exception as e:
    print("FA bench failed:", type(e).__name__, str(e)[:120])

# ---- 反向 ----
g = torch.randn(B, LQ, DM, device=DEV, dtype=torch.bfloat16)


def bwd_sdpa():
    qq, kk, vv = q3.clone().requires_grad_(True), k3.clone().requires_grad_(True), v3.clone().requires_grad_(True)
    y = path_sdpa_with(qq, kk, vv)
    y.backward(g)
    return qq.grad, kk.grad, vv.grad


def path_sdpa_with(qq, kk, vv):
    y = torch.nn.functional.scaled_dot_product_attention(
        to_bnsd(qq), to_bnsd(kk), to_bnsd(vv), attn_mask=mask[None, None], dropout_p=0.0)
    y = y.transpose(1, 2).contiguous()
    return y.view(y.size(0), -1, DM)


def bwd_fa():
    qq, kk, vv = q3.clone().requires_grad_(True), k3.clone().requires_grad_(True), v3.clone().requires_grad_(True)
    y = torch_npu.npu_fusion_attention(qq.view(B, LQ, H, D), kk.view(B, LK, H, D), vv.view(B, LK, H, D),
                                       H, "BSH", atten_mask=mask, scale=D ** -0.5, keep_prob=1.0,
                                       sparse_mode=1)[0]
    y.view(B, -1, DM).backward(g)
    return qq.grad, kk.grad, vv.grad


try:
    g1 = bwd_sdpa()
    g2 = bwd_fa()
    for nm, x, y in zip(("dq", "dk", "dv"), g1, g2):
        d = (x.float() - y.float()).abs()
        rel = (d / x.float().abs().clamp_min(1e-2)).mean().item()
        print(f"bwd {nm}: max|Δ| {d.max().item():.3e}  mean rel {rel:.3e}")
    b_s = bench(bwd_sdpa, "fwd+bwd: 现状 SDPA+拷贝")
    b_f = bench(bwd_fa, "fwd+bwd: 候选 FA BSH")
    print(f"fwd+bwd 差 {b_f - b_s:+.3f} ms（{(b_f - b_s) / b_s * 100:+.1f}%）")
except Exception as e:
    print("BWD compare failed:", type(e).__name__, str(e)[:200])
