#!/usr/bin/env python3
# coding=utf-8
"""验证 (a)：`npu_fusion_attention` 的 `sparse_mode` 能否取代显式底右因果 mask（q<k 的层）。

现状：`paper_causal` 用显式 bool mask（底右对齐：col <= row + (key_len - query_len)）；
SDPA 的 `is_causal=True` 是左上对齐，所以 q<k 的层只能用 mask（G1 只覆盖了 q==k 的层）。
本探针直接调 `torch_npu.npu_fusion_attention`，扫 sparse_mode ∈ {2,3,4}，找出与底右 mask 等价的那个，
同时比 fwd / fwd+bwd 的耗时。

判据：fwd max|Δ| ≤ 1e-2（bf16）、反向 dq/dk/dv 相对误差 ≤ 1e-2；耗时 ≤ 现路径。
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lq", type=int, default=480)
ap.add_argument("--lk", type=int, default=1203)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
DEV = a.device
if DEV == "npu":
    import torch_npu  # noqa: F401,E402

torch.manual_seed(0)
B, H, LQ, LK, D = a.batch, a.heads, a.lq, a.lk, a.dim
dt = torch.bfloat16
q0 = torch.randn(B, H, LQ, D, device=DEV, dtype=dt)
k0 = torch.randn(B, H, LK, D, device=DEV, dtype=dt)
v0 = torch.randn(B, H, LK, D, device=DEV, dtype=dt)

row = torch.arange(LQ, device=DEV).unsqueeze(1)
col = torch.arange(LK, device=DEV).unsqueeze(0)
mask_br = (col <= row + (LK - LQ))[None, None]          # 底右对齐（paper_causal）

print(f"device={DEV} torch={torch.__version__} shape B{B} H{H} Lq{LQ} Lk{LK} D{D} {dt}")
print("npu_fusion_attention 可用:", hasattr(torch_npu, "npu_fusion_attention"))

with torch.no_grad():
    y_ref = torch.nn.functional.scaled_dot_product_attention(
        q0, k0, v0, attn_mask=mask_br, dropout_p=0.0, is_causal=False)

for sm in (2, 3, 4):
    try:
        out = torch_npu.npu_fusion_attention(
            q0, k0, v0, H, "BNSD", atten_mask=None, scale=D ** -0.5,
            keep_prob=1.0, sparse_mode=sm,
        )
        y = out[0]
        d = (y.float() - y_ref.float()).abs().max().item()
        rel = ((y.float() - y_ref.float()).abs() / y_ref.float().abs().clamp_min(1e-2)).mean().item()
        print(f"sparse_mode={sm}: fwd max|Δ| = {d:.3e}  mean rel = {rel:.3e}  {'<== 等价' if d < 1e-2 else ''}")
    except Exception as e:
        print(f"sparse_mode={sm}: FAILED {type(e).__name__} {str(e)[:150]}")

# 反向对照（sparse_mode=3）
g = torch.randn_like(y_ref)


def run_mask():
    qq = q0.clone().requires_grad_(True)
    kk = k0.clone().requires_grad_(True)
    vv = v0.clone().requires_grad_(True)
    y = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask_br, dropout_p=0.0)
    y.backward(g)
    return qq.grad, kk.grad, vv.grad


def run_sparse(sm=3):
    qq = q0.clone().requires_grad_(True)
    kk = k0.clone().requires_grad_(True)
    vv = v0.clone().requires_grad_(True)
    y = torch_npu.npu_fusion_attention(qq, kk, vv, H, "BNSD", atten_mask=None,
                                       scale=D ** -0.5, keep_prob=1.0, sparse_mode=sm)[0]
    y.backward(g)
    return qq.grad, kk.grad, vv.grad


try:
    gm = run_mask()
    gs = run_sparse(3)
    for nm, x, y in zip(("dq", "dk", "dv"), gm, gs):
        d = (x.float() - y.float()).abs()
        rel = (d / x.float().abs().clamp_min(1e-2)).mean().item()
        print(f"bwd {nm}: max|Δ| = {d.max().item():.3e}  mean rel = {rel:.3e}")
except Exception as e:
    print("BWD_FAILED:", type(e).__name__, str(e)[:200])


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
    print(f"{tag:34s} {ms:.3f} ms")
    return ms


with torch.no_grad():
    m_mask = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
        q0, k0, v0, attn_mask=mask_br, dropout_p=0.0), "fwd: SDPA + 显式 mask")
    m_iso = bench(lambda: torch.nn.functional.scaled_dot_product_attention(
        q0, k0, v0, attn_mask=None, is_causal=True), "fwd: SDPA is_causal(左上, 仅参考)")
    try:
        m_sm = bench(lambda: torch_npu.npu_fusion_attention(
            q0, k0, v0, H, "BNSD", atten_mask=None, scale=D ** -0.5, keep_prob=1.0, sparse_mode=3)[0],
            "fwd: FA sparse_mode=3")
        print(f"fwd 差 {m_sm - m_mask:+.3f} ms（{(m_sm - m_mask) / m_mask * 100:+.1f}%）")
    except Exception as e:
        print("FA bench failed:", type(e).__name__, str(e)[:150])

b_mask = bench(run_mask, "fwd+bwd: SDPA + 显式 mask")
try:
    b_sm = bench(lambda: run_sparse(3), "fwd+bwd: FA sparse_mode=3")
    print(f"fwd+bwd 差 {b_sm - b_mask:+.3f} ms（{(b_sm - b_mask) / b_mask * 100:+.1f}%）")
except Exception as e:
    print("FA bwd bench failed:", type(e).__name__, str(e)[:150])
