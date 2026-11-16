#!/usr/bin/env python3
# coding=utf-8
"""探针（G1）：attention 用 `is_causal=True` 替代显式 bool mask —— 语义是否一致 + 快多少。

我们的 `paper_causal` 掩码是「底右对齐」因果：allowed = (col <= row + (key_len - query_len))。
需要先确认 PyTorch SDPA 的 is_causal 在 query_len != key_len 时用的是哪种对齐。
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu")
ap.add_argument("--batch", type=int, default=512)
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
q = torch.randn(a.batch, a.heads, a.lq, a.dim, device=DEV, dtype=torch.bfloat16)
k = torch.randn(a.batch, a.heads, a.lk, a.dim, device=DEV, dtype=torch.bfloat16)
v = torch.randn(a.batch, a.heads, a.lk, a.dim, device=DEV, dtype=torch.bfloat16)

row = torch.arange(a.lq, device=DEV).unsqueeze(1)
col = torch.arange(a.lk, device=DEV).unsqueeze(0)
mask_br = (col <= row + (a.lk - a.lq))[None, None]      # 底右对齐（我们的 paper_causal）
mask_tl = (col <= row)[None, None]                      # 左上对齐

with torch.no_grad():
    y_br = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask_br, dropout_p=0.0, is_causal=False)
    y_tl = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask_tl, dropout_p=0.0, is_causal=False)
    y_ca = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

d_br = (y_ca.float() - y_br.float()).abs().max().item()
d_tl = (y_ca.float() - y_tl.float()).abs().max().item()
print(f"shape q=({a.batch},{a.heads},{a.lq},{a.dim}) k/v L={a.lk}  dtype=bf16  device={DEV}")
print(f"is_causal vs 底右对齐 mask: max|Δ| = {d_br:.3e}")
print(f"is_causal vs 左上对齐 mask: max|Δ| = {d_tl:.3e}")
print("⇒ is_causal 的对齐方式:", "底右（与我们的 paper_causal 一致）" if d_br < d_tl else "左上（不等价，需要改 mask 或不用 is_causal）")


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


with torch.no_grad():
    m_mask = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask_br, dropout_p=0.0, is_causal=False), "显式 bool mask")
    m_ca = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True), "is_causal=True")
print(f"差 {m_mask - m_ca:+.3f} ms/call（{(m_mask - m_ca) / m_mask * 100:+.1f}%）")
