#!/usr/bin/env python3
# coding=utf-8
"""图模式第三轮：给 `torch.npu.graph` 换 capture 模式（auto_dispatch_capture / capture_error_mode）。

已定位：默认 `torch.npu.graph(g)` 捕获 attention 算子必报 `capture_end ... 107025`
（FA fwd/bwd、SDPA 两种都失败；linear/rms_norm/silu 都 OK）⇒ 旁路 stream 在 attention 内部。
本探针把 `torch.npu.graph` 的另外两个参数（auto_dispatch_capture、capture_error_mode）
依次打开，看能不能把旁路 stream 纳入捕获。
"""
import argparse
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--seq", type=int, default=1203)
ap.add_argument("--lq", type=int, default=480)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--heads", type=int, default=4)
a = ap.parse_args()
DEV = "npu"
B, L, LQ, D, H = a.batch, a.seq, a.lq, a.dim, a.heads
HD = D // H
torch.manual_seed(0)

qb = torch.randn(B, LQ, D, device=DEV, dtype=torch.bfloat16)
kb = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
mb = (torch.arange(L, device=DEV).unsqueeze(0) > (torch.arange(LQ, device=DEV).unsqueeze(1) + (L - LQ)))
q4 = qb.view(B, LQ, H, HD).transpose(1, 2)
k4 = kb.view(B, L, H, HD).transpose(1, 2)
m4 = mb.unsqueeze(0).unsqueeze(0)


def fa_fwd():
    with torch.no_grad():
        return torch_npu.npu_fusion_attention(qb, kb, kb, H, "BSH", atten_mask=mb,
                                              scale=HD ** -0.5, keep_prob=1.0, sparse_mode=1)[0]


def sdpa_mask():
    return torch.nn.functional.scaled_dot_product_attention(q4, k4, k4, attn_mask=m4)


def try_graph(fn, tag, **gkw):
    g = torch.npu.NPUGraph()
    try:
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        with torch.npu.graph(g, **gkw):
            fn()
        torch.npu.synchronize()
        g.replay()
        torch.npu.synchronize()
        print(f"  [OK  ] {tag:52s} {gkw}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        first = next((x for x in str(e).splitlines() if x.strip()), type(e).__name__)
        print(f"  [FAIL] {tag:52s} {gkw} <- {first[:120]}", flush=True)
        return False
    finally:
        del g


print(f"=== capture 模式矩阵（B{B} LQ{LQ} LK{L} H{H}）")
for fn, nm in ((fa_fwd, "FA BSH fwd+mask"), (sdpa_mask, "SDPA 显式掩码")):
    for gkw in (dict(), dict(auto_dispatch_capture=True),
                dict(capture_error_mode="thread_local"),
                dict(auto_dispatch_capture=True, capture_error_mode="thread_local")):
        try_graph(fn, nm, **gkw)
