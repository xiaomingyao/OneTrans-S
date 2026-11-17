#!/usr/bin/env python3
# coding=utf-8
"""图模式失败点二分（单 die）：逐个算子单独 capture，找出「旁路 stream 未 join」的真凶。

背景：整步 capture 报 `rtStreamEndCapture ... contains a stream that was not joined`；
只捕获前向也报同一错 ⇒ 旁路 stream 来自某个算子内部。这里把候选算子逐个单独捕获，
每个都做 warmup → capture → replay，打印 OK/FAIL 与错误首行。

用法：
  python3 tools/probes/probe_graph_op_bisect.py [--batch 256] [--skip-fa-bwd]
"""
import argparse
import traceback

import torch
import torch_npu  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--seq", type=int, default=1203)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--skip-fa-bwd", action="store_true")
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--only", type=str, default="", help="只跑指定项（逗号分隔关键字，如 sdpa_mask,fa_fwd）；一个进程里 SDPA is_causal 失败后上下文会坏，需分进程")
a = ap.parse_args()
DEV = "npu"
B, L, D, H = a.batch, a.seq, a.dim, a.heads
HD = D // H
torch.manual_seed(0)


ONLY = [t for t in a.only.split(",") if t]


def want(key):
    return (not ONLY) or any(k in key for k in ONLY)


def try_graph(fn, tag, warmup=3):
    """warmup 若干次后 capture 一次、replay 一次。返回 (ok, 说明)。"""
    g = torch.npu.NPUGraph()
    try:
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        with torch.npu.graph(g):
            fn()
        torch.npu.synchronize()
        for _ in range(a.repeat):
            g.replay()
        torch.npu.synchronize()
        print(f"  [OK  ] {tag}", flush=True)
        return True, ""
    except Exception as e:  # noqa: BLE001
        line = str(e).strip().splitlines()
        first = next((x for x in line if x.strip()), type(e).__name__)
        print(f"  [FAIL] {tag}  <- {type(e).__name__}: {first[:150]}", flush=True)
        return False, first[:300]
    finally:
        del g


print(f"=== op 级 capture 二分（B{B} L{L} d{D} H{H}）")

# ---------- 1. 纯 GEMM ----------
x = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
w = torch.randn(D, D, device=DEV, dtype=torch.bfloat16)
bias = torch.randn(D, device=DEV, dtype=torch.bfloat16)
if want("linear"): try_graph(lambda: torch.nn.functional.linear(x, w, bias), "F.linear bf16")

# ---------- 2. RMSNorm（融合 vs 手写） ----------
g_w = torch.randn(D, device=DEV, dtype=torch.float32)


def fused_norm():
    return torch_npu.npu_rms_norm(x, g_w.to(x.dtype), 1e-6)[0]


if want("rms_norm"): try_graph(fused_norm, "npu_rms_norm(融合)")


def manual_norm():
    v = x.float()
    return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * g_w).to(x.dtype)


if want("manual_norm"): try_graph(manual_norm, "手写 rmsnorm(cast 链)")

# ---------- 3. add + rmsnorm 融合 ----------
res = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)


def add_norm():
    return torch_npu.npu_add_rms_norm(res, x, g_w.to(x.dtype), 1e-6)


if want("add_norm"): try_graph(add_norm, "npu_add_rms_norm(融合)")

# ---------- 4. SDPA：q==k 因果 / q≠k 掩码 ----------
q4 = torch.randn(B, H, L, HD, device=DEV, dtype=torch.bfloat16)
mask_blk = (torch.arange(L, device=DEV).unsqueeze(0) > torch.arange(L, device=DEV).unsqueeze(1))


def sdpa_same():
    return torch.nn.functional.scaled_dot_product_attention(q4, q4, q4, is_causal=True)


if want("sdpa_same"): try_graph(sdpa_same, "SDPA is_causal (q==k)")

LQ = 480
q_short = torch.randn(B, H, LQ, HD, device=DEV, dtype=torch.bfloat16)
m2 = (torch.arange(L, device=DEV).unsqueeze(0) > (torch.arange(LQ, device=DEV).unsqueeze(1) + (L - LQ)))
m2 = m2.unsqueeze(0).unsqueeze(0)


def sdpa_mask():
    return torch.nn.functional.scaled_dot_product_attention(q_short, q4, q4, attn_mask=m2)


if want("sdpa_mask"): try_graph(sdpa_mask, "SDPA 显式掩码 (q≠k)")

# ---------- 5. FA BSH 前向 ----------
qb = torch.randn(B, LQ, H * HD, device=DEV, dtype=torch.bfloat16)
kb = torch.randn(B, L, H * HD, device=DEV, dtype=torch.bfloat16)
mb = (torch.arange(L, device=DEV).unsqueeze(0) > (torch.arange(LQ, device=DEV).unsqueeze(1) + (L - LQ)))


def fa_fwd():
    with torch.no_grad():
        return torch_npu.npu_fusion_attention(qb, kb, kb, H, "BSH", atten_mask=mb,
                                              scale=HD ** -0.5, keep_prob=1.0, sparse_mode=1)[0]


if want("fa_fwd"): try_graph(fa_fwd, "npu_fusion_attention BSH 前向(带掩码)")

# ---------- 6. FA 前向+反向（捕获含 autograd） ----------
if not a.skip_fa_bwd and want("fa_fb"):
    def fa_fb():
        qq = qb.detach().clone().requires_grad_(True)
        kk = kb.detach().clone().requires_grad_(True)
        vv = kb.detach().clone().requires_grad_(True)
        yy = torch_npu.npu_fusion_attention(qq, kk, vv, H, "BSH", atten_mask=mb,
                                            scale=HD ** -0.5, keep_prob=1.0, sparse_mode=1)[0]
        yy.float().sum().backward()
        return yy

    try_graph(fa_fb, "npu_fusion_attention 前向+反向")

# ---------- 7. 逐元素 / 归约 ----------
if want("silu"): try_graph(lambda: torch.nn.functional.silu(x), "silu")
if want("mean"): try_graph(lambda: x.float().mean(), "mean→fp32 (含 cast)")
if want("zeros"): try_graph(lambda: torch.zeros(B, L, D, device=DEV, dtype=torch.bfloat16), "zeros(分配)")

# ---------- 8. 小段真实 block（如果单算子都 OK 而整模不行，问题在组合） ----------
if want("block"):
  try:
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU"))
    from main_pytorch import CausalMaskAttention  # noqa: E402

    blk = CausalMaskAttention(ns_len=12, d_model=D, num_heads=H, mask_type="paper_causal").to(DEV).bfloat16()
    xin = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
    try_graph(lambda: blk(xin), "CausalMaskAttention 整块 (bf16 参数)")
  except Exception:  # noqa: BLE001
    traceback.print_exc(limit=1)
