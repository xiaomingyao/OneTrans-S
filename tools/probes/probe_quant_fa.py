#!/usr/bin/env python3
"""probe_quant_fa.py — 昇腾有没有「量化版 FlashAttention」可用（决定 FA 能否 int8/fp8）

背景：论文 4 项优化里 FlashAttention-2 在昇腾的等价物是 aclnnFlashAttentionScoreV4（已用，fwd 44.8% / bwd 35.0%）。
若 torch_npu 的 npu_quant_fusion_attention(+backward) 能吃 int8/fp8，则 FAQ 的"量化受限"结论需要更新。
本探针只查能力与签名，不做性能结论。
"""
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"

print("=" * 100)
for n in ("npu_quant_fusion_attention", "npu_quant_fusion_attention_backward"):
    try:
        print(f"{n}{getattr(torch.ops.npu, n).default._schema}")
    except Exception as e:  # noqa: BLE001
        print(f"{n}: schema 取不到 {type(e).__name__}: {str(e)[:120]}")

print("\n[1] 用 bf16 正常输入探底（排除参数名错误）")
B, Hh, S, D = 2, 4, 64, 64
q = torch.randn(B, S, Hh * D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, S, Hh * D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, S, Hh * D, device=dev, dtype=torch.bfloat16)
sq = torch.ones(Hh, device=dev, dtype=torch.float32)
sk = torch.ones(Hh, device=dev, dtype=torch.float32)
sv = torch.ones(Hh, device=dev, dtype=torch.float32)
try:
    out = torch_npu.npu_quant_fusion_attention(q, k, v, Hh, "BSH", d_scale_q=sq, d_scale_k=sk, d_scale_v=sv)
    print(f"  bf16 OK → {len(out)} 个返回, out[0].shape={tuple(out[0].shape)} dtype={out[0].dtype}")
except Exception as e:  # noqa: BLE001
    print(f"  bf16 FAIL {type(e).__name__}: {str(e).splitlines()[0][:160]}")

print("\n[2] int8 输入")
for dt in (torch.int8, torch.float8_e4m3fn):
    try:
        qi = (q * 100).to(dt)
        ki = (k * 100).to(dt)
        vi = (v * 100).to(dt)
        out = torch_npu.npu_quant_fusion_attention(qi, ki, vi, Hh, "BSH", d_scale_q=sq, d_scale_k=sk, d_scale_v=sv)
        print(f"  {dt}: OK → dtype={out[0].dtype} shape={tuple(out[0].shape)}")
    except Exception as e:  # noqa: BLE001
        print(f"  {dt}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:150]}")

print("\n[3] 参考：厂商 bf16 FA v4 在同一形状的耗时（只为说明它存在且可用）")
import time
def bench(fn, it=10, w=3):
    for _ in range(w): fn()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.npu.synchronize(); return (time.perf_counter() - t0) / it * 1000
try:
    t = bench(lambda: torch_npu.npu_fusion_attention(q, k, v, Hh, "BSH")[0])
    print(f"  npu_fusion_attention bf16: {t:.3f} ms (S={S}, 小形状仅供参考)")
except Exception as e:  # noqa: BLE001
    print("  FAIL", e)
