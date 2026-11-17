#!/usr/bin/env python3
"""probe_flopcounter_attn.py — FlopCounterMode 到底数不数 attention？

目的：解释 31.529 GFLOP/sample 这个记账值是否包含 attention。
  · 本仓 GPU_Reference/CPU 路径用 F.scaled_dot_product_attention；
  · 昇腾路径用 torch_npu.npu_fusion_attention（自定义算子，FlopCounterMode 无 handler）。
测三件事：SDPA 单算子、nn.Linear、以及一个含 attention 的小 block。
"""
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

torch.manual_seed(0)


def count(fn):
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    return fc.get_total_flops()


B, H, S, D = 4, 4, 128, 64

q = torch.randn(B, H, S, D)
k = torch.randn(B, H, S, D)
v = torch.randn(B, H, S, D)

sdpa = count(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
expect_sdpa = 4 * B * H * S * S * D  # 两次 matmul
print(f"[1] SDPA(is_causal)        counted={sdpa/1e6:8.3f} MFLOP   理论两次matmul={expect_sdpa/1e6:8.3f} MFLOP"
      f"  -> {'计数' if sdpa > 0 else '**不计数**'}")

lin = torch.nn.Linear(256, 256)
x = torch.randn(1024, 256)
lin_flops = count(lambda: lin(x))
print(f"[2] nn.Linear(256->256)    counted={lin_flops/1e6:8.3f} MFLOP  理论={2*1024*256*256/1e6:8.3f} MFLOP"
      f"  -> {'计数' if lin_flops > 0 else '**不计数**'}")

mm = count(lambda: x @ x.t())
print(f"[3] torch.mm               counted={mm/1e6:8.3f} MFLOP  理论={2*1024*1024*256/1e6:8.3f} MFLOP")

# [4] 昇腾 FA 自定义算子（无 NPU 时跳过）
try:
    import torch_npu  # noqa: F401
    qq = torch.randn(4, 128, 256, device="npu", dtype=torch.bfloat16)
    fa = count(lambda: torch_npu.npu_fusion_attention(qq, qq, qq, 4, "BSH"))
    print(f"[4] npu_fusion_attention   counted={fa/1e6:8.3f} MFLOP  -> {'计数' if fa > 0 else '**不计数**'}")
except Exception as e:  # noqa: BLE001
    print(f"[4] npu_fusion_attention   跳过/失败: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
