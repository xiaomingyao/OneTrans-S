#!/usr/bin/env python3
# coding=utf-8
"""确认 OneTrans 的 SDPA 在 950 上落到哪个 attention kernel（float mask vs bool mask）"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch_npu  # noqa: E402
import npu_compat  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

npu_compat.init_device("npu")
torch.manual_seed(0)
m = TAACOneTransClassifier(non_seq_dim=64, seq_feature_dim=32, num_classes=5, seq_len=32,
                           ns_len=4, d_model=64, num_heads=4, ffn_hidden=128, multi_num=1,
                           mask_type="paper_causal", num_pyramid_layers=2,
                           pyramid_align=32, use_sep_token=True).to("npu").eval()
ns = torch.randn(8, 64, device="npu")
sq = torch.randn(8, 32, 32, device="npu")

# 直接对 SDPA 做三种 mask 的对照，避免整模型噪声
B, H, S, D = 8, 4, 64, 16
q = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device="npu", dtype=torch.bfloat16)
fmask = torch.zeros(S, S, device="npu", dtype=torch.bfloat16)
bmask = torch.triu(torch.ones(S, S, device="npu", dtype=torch.bool), 1)

for tag, mask in (("float_mask", fmask), ("bool_mask", bmask), ("none", None)):
    with torch.no_grad():
        torch.npu.synchronize()
        exp = f"/tmp/prof_sdpa_{tag}"
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],
                                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(exp)):
            for _ in range(3):
                torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
            torch.npu.synchronize()
    print("PROF", tag, exp, flush=True)
print("DONE")
