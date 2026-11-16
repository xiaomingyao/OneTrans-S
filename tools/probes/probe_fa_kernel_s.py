#!/usr/bin/env python3
# coding=utf-8
"""确认 OneTrans-S 形状下、训练一步里 attention 实际落到哪个昇腾算子（FAv4 融合 or 拆解链）。

判据（从 profiler 的 kernel_details.csv 里数）：
  命中 `FlashAttentionScore` 且带 `V4` ⇒ 走 FAv4 融合算子；
  只看到 `Softmax`/`BatchMatMul`/`Mul` 组合而没有 FlashAttention ⇒ 回退成手工路径。

用法（1 die）：
  ONETRANS_ATTN_MASK=bool python3 tools/probes/probe_fa_kernel_s.py --batch 16 --steps 3
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("ONETRANS_ATTN_MASK", "bool")

ROOT = Path(__file__).resolve().parents[2] / "OneTrans_NPU"
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch_npu  # noqa: E402
import npu_compat  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

P = argparse.ArgumentParser()
P.add_argument("--batch", type=int, default=16)
P.add_argument("--steps", type=int, default=3)
P.add_argument("--out", default="/tmp/prof_fa_s")
a = P.parse_args()

npu_compat.init_device("npu")
torch.manual_seed(0)
m = TAACOneTransClassifier(
    non_seq_dim=256, seq_feature_dim=64, num_classes=2,
    seq_len=1190, ns_len=12, d_model=256, num_heads=4, ffn_hidden=1536,
    multi_num=1, mask_type="paper_causal", num_pyramid_layers=6,
    pyramid_align=32, use_sep_token=True,
).to("npu")
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
crit = torch.nn.CrossEntropyLoss()
ns = torch.randn(a.batch, 256, device="npu")
sq = torch.randn(a.batch, 1190, 64, device="npu")
lb = torch.randint(0, 2, (a.batch,), device="npu")

def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        loss = crit(m(ns, sq), lb)
    loss.backward()
    opt.step()

step()  # 预热（编译器/图缓存）
torch.npu.synchronize()
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(a.out),
):
    for _ in range(a.steps):
        step()
    torch.npu.synchronize()

csvs = glob.glob(os.path.join(a.out, "**", "kernel_details*.csv"), recursive=True)
if not csvs:
    print("NO_CSV", a.out)
    sys.exit(2)
agg: dict[str, list[float]] = defaultdict(list)
for p in csvs:
    with open(p, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            name = (row.get("Name") or row.get("OP Type") or "").strip()
            dur = row.get("Duration(us)") or row.get("Duration") or "0"
            try:
                agg[name].append(float(dur))
            except ValueError:
                pass
total = sum(sum(v) for v in agg.values())
print(f"CSV={csvs[0]}  kernels={sum(len(v) for v in agg.values())}  total_us={total:.1f}")
fa = {k: v for k, v in agg.items() if "FlashAttention" in k}
print("\n[FlashAttention* 命中]")
for k, v in sorted(fa.items(), key=lambda kv: -sum(kv[1])):
    print(f"  {k:<60} n={len(v):<4} sum_us={sum(v):8.1f}")
other = {k: v for k, v in agg.items() if any(s in k for s in ("Softmax", "BatchMatMul", "Mul"))}
print("\n[可能的手工 attention 拆解算子]")
for k, v in sorted(other.items(), key=lambda kv: -sum(kv[1]))[:10]:
    print(f"  {k:<60} n={len(v):<4} sum_us={sum(v):8.1f}")
print("\nTOP10 by 累计耗时")
for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:10]:
    print(f"  {k:<60} n={len(v):<4} sum_us={sum(v):8.1f}")
print("\nVERDICT:", "FAV4=YES" if any("FlashAttentionScoreV4" in k for k in fa) else ("FA_OTHER_ONLY" if fa else "NO_FA"))
