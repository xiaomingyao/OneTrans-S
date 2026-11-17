#!/usr/bin/env python3
"""param_breakdown.py — 把 91,026,946 逐模块拆开（供与论文 Table 2/3 对账）"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

CFG = dict(non_seq_dim=256, seq_feature_dim=64, num_classes=2,
           seq_len=1190, ns_len=12, d_model=256, num_heads=4,
           ffn_hidden=1536, multi_num=1, num_pyramid_layers=6,
           pyramid_align=32, use_sep_token=True, mask_type="paper_causal")

m = TAACOneTransClassifier(**CFG)
tot = sum(p.numel() for p in m.parameters())
print(f"total = {tot:,}")

buckets = {}
for name, p in m.named_parameters():
    key = name.split(".")[0]
    buckets[key] = buckets.get(key, 0) + p.numel()
for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"  {k:<24} {v:>12,}  {v/tot*100:5.2f}%")

# 逐 block 拆分：QKV 组 / FFN 组
print("\n按算子族：")
kqv = sum(p.numel() for n, p in m.named_parameters() if "kqv" in n)
ffn = sum(p.numel() for n, p in m.named_parameters() if "ffn" in n)
print(f"  QKV（13 组）  {kqv:,}")
print(f"  FFN（13 组）  {ffn:,}")
print(f"  其余                    {tot-kqv-ffn:,}")

# 论文 Table 3 Row 4 = 24 M：只共享 FFN（QKV 仍 13 组）
nb = 7
per_kqv13 = sum(p.numel() for n, p in m.named_parameters() if n.startswith("base_block") and "kqv" in n)
per_ffn13 = sum(p.numel() for n, p in m.named_parameters() if n.startswith("base_block") and "ffn" in n)
per_ffn1 = sum(p.numel() for n, p in m.named_parameters()
               if n.startswith("base_block") and "ffn_list.0." in n)
share_ffn_only = tot - nb * (per_ffn13 - per_ffn1)
print(f"\n每 block：QKV13={per_kqv13:,}  FFN13={per_ffn13:,}  FFN1={per_ffn1:,}")
print(f"若只共享 FFN（QKV 保持 13 组）：{share_ffn_only:,}  ← 论文 Table 3 Row4 = 24 M")
