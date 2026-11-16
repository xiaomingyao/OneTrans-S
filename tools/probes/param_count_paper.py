#!/usr/bin/env python3
# coding=utf-8
"""按论文 OneTransS 规格数参数量（CPU，无需 NPU）。

论文锚点（arXiv:2510.26104v3）:
  - §4.1.4: 6 stacked OneTrans blocks, width d=256, H=4 heads, pyramid 1190 -> 12
  - Table 2: OneTransS = 91 M params（§4.1.2 口径: "Params (model parameters excluding sparse embeddings)"）
"""
import itertools
import sys
import os

sys.path.insert(0, os.environ["ONETRANS_NPU_DIR"])

import torch
from models.taac_onetrans import TAACOneTransClassifier


def count(d=256, h=4, ffn=1024, multi=1, pyra=6, ns=12, seq=1190, ns_dim=64, seq_dim=64, sep=True):
    m = TAACOneTransClassifier(
        non_seq_dim=ns_dim, seq_feature_dim=seq_dim, num_classes=1,
        seq_len=seq, ns_len=ns, d_model=d, num_heads=h, ffn_hidden=ffn,
        multi_num=multi, num_pyramid_layers=pyra, use_sep_token=sep,
    )
    return sum(p.numel() for p in m.parameters()), m


print(f"torch={torch.__version__}")
base, m = count()
print(f"\n[paper-shape] d=256 h=4 ffn=1024 multi=1 pyra=6 ns=12 seq=1190 -> {base:,}  ({base/1e6:.1f} M)")
print("n_stage_blocks:", len(m.stack_blocks))

print("\n[grid] 目标 91 M（论文 Table 2），扫 ffn / ns / 层数组合")
rows = []
for ffn, ns, pyra, multi in itertools.product((512, 1024, 1536, 2048), (8, 12, 16, 32), (5, 6), (1, 2)):
    n, _ = count(ffn=ffn, ns=ns, pyra=pyra, multi=multi)
    rows.append((abs(n - 91e6), ffn, ns, pyra, multi, n))
rows.sort()
for d, ffn, ns, pyra, multi, n in rows[:8]:
    print(f"  Δ={d:>10,.0f}  ffn={ffn:<5} ns={ns:<3} pyra={pyra} multi={multi}  params={n:,} ({n/1e6:.1f} M)")

n, _ = count(sep=False)
print(f"\n[no-sep] d=256 ffn=1024 ns=12 seq=1190 -> {n:,} ({n/1e6:.1f} M)")
n, _ = count(seq=1500, ns=16)
print(f"[OneTransL-shape d=384? skip]  seq=1500 ns=16 d=256 -> {n:,} ({n/1e6:.1f} M)")
