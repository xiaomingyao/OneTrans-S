#!/usr/bin/env python3
# coding=utf-8
"""口径无关的结构自证：pyramid 开关的 TFLOPs 比值（论文 Table 2 vs Table 3: 8.08 / 2.64 = 3.06×）。

我们侧同样数一遍：全 6 层都用满 1190 token（关 pyramid）vs 线性 schedule（开 pyramid）。
"""
import os
import sys

sys.path.insert(0, os.environ["ONETRANS_NPU_DIR"])

import torch
from torch.utils.flop_counter import FlopCounterMode
from models.taac_onetrans import TAACOneTransClassifier

B, D, H, FFN, NS, SEQ, PYRA = 2, 256, 4, 1536, 12, 1190, 6


def build():
    return TAACOneTransClassifier(
        non_seq_dim=64, seq_feature_dim=64, num_classes=1,
        seq_len=SEQ, ns_len=NS, d_model=D, num_heads=H, ffn_hidden=FFN,
        multi_num=1, num_pyramid_layers=PYRA, use_sep_token=True,
    )


def flops(m):
    m.eval()
    non_seq = torch.randn(B, 64)
    seq = torch.randn(B, SEQ, 64)
    fc = FlopCounterMode(display=False)
    with fc:
        out = m(non_seq, seq)
        out.float().pow(2).mean().backward()
    return fc.get_total_flops()


m1 = build()
f_pyramid = flops(m1)

m2 = build()
n_override, shapes = 0, []
for blk in m2.stack_blocks:
    for otb in blk.otb_list:
        if hasattr(otb, "pyramid_stack_len"):
            otb.pyramid_stack_len = NS + SEQ + 1   # 满长：不剪
            n_override += 1
        shapes.append(getattr(otb, "pyramid_stack_len", None))
f_full = flops(m2)

print(f"patched blocks={n_override}  last shapes={shapes[-3:]}")
print(f"pyramid_on  = {f_pyramid/B/1e9:.3f} GFLOP/sample")
print(f"pyramid_off = {f_full/B/1e9:.3f} GFLOP/sample")
print(f"ratio off/on = {f_full/f_pyramid:.2f}x   (论文 Table3/Table2 = 8.08/2.64 = {8.08/2.64:.2f}x)")
