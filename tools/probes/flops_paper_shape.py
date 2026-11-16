#!/usr/bin/env python3
# coding=utf-8
"""按论文 OneTransS 形状量每样本训练 FLOPs（前向 + loss + 反向），用于与论文 2.64 TFLOPs/2048 对齐。

CPU 上跑 batch=2，输出 per-sample GFLOP（线性外推到 batch 2048 需在报告里标注「线性外推」）。
"""
import os
import sys

sys.path.insert(0, os.environ["ONETRANS_NPU_DIR"])

import torch
from torch.utils.flop_counter import FlopCounterMode
from models.taac_onetrans import TAACOneTransClassifier

B, D, H, FFN, NS, SEQ, PYRA = 2, 256, 4, 1536, 12, 1190, 6

m = TAACOneTransClassifier(
    non_seq_dim=64, seq_feature_dim=64, num_classes=1,
    seq_len=SEQ, ns_len=NS, d_model=D, num_heads=H, ffn_hidden=FFN,
    multi_num=1, num_pyramid_layers=PYRA, use_sep_token=True,
)
m.eval()
n_par = sum(p.numel() for p in m.parameters())

non_seq = torch.randn(B, 64)
seq = torch.randn(B, SEQ, 64)

fc = FlopCounterMode(display=False)
with fc:
    out = m(non_seq, seq)
    loss = out.float().pow(2).mean()
    loss.backward()
tot = fc.get_total_flops()
print(f"params={n_par:,} ({n_par/1e6:.1f} M)  batch={B}")
print(f"train_flops_total={tot/1e12:.6f} TFLOP  per_sample={tot/B/1e9:.4f} GFLOP")
print(f"-> 推算 batch2048 per-GPU = {tot/B*2048/1e12:.3f} TFLOP  (论文 OneTransS: 2.64 TFLOP)")
