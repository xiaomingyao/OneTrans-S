#!/usr/bin/env python3
# coding=utf-8
"""打印每个 OneTrans block 的输入长度与 query 长度（确认 pyramid 到底剪了什么）。

实测（S 形状，schedule [1203,960,736,480,256,12]）：
  base 输入 1203；stack0 输入 1203(Lq 1203)；stack1 输入 1203(Lq 960)；stack2 输入 960(Lq 736)…
⇒ 每个 block 内 **K/V = 该层输入的全序列**，只有 query 取尾部 L′ 行；输出长度变成 L′，
  于是下一层的「全长」随深度收缩 —— 就是论文 §3.4 的 "reducing the token length to L′"。
"""
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.getcwd() + "/OneTrans_NPU")
os.environ.setdefault("ONETRANS_ATTN_MASK", "bool")
import torch
from models.taac_onetrans import TAACOneTransClassifier, linear_pyramid_schedule
CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
           d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=6, pyramid_align=32, use_sep_token=True)
m = TAACOneTransClassifier(**CFG).eval()
print("schedule =", linear_pyramid_schedule(1190+12+1, 12, 6, 32))
rows = []
def wrap(mod, tag):
    orig = mod.forward
    def f(x):
        rows.append((tag, int(x.shape[1]), getattr(mod, "pyramid_stack_len", None)))
        return orig(x)
    mod.forward = f
wrap(m.base_block, "base")
for i, b in enumerate(m.stack_blocks):
    wrap(b, f"stack{i}")
with torch.no_grad():
    m(torch.randn(2, 64), torch.randn(2, 1190, 32))
print(f"{'block':8s} {'输入 L':>7s} {'pyramid_stack_len(Lq)':>22s}")
for tag, L, pl in rows:
    print(f"{tag:8s} {L:7d} {str(pl):>22s}")
