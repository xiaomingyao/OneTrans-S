#!/usr/bin/env python3
# coding=utf-8
"""数上游 TAAC2026 OneTrans 实现的参数量，判断它对应论文哪个规模（91M = OneTransS？）。

用法: python param_count.py            # 打印默认配置 + 网格搜索最接近 91M 的组合
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: F401

from models.taac_onetrans import TAACOneTransClassifier


def count(d_model, heads, ffn, multi, pyra, ns_len, seq_len, ns_dim=64, seq_dim=64):
    m = TAACOneTransClassifier(
        non_seq_dim=ns_dim, seq_feature_dim=seq_dim, num_classes=1,
        seq_len=seq_len, ns_len=ns_len, d_model=d_model, num_heads=heads,
        ffn_hidden=ffn, multi_num=multi, num_pyramid_layers=pyra,
    )
    return sum(p.numel() for p in m.parameters())


def main():
    print("== 上游 README/脚本默认配置 (d128 h4 ffn256 multi4 pyra6 ns4 seq16) ==")
    print(f"  params = {count(128, 4, 256, 4, 6, 4, 16):,}")

    print("\n== 候选放大配置 ==")
    for cfg in [(256, 8, 1024, 4, 6, 8, 64), (256, 8, 1024, 4, 6, 8, 256),
                (384, 8, 1536, 6, 6, 8, 256), (512, 8, 2048, 4, 6, 8, 512)]:
        print(f"  d={cfg[0]:<4} h={cfg[1]} ffn={cfg[2]:<5} multi={cfg[3]} seq={cfg[6]:<5} "
              f"params = {count(*cfg):,}")

    print("\n== 定点扫描（找 91M 对应的 d_model）==")
    print(f"  {'d':>4} {'h':>3} {'ffn':>6} {'multi':>5} {'seq':>5} {'params':>14}")
    for d in (128, 192, 256, 320, 384, 512):
        h = 8 if d % 8 == 0 else 4
        n = count(d, h, 4 * d, 4, 6, 8, 256)
        print(f"  {d:>4} {h:>3} {4*d:>6} {4:>5} {256:>5} {n:>14,}")

if __name__ == "__main__":
    main()
