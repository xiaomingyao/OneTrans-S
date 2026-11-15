#!/usr/bin/env python3
# coding=utf-8
"""attention mask 语义对照：GPU 参考实现（float 加性）vs NPU 迁移实现（float / bool）。

只有 paper_causal / hard_mask 两种"0 / -inf"型掩码会被 ONETRANS_ATTN_MASK=bool 改写；
origin（+1e-9 偏置）与 bimask_soft（0/1 加性偏置）是软偏置，必须保持 float。

判据：
  * float 路径：两侧逐位相同（max|Δ| == 0，同一份代码、同一种 dtype）；
  * bool 路径：`bool == True` 的集合必须等于 float 路径里「未填 -inf」的集合（即允许集一致）；
  * 软偏置：bool 开关不生效，两侧仍逐位相同。

用法：python3 tests/compare_mask_semantics.py            # 全流程
      python3 tests/compare_mask_semantics.py --side ref|npu_float|npu_bool   # 内部用
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REF_DIR = "/tmp/onetrans_ref"
NPU_DIR = "/tmp/onetrans_npu"
WORK = "/tmp/cmp_ref_npu"
MASK_TYPES = ("paper_causal", "hard_mask", "origin", "bimask_soft")
Q, K = 41, 41          # 序列 32 + sep 1 + non_seq 8 = 41
SHAPE = dict(non_seq_dim=183, seq_feature_dim=45, num_classes=2, seq_len=32, ns_len=8,
             d_model=64, num_heads=4, ffn_hidden=128, multi_num=2,
             num_pyramid_layers=3, pyramid_align=32)


def run_side(side: str) -> None:
    import torch

    sys.path.insert(0, REF_DIR if side == "ref" else NPU_DIR)
    if side == "npu_float" or side == "npu_bool":
        import torch_npu  # noqa: F401
    from main_pytorch import CausalMaskAttention   # mask 生成在 attention 类上，不在 block 上

    out = {}
    for mt in MASK_TYPES:
        attn = CausalMaskAttention(ns_len=SHAPE["ns_len"], d_model=SHAPE["d_model"],
                                   num_heads=SHAPE["num_heads"], mask_type=mt)
        m = attn.create_attention_mask(Q, K, torch.device("cpu"), torch.float32)
        out[mt] = m.cpu()
    tag = "ref" if side == "ref" else side
    torch.save(out, os.path.join(WORK, f"mask_{tag}.pt"))
    print(f"[{tag}] " + ", ".join(f"{k}:dtype={str(v.dtype)}" for k, v in out.items()))


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["ref", "npu_float", "npu_bool"])
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    if a.side:
        run_side(a.side)
        return 0

    base_env = {k: v for k, v in os.environ.items() if k != "ONETRANS_ATTN_MASK"}
    subprocess.run([sys.executable, __file__, "--side", "ref"], check=True, env=base_env)
    subprocess.run([sys.executable, __file__, "--side", "npu_float"], check=True,
                   env={**base_env, "ONETRANS_ATTN_MASK": "float", "ASCEND_RT_VISIBLE_DEVICES": "0"})
    subprocess.run([sys.executable, __file__, "--side", "npu_bool"], check=True,
                   env={**base_env, "ONETRANS_ATTN_MASK": "bool", "ASCEND_RT_VISIBLE_DEVICES": "0"})

    ref = torch.load(os.path.join(WORK, "mask_ref.pt"))
    nf = torch.load(os.path.join(WORK, "mask_npu_float.pt"))
    nb = torch.load(os.path.join(WORK, "mask_npu_bool.pt"))

    print("\n| mask_type | float 两侧 max\\|Δ\\| | bool 两侧 dtype | 允许集一致 | 语义 |")
    print("|---|---|---|---|---|")
    ok = True
    for mt in MASK_TYPES:
        r, f, b = ref[mt], nf[mt], nb[mt]
        d_float = (r - f).abs().max().item()
        hard = mt in ("paper_causal", "hard_mask")
        if hard:
            allowed_float = (r > -1e30)
            same = bool(torch.equal(allowed_float, b.bool()))
            note = "hard（True=参与）"
        else:
            same = bool(torch.equal(f, b))
            note = "软偏置，bool 开关不生效"
        ok = ok and d_float == 0 and same
        print(f"| {mt} | {d_float:.1e} | {b.dtype} | {same} | {note} |")

    # 数值示例：paper_causal 前 3 行前 6 列的允许位
    print("\npaper_causal 允许位（ref float→ 1 表示允许；bool → True 表示参与）：")
    for i in range(3):
        print("  ref ", [int(x) for x in (ref["paper_causal"][i, :6] > -1e30).tolist()],
              " bool", [int(x) for x in nb["paper_causal"][i, :6].tolist()])

    print("\nMASK-SEMANTICS " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
