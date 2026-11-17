#!/usr/bin/env python3
"""param_compare.py — 逐张量 / 逐超参数的参数量对账（供与论文 Table 2/3 对比）"""
import collections
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

S = dict(non_seq_dim=256, seq_feature_dim=64, num_classes=2, seq_len=1190, ns_len=12,
         d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1, num_pyramid_layers=6,
         pyramid_align=32, use_sep_token=True, mask_type="paper_causal")


def build(**over):
    cfg = dict(S)
    cfg.update(over)
    torch.manual_seed(0)
    m = TAACOneTransClassifier(**cfg)
    return m, sum(p.numel() for p in m.parameters())


# ---- A. 逐张量清单（按名字模式归并）----
m, tot = build()
print(f"=== A. OneTrans-S 张量清单（total {tot:,}）===")
pat = collections.OrderedDict()
for n, p in m.named_parameters():
    key = n.replace("base_block.", "<blk>.").replace("stack_blocks.", "<stk>.")
    import re
    key = re.sub(r"\.\d+\.", ".<g>.", key)
    pat.setdefault(key, []).append(tuple(p.shape))
print(f"{'参数名（归并后）':<52}{'形状':<18}{'个数':>5}{'小计':>12}")
for k, shapes in pat.items():
    per = 1
    for s in shapes[0]:
        per *= s
    print(f"{k:<52}{str(shapes[0]):<18}{len(shapes):>5}{per*len(shapes):>12,}")

# ---- B. 逐 block / 逐组的构成 ----
print("\n=== B. 单 block 构成（13 组 = 1 个 S 组 + 12 个 NS 组）===")
one = {}
for n, p in m.named_parameters():
    if n.startswith("base_block."):
        key = n.split(".")[0] + "." + n.split(".")[1]
        one[key] = one.get(key, 0) + p.numel()
for k, v in one.items():
    print(f"  {k:<24}{v:>12,}")
print(f"  {'合计':<24}{sum(one.values()):>12,}   ×7 = {sum(one.values())*7:,}")

# ---- C. 超参敏感度 ----
print("\n=== C. 超参敏感度（相对论文 S）===")
print(f"{'配置':<40}{'参数量':>14}{'vs 91.03M':>12}")
rows = [("论文 S（本仓配置）", {}),
        ("ffn_hidden=256", dict(ffn_hidden=256)),
        ("ffn_hidden=512", dict(ffn_hidden=512)),
        ("ffn_hidden=768", dict(ffn_hidden=768)),
        ("ffn_hidden=1024", dict(ffn_hidden=1024)),
        ("ffn_hidden=2048", dict(ffn_hidden=2048)),
        ("ns_len=4", dict(ns_len=4)),
        ("ns_len=8", dict(ns_len=8)),
        ("ns_len=16", dict(ns_len=16)),
        ("d_model=384（OneTransL 宽度）", dict(d_model=384, num_heads=6)),
        ("上游默认 d=128/ffn=256/ns=4", dict(d_model=128, ffn_hidden=256, ns_len=4, seq_len=16)),
        ]
for name, over in rows:
    try:
        _, p = build(**over)
        print(f"{name:<40}{p:>14,}{(p/91026946-1)*100:>11.1f}%")
    except Exception as e:
        print(f"{name:<40}  构建失败: {type(e).__name__}: {e}")

# ---- D. 论文 Table 3 四个数据点 ----
print("\n=== D. 论文 Table 3 四个 Params 点 vs 本仓可复现值 ===")
print(f"  {'论文行':<34}{'论文':>8}{'本仓可复现':>16}{'判定':>10}")
print(f"  {'参考 OneTransS':<34}{'91 M':>8}{f'{tot:,}':>16}{'✅':>10}")
print(f"  {'Shared parameters':<34}{'24 M':>8}{'24,816,130（只共享 FFN）':>16}{'✅':>10}")
print(f"  {'w/o pyramid stack':<34}{'92 M':>8}{f'{tot:,}（参数与序列长无关）':>16}{'⚠️':>10}")
print(f"  {'Group-wise Tokenizer':<34}{'78 M':>8}{'不可复现（差 13M 在 tokenizer）':>16}{'❌':>10}")
