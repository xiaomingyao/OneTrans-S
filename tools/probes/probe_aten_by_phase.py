#!/usr/bin/env python3
# coding=utf-8
"""按阶段统计 aten 算子调用（fwd / bwd / opt），定位 Add、ZerosLike、Cast、Slice 的来源。

不依赖 profiler：用 TorchFunctionMode 在 Python 层计数 + 记录形状，输出「谁在发这些 kernel」。
单 die 即可（每卡形状一致）。

用法：python3 tools/probes/probe_aten_by_phase.py [--device npu] [--steps 2]
"""
import argparse
import collections
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
os.environ.setdefault("ONETRANS_ATTN_MASK", "bool")
os.environ.setdefault("ONETRANS_NS_BMM", "1")
os.environ.setdefault("ONETRANS_FUSED_NORM", "1")
os.environ.setdefault("ONETRANS_CAT_INPLACE", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--accum", type=int, default=2)
a = ap.parse_args()
DEV = a.device
if DEV == "npu":
    import torch_npu  # noqa: E402,F401

from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
           d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=3, pyramid_align=32, use_sep_token=True)

WATCH = ("add", "zero_", "copy_", "_to_copy", "slice", "cat", "mul", "silu", "bmm", "mm", "sum", "rsqrt")


class Counter(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.c: collections.Counter = collections.Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        if any(w in name for w in WATCH):
            shapes = []
            for t in args[:3]:
                if isinstance(t, torch.Tensor):
                    shapes.append("x".join(map(str, t.shape[-3:])))
            self.c[(name, "|".join(shapes))] += 1
        return func(*args, **kwargs)


def dump(counter, title, top=12):
    c = counter.c if isinstance(counter, Counter) else counter
    print(f"\n--- {title}（前 {top}）")
    for (name, shapes), cnt in c.most_common(top):
        print(f"  {cnt:5d}  {name:<14} {shapes[:60]}")


torch.manual_seed(0)
model = TAACOneTransClassifier(**CFG).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
print(f"params tensors = {sum(1 for _ in model.parameters())}  device={DEV} batch={a.batch} accum={a.accum}")
ns = torch.randn(a.batch * a.accum, CFG["non_seq_dim"], device=DEV)
sq = torch.randn(a.batch * a.accum, CFG["seq_len"], CFG["seq_feature_dim"], device=DEV)
y = torch.randint(0, 2, (a.batch * a.accum,), device=DEV)

for step in range(2):
    ph = {"fwd": Counter(), "bwd": Counter(), "opt": Counter()}
    opt.zero_grad(set_to_none=True)
    for k in range(a.accum):
        lo, hi = k * a.batch, (k + 1) * a.batch
        with ph["fwd"], torch.autocast(DEV, dtype=torch.bfloat16):
            loss = crit(model(ns[lo:hi], sq[lo:hi]), y[lo:hi]) / a.accum
        with ph["bwd"]:
            loss.backward()
        torch.npu.synchronize() if DEV == "npu" else None
    with ph["opt"]:
        opt.step()
    if step == 1:
        for name, c in ph.items():
            dump(c, f"{name.upper()} 阶段（1 step = {a.accum} micro-batch）")
        tot = collections.Counter()
        for c in ph.values():
            tot.update(c)
        dump(tot, "整步合计", top=18)
