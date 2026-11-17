#!/usr/bin/env python3
# coding=utf-8
"""Python 层归因探针：谁在发 sum / zeros / zero_ / add_ / slice / _to_copy（带调用点）。

用途：profiler 导出的 Call Stack 列在 torch_npu 2.10 上恒为空（`--profile-with-stack` 也不生效），
所以改用 TorchFunctionMode 在 Python API 层计数 + 记录栈。

判据：Python 层计数 ≈ profiler 计数 ⇒ 来源在模型/训练脚本里，可直接改；
Python 层计数 << profiler 计数 ⇒ 其余来自 C++ autograd 内部（CopySlices / slice_backward / DDP），只能靠改图结构消除。
"""
import argparse
import collections
import os
import re
import sys
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
os.environ.setdefault("ONETRANS_ATTN_MASK", "bool")
os.environ.setdefault("ONETRANS_NS_BMM", "1")
os.environ.setdefault("ONETRANS_FUSED_NORM", "1")
os.environ.setdefault("ONETRANS_CAT_INPLACE", "1")
os.environ.setdefault("ONETRANS_SDPA_CAUSAL", "1")
os.environ.setdefault("ONETRANS_FULL_PROJ", "1")
os.environ.setdefault("ONETRANS_FA_BSH", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--pyramid", type=int, default=6)
a = ap.parse_args()
DEV = "npu"
import torch_npu  # noqa: E402,F401
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

WATCH = ("sum", "mean", "zeros", "zero_", "add_", "slice", "narrow", "_to_copy", "to", "copy_", "cat", "view", "contiguous")
SKIP = ("zeros_like", "new_zeros")


def site():
    frames = traceback.extract_stack()[:-2]
    ours = [f for f in frames if "OneTrans_NPU" in f.filename]
    f = ours[-1] if ours else (frames[-1] if frames else None)
    if f is None:
        return "?"
    return f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"


class Probe(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.c = collections.Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        if name in WATCH and not any(s == name for s in SKIP):
            shape = ""
            for t in list(args)[:1]:
                if isinstance(t, torch.Tensor):
                    shape = "x".join(map(str, t.shape[-3:]))
            self.c[(name, site(), shape)] += 1
        return func(*args, **kwargs)


CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
           d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=a.pyramid, pyramid_align=32,
           use_sep_token=True, use_checkpoint=False)
torch.manual_seed(0)
model = TAACOneTransClassifier(**CFG).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
ns = torch.randn(a.batch, 64, device=DEV)
sq = torch.randn(a.batch, 1190, 32, device=DEV)
y = torch.randint(0, 2, (a.batch,), device=DEV)

for i in range(a.steps):
    p = Probe()
    opt.zero_grad(set_to_none=True)
    with p, torch.autocast(DEV, dtype=torch.bfloat16):
        loss = crit(model(ns, sq), y)
    loss.backward()
    torch.npu.synchronize()
    if i == a.steps - 1:
        tot = collections.Counter()
        for (n, s, sh), c in p.c.items():
            tot[n] += c
        print(f"\n=== 每步 Python 层调用总数（batch={a.batch}）")
        for n, c in tot.most_common():
            print(f"  {n:10s} {c:6d}")
        print("\n=== 明细（top 30）")
        for (n, s, sh), c in p.c.most_common(30):
            print(f"  {c:5d}  {n:10s} {s:34s} {sh}")
