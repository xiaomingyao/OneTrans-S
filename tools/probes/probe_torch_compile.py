#!/usr/bin/env python3
# coding=utf-8
"""探针：`torch.compile` 能否吃掉非算力部分（同形状 1 die 单测）。

做法：同一模型、同一输入，先量 eager 的稳态步时，再量 compile 后的稳态步时。
判据：能编译通过且稳态步时下降 ≥5% 才值得上 16 卡；否则换成 npugraph_ex 再试。

用法：
  python3 tools/probes/probe_torch_compile.py --device npu --batch 1024 --accum 2 --steps 4 \
      [--backend inductor|npu] [--fullgraph] [--no-ckpt]
"""
import argparse
import os
import statistics
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
os.environ.setdefault("ONETRANS_ATTN_MASK", "bool")
os.environ.setdefault("ONETRANS_NS_BMM", "1")
os.environ.setdefault("ONETRANS_FUSED_NORM", "1")
os.environ.setdefault("ONETRANS_CAT_INPLACE", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--backend", default="inductor")
ap.add_argument("--fullgraph", action="store_true")
ap.add_argument("--no-ckpt", action="store_true")
a = ap.parse_args()
DEV = a.device
if DEV == "npu":
    import torch_npu  # noqa: E402,F401

from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
           d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=3, pyramid_align=32, use_sep_token=True,
           use_checkpoint=not a.no_ckpt)


def build():
    torch.manual_seed(0)
    return TAACOneTransClassifier(**CFG).to(DEV)


def make_data(model_bs):
    g = torch.Generator().manual_seed(7)
    ns = torch.randn(model_bs, CFG["non_seq_dim"], generator=g).to(DEV)
    sq = torch.randn(model_bs, CFG["seq_len"], CFG["seq_feature_dim"], generator=g).to(DEV)
    y = torch.randint(0, 2, (model_bs,), generator=g).to(DEV)
    return ns, sq, y


def bench(tag, steps):
    times = []
    for i in range(a.warmup + steps):
        if DEV == "npu":
            torch.npu.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for k in range(a.accum):
            lo, hi = k * a.batch, (k + 1) * a.batch
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = crit(model(ns[lo:hi], sq[lo:hi]), y[lo:hi]) / a.accum
            loss.backward()
        opt.step()
        if DEV == "npu":
            torch.npu.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= a.warmup:
            times.append(dt)
        print(f"  [{tag} step {i:02d}] {dt:8.1f} ms  loss={loss.item() * a.accum:.5f}")
    med = statistics.median(times)
    print(f"[{tag}] 稳态中位 {med:.1f} ms（n={len(times)}）")
    return med


print(f"torch={torch.__version__} device={DEV} batch={a.batch} accum={a.accum} ckpt={not a.no_ckpt} backend={a.backend}")
model = build()
n_local = a.batch * a.accum
ns, sq, y = make_data(n_local)
crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

t_eager = bench("eager", a.steps)

print(f"\n--- 开始 compile（backend={a.backend}, fullgraph={a.fullgraph}）")
try:
    model = torch.compile(model, backend=a.backend, fullgraph=a.fullgraph)
    t_comp = bench("compile", a.steps)
except Exception as e:
    print("COMPILE_FAILED:", type(e).__name__, str(e)[:500])
    sys.exit(3)

d = (t_comp - t_eager) / t_eager * 100
print(f"\n结论: eager {t_eager:.1f} ms → compile {t_comp:.1f} ms（{d:+.2f}%）")
print("VERDICT:", "GAIN" if d <= -5 else "NO_GAIN")
