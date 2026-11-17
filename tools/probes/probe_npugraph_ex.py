#!/usr/bin/env python3
# coding=utf-8
"""图模式第二条路：torch.compile(backend="npu") / npugraph_ex（单 die）。

已知：手写 `torch.npu.graph` 捕获**所有 attention 算子**（SDPA 两种 + FA fwd/bwd）都报
`capture_end ... 107025`，而 linear/rms_norm/silu/mean/zeros 全 OK ⇒ 旁路 stream 在 attention 内部，
且不是我们的代码问题。这里验证另一条官方路径：npugraph_ex（`torch.compile(backend="npu")`），
它用 tiling cache + 自己的 capture 驱动，可能绕过该限制。

同时把 NPUGraph 的可用签名打出来，便于判断有没有 capture 模式开关。
"""
import argparse
import inspect
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
for k, v in dict(ONETRANS_ATTN_MASK="bool", ONETRANS_NS_BMM="1", ONETRANS_FUSED_NORM="1",
                 ONETRANS_CAT_INPLACE="1", ONETRANS_SDPA_CAUSAL="1", ONETRANS_FULL_PROJ="1",
                 ONETRANS_FA_BSH="1").items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from torch import nn  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--warmup", type=int, default=3)
a = ap.parse_args()
DEV = "npu"

print("=== A) NPUGraph / graph 相关签名")
for obj, nm in ((torch.npu.NPUGraph, "NPUGraph.__init__"),
                (getattr(torch.npu, "graph", None), "torch.npu.graph")):
    try:
        print(f"  {nm}: {inspect.signature(obj)}")
    except Exception as e:  # noqa: BLE001
        print(f"  {nm}: <{type(e).__name__} {e}>")
print("  torch._dynamo backends:", [b for b in ("npu", "inductor", "eager", "aot_eager")])

CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
           d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1, mask_type="paper_causal",
           num_pyramid_layers=6, pyramid_align=32, use_sep_token=True, use_checkpoint=False)
torch.manual_seed(0)
model = TAACOneTransClassifier(**CFG).to(DEV)
crit = nn.CrossEntropyLoss()
ns = torch.randn(a.batch, CFG["non_seq_dim"], device=DEV)
sq = torch.randn(a.batch, CFG["seq_len"], CFG["seq_feature_dim"], device=DEV)
y = torch.randint(0, 2, (a.batch,), device=DEV)


def bench(fn, tag):
    for _ in range(a.warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(a.steps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(f"  {tag:34s} median {ts[len(ts) // 2]:8.2f} ms  min {ts[0]:8.2f}")
    return ts[len(ts) // 2]


def step_eager():
    with torch.autocast(DEV, dtype=torch.bfloat16):
        loss = crit(model(ns, sq), y)
    loss.backward()
    model.zero_grad(set_to_none=True)


t_eager = bench(step_eager, "eager + autocast(现状)")

print("\n=== B) torch.compile(backend='npu') 整步 fwd+bwd")
for kwargs in (dict(backend="npu"), dict(backend="npu", mode="reduce-overhead")):
    try:
        cm = torch.compile(model, fullgraph=False, dynamic=False, **kwargs)
        t = bench(lambda: cm(ns, sq), f"compile({kwargs}) 只前向")
        print(f"     -> 前向 {t:.2f} ms（eager 前向需另测，仅作可行性判据）")
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()
        print(f"  compile({kwargs}) FAIL {type(e).__name__}: {(msg[0] if msg else '')[:200]}")
