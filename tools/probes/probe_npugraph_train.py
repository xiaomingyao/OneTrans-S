#!/usr/bin/env python3
# coding=utf-8
"""图模式效果量化（1 die）：分开测「只捕获前向」与「make_graphed_callables 捕获 fwd+bwd」。

前一版整步 capture 失败（内部 stream 未 join / 捕获期 OOM），所以这里拆成两档定位：
  A. 纯前向 no_grad + `torch.npu.NPUGraph` 捕获 replay  —— 看 launch/间隙能省多少
  B. `torch.npu.make_graphed_callables` 捕获 fwd+bwd —— 训练可用的图模式
opt.step 不进图（它本来只占 0.2%）。

判据：任一路径 replay 比 eager 低 ≥5% 才值得往下推。
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
import torch_npu  # noqa: E402,F401
from torch import nn  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--pyramid", type=int, default=6)
a = ap.parse_args()
DEV = "npu"

CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
           d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=a.pyramid, pyramid_align=32,
           use_sep_token=True, use_checkpoint=False)

torch.manual_seed(0)
model = TAACOneTransClassifier(**CFG).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
ns = torch.randn(a.batch, CFG["non_seq_dim"], device=DEV)
sq = torch.randn(a.batch, CFG["seq_len"], CFG["seq_feature_dim"], device=DEV)
y = torch.randint(0, 2, (a.batch,), device=DEV)


def bench(fn, tag, n=None):
    n = n or a.steps
    ts = []
    for i in range(a.warmup + n):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= a.warmup:
            ts.append(dt)
    med = statistics.median(ts)
    print(f"[{tag}] 稳态中位 {med:.1f} ms（n={len(ts)}）")
    return med


print(f"torch={torch.__version__} batch={a.batch} pyramid={a.pyramid}")

# ---------- A. 纯前向 ----------
def fwd():
    with torch.no_grad():
        with torch.autocast(DEV, dtype=torch.bfloat16):
            return model(ns, sq)


t_fwd_eager = bench(fwd, "A/eager-fwd")
try:
    s = torch.npu.Stream()
    s.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(s):
        for _ in range(3):
            fwd()
    torch.npu.current_stream().wait_stream(s)
    torch.npu.synchronize()
    g = torch.npu.NPUGraph()
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.bfloat16):
        with torch.npu.graph(g):
            model(ns, sq)
    t_fwd_graph = bench(lambda: g.replay(), "A/graph-fwd-replay")
    print(f"  A 结论: {t_fwd_eager:.1f} → {t_fwd_graph:.1f} ms（{(t_fwd_graph - t_fwd_eager) / t_fwd_eager * 100:+.2f}%）")
except Exception as e:
    print("A/CAPTURE_FAILED:", type(e).__name__, str(e)[:300])
    t_fwd_graph = None

# ---------- B. fwd+bwd 图 ----------
def full_step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        loss = crit(model(ns, sq), y)
    loss.backward()
    opt.step()
    return loss


t_step_eager = bench(full_step, "B/eager-step")
try:
    torch.npu.synchronize()
    gm = torch.npu.make_graphed_callables(model, (ns, sq), num_warmup_iters=3)
    def step_graph():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            logits = gm(ns, sq)
            loss = crit(logits, y)
        loss.backward()
        opt.step()
        return loss
    loss_g = step_graph()
    print(f"B 首步 loss={loss_g.item():.5f}")
    t_step_graph = bench(step_graph, "B/graph-step")
    print(f"  B 结论: {t_step_eager:.1f} → {t_step_graph:.1f} ms（{(t_step_graph - t_step_eager) / t_step_eager * 100:+.2f}%）")
except Exception as e:
    print("B/CAPTURE_FAILED:", type(e).__name__, str(e)[:400])
