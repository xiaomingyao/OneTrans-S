#!/usr/bin/env python3
# coding=utf-8
"""探针：ACL 图捕获（`torch.npu.NPUGraph`）能否把一步训练的时间压下来。

背景：`torch.compile(inductor)` 在本负载上直接失败（NoTritonConfigsError，配置搜索无解），
所以改测「整步捕获 + replay」这条 NPU 原生路径：它不融合算子，但能消掉 host 侧 launch/间隙。

判据：replay 稳态中位数比 eager 低 ≥5% 才值得往 16 卡推。
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
ap.add_argument("--batch", type=int, default=1024)
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


def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast(DEV, dtype=torch.bfloat16):
        loss = crit(model(ns, sq), y)
    loss.backward()
    opt.step()
    return loss


def bench(fn, tag, n):
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


print(f"torch={torch.__version__} batch={a.batch} pyramid_layers={a.pyramid} ckpt=False")
loss_e = step()
print(f"eager 首步 loss={loss_e.item():.5f}")
t_eager = bench(step, "eager", a.steps)

# 多流 warmup：capture 前必须在旁路 stream 上跑过同样的 kernel，避免捕获到首次编译/分配
s = torch.npu.Stream()
s.wait_stream(torch.npu.current_stream())
with torch.npu.stream(s):
    for _ in range(3):
        step()
torch.npu.current_stream().wait_stream(s)
torch.npu.synchronize()

g = torch.npu.NPUGraph()
try:
    with torch.npu.graph(g):
        step()
    print("capture OK")
except Exception as e:
    print("CAPTURE_FAILED:", type(e).__name__, str(e)[:400])
    sys.exit(3)


def replay():
    g.replay()


t_graph = bench(replay, "graph-replay", a.steps)
d = (t_graph - t_eager) / t_eager * 100
print(f"\n结论: eager {t_eager:.1f} ms → replay {t_graph:.1f} ms（{d:+.2f}%）")
print("VERDICT:", "GAIN" if d <= -5 else "NO_GAIN")
