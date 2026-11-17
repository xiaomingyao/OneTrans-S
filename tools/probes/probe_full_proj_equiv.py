#!/usr/bin/env python3
# coding=utf-8
"""G2（全量投影 + 原地覆盖 NS 行）单卡对拍 + 计时。

同一脚本跑两遍（`ONETRANS_FULL_PROJ=0/1`），比较：
  · 每步 loss（全精度打印）
  · 参数梯度摘要（sum / absmax / 与固定随机向量的点积）
  · 稳态步时（median）
判据：loss 与梯度摘要必须**逐位相同**（0 差异）；步时看收益。
"""
import argparse
import json
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
os.environ.setdefault("ONETRANS_SDPA_CAUSAL", "1")

_EXPLICIT_BF16 = os.environ.get("ONETRANS_EXPLICIT_BF16", "0") == "1"
if _EXPLICIT_BF16:
    os.environ.setdefault("ONETRANS_WEIGHT_SHADOW", "1")

import contextlib  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from torch import nn  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402
import main_pytorch as _MP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--timing-steps", type=int, default=5)
ap.add_argument("--pyramid", type=int, default=6)
ap.add_argument("--tag", default="run")
ap.add_argument("--dump", default="")      # 保存拼接后的 fp32 梯度，用于跨进程逐元素对拍
a = ap.parse_args()
DEV = "npu"

CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
           d_model=256, num_heads=4, ffn_hidden=1536, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=a.pyramid, pyramid_align=32,
           use_sep_token=True, use_checkpoint=False)

torch.manual_seed(1234)
model = TAACOneTransClassifier(**CFG).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()
g = torch.Generator(device="cpu").manual_seed(7)
ns = torch.randn(a.batch, CFG["non_seq_dim"], generator=g).to(DEV)
sq = torch.randn(a.batch, CFG["seq_len"], CFG["seq_feature_dim"], generator=g).to(DEV)
y = torch.randint(0, 2, (a.batch,), generator=g).to(DEV)

params = [p for p in model.parameters() if p.requires_grad]
probe = torch.randn(sum(p.numel() for p in params), generator=g).to(DEV)


def grad_digest():
    cat = torch.cat([p.grad.float().flatten() for p in params if p.grad is not None])
    return dict(n=int(cat.numel()), sum=float(cat.sum()), absmax=float(cat.abs().max()),
                dot=float((cat * probe).sum()), nan=bool(torch.isnan(cat).any()))


_ctx = contextlib.nullcontext() if _EXPLICIT_BF16 else torch.autocast(DEV, dtype=torch.bfloat16)
if _EXPLICIT_BF16:
    ns = ns.bfloat16()
    sq = sq.bfloat16()
    model = model.bfloat16() if False else model      # 参数保持 fp32，权重由影子缓存 cast


def one_step():
    opt.zero_grad(set_to_none=True)
    _MP.clear_weight_shadow_cache()      # 权重影子必须每步清（runner 里也是每步末清）
    with _ctx:
        logits = model(ns, sq)
        if _EXPLICIT_BF16:
            logits = logits.float()
        loss = crit(logits, y)
    loss.backward()
    opt.step()
    return loss


losses = []
for i in range(a.steps):
    loss = one_step()
    torch.npu.synchronize()
    losses.append(float(loss))
    print(f"  step{i} loss = {float(loss):.10f}")

dg = grad_digest()
print(f"  grad: n={dg['n']} sum={dg['sum']:.10e} absmax={dg['absmax']:.10e} dot={dg['dot']:.10e} nan={dg['nan']}")

if a.dump:
    cat = torch.cat([p.grad.float().flatten() for p in params if p.grad is not None]).cpu()
    torch.save({"grad": cat,
                "shapes": [tuple(p.shape) for p in params if p.grad is not None]}, a.dump)
    print(f"  dumped {cat.numel()} grads -> {a.dump}")

ts = []
for i in range(a.warmup + a.timing_steps):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    one_step()
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    if i >= a.warmup:
        ts.append(dt)
print(f"  step ms: median {statistics.median(ts):.1f}  min {min(ts):.1f}  max {max(ts):.1f}  n={len(ts)}")
print("RESULT " + json.dumps({"tag": a.tag, "full_proj": os.environ.get("ONETRANS_FULL_PROJ", "0"),
                              "losses": losses, "grad": dg, "median_ms": statistics.median(ts)}))
