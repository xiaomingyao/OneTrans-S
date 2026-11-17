#!/usr/bin/env python3
# coding=utf-8
"""CPU 侧 aten 归因（带 shape + 栈）：torch.profiler 只开 CPU activity，
NPU 算子也会在 CPU 侧留下 aten 记录，且 `group_by_input_shape` 能给出张量形状。
用来回答「588 次 zero_ / 373 次 add_ / 158 次 sum 是谁发的、多大」。
"""
import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
for k, v in dict(ONETRANS_ATTN_MASK="bool", ONETRANS_NS_BMM="1", ONETRANS_FUSED_NORM="1",
                 ONETRANS_CAT_INPLACE="1", ONETRANS_SDPA_CAUSAL="1", ONETRANS_FULL_PROJ="1",
                 ONETRANS_FA_BSH="1").items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--pyramid", type=int, default=6)
ap.add_argument("--top", type=int, default=14)
a = ap.parse_args()
DEV = "npu"
import torch_npu  # noqa: E402,F401
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

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

# warmup
opt.zero_grad(set_to_none=True)
with torch.autocast(DEV, dtype=torch.bfloat16):
    crit(model(ns, sq), y).backward()
opt.step()
torch.npu.synchronize()

opt.zero_grad(set_to_none=True)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=False) as prof:
    with torch.autocast(DEV, dtype=torch.bfloat16):
        loss = crit(model(ns, sq), y)
    loss.backward()
    opt.step()
torch.npu.synchronize()

for name in ("aten::zero_", "aten::zeros", "aten::add_", "aten::add", "aten::sum", "aten::_to_copy",
             "aten::slice_backward", "aten::copy_"):
    ev = [e for e in prof.events() if str(e.key).startswith(name)]
    if not ev:
        print(f"\n{name}: 无")
        continue
    shapes = {}
    for e in ev:
        sh = str(getattr(e, "input_shapes", ""))
        shapes[sh] = shapes.get(sh, 0) + 1
    print(f"\n{name}: {len(ev)} 次, 不同形状 {len(shapes)} 种")
    for sh, c in sorted(shapes.items(), key=lambda kv: -kv[1])[:a.top]:
        print(f"  {c:5d}  {sh[:150]}")
