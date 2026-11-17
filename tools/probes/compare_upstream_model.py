#!/usr/bin/env python3
"""compare_upstream_model.py — 与上游 GPU 参考（WestbrookLong/OneTrans_Pytorch）结构对账

用法（必须在各自仓库根目录下跑，因为两边的模块名都叫 main_pytorch / models.taac_onetrans）：
    python3 compare_upstream_model.py --dump /tmp/up.json     # 在 GPU_Reference/ 下
    python3 compare_upstream_model.py --dump /tmp/npu.json    # 在 OneTrans_NPU/ 下

输出：参数量、state_dict 的 (key, shape) 全集、以及固定输入下的前向输出指纹。
比对两个 json 即可判定「结构是否一致」（key 集合 + 形状全等 + 前向指纹一致）。
"""
import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from models.taac_onetrans import TAACOneTransClassifier, linear_pyramid_schedule  # noqa: E402

# 与本仓 scripts/run_onetrans_s.py 的 PAPER_S 一致
CFG = dict(non_seq_dim=256, seq_feature_dim=64, num_classes=2,
           seq_len=1190, ns_len=12, d_model=256, num_heads=4,
           ffn_hidden=1536, multi_num=1, num_pyramid_layers=6,
           pyramid_align=32, use_sep_token=True, mask_type="paper_causal")

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
a = ap.parse_args()

torch.manual_seed(0)
m = TAACOneTransClassifier(**CFG)
n_par = sum(p.numel() for p in m.parameters())

schedule = linear_pyramid_schedule(CFG["ns_len"] + CFG["seq_len"] + 1, CFG["ns_len"],
                                  CFG["num_pyramid_layers"], CFG["pyramid_align"])

sd = {k: tuple(v.shape) for k, v in m.state_dict().items()}

# 固定输入前向指纹（weights 由 seed=0 决定，两边应逐位相同）
m.eval()
B = 2
non_seq = torch.randn(B, CFG["non_seq_dim"])
seq = torch.randn(B, CFG["seq_len"], CFG["seq_feature_dim"])
with torch.no_grad():
    out = m(non_seq, seq)
fp = hashlib.md5(out.detach().float().numpy().tobytes()).hexdigest()

json.dump({"params": n_par, "schedule": schedule, "n_tensors": len(sd),
           "state_dict": {k: list(v) for k, v in sd.items()},
           "forward_fp": fp, "out_shape": list(out.shape), "out_absmax": float(out.abs().max())},
          open(a.dump, "w"), indent=1)
print(f"params={n_par:,}  tensors={len(sd)}  schedule={schedule}  forward_md5={fp[:16]}  out={list(out.shape)}")
