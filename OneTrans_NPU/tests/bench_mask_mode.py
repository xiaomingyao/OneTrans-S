#!/usr/bin/env python3
# coding=utf-8
"""mask 类型对 950 上 OneTrans 训练步时的影响（float 加性 mask vs bool mask）

用法：python3 tests/bench_mask_mode.py [--batch 64] [--seq-len 64] [--iters 20]
注意：同一进程内两种模式靠环境变量切换，需分别起进程重跑（本脚本自己 spawn）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD = "--child"


def child(args):
    sys.path.insert(0, str(ROOT))
    import torch
    import npu_compat
    from models.taac_onetrans import TAACOneTransClassifier

    npu_compat.init_device("npu")
    torch.manual_seed(0)
    m = TAACOneTransClassifier(non_seq_dim=64, seq_feature_dim=32, num_classes=5,
                               seq_len=args.seq_len, ns_len=4, d_model=128, num_heads=4,
                               ffn_hidden=256, multi_num=2, mask_type="paper_causal",
                               num_pyramid_layers=3, pyramid_align=32, use_sep_token=True).to("npu")
    g = torch.Generator().manual_seed(11)
    ns = torch.randn(args.batch, 64, generator=g).to("npu")
    sq = torch.randn(args.batch, args.seq_len, 32, generator=g).to("npu")
    y = torch.randint(0, 5, (args.batch,), generator=g).to("npu")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        lf(m(ns, sq), y).backward()
        opt.step()
    torch.npu.synchronize()
    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = lf(m(ns, sq), y)
        loss.backward()
        opt.step()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print("CHILD " + json.dumps(dict(mask=os.environ.get("ONETRANS_ATTN_MASK", "float"),
                                     p50=ts[len(ts) // 2], p90=ts[int(len(ts) * 0.9)],
                                     min=ts[0], loss=loss.item())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(CHILD, action="store_true", dest="child")
    a = ap.parse_args()
    if a.child:
        child(a)
        return

    out = []
    for mode in ("float", "bool"):
        env = dict(os.environ, ONETRANS_ATTN_MASK=mode, ASCEND_RT_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
        cmd = [sys.executable, str(Path(__file__)), CHILD, "--batch", str(a.batch),
               "--seq-len", str(a.seq_len), "--iters", str(a.iters)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("CHILD ")]
        if not line:
            print(r.stdout[-800:], r.stderr[-800:])
            raise SystemExit("child failed")
        rec = json.loads(line[0][6:])
        out.append(rec)
        print(f"[{rec['mask']:5s}] p50={rec['p50']:.1f} ms  p90={rec['p90']:.1f}  min={rec['min']:.1f}  loss={rec['loss']:.6f}")
    if len(out) == 2:
        f, b = out[0], out[1]
        print(f"[delta] bool vs float: {(f['p50']-b['p50'])/f['p50']*100:+.1f}%  "
              f"({f['p50']:.1f} -> {b['p50']:.1f} ms)")
        print(f"[loss]  float={f['loss']:.6f} bool={b['loss']:.6f} Δ={abs(f['loss']-b['loss']):.3e}")


if __name__ == "__main__":
    main()
