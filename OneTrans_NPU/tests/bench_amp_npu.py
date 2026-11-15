#!/usr/bin/env python3
# coding=utf-8
"""混合精度 A/B：同一进程、同一模型、同一批数据，交替跑 fp32 / bf16 autocast。

为什么要在同一进程里交替：跨进程比容易把先后顺序、时钟/热状态、编译缓存算成精度收益。
输出每档的稳态中位步时、峰值 HBM 增量，以及 bf16 相对 fp32 的比值。

用法（容器内）：python3 tests/bench_amp_npu.py [--rounds 3] [--steps 10] [--batch 64]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_npu  # noqa: E402
import npu_compat  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

CFG = dict(
    non_seq_dim=64, seq_feature_dim=32, num_classes=5, seq_len=32, ns_len=4,
    d_model=64, num_heads=4, ffn_hidden=128, multi_num=2, mask_type="paper_causal",
    num_pyramid_layers=3, pyramid_align=32, use_sep_token=True,
)


def make_model():
    torch.manual_seed(1234)
    return TAACOneTransClassifier(**CFG)


def batch(bs, seq_len, seed=99):
    g = torch.Generator().manual_seed(seed)
    ns = torch.randn(bs, 64, generator=g)
    sq = torch.randn(bs, seq_len, 32, generator=g)
    w = torch.randn(64, 5, generator=g)
    y = (ns @ w).argmax(dim=1)
    return ns, sq, y


def run_mode(model, data, dtype, steps):
    ns, sq, y = data
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lf = torch.nn.CrossEntropyLoss()
    times = []
    for i in range(steps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if dtype is None:
            loss = lf(model(ns, sq), y)
        else:
            with torch.autocast(device_type="npu", dtype=dtype):
                loss = lf(model(ns, sq), y)
        loss.backward()
        opt.step()
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if i >= 2:          # 前 2 步是图编译/预热
            times.append(dt)
    return statistics.median(times), times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    npu_compat.init_device("npu")
    print(f"[env] {npu_compat.describe('npu')} batch={a.batch} rounds={a.rounds} steps={a.steps}", flush=True)

    ns, sq, y = (t.npu() for t in batch(a.batch, CFG["seq_len"]))
    res = {"fp32": [], "bf16": [], "hf32": []}

    for r in range(a.rounds):
        for tag, dtype in (("fp32", None), ("bf16", torch.bfloat16)):
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats()
            model = make_model().npu()
            base = torch.npu.memory_allocated() / 2**20
            med, _ = run_mode(model, (ns, sq, y), dtype, a.steps)
            peak = torch.npu.max_memory_allocated() / 2**20 - base
            res[tag].append((med, peak))
            print(f"[{tag}] round={r} median_step_ms={med:.1f} peak_hbm_delta_mib={peak:.0f}", flush=True)
            del model

    print("\n| 档位 | 步时中位(ms) | 峰值 HBM 增量(MiB) |", flush=True)
    print("|---|---:|---:|", flush=True)
    for tag in ("fp32", "bf16"):
        med = statistics.median([m for m, _ in res[tag]])
        peak = statistics.median([p for _, p in res[tag]])
        print(f"| {tag} | {med:.1f} | {peak:.0f} |", flush=True)
    fp32_ms = statistics.median([m for m, _ in res["fp32"]])
    bf16_ms = statistics.median([m for m, _ in res["bf16"]])
    print(f"\n[verdict] bf16/fp32 步时比 = {bf16_ms / fp32_ms:.3f} "
          f"({'>' if bf16_ms > fp32_ms else '<'} 1 表示 bf16 {'更慢' if bf16_ms > fp32_ms else '更快'})", flush=True)
    print("AMP-BENCH DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
