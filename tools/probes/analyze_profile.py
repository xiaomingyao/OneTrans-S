#!/usr/bin/env python3
# coding=utf-8
"""把 kernel_details.csv 切成瓶颈视图：类别构成 + 碎片化 + 通信遮盖率 + cube 效率。

用法: python3 tools/probes/analyze_profile.py results/prof16_kernel_details.csv --step-ms 1443.7 \
        [--tflops-per-step 64.5722 --peak-tf 432]
输出即 docs/PROFILE_16DIE_BOTTLENECK.md 第 1~2 节的表。
"""
from __future__ import annotations

import argparse
import collections
import csv
import re

CAT = [
    ("cube·FA 融合(含附属转置)", r"FlashAttention"),
    ("cube·MatMul", r"Matmul|Mm_|Addmm"),
    ("vector·逐元素", r"Mul|Add|Sub|Div|Silu|Swish|Pow|Exp|Sqrt|Rsqrt|Sigmoid|Gelu|Neg|Clamp|Abs|Log|Erf"),
    ("vector·归约", r"ReduceSum|ReduceMean|ReduceMax|LogSoftmax"),
    ("layout·Cast", r"Cast"),
    ("layout·切片/拷贝", r"InplaceCopy_(Slice|ViewCopy)|Slice|ViewCopy|Contiguous|Fill|Zeros"),
    ("layout·转置", r"Transpose"),
    ("layout·拼接", r"Cat|Concat"),
    ("norm 融合", r"RMSNorm|RmsNorm|LayerNorm"),
]


def fnum(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def cat_of(name: str, core: str) -> str:
    if core == "COMMUNICATION":
        return "comm"
    for c, pat in CAT:
        if re.search(pat, name, re.I):
            return c
    return "other"


def union_len(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return sum(b - a for a, b in out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--step-ms", type=float, required=True, help="同进程稳态步时（非 profile 窗口）")
    p.add_argument("--tflops-per-step", type=float, default=0.0, help="每步总 TFLOP（算 cube 效率用）")
    p.add_argument("--peak-tf", type=float, default=432.0)
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8", errors="ignore")))
    for r in rows:
        r["dur"] = fnum(r["Duration(us)"])
        r["st"] = fnum(r["Start Time(us)"])
    steps = sorted({r["Step Id"] for r in rows}, key=int)
    n = len(steps)

    print(f"# {a.csv}  steps={steps}  wall={a.step_ms:.1f} ms（同进程稳态）")
    for s in steps:
        rs = [r for r in rows if r["Step Id"] == s]
        comp = [(r["st"], r["st"] + r["dur"]) for r in rs if r["Accelerator Core"] != "COMMUNICATION"]
        comm = [(r["st"], r["st"] + r["dur"]) for r in rs if r["Accelerator Core"] == "COMMUNICATION"]
        t0 = min(r["st"] for r in rs)
        t1 = max(r["st"] + r["dur"] for r in rs)
        wall = (t1 - t0) / 1000
        uc, uk = union_len(comp) / 1000, union_len(comm) / 1000
        hidden = 0.0
        for x, y in comm:
            for c, d in comp:
                lo, hi = max(x, c), min(y, d)
                if hi > lo:
                    hidden += hi - lo
                    break
        ds = sorted(r["dur"] for r in rs)
        print(f"  step {s}: wall={wall:.1f} compute_union={uc:.1f}({100*uc/wall:.1f}%) "
              f"comm_union={uk:.2f}({100*uk/wall:.2f}%) comm_hidden={100*hidden/1000/max(uk,1e-9):.0f}% "
              f"kernels={len(rs)} median={ds[len(ds)//2]:.1f}us <20us={100*sum(1 for d in ds if d < 20)/len(ds):.0f}%")

    agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        c = cat_of(r["Name"], r["Accelerator Core"])
        agg[c][0] += r["dur"] / n
        agg[c][1] += 1 / n
        agg[c][2] += r["dur"] < 20
    tot = sum(v[0] for v in agg.values())
    print(f"\n{'类别':<26}{'ms/步':>10}{'占比':>8}{'kernel/步':>11}{'<20us':>8}")
    for c, (d, k, s) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"{c:<26}{d/1000:>10.1f}{100*d/tot:>7.1f}%{k:>11.0f}{100*s/max(round(k*n),1):>7.0f}%")
    print(f"{'合计':<26}{tot/1000:>10.1f}{100.0:>7.1f}%{sum(v[1] for v in agg.values()):>11.0f}")

    cube = agg["cube·FA 融合(含附属转置)"][0] + agg["cube·MatMul"][0]
    print(f"\ncube 类 {cube/1000:.1f} ms/步（{100*cube/tot:.1f}%）；非 cube {(tot-cube)/1000:.1f} ms/步（{100*(tot-cube)/tot:.1f}%）")
    if a.tflops_per_step > 0:
        theo = a.tflops_per_step * 1e12 / (a.peak_tf * 1e12) * 1e3
        print(f"cube 内效率 = {theo/(cube/1000)*100:.1f}%（理论 {theo:.1f} ms/步 = {a.tflops_per_step} TFLOP ÷ {a.peak_tf:.0f} TF）")


if __name__ == "__main__":
    main()
