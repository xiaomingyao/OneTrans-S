#!/usr/bin/env python3
# coding=utf-8
"""按调用栈把算子归因到代码位置（需要 profile 时 with_stack=True）。

用法: python3 tools/probes/attribute_by_stack.py <operator_details.csv> [--filter Cast]
输出: 各「用户代码帧(file:line, in func)」的算子时间与次数。
"""
from __future__ import annotations

import argparse
import collections
import csv
import re

FRAME_RE = re.compile(r"(?:File \"([^\"]+)\":(\d+), in (\w+))|(?:(/[^;()\n]+)\((\d+)\): (\w+))")


def user_frames(stack: str):
    out = []
    for m in FRAME_RE.finditer(stack or ""):
        if m.group(1):
            path, line, func = m.group(1), int(m.group(2)), m.group(3)
        else:
            path, line, func = m.group(4), int(m.group(5)), m.group(6)
        if "site-packages" in path or "python3." in path:
            continue
        out.append((path.split("/")[-1], line, func))
    return out


def _report(rows, tag: str) -> None:
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0
    agg = collections.defaultdict(lambda: [0.0, 0])
    tot = 0.0
    for r in rows:
        d = num(r.get("Device Self Duration(us)", "0"))
        tot += d
        k = frame_key(r.get("Call Stack", ""))
        agg[k][0] += d
        agg[k][1] += 1
    print(f"\n=== {tag}: {len(rows)} ops, 合计 {tot/1000:.1f} ms（{tot/3000:.1f} ms/步，3 步）")
    for k, (d, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:6]:
        print(f"  {d/3000:8.1f} ms/步 n={c:5d}  {k}")


def frame_key(stack: str) -> str:
    fr = user_frames(stack)
    if not fr:
        return "(no-user-stack)"
    f = fr[0]
    return f"{f[0]}:{f[1]} in {f[2]}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--filter", default="", help="只统计名字里含该子串的算子")
    p.add_argument("--top", type=int, default=18)
    a = p.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8", errors="ignore")))
    if "," in a.filter or (a.filter and a.filter not in ("", "__all__")):
        pats = [x for x in a.filter.split(",") if x]
        for pat in pats:
            sub = [r for r in rows if pat.lower() in r.get("Name", "").lower()]
            _report(sub, pat)
        return
    has_stack = sum(1 for r in rows if r.get("Call Stack", "").strip())
    print(f"# {a.csv}  ops={len(rows)} 有栈={has_stack} ({100*has_stack/max(len(rows),1):.0f}%)")

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    agg = collections.defaultdict(lambda: [0.0, 0])
    agg2 = collections.defaultdict(lambda: [0.0, 0])
    tot = 0.0
    for r in rows:
        name = r.get("Name", "")
        if a.filter and a.filter.lower() not in name.lower():
            continue
        d = num(r.get("Device Self Duration(us)", "0"))
        tot += d
        fr = user_frames(r.get("Call Stack", ""))
        if fr:
            k = f"{fr[0][0]}:{fr[0][1]} in {fr[0][2]}"
            agg[k][0] += d
            agg[k][1] += 1
        if len(fr) >= 2:
            k2 = f"{fr[1][0]}:{fr[1][1]} in {fr[1][2]} → {fr[0][2]}"
            agg2[k2][0] += d
            agg2[k2][1] += 1

    print(f"\n## 按最内层用户帧（{a.filter or '全部算子'}）合计 {tot/1000:.1f} ms"
          f"（分步均摊 {tot/3000:.1f} ms/步）")
    for k, (d, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"  {d/1000:9.1f} ms  n={c:5d}  {k}")
    print(f"\n## 按第二层用户帧（调用者）")
    for k, (d, c) in sorted(agg2.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"  {d/1000:9.1f} ms  n={c:5d}  {k}")


if __name__ == "__main__":
    main()
