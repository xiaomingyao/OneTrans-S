#!/usr/bin/env python3
# coding=utf-8
"""深度 profile 分析：把 kernel_details / operator_details 变成「可优化方向」的证据。

输出：
  1 整步与 kernel 并集（气泡）
  2 类别构成（cube / vector / layout / comm / other）
  3 kernel 尺寸分布（碎片化指纹）
  4 Top N 单算子
  5 aten 级归因（operator_details：哪些 python/aten 算子最贵）
  6 每类目标的理论下限与差距（可选：给 --profile tflops）

用法：
  python3 tools/probes/analyze_profile_deep.py <kernel_details.csv> [--operator op.csv]
        [--step-ms 892.5] [--tflops 16.2] [--top 20]
"""
import argparse
import collections
import csv
import re

CUBE_PAT = re.compile(r"MatMul|Matmul|FlashAttention|FA_|BatchMatMul|Conv|GEMM|Mm\b|Addmm", re.I)
COMM_PAT = re.compile(r"HCCL|hccl|ReduceScatter|AllReduce|Allreduce|AlltoAll|Send|Recv", re.I)
CAST_PAT = re.compile(r"Cast", re.I)
COPY_PAT = re.compile(r"InplaceCopy|ViewCopy|TensorMove|Memcpy|Copy", re.I)
TRANS_PAT = re.compile(r"Transpose|Broadcast|Permute", re.I)
CAT_PAT = re.compile(r"Cat|Concat|Stack|Split|Slice", re.I)
REDUCE_PAT = re.compile(r"Reduce|Sum|Mean|Max|Min|ArgMax|Softmax|RmsNorm", re.I)
ELEM_PAT = re.compile(r"Add|Mul|Sub|Div|Sqrt|Rsqrt|Square|Pow|Silu|Gelu|Exp|Log|Neg|Abs|Clamp|Where|Select|Less|Greater|Eq\b", re.I)


def classify(name: str) -> str:
    if COMM_PAT.search(name):
        return "comm"
    if CAST_PAT.search(name):
        return "layout/cast"
    if COPY_PAT.search(name):
        return "layout/copy"
    if TRANS_PAT.search(name):
        return "layout/transpose"
    if CAT_PAT.search(name):
        return "layout/split-join"
    if CUBE_PAT.search(name):
        return "cube"
    if REDUCE_PAT.search(name):
        return "vector/reduce-norm"
    if ELEM_PAT.search(name):
        return "vector/elementwise"
    return "other"


def load_kernel(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="ignore")))
    steps = {r["Step Id"] for r in rows}
    n = max(len(steps), 1)
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    per_step = collections.defaultdict(float)
    for r in rows:
        try:
            d = float(r["Duration(us)"])
        except (KeyError, ValueError):
            continue
        agg[r["Name"]][0] += d / 1000.0 / n
        agg[r["Name"]][1] += 1.0 / n
        per_step[r["Step Id"]] += d / 1000.0
    return agg, per_step, n


def load_op(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="ignore")))
    steps = {r.get("Step Id") for r in rows}
    n = max(len(steps), 1)
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        key = r.get("OP Name") or r.get("Name") or ""
        try:
            d = float(r.get("Duration(us)") or r.get("Total Duration(us)") or 0)
        except ValueError:
            continue
        agg[key][0] += d / 1000.0 / n
        agg[key][1] += 1.0 / n
    return agg, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel_csv")
    ap.add_argument("--operator", default=None)
    ap.add_argument("--step-ms", type=float, default=None)
    ap.add_argument("--tflops", type=float, default=None, help="每步每卡实际用于统计的 TFLOP（默认按 31.529 GFLOP/sample × 每卡 batch）")
    ap.add_argument("--per-card-batch", type=int, default=None)
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    agg, per_step, n = load_kernel(a.kernel_csv)
    total_kernel = sum(v[0] for v in agg.values())
    total_cnt = sum(v[1] for v in agg.values())
    step_ms = a.step_ms or (sum(per_step.values()) / max(len(per_step), 1))

    print(f"# 深度 profile 分析")
    print(f"profile 文件: {a.kernel_csv}  步数={n}")
    print(f"步内 kernel 并集 = {total_kernel:.1f} ms | 挂钟步时 = {step_ms:.1f} ms | 气泡 = {step_ms - total_kernel:.1f} ms "
          f"({(step_ms - total_kernel) / step_ms * 100:.1f}%)")
    print(f"kernel 总数 = {total_cnt:.0f}/步 | 中位 {total_cnt and 0 or 0}")

    # 2 类别
    cat = collections.defaultdict(lambda: [0.0, 0.0])
    for name, (ms, cnt) in agg.items():
        c = classify(name)
        cat[c][0] += ms
        cat[c][1] += cnt
    print("\n## 2 类别构成")
    print("| 类别 | ms/步 | 占比 | kernel/步 | 均耗时 µs |")
    print("|---|---:|---:|---:|---:|")
    for c, (ms, cnt) in sorted(cat.items(), key=lambda kv: -kv[1][0]):
        print(f"| {c} | {ms:.1f} | {ms / total_kernel * 100:.1f}% | {cnt:.0f} | {ms / max(cnt, 1e-9) * 1000:.1f} |")

    # 3 尺寸分布
    buckets = [(0, 10), (10, 20), (20, 50), (50, 200), (200, 10 ** 9)]
    print("\n## 3 kernel 尺寸分布（碎片化指纹）")
    print("| 区间 µs | kernel/步 | 占比 | ms/步 | 占时比 |")
    print("|---|---:|---:|---:|---:|")
    for lo, hi in buckets:
        sel = [(nm, v) for nm, v in agg.items() if lo <= v[0] * 1000 / max(v[1], 1e-9) < hi]
        cnt = sum(v[1] for _, v in sel)
        ms = sum(v[0] for _, v in sel)
        lab = f"{lo}–{hi}" if hi < 10 ** 9 else f">={lo}"
        print(f"| {lab} | {cnt:.0f} | {cnt / total_cnt * 100:.1f}% | {ms:.1f} | {ms / total_kernel * 100:.1f}% |")

    # 4 top
    print(f"\n## 4 Top {a.top} 单算子")
    print("| 算子 | ms/步 | 次数/步 | 均 µs | 类别 |")
    print("|---|---:|---:|---:|---|")
    for name, (ms, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(f"| `{name[:52]}` | {ms:.1f} | {cnt:.0f} | {ms / max(cnt, 1e-9) * 1000:.1f} | {classify(name)} |")

    # 5 aten
    if a.operator:
        opagg, on = load_op(a.operator)
        otot = sum(v[0] for v in opagg.values())
        print(f"\n## 5 aten 级归因（{a.operator}，{on} 步，合计 {otot:.1f} ms/步）")
        print("| aten 算子 | ms/步 | 次数/步 | 均 µs |")
        print("|---|---:|---:|---:|")
        for name, (ms, cnt) in sorted(opagg.items(), key=lambda kv: -kv[1][0])[: a.top]:
            print(f"| `{name[:56]}` | {ms:.1f} | {cnt:.0f} | {ms / max(cnt, 1e-9) * 1000:.1f} |")

    # 6 FLOPs 视角
    if a.per_card_batch:
        tflop = a.tflops if a.tflops else 31.529 * a.per_card_batch / 1000.0
        cube_ms = cat["cube"][0]
        print("\n## 6 算力视角")
        print(f"每卡每步算力（口径 31.529 GFLOP/sample × {a.per_card_batch}）= **{tflop:.2f} TFLOP**")
        print(f"cube 类 kernel 时间 = {cube_ms:.1f} ms ⇒ 实测 cube 吞吐 = **{tflop / (cube_ms / 1000):.1f} TF/s**"
              f"（峰值 432 TF ⇒ 效率 {tflop / (cube_ms / 1000) / 432 * 100:.1f}%）")
        print(f"整步有效吞吐 = {tflop / (step_ms / 1000):.1f} TF/s ⇒ MFU {tflop / (step_ms / 1000) / 432 * 100:.2f}%")
        non_cube = total_kernel - cube_ms
        print(f"非 cube 时间 = {non_cube:.1f} ms（{non_cube / total_kernel * 100:.1f}%）；"
              f"若把非 cube 全部藏到 cube 后面，步时下限 ≈ {cube_ms:.1f} ms")


if __name__ == "__main__":
    main()
