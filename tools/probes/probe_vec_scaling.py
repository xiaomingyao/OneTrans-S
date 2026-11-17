#!/usr/bin/env python3
"""probe_vec_scaling.py — 950DT 向量侧：区分「下发受限」与「带宽受限」

对同一算子扫 1M / 4M / 16M / 64M 元素：
  - 若 µs 不随 N 变化 ⇒ 下发底（~5 µs）
  - 若 GB/s 稳定在 3.2 TB/s 量级 ⇒ 带宽受限
  - 只有两者都不成立才是「算力受限」，此时 Gop/s 才是向量算力
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

dev = "npu:0"


def bench(fn, iters, warmup=3, reps=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters)
    return statistics.median(ts)


print(f"{'op':>6} {'N':>8} {'iter':>6} {'µs/次':>9} {'GB/s':>8} {'Gop/s':>8}")
for op in ("add", "exp"):
    for N, iters in ((1 << 20, 300), (4 << 20, 200), (16 << 20, 100), (64 << 20, 40)):
        x = torch.rand(1, N, device=dev, dtype=torch.float32) + 1.0
        y = torch.rand(1, N, device=dev, dtype=torch.float32) + 1.0
        z = torch.empty_like(x)
        if op == "add":
            fn = lambda: torch.add(x, y, out=z)          # 3N×4 B
            by = N * 4 * 3
        else:
            fn = lambda: torch.exp(x, out=z)             # 2N×4 B
            by = N * 4 * 2
        t = bench(fn, iters)
        print(f"{op:>6} {N/1e6:8.1f}M {iters:6d} {t*1e6:9.2f} {by/t/1e9:8.0f} {N/t/1e9:8.1f}")

print()
print("cube 参照（隔离，大 M）:")
a = torch.randn(8192, 1024, device=dev, dtype=torch.bfloat16)
b = torch.randn(1024, 4096, device=dev, dtype=torch.bfloat16)
t = bench(lambda: a @ b, 100)
fl = 2 * 8192 * 1024 * 4096
print(f"  (8192,1024)@(1024,4096) bf16: {t*1e6:.0f} µs | {fl/t/1e12:.0f} TF/s ({fl/t/1e12/432*100:.1f}% of 432)")
