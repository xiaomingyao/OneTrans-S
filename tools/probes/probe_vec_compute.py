#!/usr/bin/env python3
"""probe_vec_compute.py — 测 950DT 单 die 向量侧算力（区分算力受限 / 带宽受限）

三个量：
  A. 大张量 elementwise（HBM 带宽受限）：z = x + y, N=64M fp32，报告 GB/s 与等效 TFLOP/s
  B. cache 驻留 elementwise（算力受限）：N=1M fp32 反复 add，报告 GFLOP/s 与上限
  C. cache 驻留超越函数（SFU 受限）：N=1M fp32 反复 exp，报告 Gexp/s
每组前 3 次 warmup，取 5 次中位；每次 launch 前后 sync。
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401

dev = "npu:0"


def bench(fn, iters=100, warmup=3, reps=5):
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


print("=" * 78)
print("A 大张量 elementwise（HBM 带宽受限）")
N = 64 * 1024 * 1024
x = torch.randn(N, device=dev, dtype=torch.float32)
y = torch.randn(N, device=dev, dtype=torch.float32)
z = torch.empty_like(x)
t = bench(lambda: torch.add(x, y, out=z), iters=20)
bytes_moved = N * 4 * 3          # 读 x + 读 y + 写 z
print(f"  add  fp32 N={N/1e6:.0f}M: {t*1e3:.3f} ms | {bytes_moved/t/1e9:.0f} GB/s | "
      f"等效 {N/t/1e12:.2f} TFLOP/s")

xb = torch.randn(N, device=dev, dtype=torch.bfloat16)
yb = torch.randn(N, device=dev, dtype=torch.bfloat16)
zb = torch.empty_like(xb)
t = bench(lambda: torch.add(xb, yb, out=zb), iters=20)
print(f"  add  bf16 N={N/1e6:.0f}M: {t*1e3:.3f} ms | {N*2*3/t/1e9:.0f} GB/s | "
      f"等效 {N/t/1e12:.2f} TFLOP/s")

print("=" * 78)
print("B cache 驻留 elementwise（算力受限，数据 4 MB 常驻）")
M = 1024 * 1024
xm = torch.randn(M, device=dev, dtype=torch.float32)
ym = torch.randn(M, device=dev, dtype=torch.float32)
zm = torch.empty_like(xm)
t = bench(lambda: torch.mul(xm, ym, out=zm), iters=500)
print(f"  mul  fp32 N={M/1e6:.0f}M: {t*1e6:.2f} µs/次 | {M/t/1e9:.0f} GFLOP/s "
      f"(若 500 次全 cache 命中则视为向量算力上限)")
t2 = bench(lambda: torch.mul(xm, ym, out=zm), iters=2000)
print(f"  同上 iters=2000: {t2*1e6:.2f} µs/次 | {M/t2/1e9:.0f} GFLOP/s")

print("=" * 78)
print("C cache 驻留超越函数（SFU 受限）")
t = bench(lambda: torch.exp(xm, out=zm), iters=500)
print(f"  exp  fp32 N={M/1e6:.0f}M: {t*1e6:.2f} µs/次 | {M/t/1e9:.1f} Gexp/s")
t = bench(lambda: torch.rsqrt(xm, out=zm), iters=500)
print(f"  rsqrt fp32 N={M/1e6:.0f}M: {t*1e6:.2f} µs/次 | {M/t/1e9:.1f} Grsqrt/s")

print("=" * 78)
print("D 小 M 矩阵乘（cube 侧参照，隔离）")
a = torch.randn(4096, 256, device=dev, dtype=torch.bfloat16)
b = torch.randn(256, 1024, device=dev, dtype=torch.bfloat16)
t = bench(lambda: a @ b, iters=200)
fl = 2 * 4096 * 256 * 1024
print(f"  (4096,256)@(256,1024) bf16: {t*1e6:.1f} µs | {fl/t/1e12:.1f} TF/s")
