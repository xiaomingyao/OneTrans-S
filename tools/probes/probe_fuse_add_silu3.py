#!/usr/bin/env python3
"""probe_fuse_add_silu3.py — 修计时：逐次 sync（前两版把 Triton 的异步下发当成 0.03 ms，算出 20 万 GB/s）

判据：任何 > 4000 GB/s 的结果直接标 INVALID（950DT HBM 峰值约 3.2~4.0 TB/s）。
计时方式：每次 launch 前后各 sync 一次，取 7 次中位数 —— 慢但不会被异步骗。
"""
import statistics
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

torch.npu.set_device(0)
dev = "npu:0"
PEAK = 4000.0  # GB/s，超了就是测量错


@triton.jit
def _add_silu_kernel(a_ptr, b_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    a = tl.load(a_ptr + off, mask=mask)
    b = tl.load(b_ptr + off, mask=mask)
    h = a + b
    y = h * tl.sigmoid(h)
    tl.store(y_ptr + off, y, mask=mask)


def bench_sync(fn, iters=7, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main():
    for M, N in ((1231872, 1024), (1231872, 256)):
        a = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        b = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        y = torch.empty_like(a)
        n = M * N
        GB = n * 2 / 1e9
        sep = torch.nn.functional.silu(a + b)
        print(f"\n=== {M}x{N}  张量 {GB:.2f} GB ===")
        ta = bench_sync(lambda: torch.nn.functional.silu(a + b))
        print(f"  A 分开 add→silu     {ta:7.3f} ms  5单位 {5*GB/ta*1000:6.0f} GB/s"
              f"{'  <-- INVALID' if 5*GB/ta*1000 > PEAK else ''}")
        for BLOCK in (4096, 8192, 16384, 32768):
            grid = (triton.cdiv(n, BLOCK),)
            def fused(BLOCK=BLOCK, grid=grid):
                _add_silu_kernel[grid](a, b, y, n, BLOCK=BLOCK)
                return y
            try:
                out = fused()
                torch.npu.synchronize()
                err = (out.float() - sep.float()).abs().max().item()
                t = bench_sync(fused)
            except Exception as e:  # noqa: BLE001
                print(f"  B 融合 BLOCK={BLOCK:<6d} FAIL {type(e).__name__}: {str(e)[:60]}")
                continue
            bw = 3 * GB / t * 1000
            print(f"  B 融合 BLOCK={BLOCK:<6d} {t:7.3f} ms  3单位 {bw:6.0f} GB/s"
                  f"{'  <-- INVALID' if bw > PEAK else ''}  B/A {ta/t:5.2f}x  max|Δ| {err:.2e}")
        del a, b, y, sep
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
