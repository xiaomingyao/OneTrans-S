#!/usr/bin/env python3
"""probe_fuse_add_silu2.py — 真实尺寸（GB 级）下 add+silu 融合 vs 分开

第一版教训：形状只有 32 MiB 时，算出的"带宽" 4455 GB/s 已超 HBM 峰值 ⇒ 数据落在 cache 里、
计时被下发开销主导，结论无效。必须用真实尺寸（档 B 的 FFN 激活 1.23M×1024 bf16 = 2.5 GB）。

字节账：Add(3) + SiLU(2) = 5 单位；融合 add+silu = 3 单位 ⇒ 理论上限 1.67×。
"""
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

torch.npu.set_device(0)
dev = "npu:0"


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


def bench(fn, iters=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    print("=" * 104)
    print("add+silu：真实尺寸（bf16）· 5 单位 vs 3 单位")
    print("=" * 104)
    for M, N in ((1231872, 1024), (1231872, 256), (983040, 1024)):
        a = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        b = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        y = torch.empty_like(a)
        n = M * N
        GB = n * 2 / 1e9
        sep = torch.nn.functional.silu(a + b)

        ta = bench(lambda: torch.nn.functional.silu(a + b))
        tc = bench(lambda: torch.nn.functional.silu(a + b, inplace=True))
        print(f"\n{M}x{N}  张量 {GB:.2f} GB/个")
        print(f"  A 分开 (Add→SiLU)        {ta:7.3f} ms   有效带宽(5单位) {5*GB/ta*1000:6.0f} GB/s")
        print(f"  C 分开+inplace            {tc:7.3f} ms   有效带宽(5单位) {5*GB/tc*1000:6.0f} GB/s")
        best = None
        for BLOCK in (2048, 4096, 8192, 16384):
            grid = (triton.cdiv(n, BLOCK),)
            def fused(BLOCK=BLOCK, grid=grid):
                _add_silu_kernel[grid](a, b, y, n, BLOCK=BLOCK)
                return y
            try:
                out = fused()
                err = (out.float() - sep.float()).abs().max().item()
                t = bench(fused)
            except Exception as e:  # noqa: BLE001
                print(f"  B 融合 BLOCK={BLOCK:<6d}    FAIL {type(e).__name__}")
                continue
            rel = 3 * GB / t * 1000
            mark = ""
            if best is None or t < best[1]:
                best = (BLOCK, t)
            print(f"  B 融合 BLOCK={BLOCK:<6d}      {t:7.3f} ms   有效带宽(3单位) {rel:6.0f} GB/s   B/A {ta/t:5.2f}x  max|Δ| {err:.2e}")
        if best:
            print(f"  ⇒ best BLOCK={best[0]}  {best[1]:.3f} ms，A/best = {ta/best[1]:.2f}x（理论上限 1.67x）")
        del a, b, y, sep
        torch.npu.empty_cache()

    print()
    print("另测厂商 inplace silu 是否贴峰值（单算子 2 单位）")
    x = torch.randn(1231872, 1024, device=dev, dtype=torch.bfloat16)
    GB = x.numel() * 2 / 1e9
    t = bench(lambda: torch.nn.functional.silu(x, inplace=True))
    print(f"  silu_(1231872x1024)      {t:7.3f} ms   有效带宽(2单位) {2*GB/t*1000:6.0f} GB/s")


if __name__ == "__main__":
    main()
