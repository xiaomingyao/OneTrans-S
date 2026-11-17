#!/usr/bin/env python3
"""probe_fuse_add_silu.py — 向量侧「已在峰值带宽」的判据 + add+silu 融合的字节收益上界

背景（16 die profile）：向量侧 ~240 ms/步，其中 Add 57.7（479 次, 119us）、Silu 56.5（fwd 22.5 + bwd 34.0）。
实测单算子带宽：SiLU 3178 GB/s、Add 3222 GB/s、sum(0) 3189~3358 GB/s ⇒ **这些算子本身已经贴 HBM 峰值**，
所以"把小算子写得更快"没空间，唯一出路是**减少搬运字节**（融合）。

热力学账（每个元素 1 单位 = 读或写一次）：
    Add  (读a + 读b + 写h)      = 3 单位
    SiLU (读h + 写y)            = 2 单位
    合计 = 5 单位
    融合 add+silu (读a + 读b + 写y) = **3 单位**  ⇒ 理论 −40%

本探针在真实形状上验证：融合内核是否真能拿到这 40%（用 Triton 写，元素级不需要 AscendC）。
若能，再决定是否落到模型里（Add/SiLU 的配对位置在 FFN 与残差链）。
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
    # silu(h) = h * sigmoid(h)；bf16 域算，与 aten 一致顺序：先乘再加
    y = h * tl.sigmoid(h)
    tl.store(y_ptr + off, y, mask=mask)


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    print("=" * 100)
    print("add+silu 融合 vs 分开：真实形状（bf16，16384×1024 = 32 MiB 起）")
    print("=" * 100)
    print(f"{'shape':>16s} {'单位':>5s} {'A 分开':>9s} {'B 融合':>9s} {'C 分开inplace':>13s} {'B/A':>6s} {'GB/s(A)':>8s} {'GB/s(B)':>8s} {'max|Δ|':>9s}")
    for M, N in ((16384, 1024), (16384, 256), (65536, 256), (4096, 1024)):
        a = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        b = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        y = torch.empty_like(a)
        n = M * N
        grid = (triton.cdiv(n, 8192),)

        def fused():
            _add_silu_kernel[grid](a, b, y, n, BLOCK=8192)
            return y

        ref = fused().clone()
        sep = torch.nn.functional.silu(a + b)
        err = (sep.float() - ref.float()).abs().max().item()

        ta = bench(lambda: torch.nn.functional.silu(a + b))
        tb = bench(fused)

        def sep_inplace():
            h = a + b
            return torch.nn.functional.silu(h, inplace=True)
        tc = bench(sep_inplace)

        units_sep = 5
        bytes_unit = n * 2
        print(f"{f'{M}x{N}':>16s} {units_sep:5d} {ta:9.3f} {tb:9.3f} {tc:13.3f} {ta/tb:5.2f}x "
              f"{units_sep*bytes_unit/ta/1e9*1000:8.0f} {3*bytes_unit/tb/1e9*1000:8.0f} {err:9.2e}")

    print()
    print("结论判据：B/A ≥ 1.25 才算真拿到了『少 2 个字节单位』的收益（理论上限 5/3 = 1.67x）")


if __name__ == "__main__":
    main()
