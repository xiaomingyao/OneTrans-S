#!/usr/bin/env python3
"""probe_bias_grad_reduce2.py — 为什么 profile 里 ReduceSum 是 1.17 ms 而隔离测 0.196 ms?

第一版结论（已得）：contiguous 上 sum(0) 已贴峰值（630 MB / 0.196 ms = 3.2 TB/s），
  ones@dy 与 addmm 累加都没有收益（1.00x / 0.97x）⇒ 「用 mm 替换 sum」这条路**不成立**。

本版要回答的是：1.17 ms 的那个 ReduceSum 到底是什么形状/布局。
候选解释：
  H1 dy 更宽：FFN up 投影 out=1024、qkv out=768 ⇒ dy 是 (M,1024)/(M,768) 而非 (M,256)
  H2 dy 非连续：反向里 dy 常是 view/permute/narrow 的结果
  H3 sum(0) 带 dtype=fp32 提升（autograd 把 bf16 梯度累加进 fp32 bias）
"""
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"
B, L, I = 1024, 1203, 256
M = B * L


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


print("=" * 92)
print(f"M = {M} 行 (batch {B} × L {L})")
print("=" * 92)
print(f"{'case':<44s} {'GB流量':>8s} {'ms':>8s} {'GB/s':>9s}")

cases = []
for O in (256, 768, 1024):
    cases.append((f"contig (M,{O}) bf16  sum(0)", O, "contig", torch.bfloat16))
cases.append(("contig (M,1024) fp32 sum(0)", 1024, "contig", torch.float32))
cases.append(("contig (M,1024) bf16 sum(0,dtype=fp32)", 1024, "contig_f32out", torch.bfloat16))
cases.append(("non-contig .t().t() (M,1024) bf16", 1024, "nc_tt", torch.bfloat16))
cases.append(("view (B,L,1024).sum((0,1)) bf16", 1024, "3d", torch.bfloat16))

for name, O, kind, dt in cases:
    if kind == "3d":
        x = torch.randn(B, L, O, device=dev, dtype=dt).abs()
        fn = lambda x=x: x.sum(dim=(0, 1))
        flow = x.numel() * x.element_size() / 1e9
    else:
        base = torch.randn(M, O, device=dev, dtype=dt).abs()
        if kind == "contig":
            x = base; fn = lambda x=x: x.sum(0)
        elif kind == "contig_f32out":
            x = base; fn = lambda x=x: x.sum(0, dtype=torch.float32)
        else:
            x = base.t().t()  # 保持非连续? .t().t() 会恢复连续
            x = base.t().contiguous().t()  # 真正 stride 转置
            fn = lambda x=x: x.sum(0)
        flow = base.numel() * base.element_size() / 1e9
    ms = bench(fn)
    print(f"{name:<44s} {flow:8.2f} {ms:8.3f} {flow/ms*1000:9.0f}")

print()
print("对照：把 wide dy 的 bias 梯度改成 'ones@dy'（cube）是否更快")
for O in (768, 1024):
    x = torch.randn(M, O, device=dev, dtype=torch.bfloat16).abs()
    ones = torch.ones(1, M, device=dev, dtype=torch.bfloat16)
    a = bench(lambda: x.sum(0))
    b = bench(lambda: torch.mm(ones, x))
    flow = x.numel() * 2 / 1e9
    print(f"  O={O:5d}  sum(0) {a:.3f} ms ({flow/a*1000:.0f} GB/s) | mm(ones) {b:.3f} ms ({flow/b*1000:.0f} GB/s) | {a/b:.2f}x")
