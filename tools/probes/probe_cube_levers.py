#!/usr/bin/env python3
# coding=utf-8
"""Cube 侧三个候选的可行性探针（单 die）：

1. **qkv 投影合并**：现在是同一输入 x 上连做 3 次 `F.linear(256→256)`；改成一次 `F.linear(256→768)`
   （权重按输出维 concat）⇒ 少 2 次 launch、N 维从 256 变 768，cube 利用率通常更高。
2. **FA 的 `inner_precise`**：0=高精度（默认）/1=高性能，看反向能不能明显加速。
3. **mask dtype**：bool vs uint8（同样的底右因果掩码）。

判据：耗时相对差；精度只做「是否逐位」的定性判断（真正门禁在模型级）。
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seq", type=int, default=1203)
ap.add_argument("--lq", type=int, default=480)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=64)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
DEV = "npu"
import torch_npu  # noqa: E402,F401

torch.manual_seed(0)
B, L, D = a.batch, a.seq, a.dim


def bench(fn, tag, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / a.reps
    print(f"  {tag:44s} {ms:8.3f} ms")
    return ms


print(f"=== 1) qkv 投影合并（B{B}·L{L}·d{D}, bf16）")
x = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
w1 = torch.randn(D, D, device=DEV, dtype=torch.bfloat16)
w2 = torch.randn(D, D, device=DEV, dtype=torch.bfloat16)
w3 = torch.randn(D, D, device=DEV, dtype=torch.bfloat16)
wcat = torch.cat([w1, w2, w3], dim=0).contiguous()          # (768, 256)  ← F.linear 的 weight 是 (out, in)


def three():
    return torch.nn.functional.linear(x, w1), torch.nn.functional.linear(x, w2), torch.nn.functional.linear(x, w3)


def one():
    y = torch.nn.functional.linear(x, wcat)
    return y[:, :, :D], y[:, :, D:2 * D], y[:, :, 2 * D:]


t3 = bench(three, "3 × linear(256→256)")
t1 = bench(one, "1 × linear(256→768)")
print(f"  合并收益 {t1 - t3:+.3f} ms（{(t1 - t3) / t3 * 100:+.1f}%）")
with torch.no_grad():
    y3 = three()
    y1 = one()
    d = max((p.float() - q.float()).abs().max().item() for p, q in zip(y3, y1))
    print(f"  数值一致性 max|Δ| = {d:.3e}（同权重切片 ⇒ 只差 matmul 分块/累加顺序）")

print(f"\n=== 2/3) FA：inner_precise × mask dtype（B{B} H{a.heads} Lq{a.lq} Lk{L} Dh{a.head_dim}）")
q = torch.randn(B, a.lq, a.heads * a.head_dim, device=DEV, dtype=torch.bfloat16)
k = torch.randn(B, L, a.heads * a.head_dim, device=DEV, dtype=torch.bfloat16)
v = torch.randn(B, L, a.heads * a.head_dim, device=DEV, dtype=torch.bfloat16)
row = torch.arange(a.lq, device=DEV).unsqueeze(1)
col = torch.arange(L, device=DEV).unsqueeze(0)
blk = col > (row + (L - a.lq))                    # True = 屏蔽
scale = float(a.head_dim) ** -0.5


def fa(mask, ip, grad=False):
    if not grad:
        def run():
            with torch.no_grad():
                torch_npu.npu_fusion_attention(q, k, v, a.heads, "BSH", atten_mask=mask,
                                               scale=scale, keep_prob=1.0, sparse_mode=1,
                                               inner_precise=ip)
        return run

    def run():
        qq = q.clone().requires_grad_(True)
        kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)
        y = torch_npu.npu_fusion_attention(qq, kk, vv, a.heads, "BSH", atten_mask=mask,
                                           scale=scale, keep_prob=1.0, sparse_mode=1,
                                           inner_precise=ip)[0]
        y.float().sum().backward()
    return run


for mk, tag in ((blk, "bool"), (blk.to(torch.uint8), "uint8")):
    for ip in (0, 1):
        try:
            f = bench(fa(mk, ip), f"fwd   mask={tag} inner_precise={ip}")
        except Exception as e:
            print(f"  fwd mask={tag} ip={ip} FAIL {type(e).__name__} {str(e)[:90]}")
            continue
        try:
            bench(fa(mk, ip, grad=True), f"f+b   mask={tag} inner_precise={ip}")
        except Exception as e:
            print(f"  f+b mask={tag} ip={ip} FAIL {type(e).__name__} {str(e)[:90]}")
