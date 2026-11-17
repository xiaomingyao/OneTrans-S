#!/usr/bin/env python3
"""probe_int8_quant_precision.py — W8A8 的精度代价（int8 GEMM 已实测 802 TF/s = bf16 的 1.99×）

指标口径：反量化回 bf16 后与原始的比较
  mean rel = mean(|Δ| / max(|x|,1e-3))      —— 逐元素相对误差
  p99 rel / max rel
三种量化粒度 × 两种分布（高斯 / 带 10σ 离群），外加「matmul 端到端」误差（这才是训练真正感受到的）。
"""
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"


def q_tensor(x):  # 全张量一个 scale
    s = x.abs().max().item() / 127.0
    xq = torch.round(x / s).clamp(-127, 127).to(torch.int8)
    return xq, s


def q_token(x):  # 每行（每个 token）一个 scale
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    xq = torch.round(x / s).clamp(-127, 127).to(torch.int8)
    return xq, s


def q_block(x, blk=32):  # 每 32 个元素一个 scale（mx 风格）
    M, N = x.shape
    assert N % blk == 0
    xr = x.view(M, N // blk, blk)
    s = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    xq = torch.round(xr / s).clamp(-127, 127).to(torch.int8)
    return xq, s


def report(tag, x, xq, s, xshape):
    sc = s if isinstance(s, float) else s
    back = xq.float() * sc
    back = back.reshape(x.shape).to(torch.bfloat16)
    err = (back.float() - x.float()).abs()
    rel = err / x.float().abs().clamp_min(1e-3)
    flat = rel.flatten()
    p99 = flat.kthvalue(int(flat.numel() * 0.99)).values.item()
    print(f"    {tag:<16s} mean rel {rel.mean().item()*100:6.3f}%   p99 {p99*100:6.2f}%   max {rel.max().item()*100:7.2f}%")


print("=" * 100)
print("[1] 单张量量化误差（bf16 输入 → int8 → 反量化）")
for name, dist in (("高斯 N(0,1)", "g"), ("10σ 离群×2", "o")):
    x = torch.randn(4096, 256, device=dev, dtype=torch.bfloat16)
    if dist == "o":
        x[0, 0] = 10.0
        x[7, 3] = -10.0
    print(f"  分布 {name}:")
    for tag, fn in (("per-tensor(全张量)", q_tensor), ("per-token(每行)", q_token), ("per-32block", q_block)):
        xq, s = fn(x)
        report(tag, x, xq, s, x.shape)

print("\n[2] 端到端 matmul 误差（A: 4096×256 激活, B: 256×1024 权重，量化后 int8 乘 + 反量化）")
A = torch.randn(4096, 256, device=dev, dtype=torch.bfloat16)
W = torch.randn(256, 1024, device=dev, dtype=torch.bfloat16) * 0.0625
ref = (A @ W).float()
for atag, afn in (("per-tensor", q_tensor), ("per-token", q_token), ("per-32block", q_block)):
    Aq, sa = afn(A)
    Wq, sw = q_tensor(W)   # 权重用 per-tensor（通用做法）
    sa_t = sa if isinstance(sa, float) else sa
    if isinstance(sa_t, torch.Tensor) and sa_t.numel() > 1:
        deq = (Aq.float() * sa_t.reshape(-1, 1)) @ (Wq.float() * sw)
    else:
        deq = (Aq.float() * float(sa_t)) @ (Wq.float() * float(sw))
    err = (deq - ref).abs()
    rel = err / ref.abs().clamp_min(1e-2)
    print(f"    A={atag:<12s} mean rel {rel.mean().item()*100:6.3f}%   p99 {rel.flatten().kthvalue(int(rel.numel()*0.99)).values.item()*100:6.2f}%"
          f"   max|Δ| {err.max().item():.4e}")

print("\n[3] 与 bf16 自身误差对比（bf16 matmul vs fp32 参考）")
ref64 = (A.double() @ W.double())
err = ((A @ W).double() - ref64).abs()
rel = (err / ref64.abs().clamp_min(1e-2))
print(f"    bf16           mean rel {rel.mean().item()*100:6.3f}%   "
      f"p99 {rel.flatten().kthvalue(int(rel.numel()*0.99)).values.item()*100:6.2f}%")
