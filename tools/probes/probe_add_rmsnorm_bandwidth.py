#!/usr/bin/env python3
# coding=utf-8
"""S1a：确认「单遍融合」到底能多快 —— 厂商 `npu_add_rms_norm` 前向 vs `add + npu_rms_norm`。

判据：若 `npu_add_rms_norm` 明显快（目标 ≤0.9 ms，现状 1.46 ms），说明单遍融合在这台机器上
确实能吃到带宽差 ⇒ 值得为它写自定义反向（厂商融合算子没有 autograd kernel，梯度全 None）。
本探针只测前向，不涉及梯度。
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seq", type=int, default=1203)
ap.add_argument("--dim", type=int, default=256)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
DEV = "npu"
B, L, D = a.batch, a.seq, a.dim
torch.manual_seed(0)

x1 = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
x2 = torch.randn(B, L, D, device=DEV, dtype=torch.bfloat16)
g = torch.randn(D, device=DEV, dtype=torch.float32)
gb = g.to(torch.bfloat16)
nbytes = x1.numel() * 2


def bench(fn, tag, rw, extra=0):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.reps):
        fn()
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / a.reps
    print(f"  {tag:44s} {ms:7.3f} ms  {rw * nbytes / (ms / 1000) / 1e9:7.0f} GB/s")
    return ms


print(f"=== 形状 {B}×{L}×{D} bf16（单张量 {nbytes / 2**20:.0f} MiB），只测前向")

t_add = bench(lambda: x1 + x2, "① Add 残差（读2写1）", 3)
t_sep = bench(lambda: torch_npu.npu_rms_norm(x1 + x2, gb, 1e-6)[0], "② Add + npu_rms_norm 分开（读2+读1写1）", 6)
t_fus = bench(lambda: torch_npu.npu_add_rms_norm(x1, x2, gb, 1e-6), "③ npu_add_rms_norm 融合（理论读2写2）", 4)
print(f"\n  现状②={t_sep:.3f} ms；融合③={t_fus:.3f} ms ⇒ 差 {(t_fus - t_sep) / t_sep * 100:+.1f}%")
print(f"  判据（≤0.9 ms）：{'通过' if t_fus <= 0.9 else '未达'}")

# 精度：融合 vs 分开（输出分别是 (h, y) 与 y）
with torch.no_grad():
    _o = torch_npu.npu_add_rms_norm(x1, x2, gb, 1e-6)
    print(f"  npu_add_rms_norm 返回 {len(_o)} 个: " + ", ".join(f"{tuple(t.shape)}/{t.dtype}" for t in _o))
    h_ref = x1 + x2
    y_ref = torch_npu.npu_rms_norm(h_ref, gb, 1e-6)[0]
    # 顺序按 aclnn 约定应为 (y, rstd, x)；这里不靠约定，逐个与参考对拍确认
    for i, t in enumerate(_o):
        if t.numel() != h_ref.numel():
            continue
        dh = (t.float() - h_ref.float()).abs().max().item()
        dy = (t.float() - y_ref.float()).abs().max().item()
        role = "h(residual)" if dh < dy else "y(norm)"
        print(f"  out[{i}]: vs h max|Δ|={dh:.3e} | vs y max|Δ|={dy:.3e}  ⇒ 判定为 {role}")

# 反向：确认厂商算子确实没有 autograd（不能直接用）
a1 = x1.clone().requires_grad_(True)
a2 = x2.clone().requires_grad_(True)
gg = g.clone().requires_grad_(True)
try:
    out = torch_npu.npu_add_rms_norm(a1, a2, gg, 1e-6)
    out[1].float().sum().backward()
    print(f"  反向检查: g1={'None' if a1.grad is None else 'ok'} g2={'None' if a2.grad is None else 'ok'} "
          f"gg={'None' if gg.grad is None else 'ok'}  ⇒ "
          f"{'无 autograd（必须自己写反向）' if a1.grad is None else '有 autograd'}")
except Exception as e:  # noqa: BLE001
    print(f"  反向检查: 抛错 {type(e).__name__}: {str(e).splitlines()[0][:120]}")
