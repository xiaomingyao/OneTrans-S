#!/usr/bin/env python3
# coding=utf-8
"""E1b 前置探针（单 die）：torch_npu.npu_rms_norm 是否支持反向 + 与手写链的精度/耗时对比。

判据：
  ① fwd/bwd 都能跑通（x.grad、w.grad 非 None）；
  ② 与 fp64 参考比，融合算子的相对误差不大于手写链（否则精度有损，不合）；
  ③ 单算子耗时：融合是否真的比「float→square→mean→rsqrt→mul→cast」短。
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch_npu  # noqa: F401

torch.manual_seed(0)
DEV = "npu"
D, B, T = 256, 64, 1203
EPS = 1e-6

x = torch.randn(B, T, D, device=DEV, dtype=torch.bfloat16)
w = torch.randn(D, device=DEV, dtype=torch.bfloat16)


def manual(x, w):
    xd = x.dtype
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).add(EPS).rsqrt()
    return (xf * rms).to(xd) * w


def fused(x, w):
    return torch_npu.npu_rms_norm(x, w, EPS)[0]


# ① 反向可用性
ok_bwd = {}
for name, fn in (("manual", manual), ("fused", fused)):
    xx = x.clone().requires_grad_(True)
    ww = w.clone().requires_grad_(True)
    try:
        y = fn(xx, ww)
        y.float().pow(2).mean().backward()
        ok_bwd[name] = (xx.grad is not None) and (ww.grad is not None)
        print(f"[{name}] fwd out={y.dtype} bwd_ok={ok_bwd[name]}")
    except Exception as e:  # noqa: BLE001
        ok_bwd[name] = False
        print(f"[{name}] FAIL {type(e).__name__}: {str(e)[:160]}")

# ② 精度（相对 fp64）
with torch.no_grad():
    x64 = x.double()
    ref = x64 * x64.pow(2).mean(-1, keepdim=True).add(EPS).rsqrt() * w.double()
    for name, fn in (("manual", manual), ("fused", fused)):
        try:
            y = fn(x, w).double()
            rel = ((y - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
            rms = ((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            print(f"[{name}] vs fp64: max_rel={rel:.3e} rms_rel={rms:.3e}")
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] 精度不可测: {str(e)[:120]}")

# ③ 耗时
for name, fn in (("manual", manual), ("fused", fused)):
    if not ok_bwd.get(name):
        continue
    with torch.no_grad():
        for _ in range(3):
            fn(x, w)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn(x, w)
        torch.npu.synchronize()
        print(f"[{name}] {(time.perf_counter()-t0)/20*1000:.3f} ms/call (b{B}, seq{T})")
