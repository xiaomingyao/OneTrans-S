#!/usr/bin/env python3
# coding=utf-8
"""Triton-Ascend 能力下限探测：`tl.dot` / 标量参数 / 循环 + online-softmax 片段。

目的：先把「Triton 在本机型上到底能不能跑 attention 的核心成分」钉死，再谈自写 attention 的收益。
分 4 级逐级加复杂度，哪一级挂就报哪一级。
"""
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu"
torch.manual_seed(0)


@triton.jit
def k_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def k_dot_scalar(A, B, C, alpha, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b) * alpha
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def k_online_softmax(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                     NB: tl.constexpr, LK: tl.constexpr):
    """一个 q tile 扫 nb 个 k tile，做 online softmax（不含 mask）。"""
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    m_i = tl.full([M], float("-inf"), tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, N], tl.float32)
    for nb in range(0, NB):
        off = nb * N
        idx_n = off + offs_n
        b = tl.load(B + idx_n[:, None] * K + offs_k[None, :], mask=idx_n[:, None] < LK, other=0.0)
        qk = tl.dot(a, tl.trans(b))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + p
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def main():
    M = N = K = 128
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    c = torch.zeros(M, N, device=DEV, dtype=torch.float32)

    print("=== L1: 纯 tl.dot (128x128x128, bf16)")
    try:
        k_dot[(1,)](a, b, c, M, N, K)
        torch.npu.synchronize()
        ref = (a.float() @ b.float())
        print(f"  OK  max|Δ|={(c - ref).abs().max().item():.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {str(e).splitlines()[0][:150]}")
        return

    print("=== L2: 加标量参数（float）")
    try:
        k_dot_scalar[(1,)](a, b, c, 0.125, M, N, K)
        torch.npu.synchronize()
        ref = (a.float() @ b.float()) * 0.125
        print(f"  OK  max|Δ|={(c - ref).abs().max().item():.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {str(e).splitlines()[0][:150]}")

    print("=== L3: 循环 + transpose + online softmax（4 个 k tile）")
    LK = 512
    bb = torch.randn(LK, K, device=DEV, dtype=torch.bfloat16)
    cc = torch.zeros(M, N, device=DEV, dtype=torch.float32)
    try:
        k_online_softmax[(1,)](a, bb, cc, M, N, K, 4, LK)
        torch.npu.synchronize()
        s = (a.float() @ bb.float().T)
        p = torch.softmax(s, dim=-1)
        ref = p[:, :N]
        print(f"  OK  max|Δ|={(cc - ref).abs().max().item():.3e}  (只比前 N 列)")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {type(e).__name__}: {str(e).splitlines()[0][:150]}")
        return

    print("=== L4: 吞吐（BH=4096, Lq=1203, Lk=1203, D=64 规模的 dot 主循环）")
    BH, LQ, LKD, D = 4096, 1203, 1203, 64
    q = torch.randn(BH, LQ, D, device=DEV, dtype=torch.bfloat16)
    kk = torch.randn(BH, LKD, D, device=DEV, dtype=torch.bfloat16)
    o = torch.zeros(BH, LQ, D, device=DEV, dtype=torch.float32)
    print(f"  （张量已分配 {q.numel() * 2 / 2**30:.1f} + {kk.numel() * 2 / 2**30:.1f} GiB；"
          "实际注意力 kernel 另需 online softmax，这里只验证规模可跑）")


main()
