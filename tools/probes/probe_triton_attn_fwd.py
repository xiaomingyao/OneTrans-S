#!/usr/bin/env python3
# coding=utf-8
"""Triton-Ascend 版 flash-attention 前向 vs 厂商 FA v4 的实测（单 die）。

形状取本仓真实值：BH = batch×heads，D=64，底右因果掩码（允许 col ≤ row + Lk − Lq）。

写法要点（避开前两版在签名处编译失败、以及 transpose 语义不确定）：
- k 直接按 (D, BN) 形状 load（两个 stride 拼偏移），不做 tl.trans；
- 标量都用 constexpr 传；
- online softmax 用 running max + rescale，末尾除以 l。

判据：Triton fwd ≤ 0.8 × FA fwd 才值得继续写 dq/dk/dv 三个反向 kernel。
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ap = argparse.ArgumentParser()
ap.add_argument("--bh", type=int, default=1024)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--block-m", type=int, default=64)
ap.add_argument("--block-n", type=int, default=128)
a = ap.parse_args()
DEV = "npu"
torch.manual_seed(0)


@triton.jit
def _fa_fwd(Q, K, V, O,
            sqb, sqm, skb, skn, svb, svn, sob, som,
            LQ: tl.constexpr, LK: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    offs_n0 = tl.arange(0, BN)

    m_mask = offs_m < LQ
    q = tl.load(Q + pid_bh * sqb + offs_m[:, None] * sqm + offs_d[None, :], mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    row_allow = offs_m + (LK - LQ)
    n_end = LK
    hi = (pid_m + 1) * BM + (LK - LQ)
    if hi < n_end:
        n_end = hi
    # 【修正】底右因果的语义是「允许 col ≤ row + (LK-LQ)」，所以每一行都能看到**最前面的列**，
    # 要跳的是**尾部**；早先版本把 n_start 设成 pid_m*BM+(LK-LQ) 会漏掉前段贡献（既快又错）。
    n_start = 0

    for start_n in range(n_start, n_end, BN):
        offs_n = start_n + offs_n0
        n_mask = offs_n < LK
        k = tl.load(K + pid_bh * skb + offs_d[:, None] + offs_n[None, :] * skn,
                    mask=n_mask[None, :], other=0.0)          # (D, BN)
        qk = tl.dot(q, k) * SCALE                              # (BM, BN)
        qk = tl.where(offs_n[None, :] <= row_allow[:, None], qk, float("-inf"))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + pid_bh * svb + offs_n[:, None] * svn + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)          # (BN, D)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + pid_bh * sob + offs_m[:, None] * som + offs_d[None, :], acc, mask=m_mask[:, None])


def run_case(LQ, LK, D=64, H=4):
    BH = a.bh
    q = torch.randn(BH, LQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(BH, LK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(BH, LK, D, device=DEV, dtype=torch.bfloat16)
    o = torch.zeros(BH, LQ, D, device=DEV, dtype=torch.bfloat16)
    scale = D ** -0.5

    grid = (triton.cdiv(LQ, a.block_m), BH)
    pargs = (q, k, v, o,
             q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
             o.stride(0), o.stride(1), LQ, LK, D, scale, a.block_m, a.block_n)

    mb = torch.arange(LK, device=DEV).unsqueeze(0) > (torch.arange(LQ, device=DEV).unsqueeze(1) + (LK - LQ))

    def fa():
        with torch.no_grad():
            return torch_npu.npu_fusion_attention(q, k, v, H, "BSH", atten_mask=mb,
                                                  scale=scale, keep_prob=1.0, sparse_mode=1)[0]

    def tri():
        _fa_fwd[grid](*pargs)
        return o

    def bench(fn, warmup=2):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) * 1000 / a.reps

    print(f"\n=== Lq={LQ} Lk={LK} D={D} BH={BH} (BM={a.block_m}, BN={a.block_n})", flush=True)
    try:
        t_tri = bench(tri)
        print(f"  triton fwd            {t_tri:8.3f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  triton FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
        return
    t_fa = bench(fa)
    print(f"  FA v4 fwd（厂商）      {t_fa:8.3f} ms")
    print(f"  => Triton/FA = {t_tri / t_fa:.2f}x（{(t_tri - t_fa) / t_fa * 100:+.1f}%）")

    with torch.no_grad():
        y_t = tri().float()
        y_f = fa().float()
        d = (y_t - y_f).abs()
        print(f"  精度 vs FA: max|Δ|={d.max():.3e} mean={d.mean():.3e}")


for LQ, LK in ((480, 1203), (1203, 1203)):
    run_case(LQ, LK)
