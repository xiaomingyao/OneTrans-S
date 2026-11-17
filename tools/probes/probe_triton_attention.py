#!/usr/bin/env python3
# coding=utf-8
"""用 Triton 验证「自写 attention」有没有戏（单 die）。

对照对象是厂商融合算子 `torch_npu.npu_fusion_attention`（BSH，带底右因果掩码），
形状取本仓真实形状：
  A) Lq=480,  Lk=1203（pyramid 层，跨步掩码）
  B) Lq=1203, Lk=1203（全长度层）

只做**前向**：如果连前向都赢不了 FA，自写（还要自己写 dq/dk/dv 三个反向 kernel、再追 131 ms 的反向）
就没有意义。判据：Triton fwd ≤ 0.8 × FA fwd 才继续往反向走。

掩码语义：本仓 paper_causal（底右对齐）= 允许 col ≤ row + (Lk − Lq)。
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ap = argparse.ArgumentParser()
ap.add_argument("--bh", type=int, default=1024, help="batch*heads（单次调用）")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--block-m", type=int, default=64)
ap.add_argument("--block-n", type=int, default=64)
a = ap.parse_args()
DEV = "npu"
torch.manual_seed(0)


@triton.jit
def _fa_fwd(Q, K, V, O, sm_scale,
            LQ: tl.constexpr, LK: tl.constexpr, D: tl.constexpr,
            stride_qb, stride_qm, stride_kb, stride_kn, stride_vb, stride_vn, stride_ob, stride_om,
            BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + pid_bh * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=offs_m[:, None] < LQ, other=0.0)

    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    # 底右因果：允许 col <= row + (LK - LQ)
    row_allow = offs_m + (LK - LQ)
    n_end = LK if not CAUSAL else min(LK, (pid_m + 1) * BM + (LK - LQ))
    for start_n in range(0, n_end, BN):
        offs_n = start_n + tl.arange(0, BN)
        k = tl.load(K + pid_bh * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :],
                    mask=offs_n[:, None] < LK, other=0.0)
        v = tl.load(V + pid_bh * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :],
                    mask=offs_n[:, None] < LK, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if CAUSAL:
            qk = tl.where(offs_n[None, :] <= row_allow[:, None], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < LK, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + pid_bh * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :],
             acc, mask=offs_m[:, None] < LQ)


def run_case(LQ, LK, D=64, H=4, causal=True):
    BH = a.bh
    q = torch.randn(BH, LQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(BH, LK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(BH, LK, D, device=DEV, dtype=torch.bfloat16)
    o = torch.empty(BH, LQ, D, device=DEV, dtype=torch.bfloat16)
    sm = D ** -0.5

    grid = (triton.cdiv(LQ, a.block_m), BH)
    args = (q, k, v, o, sm, LQ, LK, D,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            o.stride(0), o.stride(1), a.block_m, a.block_n, causal)

    def tri():
        _fa_fwd[grid](*args)
        return o

    # FA 对照（BSH）
    qb = q.transpose(0, 1).reshape(LQ, BH, D).transpose(0, 1).contiguous().view(BH, LQ, D)
    kb = k.view(BH, LK, D)
    vb = v.view(BH, LK, D)
    mb = (torch.arange(LK, device=DEV).unsqueeze(0) > (torch.arange(LQ, device=DEV).unsqueeze(1) + (LK - LQ))) if causal else None

    def fa():
        with torch.no_grad():
            return torch_npu.npu_fusion_attention(qb, kb, vb, H, "BSH", atten_mask=mb,
                                                  scale=sm, keep_prob=1.0, sparse_mode=1)[0]

    def bench(fn, tag, warmup=2):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            fn()
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / a.reps
        print(f"  {tag:28s} {ms:8.3f} ms")
        return ms

    print(f"\n=== Lq={LQ} Lk={LK} D={D} BH={BH} causal={causal}")
    try:
        t_tri = bench(tri, f"triton fwd (BM={a.block_m},BN={a.block_n})")
    except Exception as e:  # noqa: BLE001
        print(f"  triton FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return
    t_fa = bench(fa, "FA v4 fwd (厂商融合)")
    print(f"  => Triton/FA = {t_tri / t_fa:.2f}x  ({(t_tri - t_fa) / t_fa * 100:+.1f}%)")

    # 精度（bf16 输出，按 FA 为基准）
    with torch.no_grad():
        y_t = tri().float()
        y_f = fa().float().reshape(BH, LQ, D)
        d = (y_t - y_f).abs()
        print(f"  精度 vs FA: max|Δ|={d.max():.3e} mean={d.mean():.3e} (输出本身 O(1)，bf16 ULP≈7.8e-3)")


for (LQ, LK) in ((480, 1203), (1203, 1203)):
    run_case(LQ, LK)
