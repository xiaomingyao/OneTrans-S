#!/usr/bin/env python3
# coding=utf-8
"""probe_triton_mixed_mha.py — Triton 融合 MixedMHA（QK^T → 底右因果掩码 → online softmax → PV）vs 厂商 FA v4

v2 修正（v1 的整轮实测**作废**，证据见 docs/OPT_TRITON_MIXED_MHA_ASSESSMENT.md §6）：
  · **双侧下界断言**：Triton ≥ 内存下界×0.5 且 FA ≥ 算力下界×0.5，任一不满足直接 INVALID。
    v1 只保护了 Triton 侧，最终是 FA 报出 0.18 ms/1.518 TFLOP（≈8400 TF/s，峰值 432）才暴露"kernel 根本没执行"。
  · **逐形状独立进程**（`--only N`）：v1 里第一个形状 RuntimeError 之后，后续所有 case 都变成 0.06 ms 的下发时间。
  · 保留：NaN 哨兵（必须被覆盖）、带宽 > 4000 GB/s 判 INVALID、与 FA 对拍数值。
计时：每次 launch 前后各 sync（慢，但不会被 Triton 异步下发骗）。
"""
import argparse
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu"
torch.manual_seed(0)
H, D, B = 4, 64, 1024
BH = B * H
PEAK_GBS = 4000.0
PK = 432e12      # bf16 峰值 TF/s
BW = 3.2e12      # 实测有效带宽 B/s

SHAPES = [
    ("base  ", 1203, 1203),
    ("stack0", 1203, 1203),
    ("stack1", 960, 1203),
    ("stack2", 736, 960),
    ("stack3", 480, 736),
    ("stack4", 256, 480),
    ("stack5", 12, 256),
]
SWEEPS = ((32, 128), (64, 64), (64, 128), (128, 64), (128, 128))


@triton.jit
def _mha_fwd(Q, K, V, O,
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

    row_allow = offs_m + (LK - LQ)          # 底右因果：允许 col ≤ row + (Lk−Lq)
    hi = (pid_m + 1) * BM + (LK - LQ)
    n_end = LK if hi >= LK else hi
    n_start = 0                              # 每行都能看到最前面的列，只能跳尾部

    for start_n in range(n_start, n_end, BN):
        offs_n = start_n + offs_n0
        n_mask = offs_n < LK
        k = tl.load(K + pid_bh * skb + offs_d[:, None] + offs_n[None, :] * skn,
                    mask=n_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * SCALE
        qk = tl.where(offs_n[None, :] <= row_allow[:, None], qk, float("-inf"))
        qk = tl.where(n_mask[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + pid_bh * svb + offs_n[:, None] * svn + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + pid_bh * sob + offs_m[:, None] * som + offs_d[None, :], acc, mask=m_mask[:, None])


def bench_sync(fn, reps=3, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def make(LQ, LK):
    q = torch.randn(B * H, LQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B * H, LK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B * H, LK, D, device=DEV, dtype=torch.bfloat16)
    mb = torch.arange(LK, device=DEV).unsqueeze(0) > (torch.arange(LQ, device=DEV).unsqueeze(1) + (LK - LQ))
    return q, k, v, mb


def run(LQ, LK, BM, BN, tag="", verbose=True):
    q, k, v, mb = make(LQ, LK)
    scale = D ** -0.5
    o = torch.full((B * H, LQ, D), float("nan"), device=DEV, dtype=torch.bfloat16)
    grid = (triton.cdiv(LQ, BM), B * H)
    pargs = (q, k, v, o, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
             v.stride(0), v.stride(1), o.stride(0), o.stride(1), LQ, LK, D, scale, BM, BN)

    def tri():
        _mfa_kernel_launch(pargs, grid)

    def fa():
        with torch.no_grad():
            return torch_npu.npu_fusion_attention(q, k, v, H, "BSH", atten_mask=mb,
                                                  scale=scale, keep_prob=1.0, sparse_mode=1)[0]

    # --- 防伪 1：哨兵必须被覆盖 ---
    try:
        tri()
        torch.npu.synchronize()
        nan_left = int(torch.isnan(o).sum().item())
    except Exception as e:  # noqa: BLE001
        print(f"  [{tag}] Lq={LQ} Lk={LK} BM={BM} BN={BN}  triton FAIL {type(e).__name__}: {str(e).splitlines()[0][:120]}")
        return None
    if nan_left:
        print(f"  [{tag}] Lq={LQ} Lk={LK} BM={BM} BN={BN}  INVALID：输出仍有 {nan_left} 个 NaN ⇒ kernel 没写全")
        return None

    t_tri = bench_sync(tri, reps=3, warmup=1)
    t_fa = bench_sync(fa, reps=3, warmup=1)

    # --- 防伪 2/3：双侧下界 + 带宽 + 数值 ---
    flops = 4 * LQ * LK * D * BH
    gb = (2 * LQ + 2 * LK) * D * 2 * BH / 1e9
    floor_ms = flops / PK * 1000      # 算力下界
    mem_ms = gb / BW * 1000           # 内存下界
    gbs = gb / t_tri * 1000

    bad = []
    if t_tri < 0.5 * mem_ms:
        bad.append(f"triton {t_tri:.3f}<内存下界 {mem_ms:.3f}×0.5")
    if gbs > PEAK_GBS:
        bad.append(f"带宽 {gbs:.0f} GB/s 超峰值")
    if t_fa < 0.5 * floor_ms:
        bad.append(f"FA {t_fa:.3f}<算力下界 {floor_ms:.3f}×0.5")

    with torch.no_grad():
        y_t = o.clone().float()
        y_f = fa().float()
    d = (y_t - y_f).abs()
    if not bool(torch.isfinite(d).all()):
        bad.append("对拍含 NaN/Inf")
    elif d.mean().item() > 5e-3:
        bad.append(f"数值 mean|Δ| {d.mean().item():.3e}")

    if verbose:
        status = ("INVALID: " + "; ".join(bad)) if bad else "OK"
        print(f"  [{tag}] Lq={LQ:5d} Lk={LK:5d} BM={BM:3d} BN={BN:4d} | triton {t_tri:8.3f} ms | "
              f"FA {t_fa:8.3f} ms | 比值 {t_tri / t_fa:5.2f}x | 下界 算力 {floor_ms:6.3f} / 内存 {mem_ms:6.3f} ms | "
              f"max|Δ| {d.max().item():.3e} | {status}", flush=True)
    del q, k, v, o
    torch.npu.empty_cache()
    return None if bad else (t_tri, t_fa)


def _mfa_kernel_launch(pargs, grid):
    _mha_fwd[grid](*pargs)


ap = argparse.ArgumentParser()
ap.add_argument("--only", type=int, default=None, help="只跑第 N 个形状（取证时请外层 shell 循环，逐形状独立进程）")
ap.add_argument("--sweep", action="store_true", help="对 --only 指定的形状扫块尺寸")
ARGS = ap.parse_args()

if ARGS.only is None:
    print("[提示] 未指定 --only：全部形状顺序跑在本进程内。v1 经验：首个形状失败会污染后续所有测量"
          "（后续变成 0.06 ms 下发时间），正式取证请逐形状独立进程。", flush=True)
    print("=" * 132)
    print(f"[1] 7 个 block 形状（B={B}, H={H}, D={D}, BH={BH}, 底右因果）· BM=64 BN=128")
    print("=" * 132)
    tot_t = tot_f = 0.0
    n_ok = 0
    for tag, LQ, LK in SHAPES:
        r = run(LQ, LK, 64, 128, tag)
        if r:
            tot_t += r[0]
            tot_f += r[1]
            n_ok += 1
    if n_ok:
        print(f"\n  有效 {n_ok}/{len(SHAPES)} 个: triton {tot_t:.3f} ms | FA {tot_f:.3f} ms | 比值 {tot_t / tot_f:.2f}x")
    else:
        print("\n  没有任何一个形状通过防伪判据 —— 本轮数据不可用。")
else:
    tag, LQ, LK = SHAPES[ARGS.only]
    if ARGS.sweep:
        print(f"-- sweep Lq={LQ} Lk={LK}", flush=True)
        for BM, BN in SWEEPS:
            run(LQ, LK, BM, BN, "sweep")
    else:
        run(LQ, LK, 64, 128, tag)
