#!/usr/bin/env python3
"""probe_fp8_mm2.py — 阶段 3：修 scale 传参 + 补 dW 方向 GEMM + 试 npu_quant_matmul 走 fp8

阶段 2 的两个发现：
  ① `torch._scaled_mm` 的 scale_a/scale_b 必须传 **Tensor**（不是 float）
  ② 隔离测 bf16 全尺寸 GEMM 能跑到 **386~426 TF/s**，而模型里 GEMM 平均只有 277 TF/s
     ⇒ 差距很可能来自 **dW 方向**（M=256, K=1.23M，极小 M），本脚本把它单独测出来。
"""
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"
SC = lambda v: torch.tensor([v], device=dev, dtype=torch.float32)  # noqa: E731


def bench(fn, iters=8, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def q(x, dt=torch.float8_e4m3fn, mx=448.0):
    s = x.abs().max().item() / mx
    return (x / s).to(dt), SC(s)


print("=" * 108)
print("[A] GEMM 两个方向对比（bf16，全尺寸，单 die）")
print(f"{'方向':<34s} {'M':>9s} {'K':>9s} {'N':>5s} {'ms':>8s} {'TF/s':>8s}")
CASES = [
    ("fwd/act-grad (M大)", 1231872, 256, 1024),
    ("weight-grad (M=out)", 256, 1231872, 1024),
    ("fwd/act-grad up", 1231872, 256, 768),
    ("weight-grad qkv", 256, 1231872, 768),
    ("fwd/act-grad down", 1231872, 1024, 256),
    ("weight-grad down", 1024, 1231872, 256),
]
res = {}
for tag, M, K, N in CASES:
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    t = bench(lambda: A @ B)
    tf = 2 * M * K * N / t / 1e9
    print(f"{tag:<34s} {M:9d} {K:9d} {N:5d} {t:8.3f} {tf:8.1f}")
    res[tag] = (M, K, N, t, tf, A, B)
    del A, B
    torch.npu.empty_cache()

print("\n[B] torch._scaled_mm（scale 传 Tensor）")
for tag in ("fwd/act-grad (M大)", "weight-grad (M=out)"):
    M, K, N, t_bf16, tf_bf16, A, B = res[tag]
    flops = 2 * M * K * N
    Aq, sa = q(A)
    Bq, sb = q(B)
    done = False
    for label, b2 in (("Bq", Bq), ("Bq.t()", Bq.t())):
        try:
            out = torch._scaled_mm(Aq, b2, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            t = bench(lambda b2=b2: torch._scaled_mm(Aq, b2, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
            print(f"  {tag:<24s} mat2={label:<7s} {t:7.3f} ms  {flops/t/1e9:7.1f} TF/s  加速 {t_bf16/t:5.2f}x")
            done = True
            break
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:<24s} mat2={label:<7s} FAIL {type(e).__name__}: {str(e)[:90]}")
    if not done:
        print(f"  {tag:<24s} _scaled_mm 不可用")

print("\n[C] torch_npu.npu_quant_matmul 直接吃 fp8（HIF8 路径）")
M, K, N, t_bf16, tf_bf16, A, B = res["fwd/act-grad (M大)"]
Aq, sa = q(A)
Bq, sb = q(B)
flops = 2 * M * K * N
for x1dt, x2dt in ((torch.float8_e4m3fn, torch.float8_e4m3fn), (torch.int8, torch.int8)):
    try:
        y = torch_npu.npu_quant_matmul(Aq, Bq, sb, x1_dtype=x1dt, x2_dtype=x2dt,
                                       output_dtype=torch.bfloat16.name if hasattr(torch.bfloat16, 'name') else None)
        print(f"  dtypes={x1dt}/{x2dt} OK  shape={tuple(y.shape)} dtype={y.dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"  dtypes={x1dt}/{x2dt} FAIL {type(e).__name__}: {str(e)[:150]}")

print("\n[D] fp8 量化精度（真实激活分布：N(0,1) 与 带离群 N(0,1)+10σ 两个离群值）")
for tag, dist in (("N(0,1)", "gauss"), ("离群(2个10σ)", "outlier")):
    x = torch.randn(4096, 256, device=dev, dtype=torch.bfloat16)
    if dist == "outlier":
        x[0, 0] = 10.0
        x[7, 3] = -10.0
    xq, s = q(x)
    back = (xq.float() * s).to(torch.bfloat16)
    err = (back.float() - x.float()).abs()
    rel = (err / x.float().abs().clamp_min(1e-3))
    print(f"  {tag:<14s} max|Δ|={err.max().item():.4e}  mean rel={rel.mean().item():.4f}  "
          f"p99 rel={rel.flatten().kthvalue(int(rel.numel()*0.99)).values.item():.4f}")
