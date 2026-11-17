#!/usr/bin/env python3
"""probe_fp8_mm.py — 阶段 2：fp8 matmul 的正确入口 + 真实形状速度/精度

阶段 1 已知：
  · float8_e4m3fn / e5m2 在 950DT 上可创建（element_size=1）；*fnuz 不支持
  · torch.mm 直接吃 fp8 → 161002（不支持）
  · 但 torch_npu 有一整套量化栈：npu_quant_matmul / npu_quantize / npu_dynamic_quant /
    npu_anti_quant / npu_quant_fusion_attention(+backward) / npu_rms_norm_quant / npu_swiglu_quant …
本阶段：① 打 schema 找正确签名；② 真实形状（FFN up/down、qkv、out）测 TF/s 对比 bf16；③ 精度。
"""
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def show_schema(name):
    try:
        op = getattr(torch.ops.npu, name)
        print(f"  {name}{op.default._schema}")
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: 取 schema 失败 {type(e).__name__}: {str(e)[:100]}")


print("=" * 110)
print("[A] 关键算子 schema")
for n in ("npu_quant_matmul", "npu_quantize", "npu_dynamic_quant", "npu_anti_quant",
          "npu_add_quant_matmul", "npu_quant_fusion_attention", "npu_rms_norm_quant"):
    show_schema(n)

# ---------------- _scaled_mm 正确用法 ----------------
print("\n[B] torch._scaled_mm 布局试错（mat2 给 (K,N) 而不是 (N,K)）")
M, K, N = 8192, 256, 1024
A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
sa = A.abs().max().item() / 448.0
sb = B.abs().max().item() / 448.0
Aq = (A / sa).to(torch.float8_e4m3fn)
Bq = (B / sb).to(torch.float8_e4m3fn)
ref = (A @ B).float()
ok_layout = None
for label, b2 in (("(K,N)", Bq), ("(K,N).t()", Bq.t())):
    try:
        out = torch._scaled_mm(Aq, b2, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        deq = out.float()
        err = (deq - ref).abs()
        rel = (err / ref.abs().clamp_min(1e-2)).median().item()
        print(f"  mat2={label:10s} OK  max|Δ|={err.max().item():.4e} 中位相对误差={rel:.4f}")
        if ok_layout is None:
            ok_layout = b2
    except Exception as e:  # noqa: BLE001
        print(f"  mat2={label:10s} FAIL {type(e).__name__}: {str(e)[:120]}")

# ---------------- 真实形状速度 ----------------
print("\n[C] 真实形状：bf16 vs fp8（TF/s）")
print(f"{'shape (M,K)x(K,N)':>28s} {'bf16 ms':>9s} {'bf16 TF/s':>10s} {'fp8 ms':>9s} {'fp8 TF/s':>9s} {'加速':>7s}")
for tag, (M, K, N) in (
    ("FFN up   1.23M,256x256x1024", (1231872, 256, 1024)),
    ("FFN down 1.23M,1024x1024x256", (1231872, 1024, 256)),
    ("qkv      1.23M,256x256x768", (1231872, 256, 768)),
    ("outproj  1.23M,256x256x256", (1231872, 256, 256)),
):
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    flops = 2 * M * K * N
    t_bf16 = bench(lambda: A @ B)
    sa = A.abs().max().item() / 448.0
    sb = B.abs().max().item() / 448.0
    Aq = (A / sa).to(torch.float8_e4m3fn)
    Bq = (B / sb).to(torch.float8_e4m3fn)
    line = f"{tag:>28s} {t_bf16:9.3f} {flops/t_bf16/1e9:10.1f}"
    try:
        if ok_layout is None:
            raise RuntimeError("_scaled_mm 无可用布局")
        b2 = Bq if ok_layout.shape == (K, N) else Bq.t()
        t_fp8 = bench(lambda: torch._scaled_mm(Aq, b2, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        line += f" {t_fp8:9.3f} {flops/t_fp8/1e9:9.1f} {t_bf16/t_fp8:6.2f}x"
        # 量化本身的开销（cast + absmax）
        t_q = bench(lambda: (A / (A.abs().max().item() / 448.0)).to(torch.float8_e4m3fn))
        line += f"   [量化附带 {t_q:.3f} ms]"
    except Exception as e:  # noqa: BLE001
        line += f"  FAIL {type(e).__name__}: {str(e)[:60]}"
    print(line)
    del A, B
    torch.npu.empty_cache()
