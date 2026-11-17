#!/usr/bin/env python3
"""probe_fp8_capability.py — 950DT 上 FP8 到底能用什么（阶段 1：能力枚举）

回答三件事：
  1. fp8 dtype 能不能创建 / 搬运 / 打印（e4m3fn / e5m2）
  2. 有没有可用的 fp8 matmul 入口（torch._scaled_mm / torch_npu.* / aclnn）
  3. 支持的话，真实形状下 TF/s 对比 bf16，是否真拿到 2× 峰值

任何一步失败都单独捕获，不影响后续（本环境算子不可用时常常是静默失败，必须显式打印）。
"""
import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"

print("=" * 100)
print(f"torch {torch.__version__} | torch_npu {getattr(torch_npu, '__version__', '?')} | device {torch.npu.get_device_name(0)}")
print("=" * 100)

# ---------- 1) fp8 dtype 基础能力 ----------
print("\n[1] fp8 dtype 基础能力")
for nm in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    dt = getattr(torch, nm, None)
    if dt is None:
        print(f"  {nm:18s} torch 无此 dtype")
        continue
    try:
        x = torch.randn(64, 64, device=dev).to(dt)
        y = (x.float() + 1.0).to(dt)
        back = x.float()
        print(f"  {nm:18s} 创建/运算 OK  element_size={x.element_size()}  x[0,0]={back[0,0].item():.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"  {nm:18s} FAIL {type(e).__name__}: {str(e)[:80]}")

# ---------- 2) fp8 matmul 入口枚举 ----------
print("\n[2] fp8 matmul 入口")
cands = []
for attr in dir(torch._C):
    if "scaled_mm" in attr or "fp8" in attr.lower():
        cands.append(f"torch._C.{attr}")
for attr in dir(torch):
    if "scaled_mm" in attr or "fp8" in attr.lower():
        cands.append(f"torch.{attr}")
for attr in dir(torch_npu):
    if any(k in attr.lower() for k in ("quant", "fp8", "scaled", "mm")):
        cands.append(f"torch_npu.{attr}")
print("  候选:", ", ".join(sorted(set(cands))) or "（无）")

print("\n  torch._scaled_mm 调用测试（fp8 e4m3 × e4m3，per-tensor scale）:")
A = torch.randn(1024, 256, device=dev, dtype=torch.bfloat16)
B = torch.randn(256, 512, device=dev, dtype=torch.bfloat16)
sa = A.abs().max() / 448.0
sb = B.abs().max() / 448.0
Aq = (A / sa).to(torch.float8_e4m3fn)
Bq = (B / sb).to(torch.float8_e4m3fn)
ref = (A @ B).float()
try:
    out = torch._scaled_mm(Aq, Bq.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    err = (out.float() - ref).abs()
    rel = (err / ref.abs().clamp_min(1e-3)).median().item()
    print(f"    OK  shape {tuple(out.shape)}  max|Δ|={err.max().item():.4e}  中位相对误差={rel:.4f}")
except Exception as e:  # noqa: BLE001
    print(f"    FAIL {type(e).__name__}: {str(e)[:160]}")

print("\n  普通 torch.mm 吃 fp8 输入:")
try:
    out2 = torch.mm(Aq, Bq)
    print(f"    OK dtype={out2.dtype}  （注意：无 scale，结果只是量化后的裸乘）")
except Exception as e:  # noqa: BLE001
    print(f"    FAIL {type(e).__name__}: {str(e)[:160]}")

# ---------- 3) NPU 侧 fp8 相关 aclnn 算子 ----------
print("\n[3] torch.ops.npu 里名字带 quant/fp8/scale 的算子（前 40 个）")
try:
    names = [n for n in dir(torch.ops.npu) if any(k in n.lower() for k in ("quant", "fp8", "scaled", "antiquant"))]
    print("  ", ", ".join(names[:40]) if names else "（无）")
except Exception as e:  # noqa: BLE001
    print("  FAIL", e)

print("\n[4] FA / rmsnorm 等是否会拒绝 fp8（抽查 2 个）")
try:
    q = torch.randn(2, 4, 64, 64, device=dev, dtype=torch.float8_e4m3fn)
    o = torch_npu.npu_fusion_attention(q, q, q, 4, "BSH")
    print("    FA fp8: OK（意外）")
except Exception as e:  # noqa: BLE001
    print(f"    FA fp8: 不支持 → {type(e).__name__}: {str(e)[:100]}")
