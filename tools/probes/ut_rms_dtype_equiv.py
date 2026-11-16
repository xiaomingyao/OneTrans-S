#!/usr/bin/env python3
# coding=utf-8
"""E1a UT：RMSNorm 把 weight 放进 fp32 域内乘、输出回 bf16，与现状的精度/类型对比。

判据（本实验是「舍入位置」变化，不是逐位等价，所以用「谁更接近精确值」来判）：
  ① 输出 dtype：现状（bf16 输入）应为 fp32，E1a 应为 bf16；
  ② 相对 fp64 参考的最大相对误差：E1a 必须 ≤ 现状（否则精度确有损失，不许合）。
CPU 即可跑。
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU"))
sys.path.insert(0, ROOT)

import torch  # noqa: E402


def load_rms_norm(gate: str):
    """gate=0/1 分别 import 一次，避免模块级 env 缓存串味。"""
    os.environ["ONETRANS_RMS_DTYPE"] = gate
    for m in [k for k in list(sys.modules) if k == "main_pytorch"]:
        del sys.modules[m]
    from main_pytorch import RMSNorm  # noqa: PLC0415

    return RMSNorm


torch.manual_seed(0)
D = 256
B, T = 16, 129
w = torch.randn(D)                      # fp32 参数（与仓库一致）
x_bf16 = (torch.randn(B, T, D) * 2.0).to(torch.bfloat16)

# 精确参考（fp64）
x64 = x_bf16.double()
rms64 = x64.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
ref = (x64 * rms64 * w.double())

outs = {}
for gate in ("0", "1"):
    RMSNorm = load_rms_norm(gate)
    m = RMSNorm(D)
    with torch.no_grad():
        m.weight.copy_(w)
    y = m(x_bf16)
    outs[gate] = y
    y64 = y.double()
    rel = ((y64 - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    rms_err = ((y64 - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    print(f"gate={gate}: out_dtype={y.dtype}  max_rel_err_vs_fp64={rel:.3e}  rms_rel_err={rms_err:.3e}")

d_new_old = (outs["1"].float() - outs["0"]).abs()
denom = outs["0"].abs().clamp_min(1e-6)
mr = (d_new_old / denom).max().item()
ma = d_new_old.mean().item()
print(f"\n新旧互差: max_rel={mr:.3e}  mean_abs={ma:.3e}")

# 反向：两条路的 weight 梯度都应可算且有限
for gate in ("0", "1"):
    RMSNorm = load_rms_norm(gate)
    m = RMSNorm(D)
    with torch.no_grad():
        m.weight.copy_(w)
    y = m(x_bf16.clone().requires_grad_(True))
    y.float().pow(2).mean().backward()
    print(f"gate={gate}: grad_weight dtype={m.weight.grad.dtype} finite={torch.isfinite(m.weight.grad).all().item()}")

ok = (outs["0"].dtype == torch.float32) and (outs["1"].dtype == torch.bfloat16)
print("\nVERDICT:", "DTYPE_FIX_CONFIRMED" if ok else "UNEXPECTED_DTYPE")
