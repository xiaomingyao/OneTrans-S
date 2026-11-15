#!/usr/bin/env python3
# coding=utf-8
"""混合精度（AMP）验证探针：确认"以为开了 AMP"和"真的开了 AMP"是一回事。

要回答四个问题：
  1. NPU 上 autocast 是否真的生效（不是只 WARNING 一下然后静默禁用）；
  2. 不显式给 dtype 时默认是不是 fp16（若是，训练里必须显式 bfloat16）；
  3. bf16 autocast 下算子输出 dtype 是否真是 bf16（用 matmul 这类会被 autocast 覆盖的算子）；
  4. no-op scaler 的调用形态是否与真 GradScaler 等价（scale/unscale_/step/update 都能调）。

用法（容器内）：
  python3 tests/probe_amp_npu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_npu  # noqa: E402
import npu_compat  # noqa: E402


def main() -> int:
    npu_compat.init_device("npu")
    print(f"[env] {npu_compat.describe('npu')}", flush=True)

    # 2) 不传 dtype 时的默认值（历史坑：npu 默认 fp16，训练必须显式 bfloat16）
    try:
        default_dtype = torch.get_autocast_dtype("npu")
    except Exception as exc:  # 老版本没有该接口
        default_dtype = f"n/a ({exc.__class__.__name__})"
    print(f"[default] torch.get_autocast_dtype('npu') = {default_dtype}", flush=True)

    a = torch.randn(256, 256, device="npu")
    b = torch.randn(256, 256, device="npu")

    # 1) + 3) 显式 bf16：生效标志 + 实际算子输出 dtype
    with torch.autocast(device_type="npu", dtype=torch.bfloat16):
        enabled = torch.is_autocast_enabled("npu")
        c_bf16 = a @ b
    print(f"[bf16] autocast_enabled={enabled} matmul_out_dtype={c_bf16.dtype}", flush=True)

    # 对照：不带 autocast
    c_fp32 = a @ b
    print(f"[fp32] matmul_out_dtype={c_fp32.dtype}", flush=True)

    # 不传 dtype 的默认行为
    with torch.autocast(device_type="npu"):
        c_default = a @ b
    print(f"[default] matmul_out_dtype={c_default.dtype}", flush=True)

    # 4) 兼容层给 NPU 的 no-op scaler：调用形态必须与真 GradScaler 一致
    scaler = npu_compat.build_scaler("npu", True, torch.bfloat16)
    param = torch.nn.Parameter(torch.ones(2, 2, device="npu"))
    opt = torch.optim.AdamW([param], lr=1e-3)
    loss = (param * 3).sum()      # 必须依赖参数，否则 backward 无 grad_fn
    loss.backward()
    scaler.scale(loss)
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    print(f"[scaler] is_enabled={scaler.is_enabled()} state_dict={scaler.state_dict()} "
          f"shape_ok={scaler.scale(loss).shape == loss.shape}", flush=True)

    # 兼容层放行规则
    for dev, dt in (("npu", "bf16"), ("npu", "fp16"), ("cuda", "fp16"), ("cpu", "bf16")):
        print(f"[gate] {dev}/{dt} -> amp={npu_compat.should_enable_amp(dev, True, dt)}", flush=True)

    print("AMP-PROBE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
