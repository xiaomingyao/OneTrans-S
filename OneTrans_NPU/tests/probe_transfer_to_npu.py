#!/usr/bin/env python3
# coding=utf-8
"""三种导入方式下，CUDA API 到底被替换了多少？

模式：
  bare      —— 只 import torch（对照：注意机器上根本没 GPU）
  npu       —— import torch_npu（只注册后端，不改命名空间）
  transfer  —— from torch_npu.contrib import transfer_to_npu（官方"一行迁移"猴子补丁）

每个模式在**独立子进程**里跑，避免 monkey-patch 互相污染。
用法：python3 tests/probe_transfer_to_npu.py [--mode bare|npu|transfer|all]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BODY = r"""
import json, sys, warnings
warnings.filterwarnings("ignore")
mode = sys.argv[1]
import torch

if mode == "npu":
    import torch_npu
elif mode == "transfer":
    from torch_npu.contrib import transfer_to_npu
elif mode != "bare":
    raise SystemExit("bad mode")

out = {"mode": mode}

def rec(k, fn):
    try:
        v = fn()
        out[k] = v if isinstance(v, (bool, int, float, str, list)) else str(v)
    except Exception as e:
        out[k] = f"EXC {type(e).__name__}: {e}"

rec("cuda_is_available", lambda: torch.cuda.is_available())
rec("device_count", lambda: torch.cuda.device_count())
rec("device_ctor_cuda", lambda: torch.device("cuda").type)
rec("device_ctor_cuda0", lambda: torch.device("cuda:0").type)
rec("npu_is_available", lambda: torch.npu.is_available() if hasattr(torch, "npu") else "no torch.npu")
rec("tensor_on_cuda", lambda: str(torch.randn(2, 2, device="cuda").device))
rec("tensor_on_npu", lambda: str(torch.randn(2, 2, device="npu").device))
rec("cuda_set_device", lambda: (torch.cuda.set_device(0) or "ok"))
rec("cuda_manual_seed_all", lambda: (torch.cuda.manual_seed_all(0) or "ok"))
rec("cuda_bf16_supported", lambda: torch.cuda.is_bf16_supported())
rec("cuda_amp_autocast", lambda: str(torch.cuda.amp.autocast))
rec("amp_gradscaler_cuda", lambda: type(torch.amp.GradScaler(device="cuda", enabled=True)).__name__)
rec("sdp_backend", lambda: str(torch.nn.attention.sdpa_kernel))
rec("dist_backend_nccl", lambda: str(torch.distributed.Backend.NCCL))
print("JSON " + json.dumps(out, ensure_ascii=False))
"""


def run(mode: str) -> dict:
    p = subprocess.run([sys.executable, "-c", BODY, mode], capture_output=True, text=True, timeout=180)
    for line in p.stdout.splitlines():
        if line.startswith("JSON "):
            return json.loads(line[5:])
    return {"mode": mode, "ERROR": (p.stdout + p.stderr)[-600:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    a = ap.parse_args()
    modes = ["bare", "npu", "transfer"] if a.mode == "all" else [a.mode]

    res = [run(m) for m in modes]
    keys = [k for k in res[0] if k != "mode"]
    for r in res[1:]:
        for k in r:
            if k not in keys and k not in ("mode", "ERROR"):
                keys.append(k)

    w = max(len(k) for k in keys) + 2
    print("| " + "检查项".ljust(w) + " | " + " | ".join(f"{r['mode']}" for r in res) + " |")
    print("|" + "-" * (w + 2) + "|" + "|".join("-" * max(len(r["mode"]) + 2, 8) for r in res) + "|")
    for k in keys:
        row = []
        for r in res:
            v = r.get(k, "-")
            row.append(str(v)[:46])
        print("| " + k.ljust(w) + " | " + " | ".join(row) + " |")
    print("\nTRANSFER-PROBE DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
