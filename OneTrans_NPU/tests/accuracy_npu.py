#!/usr/bin/env python3
# coding=utf-8
"""精度测试：CPU(fP32) 作为 golden，对齐 NPU(fp32) 与 NPU(bf16 autocast)。

三项判据：
  A. 前向逐元素：NPU fp32 vs CPU fp32 —— max|Δ| ≤ 1e-4、cos ≥ 0.99999（同 dtype 应几乎一致）
  B. 前向逐元素：NPU bf16 vs CPU fp32 —— max|Δ| ≤ 2e-2、cos ≥ 0.9995（与训练侧 bf16 口径一致）
  C. 训练轨迹：同 init、同数据、同 lr 各跑 N 步 —— 末步 loss 差 ≤ 1e-2、精度差 ≤ 2 个百分点

用法（容器内）：python3 tests/accuracy_npu.py [--steps 10]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import npu_compat  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

CFG = dict(
    non_seq_dim=64,
    seq_feature_dim=32,
    num_classes=5,
    seq_len=32,
    ns_len=4,
    d_model=64,
    num_heads=4,
    ffn_hidden=128,
    multi_num=2,
    mask_type="paper_causal",
    num_pyramid_layers=3,
    pyramid_align=32,
    use_sep_token=True,
)


def make_model():
    torch.manual_seed(1234)
    return TAACOneTransClassifier(**CFG)


def batch(bs=16, seq_len=32, seed=99):
    g = torch.Generator().manual_seed(seed)
    non_seq = torch.randn(bs, 64, generator=g)
    seq = torch.randn(bs, seq_len, 32, generator=g)
    w = torch.randn(64, 5, generator=g)
    labels = (non_seq @ w).argmax(dim=1)
    return non_seq, seq, labels


def stats(a: torch.Tensor, b: torch.Tensor):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    return dict(max_abs=(a - b).abs().max().item(),
                cos=F.cosine_similarity(a, b, dim=0).item(),
                rel=((a - b).abs().mean() / b.abs().mean().clamp_min(1e-12)).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()
    npu_compat.init_device("npu")

    non_seq, seq, labels = batch()
    cpu_model = make_model().eval()

    npu_model = copy.deepcopy(cpu_model).to("npu").eval()

    with torch.no_grad():
        ref = cpu_model(non_seq, seq)
        out_fp32 = npu_model(non_seq.to("npu"), seq.to("npu")).float().cpu()
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            out_bf16 = npu_model(non_seq.to("npu"), seq.to("npu")).float().cpu()

    sA = stats(out_fp32, ref)
    sB = stats(out_bf16, ref)
    print(f"[A] NPU fp32 vs CPU fp32 : max|Δ|={sA['max_abs']:.3e} cos={sA['cos']:.8f} rel={sA['rel']:.3e}")
    print(f"[B] NPU bf16 vs CPU fp32 : max|Δ|={sB['max_abs']:.3e} cos={sB['cos']:.8f} rel={sB['rel']:.3e}")

    # C. 训练轨迹
    cpu_t = make_model()
    npu_t = copy.deepcopy(cpu_t).to("npu")
    opt_c = torch.optim.AdamW(cpu_t.parameters(), lr=1e-3)
    opt_n = torch.optim.AdamW(npu_t.parameters(), lr=1e-3)
    lf = torch.nn.CrossEntropyLoss()
    losses_c, losses_n = [], []
    for _ in range(a.steps):
        opt_c.zero_grad(set_to_none=True)
        lc = lf(cpu_t(non_seq, seq), labels)
        lc.backward()
        torch.nn.utils.clip_grad_norm_(cpu_t.parameters(), 1.0)
        opt_c.step()
        losses_c.append(lc.item())

        opt_n.zero_grad(set_to_none=True)
        ln = lf(npu_t(non_seq.to("npu"), seq.to("npu")), labels.to("npu"))
        ln.backward()
        torch.nn.utils.clip_grad_norm_(npu_t.parameters(), 1.0)
        opt_n.step()
        losses_n.append(ln.item())

    with torch.no_grad():
        acc_c = (cpu_t(non_seq, seq).argmax(-1) == labels).float().mean().item()
        acc_n = (npu_t(non_seq.to("npu"), seq.to("npu")).argmax(-1) == labels.to("npu")).float().mean().item()
    print(f"[C] loss  CPU {losses_c[0]:.6f}→{losses_c[-1]:.6f} | NPU {losses_n[0]:.6f}→{losses_n[-1]:.6f} "
          f"| Δloss={abs(losses_c[-1]-losses_n[-1]):.3e}")
    print(f"[C] acc   CPU {acc_c:.4f} | NPU {acc_n:.4f} | Δ={abs(acc_c-acc_n)*100:.2f} pp")

    okA = sA["max_abs"] <= 1e-4 and sA["cos"] >= 0.99999
    okB = sB["max_abs"] <= 2e-2 and sB["cos"] >= 0.9995
    okC = abs(losses_c[-1] - losses_n[-1]) <= 1e-2 and abs(acc_c - acc_n) <= 0.02
    print(f"[verdict] A={okA} B={okB} C={okC}")
    print("ACCURACY " + ("PASS" if (okA and okB and okC) else "FAIL"))
    return 0 if (okA and okB and okC) else 1


if __name__ == "__main__":
    raise SystemExit(main())
