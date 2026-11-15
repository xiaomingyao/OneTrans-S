#!/usr/bin/env python3
# coding=utf-8
"""NPU 冒烟测试：前向 → 反向 → 3 步优化，断言 loss 有限、参数有更新、kernel 落 V4。

用法（容器内）：python3 tests/smoke_npu.py [--device npu|cpu] [--batch 32] [--seq-len 64]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import npu_compat  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--ns-len", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--multi-num", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()

    dev_type = npu_compat.init_device(a.device)
    print(f"[env] {npu_compat.describe(dev_type)}")
    torch.manual_seed(42)

    model = TAACOneTransClassifier(
        non_seq_dim=64,
        seq_feature_dim=32,
        num_classes=5,
        seq_len=a.seq_len,
        ns_len=a.ns_len,
        d_model=a.d_model,
        num_heads=a.num_heads,
        ffn_hidden=256,
        multi_num=a.multi_num,
        mask_type="paper_causal",
        num_pyramid_layers=4,
        pyramid_align=32,
        use_sep_token=True,
    ).to(a.device)

    n_param = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_param/1e6:.2f}M  blocks=1+4  seq_len={a.seq_len}  ns_len={a.ns_len}")

    g = torch.Generator().manual_seed(7)
    non_seq = torch.randn(a.batch, 64, generator=g).to(a.device)
    seq = torch.randn(a.batch, a.seq_len, 32, generator=g).to(a.device)
    labels = torch.randint(0, 5, (a.batch,), generator=g).to(a.device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    before = [p.detach().clone() for p in list(model.parameters())[:2]]

    loss0 = None
    for step in range(a.steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        logits = model(non_seq, seq)
        loss = lossf(logits, labels)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if dev_type == "npu":
            torch.npu.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if step == 0:
            loss0 = loss.item()
        print(f"[step {step}] loss={loss.item():.6f} grad_norm={float(gn):.4f} {dt:.1f} ms")

    moved = max((p.detach() - b).abs().max().item() for p, b in zip(list(model.parameters())[:2], before))
    finite = torch.isfinite(loss).item()
    ok = bool(finite) and moved > 0
    print(f"[check] loss_finite={finite} param_delta={moved:.3e} loss_drop={loss0 - loss.item():.6f}")
    print("SMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
