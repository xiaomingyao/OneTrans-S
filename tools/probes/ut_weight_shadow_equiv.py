#!/usr/bin/env python3
# coding=utf-8
"""UT：ONETRANS_WEIGHT_SHADOW=1（同一 step 内复用一份 bf16 权重）与 autocast 每次调用现 cast 是否等价。

判据（逐位）：
  ① 影子臂每个 Parameter 都拿到梯度（否则 DDP 不会同步 ⇒ 直接失败）；
  ② 2 步 × 2 micro-batch 的 loss 逐位相同（rel = 0）；
  ③ 一步反向后所有参数梯度逐位相同。
已知踩坑（本 UT 就是为它写的）：bmm 路径的 bias 必须保持 fp32 —— baseline 里 `bf16 matmul + fp32 bias`
走 type-promotion 变回 fp32，把 bias 一起 cast 成 bf16 会改成 bf16 域加法，差 1 ULP（9.766e-04）。

用法：python3 tools/probes/ut_weight_shadow_equiv.py [--device npu|cpu]
"""
import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
os.environ.setdefault("ONETRANS_ATTN_MASK", "float")
os.environ.setdefault("ONETRANS_NS_BMM", "1")      # 与线上最优档一致（ns 走 bmm 权重切片）

import torch  # noqa: E402
from torch import nn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="npu", choices=["npu", "cpu"])
args = ap.parse_args()
DEV = args.device
if DEV == "npu":
    import torch_npu  # noqa: E402,F401

from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402
import main_pytorch  # noqa: E402

CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=32, ns_len=4,
           d_model=64, num_heads=4, ffn_hidden=128, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=2, pyramid_align=32, use_sep_token=True)
B, ACCUM, STEPS = 6, 2, 2


def make_inputs(seed=7):
    g = torch.Generator().manual_seed(seed)
    ns = torch.randn(B * ACCUM, CFG["non_seq_dim"], generator=g).to(DEV)
    seq = torch.randn(B * ACCUM, CFG["seq_len"], CFG["seq_feature_dim"], generator=g).to(DEV)
    y = torch.randint(0, CFG["num_classes"], (B * ACCUM,), generator=g).to(DEV)
    return ns, seq, y


def run(shadow: bool):
    torch.manual_seed(4321)
    model = TAACOneTransClassifier(**CFG).to(DEV)
    main_pytorch._WEIGHT_SHADOW = shadow
    main_pytorch.clear_weight_shadow_cache()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    ns, seq, y = make_inputs()
    losses, grads = [], None
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for k in range(ACCUM):
            lo, hi = k * B, (k + 1) * B
            with torch.autocast(DEV, dtype=torch.bfloat16):
                loss = crit(model(ns[lo:hi], seq[lo:hi]), y[lo:hi]) / ACCUM
            loss.backward()
            losses.append(loss.item() * ACCUM)
        main_pytorch.clear_weight_shadow_cache()      # 与训练脚本同一位置（opt.step 之后）
        if grads is None:
            grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
        opt.step()
    return losses, grads, model


print(f"torch={torch.__version__} device={DEV} NS_BMM={os.environ['ONETRANS_NS_BMM']} Batch={B} Accum={ACCUM} Steps={STEPS}")
l0, g0, _ = run(False)
l1, g1, m1 = run(True)

n_all = sum(1 for _ in m1.parameters())
n_grad = sum(1 for p in m1.parameters() if p.grad is not None)
print(f"[1] 影子臂参数拿到梯度: {n_grad}/{n_all}")

print("[2] loss: 无影子 =", [f"{x:.9f}" for x in l0])
print("    影子   =", [f"{x:.9f}" for x in l1])
rel_l = max(abs(a - b) / max(abs(a), 1e-12) for a, b in zip(l0, l1))
print(f"    max rel|Δloss| = {rel_l:.3e}")

worst, worst_name, worst_abs = 0.0, "", 0.0
for n in g0:
    if n not in g1:
        print(f"    !! 影子臂缺少梯度: {n}")
        worst = float("inf")
        worst_name = n
        break
    a, b = g0[n], g1[n]
    d = (a - b).abs().max().item()
    worst_abs = max(worst_abs, d)
    mask = a.abs() >= 1e-3 * a.abs().max().clamp_min(1e-12)     # 只对非退化元素比相对差
    if mask.any():
        r = ((a - b).abs()[mask] / a.abs()[mask]).max().item()
        if r > worst:
            worst, worst_name = r, n
print(f"[3] 梯度: max|Δg| = {worst_abs:.3e} | worst rel|Δg|（非退化元素）= {worst:.3e}  ({worst_name})")

ok = (n_grad == n_all) and rel_l == 0.0 and worst == 0.0
print("VERDICT:", "PASS（逐位相同）" if ok else "FAIL")
sys.exit(0 if ok else 1)
