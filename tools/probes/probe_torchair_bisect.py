#!/usr/bin/env python3
# coding=utf-8
"""图模式第四条路：TorchAir（`torch.compile(backend="npu")`）失败点二分。

现状：整模 compile 报 GE 图编译错误
  `[InferShape] The k-axis of a(1191) and b(256) tensors must be the same`（MatMul）。
1191 = 1203 − 12，说明某个子模块在 GE 侧把「序列维」当成 matmul 的 k 轴了（eager 下走的是别的分支）。
本探针按 3 个粒度分别 compile（静态 shape、小 batch），定位到具体子模块。

输出每个粒度的 OK / FAIL 与 GE 的首个 InferShape 报错。
"""
import argparse
import os
import sys
import traceback

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
for k, v in dict(ONETRANS_ATTN_MASK="bool", ONETRANS_NS_BMM="1", ONETRANS_FUSED_NORM="1",
                 ONETRANS_CAT_INPLACE="1", ONETRANS_SDPA_CAUSAL="1", ONETRANS_FULL_PROJ="1",
                 ONETRANS_FA_BSH="1").items():
    os.environ.setdefault(k, v)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
import main_pytorch as MP  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--case", default="all",
                choices=["attn", "block", "stack", "model", "all"])
a = ap.parse_args()
DEV = "npu"
D, H = 256, 4
B = a.batch


def run(case):
    torch.manual_seed(0)
    if case == "attn":
        m = MP.CausalMaskAttention(ns_len=12, d_model=D, num_heads=H, mask_type="paper_causal").to(DEV)
        x = torch.randn(B, 1203, D, device=DEV)
        fn = lambda: m((x, x, x))  # CausalMaskAttention 收 (k_x, q_x, v_x)  # noqa: E731
    elif case == "block":
        m = MP.OneTransBlock(ns_len=12, d_model=D, num_heads=H, ffn_units=(1536, D),
                             pyramid_stack_len=960, mask_type="paper_causal").to(DEV)
        x = torch.randn(B, 1203, D, device=DEV)
        fn = lambda: m(x)  # noqa: E731
    elif case == "stack":
        m = MP.MultiOneTransBlock(ns_len=12, d_model=D, num_heads=H, ffn_units=(1536, D),
                                  n=1, pyramid_stack_len=960,
                                  mask_type="paper_causal").to(DEV)
        x = torch.randn(B, 1203, D, device=DEV)
        fn = lambda: m(x)  # noqa: E731
    else:
        CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=1190, ns_len=12,
                   d_model=D, num_heads=H, ffn_hidden=1536, multi_num=1, mask_type="paper_causal",
                   num_pyramid_layers=6, pyramid_align=32, use_sep_token=True, use_checkpoint=False)
        m = TAACOneTransClassifier(**CFG).to(DEV)
        ns = torch.randn(B, 64, device=DEV)
        sq = torch.randn(B, 1190, 32, device=DEV)
        fn = lambda: m(ns, sq)  # noqa: E731

    # 先 eager 跑一遍（确认形状合法）
    fn()
    torch.npu.synchronize()
    try:
        cm = torch.compile(m, backend="npu", fullgraph=False, dynamic=False)
        cm(*((torch.randn(B, 1203, D, device=DEV),) if case != "model"
              else (torch.randn(B, 64, device=DEV), torch.randn(B, 1190, 32, device=DEV))))
        torch.npu.synchronize()
        print(f"  [OK  ] {case} (TorchAir 编译+执行)")
        return True
    except Exception as e:  # noqa: BLE001
        msg = [ln.strip() for ln in str(e).splitlines() if ln.strip()]
        key = next((ln for ln in msg if "k-axis" in ln or "InferShape" in ln or "EZ9999" in ln), msg[0] if msg else "")
        print(f"  [FAIL] {case} <- {key[:180]}")
        if os.environ.get("VERBOSE"):
            traceback.print_exc(limit=3)
        return False


print(f"=== TorchAir 粒度二分（B{B}）")
for c in (["attn", "block", "stack", "model"] if a.case == "all" else [a.case]):
    run(c)
