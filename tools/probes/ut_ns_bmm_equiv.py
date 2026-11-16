#!/usr/bin/env python3
# coding=utf-8
"""UT：ONETRANS_NS_BMM=1（NS-token 专有参数合并成 bmm）与默认逐 token Linear 是否逐值等价。

CPU 即可跑（不需 NPU）。判据：
  ① 参数量相同；② 前向输出 max|Δ| ≤ 1e-6（fp32）；③ 一步反向后对应参数梯度 worst rel|Δg| ≤ 1e-5
     （与仓库 DDP 梯度等价判据同标准；bmm 与 Linear 的 fp32 累加顺序不同，1e-6 级差异属数值噪声）。
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "OneTrans_NPU")
sys.path.insert(0, os.path.abspath(ROOT))
os.environ.setdefault("ONETRANS_ATTN_MASK", "float")   # 用 float mask：与上游逐字等价的默认口径

import torch  # noqa: E402
from models.taac_onetrans import TAACOneTransClassifier  # noqa: E402

CFG = dict(non_seq_dim=64, seq_feature_dim=32, num_classes=2, seq_len=32, ns_len=4,
           d_model=64, num_heads=4, ffn_hidden=128, multi_num=1,
           mask_type="paper_causal", num_pyramid_layers=2, pyramid_align=32, use_sep_token=True)


def build(gate: str):
    os.environ["ONETRANS_NS_BMM"] = gate
    torch.manual_seed(1234)
    return TAACOneTransClassifier(**CFG)


m0 = build("0")
sd0 = {k: v.clone() for k, v in m0.state_dict().items()}
n0 = sum(p.numel() for p in m0.parameters())
m1 = build("1")
n1 = sum(p.numel() for p in m1.parameters())

# 逐 tensor 搬运
with torch.no_grad():
    def blocks(m):
        out = []
        for mb in [m.base_block] + list(m.stack_blocks):
            out.extend(list(mb.otb_list))      # MultiOneTransBlock 里才是 OneTransBlock
        return out

    for b0, b1 in zip(blocks(m0), blocks(m1)):
        for g in range(CFG["ns_len"] + 1):
            for p in range(3):
                lin = b0.cma.kqv_list[g][p]
                b1.cma.kqv_w[g, p].copy_(lin.weight)
                b1.cma.kqv_b[g, p].copy_(lin.bias)
            f = b0.ffn_list[g]
            b1.ffn_w1[g].copy_(f.proj_1.weight)
            b1.ffn_b1[g].copy_(f.proj_1.bias)
            b1.ffn_w2[g].copy_(f.proj_2.weight)
            b1.ffn_b2[g].copy_(f.proj_2.bias)
    # tokenizer / 其余同名参数
    sd1 = m1.state_dict()
    for k, v in sd0.items():
        if k in sd1 and sd1[k].shape == v.shape:
            sd1[k].copy_(v)

torch.manual_seed(7)
ns = torch.randn(3, 64)
sq = torch.randn(3, 32, 32)
lb = torch.randint(0, 2, (3,))

outs = []
for m in (m0, m1):
    m.train()
    out = m(ns, sq)
    loss = torch.nn.functional.cross_entropy(out, lb)
    loss.backward()
    outs.append((out.detach().clone(), {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}))

d_out = (outs[0][0] - outs[1][0]).abs().max().item()

# 梯度对齐：把 bmm 侧梯度按 (g,p) 拼回逐 token 侧的同名位置比较
gmax = 0.0
for b0, b1 in zip(blocks(m0), blocks(m1)):
    for g in range(CFG["ns_len"] + 1):
        for p in range(3):
            lin = b0.cma.kqv_list[g][p]
            for name, t1 in (("weight", b1.cma.kqv_w.grad[g, p]), ("bias", b1.cma.kqv_b.grad[g, p])):
                g0 = getattr(lin, name).grad
                g1 = t1 if t1 is not None else torch.zeros_like(g0)
                if g0 is not None and g0.abs().max() > 0:
                    gmax = max(gmax, (g0 - g1).abs().max().item() / g0.abs().max().item())
        f0 = b0.ffn_list[g]
        for (p0, p1) in ((f0.proj_1.weight, b1.ffn_w1.grad[g]), (f0.proj_1.bias, b1.ffn_b1.grad[g]),
                         (f0.proj_2.weight, b1.ffn_w2.grad[g]), (f0.proj_2.bias, b1.ffn_b2.grad[g])):
            g0 = p0.grad
            g1 = p1 if p1 is not None else torch.zeros_like(g0)
            if g0 is not None and g0.abs().max() > 0:
                gmax = max(gmax, (g0 - g1).abs().max().item() / g0.abs().max().item())

print(f"params: linear={n0:,} bmm={n1:,} equal={n0 == n1}")
print(f"forward  max|Δout| = {d_out:.3e}")
print(f"backward worst rel|Δg| = {gmax:.3e}")
TOL_G = 1e-5   # fp32 累加顺序噪声；与仓库 DDP 梯度等价判据（worst rel|Δg| ≤ 1e-5）同标准
ok = (n0 == n1) and d_out <= 1e-6 and gmax <= TOL_G
print("VERDICT:", "EQUIVALENT_PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
