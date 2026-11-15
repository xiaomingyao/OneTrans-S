#!/usr/bin/env python3
# coding=utf-8
"""多卡等价性测试：DDP(W 卡 × 局部 batch) 的梯度 == 单进程全局 batch 的梯度。

判据：
  A. 梯度等价：all-reduce 后的平均梯度 vs 单进程 global-batch 梯度，max|Δ|/max|ref| ≤ 1e-5（fp32）
  B. batch 等价：全局 loss（各 rank 局部 loss 的均值，等分 batch）= 单进程 global-batch loss
  C. bf16 口径：--dtype bf16 时同一判据放宽到 2e-2（训练实际口径）

用法（容器内，torchrun 拉起）：
  torchrun --nproc_per_node=2 tests/ddp_equiv_npu.py --local-batch 8
  torchrun --nproc_per_node=4 tests/ddp_equiv_npu.py --local-batch 8 --dtype bf16
单卡不带 torchrun 运行会直接判 FAIL（没有可比对象），用于防止误判。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_npu  # noqa: E402
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


def make_model() -> torch.nn.Module:
    torch.manual_seed(1234)
    return TAACOneTransClassifier(**CFG)


def fixed_batch(global_bs: int, seed: int = 99):
    g = torch.Generator().manual_seed(seed)
    non_seq = torch.randn(global_bs, 64, generator=g)
    seq = torch.randn(global_bs, 32, 32, generator=g)
    w = torch.randn(64, 5, generator=g)
    labels = (non_seq @ w).argmax(dim=1)
    return non_seq, seq, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-batch", type=int, default=8)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--tol-grad", type=float, default=None)
    ap.add_argument("--tol-loss", type=float, default=None)
    a = ap.parse_args()

    if "LOCAL_RANK" not in os.environ:
        print("[ddp-equiv] FAIL: 必须用 torchrun 拉起（缺少 LOCAL_RANK），单进程没有可比对象")
        return 1

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = f"npu:{local_rank}"

    global_bs = a.local_batch * world
    non_seq, seq, labels = fixed_batch(global_bs)
    my_idx = torch.arange(global_bs)[rank::world]

    model = make_model().to(dev)
    opt_loss = torch.nn.CrossEntropyLoss()

    def forward_backward(m, ns, sq, y):
        if a.dtype == "bf16":
            with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                out = m(ns, sq)
                loss = opt_loss(out, y)
        else:
            out = m(ns, sq)
            loss = opt_loss(out, y)
        loss.backward()
        return loss.detach().float()

    local_loss = forward_backward(
        model, non_seq[my_idx].to(dev), seq[my_idx].to(dev), labels[my_idx].to(dev)
    )
    grads = [
        (p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32))
        for p in model.parameters()
    ]
    for gt in grads:
        dist.all_reduce(gt, op=dist.ReduceOp.SUM)
        gt /= world

    # 全局 loss = 各 rank 局部 loss 的均值（每 rank 等分 batch）
    loss_sum = local_loss.clone()
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    ddp_loss = (loss_sum / world).item()

    # rank0 用同一 init、同一全局 batch 重算参考梯度（同一设备，避免跨设备比较）
    ok = {"A": True, "B": True}
    worst_grad = 0.0
    if rank == 0:
        ref = make_model().to(dev)
        ref_loss = forward_backward(ref, non_seq.to(dev), seq.to(dev), labels.to(dev))
        ref_grads = [p.grad.detach().float() if p.grad is not None else torch.zeros_like(p) for p in ref.parameters()]
        for g_ddp, g_ref, p in zip(grads, ref_grads, ref.parameters()):
            denom = g_ref.abs().max().clamp_min(1e-12)
            worst_grad = max(worst_grad, ((g_ddp - g_ref).abs().max() / denom).item())
        tol_grad = a.tol_grad if a.tol_grad is not None else (2e-2 if a.dtype == "bf16" else 1e-5)
        tol_loss = a.tol_loss if a.tol_loss is not None else (2e-2 if a.dtype == "bf16" else 1e-6)
        loss_diff = abs(ddp_loss - float(ref_loss))
        ok["A"] = worst_grad <= tol_grad
        ok["B"] = loss_diff <= tol_loss
        print(f"[A] 梯度等价 world={world} local_batch={a.local_batch} global_batch={global_bs} dtype={a.dtype}")
        print(f"    单进程 |g|max={max(g.abs().max().item() for g in ref_grads):.6e} "
              f"worst rel|Δg|={worst_grad:.3e} 门槛={tol_grad:.1e}")
        print(f"[B] loss 单进程={float(ref_loss):.8f} DDP均值={ddp_loss:.8f} |Δ|={loss_diff:.3e} 门槛={tol_loss:.1e}")
        print(f"[verdict] A={ok['A']} B={ok['B']}")
        print("DDP-EQUIV " + ("PASS" if all(ok.values()) else "FAIL"))
    dist.barrier()
    dist.destroy_process_group()
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
