#!/usr/bin/env python3
"""probe_embedding_components.py — 昇腾侧 embedding 关键组件可用性（为"论文含稀疏表"的补齐做前置）"""
import traceback

import torch
import torch_npu  # noqa: F401

dev = "npu:0"


def try_run(name, fn):
    try:
        out = fn()
        print(f"  [OK]   {name}: {out}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        print(f"  [FAIL] {name}: {type(e).__name__}: {msg}")


print("=== 1. dense embedding 路径 ===")


def emb_dense():
    m = torch.nn.Embedding(1000, 64).to(dev)
    idx = torch.randint(0, 1000, (32, 8), device=dev)
    y = m(idx)
    y.sum().backward()
    return f"out {tuple(y.shape)}, grad dense={m.weight.grad.is_sparse}"


try_run("nn.Embedding(sparse=False) fwd+bwd", emb_dense)

print("=== 2. sparse embedding 路径（论文的表后端形态）===")


def emb_sparse():
    m = torch.nn.Embedding(1000, 64, sparse=True).to(dev)
    idx = torch.randint(0, 1000, (32, 8), device=dev)
    y = m(idx)
    y.sum().backward()
    g = m.weight.grad
    return f"grad is_sparse={g.is_sparse}, nnz={g._nnz() if g.is_sparse else 'n/a'}"


try_run("nn.Embedding(sparse=True) fwd+bwd", emb_sparse)

print("=== 3. EmbeddingBag（RecSDK/torchrec 的常用后端）===")


def embbag():
    m = torch.nn.EmbeddingBag(1000, 64, mode="sum").to(dev)
    idx = torch.randint(0, 1000, (32, 8), device=dev)
    y = m(idx)
    y.sum().backward()
    return f"out {tuple(y.shape)}, grad dense={not m.weight.grad.is_sparse}"


try_run("nn.EmbeddingBag fwd+bwd", embbag)

print("=== 4. sparse 梯度上的优化器（Adagrad，论文 sparse 侧）===")


def sparse_adagrad():
    m = torch.nn.Embedding(1000, 64, sparse=True).to(dev)
    opt = torch.optim.Adagrad(m.parameters(), lr=0.1)
    idx = torch.randint(0, 1000, (32, 8), device=dev)
    m(idx).sum().backward()
    opt.step()
    return "Adagrad over sparse grad OK"


try_run("torch.optim.Adagrad + sparse grad", sparse_adagrad)

print("=== 5. 稀疏梯度跨卡同步（论文 data-parallel 的必需项）===")


def sparse_allreduce():
    g = torch.sparse_coo_tensor(torch.tensor([[0, 5], [1, 2]]), torch.tensor([1.0, 2.0]),
                               (10, 3), device=dev)
    return torch.distributed.all_reduce(g)


try_run("all_reduce on sparse COO tensor", sparse_allreduce)

print("=== 6. 表参数留在 fp32 而主干 bf16（论文的混合精度形态）===")


def table_fp32_main_bf16():
    m = torch.nn.Embedding(1000, 64, sparse=True).to(dev)
    idx = torch.randint(0, 1000, (8,), device=dev)
    with torch.autocast("npu", dtype=torch.bfloat16):
        y = m(idx)
        out = torch.nn.functional.linear(y, torch.randn(16, 64, device=dev, dtype=torch.bfloat16))
    out.sum().backward()
    return f"y.dtype={y.dtype}, table.dtype={m.weight.dtype}, grad={m.weight.grad.dtype}"


try_run("autocast bf16 + fp32 表", table_fp32_main_bf16)

print("=== 7. 索引类算子（Auto-Split / gather 与 scatter）===")
try_run("torch.gather", lambda: tuple(torch.gather(torch.randn(64, 32, device=dev),
                                                 0, torch.randint(0, 64, (10, 32), device=dev)).shape))
try_run("index_copy_", lambda: torch.zeros(100, 8, device=dev).index_copy_(
    0, torch.arange(10, device=dev), torch.randn(10, 8, device=dev)).shape)
