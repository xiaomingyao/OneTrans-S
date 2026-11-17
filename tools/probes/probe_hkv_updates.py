#!/usr/bin/env python3
"""probe_hkv_updates.py — 验证 HKV 表在反向时是否真的被更新（rowwise Adagrad 融进 backward）"""
import torch
import torch_npu
from dynamic_emb import DynamicEmbTableOptions, EmbOptimType
from dynamic_emb.distributed.batched_dynamicemb_table import (
    BatchedDynamicEmbeddingTablesV2, DynamicEmbPoolingMode)

dev = torch.device("npu:0")
opts = [DynamicEmbTableOptions(
    training=True, init_capacity=1000, max_capacity=1000, dim=8,
    embedding_dtype=torch.float32, index_type=torch.int64,
    bucket_capacity=128, block_size=128, io_block_size=1024,
    global_hbm_for_values=1000 * 8 * 4, local_hbm_for_values=1000 * 8 * 4)]
mod = BatchedDynamicEmbeddingTablesV2(
    table_options=opts, table_names=["f0"], feature_table_map=[0],
    pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=torch.float32,
    device=dev, use_index_dedup=True,
    optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD, learning_rate=0.5)

ids = torch.tensor([[1], [2], [3], [7]], device=dev, dtype=torch.int64)


def gather(ids_):
    off = torch.arange(0, ids_.numel() + 1, ids_.size(0), device=dev, dtype=torch.int64)
    out = mod(ids_.reshape(-1), off)
    t = out[0] if isinstance(out, (list, tuple)) else out
    return t.view(ids_.size(0), 8)


before = gather(ids).clone()
for i in range(5):
    loss = gather(ids).sum()          # 固定梯度方向
    loss.backward()
after = gather(ids).clone()
print("表是否变化:", not torch.allclose(before, after))
print("|Δ| max = %.6f mean = %.6f" % ((after - before).abs().max().item(),
                                      (after - before).abs().mean().item()))
print("before[0,:4] =", before[0, :4].tolist())
print("after [0,:4] =", after[0, :4].tolist())
print("未访问行 (id=0) 是否变化:",
      not torch.allclose(gather(torch.tensor([[0]], device=dev, dtype=torch.int64)),
                         gather(torch.tensor([[0]], device=dev, dtype=torch.int64))))
