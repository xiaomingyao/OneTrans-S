#!/usr/bin/env python3
"""bench_dynamicemb_min.py — dynamic_emb 最小可测（单次构建，避免同进程多次建表触发 aclrt 错误）"""
import statistics
import time

import torch
import torch_npu
from dynamic_emb import DynamicEmbTableOptions, EmbOptimType
from dynamic_emb.distributed.batched_dynamicemb_table import (
    BatchedDynamicEmbeddingTablesV2, DynamicEmbPoolingMode)

DEV = torch.device("npu:0")
T, ROWS, DIM, PER = 12, 1_000_000, 64, 2048

opts = [DynamicEmbTableOptions(
    training=True, init_capacity=ROWS, max_capacity=ROWS, dim=DIM,
    embedding_dtype=torch.float32, index_type=torch.int64,
    bucket_capacity=128, block_size=128, io_block_size=1024,
    global_hbm_for_values=ROWS * DIM * 4, local_hbm_for_values=ROWS * DIM * 4)
    for _ in range(T)]

mod = BatchedDynamicEmbeddingTablesV2(
    table_options=opts, table_names=[f"t{i}" for i in range(T)],
    feature_table_map=list(range(T)), pooling_mode=DynamicEmbPoolingMode.NONE,
    output_dtype=torch.float32, device=DEV, use_index_dedup=True,
    optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD, learning_rate=0.1)

idx = torch.randint(0, ROWS, (T, PER), device=DEV, dtype=torch.int64)
flat = idx.reshape(-1)
off = torch.arange(0, T * PER + 1, PER, device=DEV, dtype=torch.int64)

out = mod(flat, off)
outs = out if isinstance(out, (list, tuple)) else [out]
print("输出:", len(outs), "个", [tuple(o.shape) for o in outs[:3]], "dtype", outs[0].dtype)
print("显存:", torch_npu.npu.max_memory_allocated(DEV) / 2**30, "GiB")


def step():
    o = mod(flat, off)
    acc = None
    for v in (o if isinstance(o, (list, tuple)) else [o]):
        acc = v.float().sum() if acc is None else acc + v.float().sum()
    acc.backward()


step()
torch.npu.synchronize()
ts = []
for _ in range(8):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    step()
    torch.npu.synchronize()
    ts.append(time.perf_counter() - t0)
t = statistics.median(ts)
print(f"dynamic_emb: {t*1e3:.2f} ms/步 | {T*PER/t/1e6:.2f} Mlookup/s | 中位 {t*1e3:.1f} ms "
      f"(min {min(ts)*1e3:.1f} / max {max(ts)*1e3:.1f})")
