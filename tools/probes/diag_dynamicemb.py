#!/usr/bin/env python3
"""diag_dynamicemb.py — 诊断 dynamic_emb 单步 1.4 s 的来源（是否为脚本误用）"""
import statistics, time, torch, torch_npu
from dynamic_emb import DynamicEmbTableOptions, EmbOptimType
from dynamic_emb.distributed.batched_dynamicemb_table import (
    BatchedDynamicEmbeddingTablesV2, DynamicEmbPoolingMode)

DEV = torch.device("npu:0")
DIM, T = 64, 12


def build(rows, dedup=False):
    opts = []
    for _ in range(T):
        opts.append(DynamicEmbTableOptions(
            training=True, init_capacity=rows, max_capacity=rows, dim=DIM,
            embedding_dtype=torch.float32,
            global_hbm_for_values=rows * DIM * 4, local_hbm_for_values=rows * DIM * 4,
            bucket_capacity=128, block_size=128, io_block_size=1024,
            index_type=torch.int64))
    return BatchedDynamicEmbeddingTablesV2(
        table_options=opts, table_names=[f"t{i}" for i in range(T)],
        feature_table_map=list(range(T)), pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=torch.float32, device=DEV, use_index_dedup=dedup,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD, learning_rate=0.1)


def timeit(fn, n=5):
    for _ in range(2):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(n):
        torch.npu.synchronize(); t0 = time.perf_counter()
        fn(); torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


for rows in (100_000, 1_000_000):
    per = 2048
    mod = build(rows)
    idx = torch.randint(0, rows, (T, per), device=DEV, dtype=torch.int64)
    flat = idx.reshape(-1)
    off = torch.arange(0, T * per + 1, per, device=DEV, dtype=torch.int64)
    out = mod(flat, off)
    shapes = [tuple(o.shape) for o in out] if isinstance(out, (list, tuple)) else [tuple(out.shape)]
    t_fwd = timeit(lambda: mod(flat, off))
    def fwd_bwd():
        o = mod(flat, off)
        acc = None
        for v in (o if isinstance(o, (list, tuple)) else [o]):
            acc = v.float().sum() if acc is None else acc + v.float().sum()
        acc.backward()
    t_fb = timeit(fwd_bwd)
    print(f"rows={rows:>9,} | 前向 {t_fwd*1e3:8.2f} ms | 前向+反向 {t_fb*1e3:8.2f} ms | "
          f"输出 {len(shapes)} 个 {shapes[:2]}{'...' if len(shapes)>2 else ''}")
    # 表是否真的被更新（fused optimizer）
    r0 = mod(flat[:8], torch.tensor([0, 8], device=DEV, dtype=torch.int64))[0].float().sum().item()
    _ = timeit(fwd_bwd, n=3)
    r1 = mod(flat[:8], torch.tensor([0, 8], device=DEV, dtype=torch.int64))[0].float().sum().item()
    print(f"          表更新检查: 前 {r0:.4f} → 后 {r1:.4f}  {'✅ 已更新' if abs(r0-r1) > 1e-6 else '⚠️ 未变化'}")
