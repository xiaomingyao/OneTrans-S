#!/usr/bin/env python3
"""bench_dynamicemb_16.py — dynamic_emb(HKV) 16 die 数据并行实测

与单 die 探针同形状：12 表 × 1,000,000 行 × 64 维，每 die 每步 12×2048 = 24,576 次查表。
每 die 持有**完整副本**（无跨卡稀疏同步——NPU 不支持 sparse all_reduce，这是数据并行的真实形态），
因此本测度量的是「16 die 同时跑时的单步时延与聚合吞吐」，不是分片表。

启动（每节点 8 进程）：
  torchrun --nnodes=2 --node_rank=N --nproc_per_node=8 \
           --master_addr=192.168.0.82 --master_port=29511 bench_dynamicemb_16.py
"""
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch_npu
from dynamic_emb import DynamicEmbTableOptions, EmbOptimType
from dynamic_emb.distributed.batched_dynamicemb_table import (
    BatchedDynamicEmbeddingTablesV2, DynamicEmbPoolingMode)

T, ROWS, DIM, PER = 12, 1_000_000, 64, 2048
ITERS, WARMUP = 6, 2


def main():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("hccl")
    torch.npu.set_device(local)
    dev = torch.device(f"npu:{local}")

    opts = [DynamicEmbTableOptions(
        training=True, init_capacity=ROWS, max_capacity=ROWS, dim=DIM,
        embedding_dtype=torch.float32, index_type=torch.int64,
        bucket_capacity=128, block_size=128, io_block_size=1024,
        global_hbm_for_values=ROWS * DIM * 4, local_hbm_for_values=ROWS * DIM * 4)
        for _ in range(T)]
    mod = BatchedDynamicEmbeddingTablesV2(
        table_options=opts, table_names=[f"t{i}" for i in range(T)],
        feature_table_map=list(range(T)), pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=torch.float32, device=dev, use_index_dedup=True,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD, learning_rate=0.1)

    gen = torch.Generator(device="cpu").manual_seed(1234 + rank)
    off = torch.arange(0, T * PER + 1, PER, device=dev, dtype=torch.int64)

    def step():
        idx = torch.randint(0, ROWS, (T, PER), generator=gen).to(dev)
        out = mod(idx.reshape(-1), off)
        acc = None
        for v in (out if isinstance(out, (list, tuple)) else [out]):
            acc = v.float().sum() if acc is None else acc + v.float().sum()
        acc.backward()

    for _ in range(WARMUP):
        step()
    torch.npu.synchronize()
    dist.barrier()

    ts = []
    for _ in range(ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        step()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    dist.barrier()

    med = statistics.median(ts)
    print(f"[rank {rank:>2}/{world-1}] local_rank={local} | {med*1e3:8.2f} ms/步 "
          f"(min {min(ts)*1e3:.2f} / max {max(ts)*1e3:.2f}) | "
          f"{T*PER/med/1e6:5.2f} Mlookup/s | HBM峰值 "
          f"{torch_npu.npu.max_memory_allocated(dev)/2**30:.1f} GiB", flush=True)

    # 汇总到 rank0
    payload = torch.tensor([med], dtype=torch.float64, device=dev)
    gathered = [torch.zeros_like(payload) for _ in range(world)]
    dist.all_gather(gathered, payload)
    if rank == 0:
        per_rank = [float(g.item()) for g in gathered]
        worst = max(per_rank)
        mean = sum(per_rank) / len(per_rank)
        print("\n===== 16 die 汇总 =====")
        print(f"逐 rank 单步(ms): {[round(x*1e3, 2) for x in per_rank]}")
        print(f"中位 {statistics.median(per_rank)*1e3:.2f} ms | 最慢 rank {worst*1e3:.2f} ms | "
              f"最慢/中位 = {worst/statistics.median(per_rank):.3f}")
        print(f"聚合吞吐: {world*T*PER/statistics.median(per_rank)/1e6:.2f} Mlookup/s "
              f"({world*T*PER/statistics.median(per_rank)/1e6/ (T*PER/0.01062/1e6):.2f}× 单 die)")
        json.dump({"per_rank_ms": [x * 1e3 for x in per_rank],
                   "world": world, "rows": ROWS, "dim": DIM, "per": PER,
                   "lookups_per_die": T * PER},
                  open("/tmp/hkv16_result.json", "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()
    os._exit(0)          # 规避 NPU teardown SIGSEGV


if __name__ == "__main__":
    main()
