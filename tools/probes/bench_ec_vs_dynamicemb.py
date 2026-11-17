#!/usr/bin/env python3
"""bench_ec_vs_dynamicemb.py — torchrec.EC vs RecSDK v2 DynamicEmbedding 对比

必须在 RecSDK 环境跑：/opt/buildtools/torch_v2_pt2.7.1/bin/python3
形状：12 张表 × 1,000,000 行 × 64 维，每步 12×2048 = 24,576 次查表（≈本模型 NS token 规模）

两臂：
  A. torchrec.EmbeddingCollection（原始 EC，dense 梯度 + torchrec.optim.Adagrad）
  B. dynamic_emb.BatchedDynamicEmbeddingTablesV2（HKV 动态表，优化器融进反向）
"""
import statistics
import sys
import time
import traceback

import torch
import torch_npu  # noqa: F401

DEV = torch.device("npu:0")
TABLES, ROWS, DIM, PER_TABLE = 12, 1_000_000, 64, 2048
ITERS, WARMUP = 10, 3


def mem_gb():
    return torch_npu.npu.memory_allocated(DEV) / 2**30


def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def arm_torchrec():
    from torchrec.modules.embedding_configs import EmbeddingConfig
    from torchrec.modules.embedding_modules import EmbeddingCollection
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    keys = [f"t{i}" for i in range(TABLES)]
    cfg = [EmbeddingConfig(num_embeddings=ROWS, embedding_dim=DIM, name=k) for k in keys]
    ec = EmbeddingCollection(tables=cfg, device=DEV)
    opt = torch.optim.Adagrad(ec.parameters(), lr=0.1)

    idx = torch.randint(0, ROWS, (TABLES, PER_TABLE), device=DEV, dtype=torch.int64)
    kjt = KeyedJaggedTensor(
        keys=keys,
        values=idx.reshape(-1),
        lengths=torch.full((TABLES,), PER_TABLE, device=DEV, dtype=torch.int32),
    )

    def step():
        opt.zero_grad(set_to_none=True)
        out = ec(kjt)                      # KeyedTensor / dict[str, JaggedTensor]
        vals = out.values() if hasattr(out, "values") else list(out.values())
        if isinstance(vals, torch.Tensor):
            vals = [vals]
        acc = None
        for v in vals:
            flat = v.values() if (hasattr(v, "values") and not isinstance(v, torch.Tensor)) else v
            acc = flat.float().sum() if acc is None else acc + flat.float().sum()
        acc.backward()
        opt.step()

    t = bench(step)
    grads = [p.grad for p in ec.parameters() if p.grad is not None]
    sparse = [g.is_sparse for g in grads]
    print(f"  [A torchrec.EC]  {t*1e3:7.2f} ms/步 | {TABLES*PER_TABLE/t/1e6:7.2f} Mlookup/s | "
          f"grads={len(grads)}, is_sparse={set(sparse)}, dense_grad_bytes={sum(g.numel()*g.element_size() for g in grads)/2**20:.0f} MiB")
    return t


def arm_dynamicemb():
    from dynamic_emb import DynamicEmbTableOptions, EmbOptimType
    from dynamic_emb.distributed.batched_dynamicemb_table import (
        BatchedDynamicEmbeddingTablesV2, DynamicEmbPoolingMode)

    keys = [f"t{i}" for i in range(TABLES)]
    opts = []
    for _ in range(TABLES):
        kw = dict(training=True, init_capacity=ROWS, max_capacity=ROWS, dim=DIM,
                  embedding_dtype=torch.float32, output_dtype=torch.float32,
                  global_hbm_for_values=ROWS * DIM * 4,
                  local_hbm_for_values=ROWS * DIM * 4,
                  bucket_capacity=128, block_size=128, io_block_size=1024,
                  index_type=torch.int64)
        while True:
            try:
                o = DynamicEmbTableOptions(**kw)
                break
            except TypeError as e:        # 逐步剔除本版本不认识的字段
                bad = str(e).split("'")[1] if "'" in str(e) else None
                if bad and bad in kw:
                    kw.pop(bad)
                    continue
                o = DynamicEmbTableOptions(training=True, init_capacity=ROWS,
                                           dim=DIM, bucket_capacity=128)
                for k in ("embedding_dtype", "output_dtype", "index_type"):
                    if hasattr(o, k):
                        setattr(o, k, kw.get(k))
                break
        opts.append(o)
    mod = None
    last_err = None
    for ot in (EmbOptimType.EXACT_ROWWISE_ADAGRAD, EmbOptimType.EXACT_ADAGRAD,
               EmbOptimType.ADAM, EmbOptimType.EXACT_SGD):
        try:
            mod = BatchedDynamicEmbeddingTablesV2(
                table_options=opts, table_names=keys, feature_table_map=list(range(TABLES)),
                pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=torch.float32,
                device=DEV, optimizer=ot, learning_rate=0.1)
            print(f"      optimizer = {ot}")
            break
        except Exception as e:            # 某些版本 optimizer 枚举映射不全
            last_err = e
            mod = None
    if mod is None:
        raise last_err

    idx = torch.randint(0, ROWS, (TABLES, PER_TABLE), device=DEV, dtype=torch.int64)
    indices = idx.reshape(-1)
    offsets = torch.arange(0, TABLES * PER_TABLE + 1, PER_TABLE, device=DEV, dtype=torch.int64)

    last = {}

    def step():
        out = mod(indices, offsets)        # List[Tensor]，优化器已融进反向
        last["o"] = out
        acc = None
        for v in out:
            acc = v.float().sum() if acc is None else acc + v.float().sum()
        acc.backward()

    t = bench(step)
    o0 = last["o"]
    n = len(o0) if isinstance(o0, (list, tuple)) else 1
    d = (o0[0] if isinstance(o0, (list, tuple)) else o0).dtype
    print(f"  [B dynamic_emb]  {t*1e3:7.2f} ms/步 | {TABLES*PER_TABLE/t/1e6:7.2f} Mlookup/s | "
          f"输出 {n} 个张量, dtype={d}")
    return t


print(f"形状: {TABLES} 表 × {ROWS:,} 行 × {DIM} 维 | 每步 {TABLES*PER_TABLE:,} 次查表 | "
      f"表显存(f32) ≈ {TABLES*ROWS*DIM*4/2**30:.1f} GiB | 基线显存 {mem_gb():.1f} GiB")
res = {}
for name, fn in (("A torchrec.EC", arm_torchrec), ("B dynamic_emb", arm_dynamicemb)):
    try:
        res[name] = fn()
        print(f"      显存 {mem_gb():.1f} GiB")
    except Exception:
        print(f"  [FAIL] {name}:")
        traceback.print_exc(limit=4)

if len(res) == 2:
    a, b = res["A torchrec.EC"], res["B dynamic_emb"]
    print(f"\n汇总: dynamic_emb / torchrec.EC = {a/b:.2f}× 加速（>1 表示 dynamic_emb 更快）")
