"""ns_backend.py — 非序列特征 tokenizer 的后端抽象（三种实现，接口一致）

    build_ns_tokenizer(kind, non_seq_dim, ns_len, d_model, num_rows=..., device=...)
      kind = "linear"       → nn.Linear（当前默认，输入为连续向量 [B, non_seq_dim]）
      kind = "ec"           → torchrec.EmbeddingCollection（ns_len 张表，输入为 id [B, ns_len]）
      kind = "dynamic_emb"  → RecSDK v2 BatchedDynamicEmbeddingTablesV2（同上）

三者 forward 输出形状统一为 [B, ns_len, d_model]；`expects_ids` 属性告诉调用方该喂哪种输入。
ec / dynamic_emb 只在装了 torchrec / dynamic_emb 的环境可用（本机为 /opt/buildtools/torch_v2_pt2.7.1）。
"""
from __future__ import annotations

import os

import torch
from torch import nn


class LinearNSTokenizer(nn.Module):
    """当前默认实现：连续特征 → ns_len×d_model 的稠密投影。"""

    expects_ids = False

    def __init__(self, non_seq_dim: int, ns_len: int, d_model: int):
        super().__init__()
        self.ns_len, self.d_model = ns_len, d_model
        self.proj = nn.Linear(non_seq_dim, ns_len * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).view(x.size(0), self.ns_len, self.d_model)


class ECNSTokenizer(nn.Module):
    """torchrec 原始 EC：每个 NS 位置一张表（num_rows × d_model）。"""

    expects_ids = True

    def __init__(self, ns_len: int, d_model: int, num_rows: int, device=None):
        super().__init__()
        from torchrec.modules.embedding_configs import EmbeddingConfig
        from torchrec.modules.embedding_modules import EmbeddingCollection

        self.ns_len, self.d_model = ns_len, d_model
        self.keys = [f"ns{i}" for i in range(ns_len)]
        tables = [EmbeddingConfig(num_embeddings=num_rows, embedding_dim=d_model, name=k)
                  for k in self.keys]
        self.ec = EmbeddingCollection(tables=tables, device=device)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        ids = ids.to(torch.int64)
        b = ids.size(0)
        kjt = KeyedJaggedTensor(
            keys=self.keys,
            values=ids.reshape(-1),
            lengths=torch.ones(self.ns_len * b, device=ids.device, dtype=torch.int32),
        )
        out = self.ec(kjt)                       # KeyedTensor: key -> JaggedTensor
        toks = [out[k].values().view(b, self.d_model) for k in self.keys]
        return torch.stack(toks, dim=1)          # [B, ns_len, d_model]


class DynamicEmbNSTokenizer(nn.Module):
    """RecSDK v2 DynamicEmbedding（HKV 动态表）：优化器融进反向，无 dense 梯度。"""

    expects_ids = True

    def __init__(self, ns_len: int, d_model: int, num_rows: int, device=None,
                 optimizer: str = "adagrad", lr: float = 0.1):
        super().__init__()
        from dynamic_emb import DynamicEmbTableOptions, EmbOptimType
        from dynamic_emb.distributed.batched_dynamicemb_table import (
            BatchedDynamicEmbeddingTablesV2, DynamicEmbPoolingMode)

        self.ns_len, self.d_model = ns_len, d_model
        opt_type = {"adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                    "adam": EmbOptimType.ADAM,
                    "sgd": EmbOptimType.EXACT_SGD}.get(optimizer, EmbOptimType.EXACT_ROWWISE_ADAGRAD)
        opts = [DynamicEmbTableOptions(
            training=True, init_capacity=num_rows, max_capacity=num_rows, dim=d_model,
            embedding_dtype=torch.float32, index_type=torch.int64,
            bucket_capacity=128, block_size=128, io_block_size=1024,
            global_hbm_for_values=num_rows * d_model * 4,
            local_hbm_for_values=num_rows * d_model * 4) for _ in range(ns_len)]
        self.mod = BatchedDynamicEmbeddingTablesV2(
            table_options=opts, table_names=[f"ns{i}" for i in range(ns_len)],
            feature_table_map=list(range(ns_len)),
            pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=torch.float32,
            device=device, use_index_dedup=True, optimizer=opt_type, learning_rate=lr)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(torch.int64)
        b = ids.size(0)
        offsets = torch.arange(0, self.ns_len * b + 1, b, device=ids.device, dtype=torch.int64)
        out = self.mod(ids.reshape(-1), offsets)     # 优化器已在反向内完成
        flat = out[0] if isinstance(out, (list, tuple)) else out
        # 输出按 feature 主序排列：[ns_len * B, d_model] → [B, ns_len, d_model]
        return flat.view(self.ns_len, b, self.d_model).transpose(0, 1).contiguous()


def build_ns_tokenizer(kind: str | None = None, *, non_seq_dim: int, ns_len: int,
                       d_model: int, num_rows: int = 0, device=None) -> nn.Module:
    kind = (kind or os.environ.get("ONETRANS_NS_BACKEND", "linear")).lower()
    if kind in ("linear", "dense", ""):
        return LinearNSTokenizer(non_seq_dim, ns_len, d_model)
    if kind == "ec":
        if num_rows <= 0:
            raise ValueError("ec 后端需要 num_rows（表行数）")
        return ECNSTokenizer(ns_len, d_model, num_rows, device=device)
    if kind in ("dynamic_emb", "dynamicemb", "dyn"):
        if num_rows <= 0:
            raise ValueError("dynamic_emb 后端需要 num_rows（表行数）")
        return DynamicEmbNSTokenizer(ns_len, d_model, num_rows, device=device)
    raise ValueError(f"未知的 NS 后端: {kind}（可选 linear / ec / dynamic_emb）")


def table_backend_params(ns_len: int, d_model: int, num_rows: int) -> int:
    """表后端的参数规模（不进 nn.Parameters，用于报告口径）。"""
    return ns_len * num_rows * d_model


class ECTableLookup(nn.Module):
    """通用 EC 查表：F 张表（行数可各不相同），输入 [B, F] id，输出 [B, F, e_dim]。"""

    expects_ids = True

    def __init__(self, cardinalities, e_dim: int, device=None):
        super().__init__()
        from torchrec.modules.embedding_configs import EmbeddingConfig
        from torchrec.modules.embedding_modules import EmbeddingCollection

        self.f, self.e_dim = len(cardinalities), e_dim
        self.keys = [f"f{i}" for i in range(self.f)]
        tables = [EmbeddingConfig(num_embeddings=int(c), embedding_dim=e_dim, name=k)
                  for c, k in zip(cardinalities, self.keys)]
        self.ec = EmbeddingCollection(tables=tables, device=device)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

        ids = ids.to(torch.int64)
        b = ids.size(0)
        kjt = KeyedJaggedTensor(
            keys=self.keys, values=ids.reshape(-1),
            lengths=torch.ones(self.f * b, device=ids.device, dtype=torch.int32))
        out = self.ec(kjt)
        return torch.stack([out[k].values().view(b, self.e_dim) for k in self.keys], dim=1)


class DynamicEmbTableLookup(nn.Module):
    """通用 HKV 查表：F 张表（行数可各不相同），输入 [B, F] id，输出 [B, F, e_dim]。"""

    expects_ids = True

    def __init__(self, cardinalities, e_dim: int, device=None,
                 optimizer: str = "adagrad", lr: float = 0.1):
        super().__init__()
        from dynamic_emb import DynamicEmbTableOptions, EmbOptimType
        from dynamic_emb.distributed.batched_dynamicemb_table import (
            BatchedDynamicEmbeddingTablesV2, DynamicEmbPoolingMode)

        self.f, self.e_dim = len(cardinalities), e_dim
        opt_type = {"adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                    "adam": EmbOptimType.ADAM,
                    "sgd": EmbOptimType.EXACT_SGD}.get(optimizer, EmbOptimType.EXACT_ROWWISE_ADAGRAD)
        opts = []
        for c in cardinalities:
            # HKV 硬约束：init_capacity 必须 ≥ bucket_capacity(128)，否则建表报
            # "num_of_buckets_per_alloc must be equal or less than initial required buckets number"
            c = max(int(c), 1024)
            opts.append(DynamicEmbTableOptions(
                training=True, init_capacity=c, max_capacity=c, dim=e_dim,
                embedding_dtype=torch.float32, index_type=torch.int64,
                bucket_capacity=128, block_size=128, io_block_size=1024,
                global_hbm_for_values=c * e_dim * 4, local_hbm_for_values=c * e_dim * 4))
        self.mod = BatchedDynamicEmbeddingTablesV2(
            table_options=opts, table_names=[f"f{i}" for i in range(self.f)],
            feature_table_map=list(range(self.f)),
            pooling_mode=DynamicEmbPoolingMode.NONE, output_dtype=torch.float32,
            device=device, use_index_dedup=True, optimizer=opt_type, learning_rate=lr)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(torch.int64)
        b = ids.size(0)
        offsets = torch.arange(0, self.f * b + 1, b, device=ids.device, dtype=torch.int64)
        out = self.mod(ids.reshape(-1), offsets)      # 优化器已在反向内完成
        flat = out[0] if isinstance(out, (list, tuple)) else out
        return flat.view(self.f, b, self.e_dim).transpose(0, 1).contiguous()


def build_table_lookup(kind: str, *, cardinalities, e_dim: int, device=None,
                       num_rows_default: int = 0) -> nn.Module:
    """按引擎类型构造通用查表模块（供 paper_input.PaperNSTokenizer 使用）。"""
    kind = (kind or "linear").lower()
    if kind == "ec":
        return ECTableLookup(cardinalities, e_dim, device=device)
    if kind in ("dynamic_emb", "dynamicemb", "dyn", "hkv"):
        return DynamicEmbTableLookup(cardinalities, e_dim, device=device)
    raise ValueError(f"paper 形态需要表引擎（ec / dynamic_emb），收到: {kind}")
