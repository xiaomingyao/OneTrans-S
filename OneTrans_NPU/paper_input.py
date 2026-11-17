"""paper_input.py — 按论文 §3.2 的输入形态构造 token（id + 分桶 → 查表 → 投影）

论文原文（§3.2.1）：
  "All features are either bucketized or one-hot encoded and then embedded."
  "Auto-Split Tokenizer: all features are concatenated and projected once by a single MLP, then split."

本模块实现这条路径：
  F 个特征域 → （数值：分桶；类别：直接 id）→ 各域查表得 e_dim 维 → concat(F×e_dim)
  → 单次 MLP → ns_len×d_model → split 成 ns_len 个 token

查表引擎可换（torchrec EC / RecSDK v2 HKV），由 ns_backend.build_table_lookup 决定。
"""
from __future__ import annotations

import torch
from torch import nn

from ns_backend import build_table_lookup


class PaperNSTokenizer(nn.Module):
    """论文形态的非序列 tokenizer（Auto-Split：查表 → concat → 单次 MLP → split）。"""

    expects_ids = True

    def __init__(self, lookup_kind: str, cardinalities: list[int], e_dim: int,
                 ns_len: int, d_model: int, num_rows_default: int = 0, device=None):
        super().__init__()
        self.ns_len, self.d_model, self.e_dim = ns_len, d_model, e_dim
        self.n_features = len(cardinalities)
        self.lookup = build_table_lookup(lookup_kind, cardinalities=cardinalities,
                                         e_dim=e_dim, device=device,
                                         num_rows_default=num_rows_default)
        self.mlp = nn.Linear(self.n_features * e_dim, ns_len * d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.lookup(ids)                       # [B, F, e_dim]
        b = emb.size(0)
        flat = emb.reshape(b, self.n_features * self.e_dim)
        return self.mlp(flat).view(b, self.ns_len, self.d_model)


class Bucketizer:
    """数值特征 → 分桶 id（论文的 bucketize 步骤）。边界等宽，落在 [0, rows-1]。"""

    def __init__(self, n_features: int, rows: int, device=None):
        self.rows = rows
        b = torch.linspace(-3.0, 3.0, rows - 1)       # 标准正态输入下覆盖 99.7%
        self.boundaries = b.to(device) if device is not None else b

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, F] 连续值 → [B, F] int64 桶 id"""
        return torch.bucketize(x, self.boundaries.to(x.device))
