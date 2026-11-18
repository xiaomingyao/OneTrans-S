"""taobao_ad_data.py — TaobaoAd_x1（reczoo 预处理版）mmap 数据源

把真实 CTR 数据接成 OneTrans-S 的输入形态：
  ns ids  : 16 个类别/计数域 + price 分桶（1024 桶）= F 17
  seq ids : [cate, brand, btag] 三字段（各查表 → 拼接 → 共享 MLP，论文 §3.2.2）
  label   : clk（0/1）

数据由 tools/prep_taobao_ad.py 预处理成 .npy / memmap 文件，按 (step, global_batch) 切片，
每个 rank 取全局 batch 内自己那一段（数据并行，无跨卡交换）。
"""
from __future__ import annotations

import json

import numpy as np
import torch


class TaobaoAdSource:
    PRICE_BUCKETS = 1024

    def __init__(self, data_dir: str, seq_len: int = 50, rank: int = 0, world_size: int = 1,
                 global_batch: int = 32768, device=None):
        self.meta = json.load(open(f"{data_dir}/meta.json"))
        self.n = int(self.meta["rows"])
        self.L = seq_len
        self.rank, self.world, self.gbs = rank, world_size, global_batch
        self.per_rank = global_batch // world_size
        if self.per_rank * world_size != global_batch:
            raise ValueError(f"global_batch {global_batch} 不能被 world_size {world_size} 整除")
        self.device = device

        self.ns = np.load(f"{data_dir}/ns_ids.i32.npy", mmap_mode="r")
        self.price = np.load(f"{data_dir}/price.f32.npy", mmap_mode="r")
        self.clk = np.load(f"{data_dir}/clk.i8.npy", mmap_mode="r")
        self.cate = np.memmap(f"{data_dir}/seq_cate.i32", np.int32, "r", shape=(self.n, seq_len))
        self.brand = np.memmap(f"{data_dir}/seq_brand.i32", np.int32, "r", shape=(self.n, seq_len))
        self.btag = np.memmap(f"{data_dir}/seq_btag.i32", np.int32, "r", shape=(self.n, seq_len))

        # price → 分位分桶边界（抽样估计，避免全量排序）
        sample = np.asarray(self.price[:: max(1, self.n // 200_000)], dtype=np.float32)
        sample = sample[np.isfinite(sample)]
        qs = np.linspace(0, 100, self.PRICE_BUCKETS + 1)[1:-1]
        self.price_edges = np.unique(np.percentile(sample, qs)).astype(np.float32)

        # 完整 batch 数（末尾不足一个全局 batch 的样本丢弃）
        self.steps_per_epoch = max(1, self.n // global_batch)
        self._zeros = None

    # ---- 元信息 ----
    @property
    def ns_cardinalities(self) -> list[int]:
        cols = self.meta["ns_cols"]
        card = [int(self.meta["ns_cardinality"][c]) + 1 for c in cols]
        card.append(int(self.price_edges.size) + 1)
        return card

    @property
    def seq_cardinalities(self) -> list[int]:
        # 预处理若已把基数写进 meta 就直接用（全量 .max() 要扫 3×12 GB，16 rank 各扫一遍太贵）
        card = self.meta.get("seq_cardinality")
        if card:
            return [int(c) for c in card]
        return [int(self.cate.max()) + 1, int(self.brand.max()) + 1, int(self.btag.max()) + 1]

    def __len__(self) -> int:
        return self.n

    # ---- 取一个全局 batch 的本 rank 分片 ----
    def batch(self, step: int) -> dict[str, torch.Tensor]:
        base = (step % self.steps_per_epoch) * self.gbs
        start = base + self.rank * self.per_rank
        idx = slice(start, start + self.per_rank)

        ns = np.array(self.ns[idx], dtype=np.int64)                     # [B, 16]
        pr = np.array(self.price[idx], dtype=np.float32)
        pr = np.where(np.isfinite(pr), pr, 0.0)
        pr_bucket = np.searchsorted(self.price_edges, pr).astype(np.int64)[:, None]
        ns = np.concatenate([ns, pr_bucket], axis=1)                    # [B, 17]

        seq = np.stack([np.array(self.cate[idx], dtype=np.int64),
                        np.array(self.brand[idx], dtype=np.int64),
                        np.array(self.btag[idx], dtype=np.int64)], axis=-1)   # [B, L, 3]
        label = np.array(self.clk[idx], dtype=np.int64)

        t_ns = torch.from_numpy(ns).to(self.device)
        t_seq = torch.from_numpy(seq).to(self.device)
        t_lab = torch.from_numpy(label).to(self.device)
        if self._zeros is None or self._zeros.size(0) != t_ns.size(0):
            self._zeros = torch.zeros(t_ns.size(0), self.L, 1, device=self.device)
        return {"ns": t_ns, "seq_ids": t_seq, "seq_x": self._zeros, "label": t_lab}
