from __future__ import annotations

import sys
from pathlib import Path

import os as _os

import torch
from torch.autograd.profiler import record_function as _RF


class _null:  # 空上下文
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pytorch import MultiOneTransBlock


def linear_pyramid_schedule(total_tokens: int, ns_len: int, num_layers: int, align_to: int = 32) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if total_tokens < ns_len:
        raise ValueError("total_tokens must be greater than or equal to ns_len")
    if align_to <= 0:
        raise ValueError("align_to must be positive")

    if num_layers == 1:
        return [ns_len]

    schedule = [total_tokens]
    for layer_idx in range(1, num_layers - 1):
        raw = total_tokens + (ns_len - total_tokens) * layer_idx / (num_layers - 1)
        target_len = int(round(raw))
        if align_to > 1 and total_tokens > align_to:
            target_len = int(round(target_len / align_to) * align_to)
        target_len = max(ns_len, min(schedule[-1], target_len))
        schedule.append(target_len)
    schedule.append(ns_len)
    return schedule


class TAACOneTransClassifier(nn.Module):
    def __init__(
        self,
        non_seq_dim: int,
        seq_feature_dim: int,
        num_classes: int,
        seq_len: int,
        ns_len: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        multi_num: int,
        mask_type: str = "paper_causal",
        num_pyramid_layers: int = 6,
        pyramid_align: int = 32,
        use_sep_token: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.seq_len = seq_len
        self.use_sep_token = use_sep_token
        # NS tokenizer 支持三种后端（默认 linear，与论文的稠密投影一致）
        #   ONETRANS_NS_BACKEND = linear | ec | dynamic_emb
        #   ONETRANS_NS_ROWS    = 表行数（ec / dynamic_emb 必填）
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from ns_backend import build_ns_tokenizer
        self.ns_backend = _os.environ.get("ONETRANS_NS_BACKEND", "linear")
        _ns_dev = None
        if self.ns_backend not in ("linear", "dense", ""):
            _dev_s = _os.environ.get("ONETRANS_NS_DEVICE", "")
            if _dev_s:
                _ns_dev = torch.device(_dev_s)
            elif hasattr(torch, "npu") and torch.npu.is_available():
                _ns_dev = torch.device(f"npu:{torch.npu.current_device()}")
            else:
                _ns_dev = torch.device("cpu")
        self.non_seq_tokenizer = build_ns_tokenizer(
            self.ns_backend, non_seq_dim=non_seq_dim, ns_len=ns_len, d_model=d_model,
            num_rows=int(_os.environ.get("ONETRANS_NS_ROWS", "0")), device=_ns_dev)
        self.seq_tokenizer = nn.Linear(seq_feature_dim, d_model)
        if use_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.base_block = MultiOneTransBlock(
            ns_len=ns_len,
            d_model=d_model,
            num_heads=num_heads,
            ffn_units=(ffn_hidden, d_model),
            n=multi_num,
            mask_type=mask_type,
            use_checkpoint=use_checkpoint,
        )
        total_tokens = ns_len + seq_len + (1 if use_sep_token else 0)
        schedule = linear_pyramid_schedule(
            total_tokens=total_tokens,
            ns_len=ns_len,
            num_layers=num_pyramid_layers,
            align_to=pyramid_align,
        )
        self.stack_blocks = nn.ModuleList(
            [
                MultiOneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=(ffn_hidden, d_model),
                    n=multi_num,
                    pyramid_stack_len=target_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for target_len in schedule
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, non_seq_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        _mk = _os.environ.get("ONETRANS_PHASE_MARKERS", "0") == "1"
        batch_size = non_seq_x.size(0)
        with _RF("P_embed") if _mk else _null():
            _t = self.non_seq_tokenizer(non_seq_x)      # linear: [B,ns,d]；表后端: 同样 [B,ns,d]
            # 表后端输出恒为 fp32（表精度），显式 bf16 干路下需转成主干 dtype；
            # 该 cast 的反向会把梯度还原成 fp32，表侧不受影响。
            _t = _t.to(self.seq_tokenizer.weight.dtype)
            ns_tokens = _t.view(batch_size, self.ns_len, -1)
        with _RF("P_embed") if _mk else _null():
            seq_tokens = self.seq_tokenizer(seq_x)
        if self.use_sep_token:
            sep_token = self.sep_token.expand(batch_size, -1, -1)
            x = torch.cat([seq_tokens, sep_token, ns_tokens], dim=1)
        else:
            x = torch.cat([seq_tokens, ns_tokens], dim=1)
        with _RF("P_blocks") if _mk else _null():
            x = self.base_block(x)
            for block in self.stack_blocks:
                x = block(x)
        with _RF("P_pool_head") if _mk else _null():
            pooled = x.mean(dim=1)
            return self.head(pooled)
