import math
import os

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# [NPU] 可选：把"0 / -inf"型加性 mask 换成 bool mask。
# 950 实测（见 docs/GPU_VS_NPU_CODE_IMPLEMENTATION.md 与 patches/npu_changes.diff）：float 加性 mask 会让 SDPA 退回 Cast+Muls+BatchMatMul+Softmax
# 手工路径；bool mask 才落 FlashAttentionScoreV4。默认 float，保持与参考实现逐字等价。
_MASK_DTYPE = os.environ.get("ONETRANS_ATTN_MASK", "float").lower()

VALID_MASK_TYPES = {"paper_causal", "origin", "hard_mask", "bimask_soft", "bimask_hard"}
_SQRT5 = math.sqrt(5.0)
# [E1a] norm 在 fp32 域内乘 weight，输出回 bf16（去掉 fp32 激活与下游重复 cast）。默认关。
_RMS_FP32_OUT = os.environ.get("ONETRANS_RMS_DTYPE", "0") == "1"
# [E5] 激活重算的作用域：all（默认）/ ffn / attn
_CKPT_SCOPE = os.environ.get("ONETRANS_CKPT_SCOPE", "all").lower()
# [E1b] 用昇腾融合 rms_norm 算子替代手写链（Square+Pow+ReduceSum+Rsqrt+Mul）
_FUSED_NORM = os.environ.get("ONETRANS_FUSED_NORM", "0") == "1"
# [E7] _project_one 的 seq/ns 合并：cat → 预分配 + 原地写入（逐位等价，只改内存写路径）
_CAT_INPLACE = os.environ.get("ONETRANS_CAT_INPLACE", "0") == "1"


class FFNLayer(nn.Module):
    def __init__(self, input_dim: int, unit_1: int = 256, unit_2: int = 128) -> None:
        super().__init__()
        self.proj_1 = nn.Linear(input_dim, unit_1)
        self.proj_2 = nn.Linear(unit_1, unit_2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_chunk = int(os.environ.get("ONETRANS_FFN_CHUNK", "0"))
        if n_chunk > 0:
            # [NPU] 按「行」（batch×token）分块做 FFN：数学逐元素等价，只为压瞬时显存
            # （每卡 batch 4096 时 unit_1 激活约 15 GiB 一口气分配会撞 96 GiB HBM 上限）
            shape = x.shape
            flat = x.reshape(-1, shape[-1])
            if flat.shape[0] > n_chunk:
                # 预分配输出 + 原地写分片：避免 torch.cat 再复制一份整张量（4096 时约省 2.5 GiB）
                out = torch.empty_like(flat)
                for s in range(0, flat.shape[0], n_chunk):
                    out[s : s + n_chunk] = self.act(self.proj_2(self.act(self.proj_1(flat[s : s + n_chunk]))))
                return out.view(shape)
        x = self.act(self.proj_1(x))
        x = self.act(self.proj_2(x))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _FUSED_NORM:
            # E1b：gamma 按输入 dtype 参与，内部 fp32 归一化。实测比手写链更接近 fp64
            #（max_rel 3.891e-03 vs 7.704e-03，b64/seq1203 单算子 0.063 ms vs 0.205 ms）
            import torch_npu  # noqa: PLC0415

            return torch_npu.npu_rms_norm(x, self.weight.to(x.dtype), self.eps)[0]
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        if _RMS_FP32_OUT:
            # E1a：weight(fp32) 在 fp32 域内乘掉，整个 norm 只做一次量化（→ x.dtype）。
            # 原式 (x_fp32*rms).to(dtype) * weight 会把输出提升成 fp32，下游 linear 在 autocast 下再 cast 回 bf16
            # ⇒ 每层多一次 fp32 激活与一次 Cast（profile: Cast 677 次/步、185.4 ms）。
            return (x_fp32 * rms * self.weight).to(input_dtype)
        return (x_fp32 * rms).to(input_dtype) * self.weight


class CausalMaskAttention(nn.Module):
    def __init__(
        self,
        ns_len: int,
        d_model: int = 128,
        num_heads: int = 4,
        if_mask: bool = True,
        mask_type: str = "paper_causal",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if mask_type not in VALID_MASK_TYPES:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.ns_len = ns_len
        self.if_mask = if_mask
        self.mask_type = mask_type
        self.dense = nn.Linear(d_model, d_model)
        # [移植自 zeus-onetrans opt_1109 补丁1 的 ns-layout 思路] 把「每个 NS-token 一组独立 Linear」
        # 合成一次 bmm：ns 段 (B, ns, d) → (ns, B, d) @ (ns, d, d)。数学逐元素等价（每个 token 仍用自己的权重），
        # 只是把 12×3 次 1-token 小 matmul 变成 3 次 bmm，减少 kernel 下发次数。默认关（ONETRANS_NS_BMM=0）。
        self.use_bmm = os.environ.get("ONETRANS_NS_BMM", "0") == "1"
        if self.use_bmm:
            self.kqv_w = nn.Parameter(torch.empty(ns_len + 1, 3, d_model, d_model))
            self.kqv_b = nn.Parameter(torch.empty(ns_len + 1, 3, d_model))
            for g in range(ns_len + 1):
                for p in range(3):
                    # 与 nn.Linear 默认初始化同分布（kaiming_uniform a=√5 + U(-1/√fan_in, 1/√fan_in)）
                    nn.init.kaiming_uniform_(self.kqv_w[g, p], a=_SQRT5)
                    bound = 1.0 / d_model ** 0.5
                    nn.init.uniform_(self.kqv_b[g, p], -bound, bound)
        else:
            self.kqv_list = nn.ModuleList(
                [nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) for _ in range(ns_len + 1)]
            )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def create_attention_mask(self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.mask_type == "paper_causal":
            row_idx = torch.arange(query_len, device=device).unsqueeze(1)
            col_idx = torch.arange(key_len, device=device).unsqueeze(0)
            q_abs = row_idx + (key_len - query_len)
            allowed = col_idx <= q_abs
            if _MASK_DTYPE == "bool":
                return allowed
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~allowed, torch.finfo(dtype).min)

        row_idx = torch.arange(query_len, device=device).unsqueeze(1)
        col_idx = torch.arange(key_len, device=device).unsqueeze(0)
        origin_allowed = (col_idx - row_idx) <= (self.ns_len - 1)
        if self.mask_type == "origin":
            return origin_allowed.to(dtype=dtype) + 1e-9
        if self.mask_type == "hard_mask":
            if _MASK_DTYPE == "bool":
                return origin_allowed
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~origin_allowed, torch.finfo(dtype).min)

        ns_query_rows = row_idx < self.ns_len
        strict_causal_allowed = col_idx <= row_idx
        if self.mask_type == "bimask_soft":
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            seq_allowed = (~ns_query_rows) & strict_causal_allowed
            return mask + seq_allowed.to(dtype=dtype)

        allowed = ns_query_rows | strict_causal_allowed
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _cal_kqv(self, x: torch.Tensor, group_idx: int, proj_idx: int) -> torch.Tensor:
        if self.use_bmm:
            return torch.nn.functional.linear(x, self.kqv_w[group_idx, proj_idx], self.kqv_b[group_idx, proj_idx])
        return self.kqv_list[group_idx][proj_idx](x)

    def _project_one(self, x: torch.Tensor, proj_idx: int) -> torch.Tensor:
        seq_len = x.size(1) - self.ns_len
        shared_group_idx = self.ns_len
        if self.use_bmm:
            parts = []
            if seq_len > 0:
                parts.append(self._cal_kqv(x[:, :seq_len, :], shared_group_idx, proj_idx))
            if self.ns_len > 0:
                ns_in = x[:, seq_len:, :].permute(1, 0, 2)                     # (ns, B, d)
                w = self.kqv_w[: self.ns_len, proj_idx]                        # (ns, d, d)
                b = self.kqv_b[: self.ns_len, proj_idx].unsqueeze(1)           # (ns, 1, d)
                parts.append((torch.bmm(ns_in, w.transpose(1, 2)) + b).permute(1, 0, 2))  # (B, ns, d)
            if len(parts) == 1:
                return parts[0]
            if _CAT_INPLACE:
                out = parts[0].new_empty((parts[0].size(0), parts[0].size(1) + parts[1].size(1), parts[0].size(-1)))
                out[:, : parts[0].size(1)] = parts[0]
                out[:, parts[0].size(1) :] = parts[1]
                return out
            return torch.cat(parts, dim=1)
        res = []
        if seq_len > 0:
            res.append(self._cal_kqv(x[:, :seq_len, :], shared_group_idx, proj_idx))
        for i in range(self.ns_len):
            start = seq_len + i
            res.append(self._cal_kqv(x[:, start : start + 1, :], i, proj_idx))
        return torch.cat(res, dim=1)

    def cal_mix_param_kqv(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._project_one(x[0], 0), self._project_one(x[1], 1), self._project_one(x[2], 2)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        seq_len_k = x[0].size(1)
        seq_len_q = x[1].size(1)

        k, q, v = self.cal_mix_param_kqv(x)
        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        attention_mask = None
        if self.if_mask:
            attention_mask = self.create_attention_mask(
                seq_len_q, seq_len_k, device=q.device, dtype=q.dtype
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.size(0), -1, self.d_model)
        return self.dense(output)


class OneTransBlock(nn.Module):
    def __init__(
        self,
        ns_len: int,
        d_model: int,
        num_heads: int = 4,
        ffn_units: tuple[int, int] | None = None,
        pyramid_stack_len: int | None = None,
        mask_type: str = "paper_causal",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.d_model = d_model
        self.pyramid_stack_len = pyramid_stack_len
        self.use_checkpoint = use_checkpoint
        if ffn_units is None:
            ffn_units = (256, d_model)
        self.rms_0 = RMSNorm(d_model)
        self.rms_1 = RMSNorm(d_model)
        self.cma = CausalMaskAttention(
            ns_len=ns_len,
            d_model=d_model,
            num_heads=num_heads,
            mask_type=mask_type,
        )
        # [ns-layout 移植] 同 attention：NS-token 的专有 FFN 合成一次 bmm（默认关）
        self.use_bmm = os.environ.get("ONETRANS_NS_BMM", "0") == "1"
        if self.use_bmm:
            self.ffn_w1 = nn.Parameter(torch.empty(ns_len + 1, ffn_units[0], d_model))
            self.ffn_b1 = nn.Parameter(torch.empty(ns_len + 1, ffn_units[0]))
            self.ffn_w2 = nn.Parameter(torch.empty(ns_len + 1, ffn_units[1], ffn_units[0]))
            self.ffn_b2 = nn.Parameter(torch.empty(ns_len + 1, ffn_units[1]))
            for g in range(ns_len + 1):
                for w, b in ((self.ffn_w1[g], self.ffn_b1[g]), (self.ffn_w2[g], self.ffn_b2[g])):
                    fan_in = w.size(-1)
                    nn.init.kaiming_uniform_(w, a=_SQRT5)
                    bound = 1.0 / fan_in ** 0.5
                    nn.init.uniform_(b, -bound, bound)
        else:
            self.ffn_list = nn.ModuleList(
                [FFNLayer(input_dim=d_model, unit_1=ffn_units[0], unit_2=ffn_units[1]) for _ in range(ns_len + 1)]
            )

    def _ffn_one(self, x: torch.Tensor, gi: int) -> torch.Tensor:
        if self.use_bmm:
            h = torch.nn.functional.silu(
                torch.nn.functional.linear(x, self.ffn_w1[gi], self.ffn_b1[gi]))
            return torch.nn.functional.silu(
                torch.nn.functional.linear(h, self.ffn_w2[gi], self.ffn_b2[gi]))
        return self.ffn_list[gi](x)

    def cal_mix_param_ffn(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1) - self.ns_len
        res = []
        if seq_len > 0:
            res.append(self._ffn_one(x[:, :seq_len, :], self.ns_len))
        if self.ns_len > 0:
            if self.use_bmm:
                ns_in = x[:, seq_len:, :].permute(1, 0, 2)                    # (ns, B, d)
                h = torch.nn.functional.silu(
                    torch.bmm(ns_in, self.ffn_w1[: self.ns_len].transpose(1, 2)) + self.ffn_b1[: self.ns_len].unsqueeze(1))
                o = torch.bmm(h, self.ffn_w2[: self.ns_len].transpose(1, 2)) + self.ffn_b2[: self.ns_len].unsqueeze(1)
                res.append(torch.nn.functional.silu(o).permute(1, 0, 2))      # (B, ns, d)
            else:
                for i in range(self.ns_len):
                    start = seq_len + i
                    res.append(self.ffn_list[i](x[:, start : start + 1, :]))
        return res[0] if len(res) == 1 else torch.cat(res, dim=1)

    def _attn_part(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rms_0(x)
        k_x, q_x, v_x = x, x, x
        if self.pyramid_stack_len is not None and self.pyramid_stack_len >= self.ns_len:
            q_x = x[:, -self.pyramid_stack_len :, :]
        origin_x = q_x
        x = self.cma((k_x, q_x, v_x))
        return origin_x + x

    def _ffn_part(self, x: torch.Tensor) -> torch.Tensor:
        origin_x = x
        x = self.rms_1(x)
        x = self.cal_mix_param_ffn(x)
        return origin_x + x

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self._ffn_part(self._attn_part(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ck = self.use_checkpoint and self.training
        if not ck or _CKPT_SCOPE == "all":
            return checkpoint(self._forward_impl, x, use_reentrant=False) if ck else self._forward_impl(x)
        # [E5] 选择性重算：只重算 attention 或只重算 FFN（算子序列与 _forward_impl 完全一致 ⇒ 逐位等价）
        if _CKPT_SCOPE == "ffn":
            return checkpoint(self._ffn_part, self._attn_part(x), use_reentrant=False)
        if _CKPT_SCOPE == "attn":
            return self._ffn_part(checkpoint(self._attn_part, x, use_reentrant=False))
        raise ValueError(f"ONETRANS_CKPT_SCOPE 只能是 all/ffn/attn，收到 {_CKPT_SCOPE!r}")


class MultiOneTransBlock(nn.Module):
    def __init__(
        self,
        ns_len: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_units: tuple[int, int] | None = None,
        n: int = 4,
        pyramid_stack_len: int | None = None,
        mask_type: str = "paper_causal",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.otb_list = nn.ModuleList(
            [
                OneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=ffn_units,
                    pyramid_stack_len=pyramid_stack_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for otb in self.otb_list:
            x = otb(x)
        return x


def main() -> None:
    torch.manual_seed(0)

    batch_size = 4
    seq_len = 3
    feat_dim = 8
    d_model = 16

    seq_feature = torch.randn(batch_size, seq_len, feat_dim)
    seq_tokenizer = nn.Linear(feat_dim, d_model)
    s_feat = seq_tokenizer(seq_feature)
    print("Sequence feature [batch_size, seq_len, feat_dim]:", tuple(seq_feature.shape))
    print("Encoded sequence feature [batch_size, seq_len, d_model]:", tuple(s_feat.shape))

    n_seq_feature = torch.randn(batch_size, 128)
    ns_len = 2
    non_seq_tokenizer = nn.Linear(128, ns_len * d_model)
    ns_feat = non_seq_tokenizer(n_seq_feature).view(batch_size, ns_len, d_model)
    print("Non-sequence feature [batch_size, random_dim]:", tuple(n_seq_feature.shape))
    print("Encoded non-sequence feature [batch_size, ns_len, d_model]:", tuple(ns_feat.shape))
    print()

    num_head = 4
    multi_num = 8
    ffn_units = (64, d_model)

    base_block = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
    )
    base_embedding = base_block(torch.cat([s_feat, ns_feat], dim=1))
    print("After base OneTrans block [batch_size, seq_len + ns_len, d_model]:", tuple(base_embedding.shape))

    base_seq_len = base_embedding.size(1)

    stack_block_1 = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
        pyramid_stack_len=base_seq_len - 1,
    )
    stack_embedding = stack_block_1(base_embedding)
    print("After compression block 1 [batch_size, seq_len + ns_len - 1, d_model]:", tuple(stack_embedding.shape))

    stack_block_2 = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
        pyramid_stack_len=base_seq_len - 2,
    )
    stack_embedding = stack_block_2(stack_embedding)
    print("After compression block 2 [batch_size, seq_len + ns_len - 2, d_model]:", tuple(stack_embedding.shape))

    stack_block_3 = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
        pyramid_stack_len=base_seq_len - 3,
    )
    stack_embedding = stack_block_3(stack_embedding)
    print("After compression block 3 [batch_size, seq_len + ns_len - 3, d_model]:", tuple(stack_embedding.shape))
    print()

    print("Final compressed result [batch_size, ns_len, d_model]:", tuple(stack_embedding.shape))
    final_embedding = stack_embedding.mean(dim=1)
    print("Pooling result before downstream task:", tuple(final_embedding.shape))


if __name__ == "__main__":
    main()
