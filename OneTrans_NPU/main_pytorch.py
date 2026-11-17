import math
import os

import torch
from torch.autograd.profiler import record_function
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
# [O1] 显式 bf16 模式：关掉 autocast 后，NS 路径的 fp32 bias 会把输出提升成 fp32（与 SDPA 的 bf16 k/v 冲突），
# 该模式下 NS bias 一起落 bf16、RMSNorm 输出与残差流也落 bf16。默认关 ⇒ 既有路径逐位不变。
_EXPLICIT_BF16 = os.environ.get("ONETRANS_EXPLICIT_BF16", "0") == "1"
# 阶段级 profiling 标记（默认关）：打开后 block 内 attn/ffn、以及训练循环里的 fwd/loss/bwd/opt 各打一个 record_function
_PHASE_MARK = os.environ.get("ONETRANS_PHASE_MARKERS", "0") == "1"
# [N1] 用 npu_rms_norm 返回的 rstd + npu_rms_norm_backward 直接算梯度，免去反向重算平方和
_RMSNORM_FN = os.environ.get("ONETRANS_RMSNORM_FN", "0") == "1"


def _nsbias(b):
    """NS 路径的 bias：显式 bf16 模式下必须与激活同 dtype，否则输出被 type-promotion 提成 fp32。"""
    return b.to(torch.bfloat16) if _EXPLICIT_BF16 else b


# [G3] attention 走 FA 的 BSH 布局：输入 (B,L,d)、输出也是 (B,L,d)，省掉 transpose(1,2)+contiguous 那次整张量往返。
# 只用在 q_len != key_len 的层（q==k 的层继续用 G1 的无掩码 is_causal，掩码反而更慢）。
# 注意 FA 的 atten_mask 语义是「True = 屏蔽」，与 SDPA 的「True = 保留」相反。
_FA_BSH = os.environ.get("ONETRANS_FA_BSH", "0") == "1"
_FA_BLOCKED_MASK = {}      # (q_len, k_len) -> bool 掩码，True = 屏蔽

# [G2] 全量投影 + 原地覆盖 NS 行：省掉「seq/ns 两路算完再 alloc+2 写」的那次整块合并。
# 语义：NS 的 12 行先用共享权重算一遍（多算 12/L ≈1% 算力），随后被独立权的结果整行覆盖 ⇒ 前向逐位不变。
_FULL_PROJ = os.environ.get("ONETRANS_FULL_PROJ", "0") == "1"

# [G1] query_len == key_len 的层用 SDPA 内置 is_causal（无掩码张量）。
# 只有在这种层里「底右对齐」与「左上对齐」是同一个掩码 ⇒ 逐位等价；q<k 的层仍走显式 mask。
_SDPA_CAUSAL = os.environ.get("ONETRANS_SDPA_CAUSAL", "0") == "1"
# [E13] attention 掩码缓存（同形状只算一次；掩码与权重/激活无关 ⇒ 逐位等价）
_MASK_CACHE_ENABLE = os.environ.get("ONETRANS_MASK_CACHE", "0") == "1"
_MASK_CACHE: dict = {}


def clear_mask_cache() -> None:
    _MASK_CACHE.clear()

# [E11] 权重影子：autocast 会在每次 F.linear 调用时把 fp32 权重 cast 一遍，重算/累加下同一份权重要 cast 3 次。
# 这里按 (param, 切片) 在一个 step 内复用同一份 bf16 权重 ⇒ 每个权重每步只 cast 一次。
# 影子靠在原 Parameter 的计算图上（p[idx].to(bf16)），不新增叶子张量 ⇒ DDP 梯度同步、激活重算、
# 梯度累加语义全部不变；step 结束时清缓存。数值上与 autocast 的 cast 逐位相同（同为 rne 舍入）。
# 显式 bf16 必须配影子权重：否则 fp32 权重与 bf16 激活相乘会被 type-promotion 提回 fp32
_WEIGHT_SHADOW = (_EXPLICIT_BF16
                  or os.environ.get("ONETRANS_WEIGHT_SHADOW", "0").lower() not in ("", "0", "false", "no"))
_SHADOW_CACHE: dict = {}


def _wget(p, key, getter):
    """同一 step 内同一份权重只 cast 一次；返回的张量仍连在原 Parameter 的计算图上。"""
    if not _WEIGHT_SHADOW:
        return getter()
    k = (id(p), key)
    t = _SHADOW_CACHE.get(k)
    if t is None:
        t = getter().to(torch.bfloat16)
        _SHADOW_CACHE[k] = t
    return t


def clear_weight_shadow_cache() -> None:
    _SHADOW_CACHE.clear()


def _lin(m, x):
    return torch.nn.functional.linear(
        x,
        _wget(m, "w", lambda: m.weight),
        None if m.bias is None else _wget(m, "b", lambda: m.bias),
    )


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
                    out[s : s + n_chunk] = self.act(_lin(self.proj_2, self.act(_lin(self.proj_1, flat[s : s + n_chunk]))))
                return out.view(shape)
        x = self.act(_lin(self.proj_1, x))
        x = self.act(_lin(self.proj_2, x))
        return x


if _RMSNORM_FN:
    class _RMSNormFn(torch.autograd.Function):
        """RMSNorm 的前向/反向都走厂商融合算子，且**复用前向已算出的 rstd**。

        autograd 默认反向会重算一遍平方和；厂商 `npu_rms_norm_backward(dy, x, gamma, rstd)`
        直接吃 rstd，实测反向 2.536 → 1.294 ms（−49%），且 dx **逐位相同**。
        精度注意：dgamma 由 bf16 域归约得到，与 autograd 路径相比是 bf16 噪声级（mean rel ~1.3e-3）。
        """

        @staticmethod
        def forward(ctx, x: torch.Tensor, gamma: torch.Tensor, eps: float):
            import torch_npu  # noqa: PLC0415  模块级不导入，保持 CPU 路径可用

            g = gamma.to(x.dtype)
            y, rstd = torch_npu.npu_rms_norm(x, g, eps)
            ctx.save_for_backward(x, g, rstd)
            return y

        @staticmethod
        def backward(ctx, dy: torch.Tensor):
            x, g, rstd = ctx.saved_tensors
            dx, dgamma = torch.ops.npu.npu_rms_norm_backward(dy.contiguous(), x, g, rstd)
            return dx, dgamma.to(g.dtype).to(torch.float32) if g.dtype != torch.float32 else dgamma, None


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

            if _RMSNORM_FN:
                _y = _RMSNormFn.apply(x, self.weight, self.eps)     # [N1] 存 rstd，反向省钱
            else:
                _y = torch_npu.npu_rms_norm(x, self.weight.to(x.dtype), self.eps)[0]
            return _y.to(torch.bfloat16) if _EXPLICIT_BF16 else _y
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

    def create_blocked_mask(self, query_len: int, key_len: int, device: torch.device) -> torch.Tensor:
        """[G3] FA 用的掩码：True = 屏蔽（底右对齐因果被屏蔽的部分）。只依赖形状 ⇒ 可缓存。"""
        key = (query_len, key_len, str(device))
        hit = _FA_BLOCKED_MASK.get(key)
        if hit is None:
            row_idx = torch.arange(query_len, device=device).unsqueeze(1)
            col_idx = torch.arange(key_len, device=device).unsqueeze(0)
            hit = col_idx > (row_idx + (key_len - query_len))
            _FA_BLOCKED_MASK[key] = hit
        return hit

    def create_attention_mask(self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """[E13] 掩码缓存：同形状（每步都一样）只算一次。

        原先每个 block、每个 micro-batch 都重算一次 `arange` 比较（7 block × 2 accum = 14 次/步，
        实测 arange 类 kernel 约 12 ms/步 + 比较/填充若干）。掩码只依赖 (mask_type, q_len, k_len, dtype)，
        与权重、激活无关 ⇒ 缓存后逐位等价。
        """
        if not _MASK_CACHE:
            return self._build_attention_mask(query_len, key_len, device, dtype)
        key = (self.mask_type, _MASK_DTYPE, query_len, key_len, str(device), str(dtype), self.ns_len)
        hit = _MASK_CACHE.get(key)
        if hit is None:
            hit = self._build_attention_mask(query_len, key_len, device, dtype)
            _MASK_CACHE[key] = hit
        return hit

    def _build_attention_mask(self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
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
            return torch.nn.functional.linear(
                x,
                _wget(self.kqv_w, (group_idx, proj_idx), lambda: self.kqv_w[group_idx, proj_idx]),
                _wget(self.kqv_b, (group_idx, proj_idx), lambda: self.kqv_b[group_idx, proj_idx]),
            )
        return self.kqv_list[group_idx][proj_idx](x)

    def _project_one(self, x: torch.Tensor, proj_idx: int) -> torch.Tensor:
        seq_len = x.size(1) - self.ns_len
        shared_group_idx = self.ns_len
        if self.use_bmm and _FULL_PROJ and self.ns_len > 0 and seq_len > 0:
            # [G2] 全量投影（NS 行先用共享权重占位），再把 NS 那几行整块覆盖成独立权重的结果。
            # 不用额外缓冲、不做 cat：省掉每个投影一次整块 alloc + 一次全量写。
            out = self._cal_kqv(x, shared_group_idx, proj_idx)
            ns_in = x[:, seq_len:, :].permute(1, 0, 2)                          # (ns, B, d)
            w = _wget(self.kqv_w, ("nsT", proj_idx),
                      lambda: self.kqv_w[: self.ns_len, proj_idx].transpose(1, 2))
            b = _nsbias(self.kqv_b[: self.ns_len, proj_idx].unsqueeze(1))
            out[:, seq_len:, :] = (torch.bmm(ns_in, w) + b).permute(1, 0, 2)
            return out
        if self.use_bmm:
            parts = []
            if seq_len > 0:
                parts.append(self._cal_kqv(x[:, :seq_len, :], shared_group_idx, proj_idx))
            if self.ns_len > 0:
                ns_in = x[:, seq_len:, :].permute(1, 0, 2)                     # (ns, B, d)
                # 注意：影子缓存的必须是「已转置」的张量——autocast 也是先 transpose 再 cast，
                # 两者布局一致才逐位相同（先 cast 后 transpose 会得到另一种 stride，bmm 分块不同 ⇒ 差 1 ULP）。
                w = _wget(self.kqv_w, ("nsT", proj_idx),
                          lambda: self.kqv_w[: self.ns_len, proj_idx].transpose(1, 2))         # (ns, d, d)^T
                b = _nsbias(self.kqv_b[: self.ns_len, proj_idx].unsqueeze(1))   # bias 保持 fp32：与 baseline 的 fp32 加法逐位一致
                parts.append((torch.bmm(ns_in, w) + b).permute(1, 0, 2))  # (B, ns, d)
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

        # 条件里的 seq_len_q > ns_len / dtype / contiguity 是必需的：Lq == ns_len 的最后一层走的是纯 NS 路径，
        # q 是 fp32 且非连续（bmm+fp32 bias 的 type-promotion），FA 会直接报 161002；那层只有 12 行，留在 SDPA 上更省事。
        if (_FA_BSH and self.if_mask and self.ns_len < seq_len_q < seq_len_k
                and q.is_contiguous() and q.dtype == k.dtype == v.dtype):
            # [G3] q<k 的层：FA 的 BSH 布局，输入输出都是 (B, L, d) 连续张量 ⇒ 不需要 transpose+contiguous
            import torch_npu  # noqa: PLC0415
            blocked = self.create_blocked_mask(seq_len_q, seq_len_k, device=q.device)
            out = torch_npu.npu_fusion_attention(
                q, k, v, self.num_heads, "BSH", atten_mask=blocked,
                scale=float(self.depth) ** -0.5, keep_prob=1.0, sparse_mode=1,
            )[0]
            return _lin(self.dense, out)

        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        attention_mask = None
        _builtin_causal = _SDPA_CAUSAL and self.if_mask and seq_len_q == seq_len_k and self.mask_type == "paper_causal"
        if self.if_mask and not _builtin_causal:
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
            is_causal=_builtin_causal,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.size(0), -1, self.d_model)
        return _lin(self.dense, output)


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
                torch.nn.functional.linear(
                    x,
                    _wget(self.ffn_w1, (gi,), lambda: self.ffn_w1[gi]),
                    _wget(self.ffn_b1, (gi,), lambda: self.ffn_b1[gi]),
                ))
            return torch.nn.functional.silu(
                torch.nn.functional.linear(
                    h,
                    _wget(self.ffn_w2, (gi,), lambda: self.ffn_w2[gi]),
                    _wget(self.ffn_b2, (gi,), lambda: self.ffn_b2[gi]),
                ))
        return self.ffn_list[gi](x)

    def cal_mix_param_ffn(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1) - self.ns_len
        if self.use_bmm and _FULL_PROJ and self.ns_len > 0 and seq_len > 0:
            # [G2] 同 _project_one：全量 FFN，再把 NS 行覆盖成独立权重的结果
            out = self._ffn_one(x, self.ns_len)
            ns_in = x[:, seq_len:, :].permute(1, 0, 2)
            h = torch.nn.functional.silu(
                torch.bmm(ns_in, _wget(self.ffn_w1, ("nsT",), lambda: self.ffn_w1[: self.ns_len].transpose(1, 2)))
                + _nsbias(self.ffn_b1[: self.ns_len].unsqueeze(1)))
            o = torch.bmm(h, _wget(self.ffn_w2, ("nsT",), lambda: self.ffn_w2[: self.ns_len].transpose(1, 2))) \
                + _nsbias(self.ffn_b2[: self.ns_len].unsqueeze(1))
            out[:, seq_len:, :] = torch.nn.functional.silu(o).permute(1, 0, 2)
            return out
        res = []
        if seq_len > 0:
            res.append(self._ffn_one(x[:, :seq_len, :], self.ns_len))
        if self.ns_len > 0:
            if self.use_bmm:
                ns_in = x[:, seq_len:, :].permute(1, 0, 2)                    # (ns, B, d)
                h = torch.nn.functional.silu(
                    torch.bmm(
                        ns_in,
                        _wget(self.ffn_w1, ("nsT",), lambda: self.ffn_w1[: self.ns_len].transpose(1, 2)),
                    )
                    + _nsbias(self.ffn_b1[: self.ns_len].unsqueeze(1)))   # bias 保持 fp32（同 kqv）
                o = torch.bmm(
                    h, _wget(self.ffn_w2, ("nsT",), lambda: self.ffn_w2[: self.ns_len].transpose(1, 2))
                ) + _nsbias(self.ffn_b2[: self.ns_len].unsqueeze(1))
                res.append(torch.nn.functional.silu(o).permute(1, 0, 2))      # (B, ns, d)
            else:
                for i in range(self.ns_len):
                    start = seq_len + i
                    res.append(self.ffn_list[i](x[:, start : start + 1, :]))
        return res[0] if len(res) == 1 else torch.cat(res, dim=1)

    def _attn_part(self, x: torch.Tensor) -> torch.Tensor:
        if _PHASE_MARK:
            with record_function("P_attn"):
                return self._attn_part_impl(x)
        return self._attn_part_impl(x)

    def _attn_part_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rms_0(x)
        k_x, q_x, v_x = x, x, x
        if self.pyramid_stack_len is not None and self.pyramid_stack_len >= self.ns_len:
            q_x = x.narrow(1, x.size(1) - self.pyramid_stack_len, self.pyramid_stack_len)
        origin_x = q_x
        x = self.cma((k_x, q_x, v_x))
        return origin_x + x

    def _ffn_part(self, x: torch.Tensor) -> torch.Tensor:
        if _PHASE_MARK:
            with record_function("P_ffn"):
                return self._ffn_part_impl(x)
        return self._ffn_part_impl(x)

    def _ffn_part_impl(self, x: torch.Tensor) -> torch.Tensor:
        origin_x = x
        x = self.rms_1(x)
        x = self.cal_mix_param_ffn(x)
        return origin_x + x

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self._ffn_part(self._attn_part(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _EXPLICIT_BF16 and x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)          # [O1] 残差流保持 bf16，避免 autocast 反复搬 dtype
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
