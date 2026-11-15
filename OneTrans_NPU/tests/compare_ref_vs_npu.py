#!/usr/bin/env python3
# coding=utf-8
"""GPU 参考实现 vs NPU 迁移实现：逐模块结构 + 逐张量数值 对照。

设计：
  * 参考实现（GPU_Reference）在 **CPU** 上跑（本机无 GPU），NPU 实现在 **npu** 上跑；
  * 每个 side 在独立子进程里跑，`sys.path` 只指向自己那份代码，避免同名模块互相串味；
  * 权重由固定 seed 构造，两侧都是同一份 `models/taac_onetrans.py`（逐字一致）⇒ 参数应逐位相同；
  * 输入由固定 seed 生成后落盘，两侧读同一份 .pt；
  * 输出 logits / loss / 每个参数的梯度，落盘后由 compare 侧统一算 max|Δ| 与 cos。

用法：
  python3 tests/compare_ref_vs_npu.py --all           # 全流程（fp32）
  python3 tests/compare_ref_vs_npu.py --side ref      # 内部用：仅跑参考侧
  python3 tests/compare_ref_vs_npu.py --side npu --dtype bf16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REF_DIR = "/tmp/onetrans_ref"
NPU_DIR = "/tmp/onetrans_npu"
WORK = "/tmp/cmp_ref_npu"

SHAPE = dict(non_seq_dim=183, seq_feature_dim=45, num_classes=2, seq_len=32, ns_len=8,
             d_model=64, num_heads=4, ffn_hidden=128, multi_num=2,
             num_pyramid_layers=3, pyramid_align=32, mask_type="paper_causal")
B, SEED = 8, 1234


# ---------------------------------------------------------------- 单侧执行
def run_side(side: str, dtype: str, mask_env: str) -> None:
    import torch

    root = REF_DIR if side == "ref" else NPU_DIR
    sys.path.insert(0, root)
    dev = torch.device("cpu") if side == "ref" else torch.device("npu:0")
    if side == "npu":
        import torch_npu  # noqa: F401  必须先于任何 npu 张量

    from models.taac_onetrans import TAACOneTransClassifier

    torch.manual_seed(SEED)
    model = TAACOneTransClassifier(**SHAPE).to(dev)
    n_param = sum(p.numel() for p in model.parameters())

    data_path = os.path.join(WORK, "inputs.pt")
    if side == "ref":                      # 参考侧负责造输入，保证两侧同一份数据
        g = torch.Generator().manual_seed(SEED)
        non_seq = torch.randn(B, SHAPE["non_seq_dim"], generator=g)
        seq = torch.randn(B, SHAPE["seq_len"], SHAPE["seq_feature_dim"], generator=g)
        labels = torch.randint(0, SHAPE["num_classes"], (B,), generator=g)
        torch.save({"non_seq": non_seq, "seq": seq, "labels": labels}, data_path)
    d = torch.load(data_path, map_location="cpu")
    non_seq, seq, labels = d["non_seq"].to(dev), d["seq"].to(dev), d["labels"].to(dev)

    model.train()
    ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if dtype == "bf16" else None
    logits, loss = None, None
    if ctx is None:
        logits = model(non_seq, seq)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    else:
        with ctx:
            logits = model(non_seq, seq)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss.backward()

    out = {
        "side": side, "dtype": dtype, "device": str(dev), "mask_env": mask_env,
        "n_param": n_param,
        "logits": logits.detach().float().cpu(),
        "loss": float(loss.detach().float().cpu()),
        "params": {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
        "grads": {k: p.grad.detach().float().cpu() for k, p in model.named_parameters() if p.grad is not None},
    }
    torch.save(out, os.path.join(WORK, f"side_{side}_{dtype}.pt"))
    print(f"[{side}/{dtype}] device={dev} params={n_param} loss={out['loss']:.8f} grad_keys={len(out['grads'])}")


# ---------------------------------------------------------------- 对照
def compare(dtype: str) -> dict:
    import torch

    ref = torch.load(os.path.join(WORK, f"side_ref_fp32.pt"), map_location="cpu")
    npu = torch.load(os.path.join(WORK, f"side_npu_{dtype}.pt"), map_location="cpu")

    def stat(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
        a, b = a.double().flatten(), b.double().flatten()
        d = (a - b).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        denom = a.abs().max().item() or 1.0
        return d, cos, d / denom

    rows = []
    d, cos, rel = stat(ref["logits"], npu["logits"])
    rows.append(("logits (B×C)", d, cos, rel))
    rows.append(("loss (标量)", abs(ref["loss"] - npu["loss"]), 1.0, abs(ref["loss"] - npu["loss"]) / (abs(ref["loss"]) or 1)))

    # 参数：两侧同一份模型代码 + 同一 seed ⇒ 应逐位相同
    same_keys = list(ref["params"]) == list(npu["params"])
    pdiff = max((ref["params"][k] - npu["params"][k]).abs().max().item() for k in ref["params"]) if same_keys else float("nan")

    # 梯度：按参数逐个比。两侧量级都低于 1e-8 的视为「数值零」（mask 乘 0 导致），
    # 单独统计、不参与 rel_max —— 否则 0 与 2.8e-10 会被算成 rel=1.0（分母陷阱）。
    all_max = max(max(g.abs().max().item() for g in ref["grads"].values()),
                  max(g.abs().max().item() for g in npu["grads"].values()))
    zero_tol = max(1e-8, all_max * 1e-10)
    gmax, per_grad, n_zero = 0.0, [], 0
    for k, gr in ref["grads"].items():
        gn = npu["grads"].get(k)
        if gn is None:
            per_grad.append((k, float("nan"), float("nan")))
            continue
        scale = max(gr.abs().max().item(), gn.abs().max().item())
        if scale < zero_tol:
            n_zero += 1
            per_grad.append((k, 0.0, 1.0))
            continue
        rel_k = (gr - gn).abs().max().item() / scale
        gmax = max(gmax, rel_k)
        per_grad.append((k, rel_k, torch.nn.functional.cosine_similarity(
            gr.double().flatten(), gn.double().flatten(), dim=0).item()))
    per_grad.sort(key=lambda x: -x[1])

    return {"rows": rows, "same_keys": same_keys, "param_max_abs": pdiff,
            "grad_max_rel": gmax, "top_grads": per_grad[:5], "n_grads": len(per_grad),
            "n_zero_grad": n_zero, "zero_tol": zero_tol, "grad_absmax": all_max}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["ref", "npu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    if a.side:
        run_side(a.side, a.dtype, os.environ.get("ONETRANS_ATTN_MASK", "float"))
        return 0

    if a.compare_only:
        _report_only()
        return 0

    env = {**os.environ, "ONETRANS_ATTN_MASK": os.environ.get("ONETRANS_ATTN_MASK", "float")}
    subprocess.run([sys.executable, __file__, "--side", "ref"], check=True, env=env)
    for dt in ("fp32", "bf16"):
        e = {**env, "ASCEND_RT_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1"}
        subprocess.run([sys.executable, __file__, "--side", "npu", "--dtype", dt], check=True, env=e)

    _print_report({dt: compare(dt) for dt in ("fp32", "bf16")})
    print("\nCOMPARE DONE")
    return 0


def _print_report(out: dict) -> None:
    print("\n===== GPU 实现(CPU 执行) vs NPU 实现：同权重同输入 =====")
    for dt, r in out.items():
        print(f"\n-- NPU {dt} --")
        for name, d, cos, rel in r["rows"]:
            print(f"  {name:22s} max|Δ|={d:.3e}  rel={rel:.2e}  cos={cos:.8f}")
        print(f"  参数逐位相同: keys_equal={r['same_keys']} max|Δ|={r['param_max_abs']:.3e}")
        print(f"  梯度 rel_max={r['grad_max_rel']:.3e} (n={r['n_grads']}，其中数值零梯度 {r['n_zero_grad']} 个，"
              f"判零阈值 {r['zero_tol']:.1e}，全局 |g|max={r['grad_absmax']:.3e})")
        for k, rk, ck in r["top_grads"]:
            print(f"    {k:46s} rel={rk:.2e} cos={ck:.8f}")
    print(json.dumps({dt: {"grad_max_rel": r["grad_max_rel"], "n_zero_grad": r["n_zero_grad"],
                           "param_max_abs": r["param_max_abs"]} for dt, r in out.items()},
                     ensure_ascii=False))


def _report_only() -> None:
    _print_report({dt: compare(dt) for dt in ("fp32", "bf16")})


if __name__ == "__main__":
    raise SystemExit(main())
