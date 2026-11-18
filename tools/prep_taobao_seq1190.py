#!/usr/bin/env python3
"""prep_taobao_seq1190.py — 把 TaobaoAd_x1 的 50 长度行为窗口拼成长序列（默认 1190）

背景：TaobaoAd_x1（reczoo x1）每个曝光只带定长 50 的行为序列（cate/brand/btag his），
没有 1190 长度的用户历史。要跑论文 S 的 seq=1190，只能用**同一用户**的多次曝光窗口拼接：

  样本 i（保留原行 i 的 NS 特征与 clk 标签）的序列 = 该用户各曝光行 his 窗口顺次拼接，
  取前 seq_len 个位置，不足补 0。
  该用户曝光行数 ~19（均值），最多拼 24 窗 × 50 = 1200 ≥ 1190。

⚠️ 语义边界（报告里必须写明）：这是"同一用户 50 窗口拼接"，不是论文的真实 1190 长历史；
  用户内相邻窗口高度重叠（每个窗口都是该用户当时的最近 50 次行为）⇒ 序列存在重复片段。

用法：
  python3 tools/prep_taobao_seq1190.py --src /tmp/taobao/prep --out /tmp/taobao/prep1190 \
      --rows 4000000 --seq-len 1190
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def build_user_index(user: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """user → CSR（uniq, offsets, order_rows）"""
    order = np.argsort(user, kind="stable")
    su = user[order]
    uniq, first = np.unique(su, return_index=True)
    offsets = np.r_[first, len(su)].astype(np.int64)
    return uniq.astype(np.int64), offsets, order.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="prep/ 目录（50 长度窗口）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4_000_000, help="输出样本数")
    ap.add_argument("--seq-len", type=int, default=1190)
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    ns = np.load(f"{args.src}/ns_ids.i32.npy", mmap_mode="r")
    price = np.load(f"{args.src}/price.f32.npy", mmap_mode="r")
    clk = np.load(f"{args.src}/clk.i8.npy", mmap_mode="r")
    n_all, n_ns = ns.shape
    cat = np.memmap(f"{args.src}/seq_cate.i32", np.int32, "r", shape=(n_all, args.window))
    bra = np.memmap(f"{args.src}/seq_brand.i32", np.int32, "r", shape=(n_all, args.window))
    btag = np.memmap(f"{args.src}/seq_btag.i32", np.int32, "r", shape=(n_all, args.window))

    M = min(args.rows, n_all)
    n_win = (args.seq_len + args.window - 1) // args.window
    out_len = n_win * args.window                       # 1200（后切到 1190）

    print(f"[prep] rows={M:,} seq_len={args.seq_len} windows={n_win} out_len={out_len}", flush=True)
    user = np.asarray(ns[:, 0], dtype=np.int64)
    uniq, offsets, order = build_user_index(user)
    print(f"[prep] users={len(uniq):,} index built {time.time()-t0:.1f}s", flush=True)

    # 每个用户的行号区间（在 user 排序后的 order 里定位）
    def user_rows(uid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.searchsorted(uniq, uid)
        lo, hi = offsets[k], offsets[k + 1]
        return lo, hi

    # 序列字段写**裸二进制**（无 .npy 头）：TaobaoAdSource 按 np.memmap 裸读，
    # 写成 .npy 会整体错位 128 字节，序列基数被算成天文数字（实测触发 16 GiB 建表 OOM）
    d_cate = np.memmap(f"{args.out}/seq_cate.i32", np.int32, "w+", shape=(M, args.seq_len))
    d_brand = np.memmap(f"{args.out}/seq_brand.i32", np.int32, "w+", shape=(M, args.seq_len))
    d_btag = np.memmap(f"{args.out}/seq_btag.i32", np.int32, "w+", shape=(M, args.seq_len))
    max_ids = [0, 0, 0]
    d_ns = np.lib.format.open_memmap(f"{args.out}/ns_ids.i32.npy", "w+", np.int32, (M, n_ns))
    d_price = np.lib.format.open_memmap(f"{args.out}/price.f32.npy", "w+", np.float32, (M,))
    d_clk = np.lib.format.open_memmap(f"{args.out}/clk.i8.npy", "w+", np.int8, (M,))

    chunk = 50_000
    win_offsets = np.arange(n_win, dtype=np.int64) * args.window
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        uid = user[s:e]
        lo, hi = user_rows(uid)
        cnt = hi - lo
        # 每样本取 min(cnt, n_win) 个窗口，不足则重复最后一个窗口
        idx = np.minimum(np.arange(n_win)[None, :], (cnt[:, None] - 1))
        picked = order[lo[:, None] + idx]                       # [b, n_win]
        c = np.asarray(cat[np.asarray(picked)])                 # [b, n_win, 50]
        b = np.asarray(bra[np.asarray(picked)])
        t = np.asarray(btag[np.asarray(picked)])
        flat_c = c.reshape(e - s, out_len)[:, : args.seq_len]
        flat_b = b.reshape(e - s, out_len)[:, : args.seq_len]
        flat_t = t.reshape(e - s, out_len)[:, : args.seq_len]
        d_cate[s:e] = flat_c
        d_brand[s:e] = flat_b
        d_btag[s:e] = flat_t
        for j, arr in enumerate((flat_c, flat_b, flat_t)):
            max_ids[j] = max(max_ids[j], int(arr.max()))
        d_ns[s:e] = np.asarray(ns[s:e])
        d_price[s:e] = np.asarray(price[s:e])
        d_clk[s:e] = np.asarray(clk[s:e])
        if (s // chunk) % 10 == 0:
            print(f"[prep] {e:,}/{M:,} {time.time()-t0:.0f}s", flush=True)

    for d in (d_cate, d_brand, d_btag, d_ns, d_price, d_clk):
        d.flush()

    # 自检：抽 3 个样本，按"该用户窗口顺次拼接"重算并比对落盘值
    sc = np.memmap(f"{args.src}/seq_cate.i32", np.int32, "r", shape=(n_all, args.window))
    for i in (0, 7, M - 1):
        uid = int(user[i])
        lo, hi = user_rows(np.array([uid]))
        rows = order[int(lo[0]):int(hi[0])][:n_win]
        exp = np.concatenate([np.asarray(sc[int(r)]) for r in rows])[: args.seq_len]
        got = np.asarray(np.memmap(f"{args.out}/seq_cate.i32", np.int32, "r",
                                   shape=(M, args.seq_len))[i])
        if not np.array_equal(exp, got):
            raise SystemExit(f"[prep][FATAL] 自检失败 sample={i} uid={uid}")
    print("[prep] self-check OK (3 samples)", flush=True)

    src_meta = json.load(open(f"{args.src}/meta.json"))
    meta = {
        "rows": int(M),
        "seq_len": int(args.seq_len),
        "seq_construction": (
            f"同用户拼接：取该用户各曝光行的 {args.window} 长度行为窗口顺次拼接，"
            f"最多 {n_win} 窗，切到 {args.seq_len}；不足补 0。"
        ),
        "seq_semantics_caveat": "同一用户窗口高度重叠，非论文真实长历史",
        "source": args.src,
        "ns_cols": src_meta.get("ns_cols"),
        "seq_cardinality": [int(m) + 1 for m in max_ids],
        "ns_cardinality": src_meta.get("ns_cardinality"),
        "ctr": src_meta.get("ctr"),
        "files": {
            "ns_ids.i32.npy": [int(M), int(n_ns)],
            "price.f32.npy": [int(M)],
            "seq_cate.i32": [int(M), int(args.seq_len)],
            "seq_brand.i32": [int(M), int(args.seq_len)],
            "seq_btag.i32": [int(M), int(args.seq_len)],
            "clk.i8.npy": [int(M)],
        },
    }
    json.dump(meta, open(f"{args.out}/meta.json", "w"), ensure_ascii=False, indent=2)
    print(f"[prep] done {M:,} rows in {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
