#!/usr/bin/env python3
"""prep_taobao_ad.py — TaobaoAd_x1 (reczoo) csv → 训练用 mmap 数组（pyarrow 向量化版）

输入 train.csv（10 GB, 21.9 M 行, 历史列表用 '^' 分隔、定长 50）
输出 out_dir：
  ns_ids.i32   [N, 16]  类别/计数特征 id（price 单独 float）
  price.f32    [N]
  seq_cate.i32 [N, L]   item 序列（cate_his）
  seq_brand.i32[N, L]   item 侧信息（brand_his）
  seq_btag.i32 [N, L]   行为类型（btag_his）
  clk.i8       [N]
  meta.json

用法：python3 tools/prep_taobao_ad.py --csv /tmp/taobao/train.csv --out /tmp/taobao/prep --seq-len 50
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

NS_COLS = ["userid", "cms_segid", "cms_group_id", "final_gender_code", "age_level",
           "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
           "adgroup_id", "cate_id", "campaign_id", "customer", "brand", "pid", "btag"]
HIS_COLS = ["cate_his", "brand_his", "btag_his"]


def his_to_array(col: pa.ChunkedArray, n: int, L: int) -> np.ndarray:
    """'a^b^c' 定长列表 → [n, L] int32（向量化：offsets + scatter）"""
    lst = pc.split_pattern(col, "^").combine_chunks()
    # 存在空串（如 "1^^3"）⇒ 先补成 "0" 再转 int32
    flat = pc.if_else(pc.equal(lst.values, ""), pa.scalar("0"), lst.values)
    offs = np.asarray(lst.offsets).astype(np.int64)          # [n+1]
    vals = np.asarray(pc.cast(flat, pa.int32())).astype(np.int32)   # [total]
    counts = np.diff(offs)
    out = np.zeros((n, L), dtype=np.int32)
    rows = np.repeat(np.arange(n, dtype=np.int64), counts)
    cols = np.arange(vals.size, dtype=np.int64) - np.repeat(offs[:-1], counts)
    keep = cols < L
    out[rows[keep], cols[keep]] = vals[keep]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    cols = ["clk"] + HIS_COLS + NS_COLS + ["price"]
    ctypes = {c: pa.string() for c in HIS_COLS}
    ctypes.update({c: pa.int32() for c in NS_COLS + ["clk"]})
    ctypes["price"] = pa.float32()
    tab = pacsv.read_csv(args.csv, read_options=pacsv.ReadOptions(use_threads=True),
                         convert_options=pacsv.ConvertOptions(include_columns=cols,
                                                              column_types=ctypes))
    n = tab.num_rows
    print(f"[prep] rows={n:,}  read {time.time()-t0:.0f}s", flush=True)

    ns = np.empty((n, len(NS_COLS)), dtype=np.int32)
    vmax = {}
    for i, c in enumerate(NS_COLS):
        a = np.asarray(tab[c].combine_chunks())
        ns[:, i] = a
        vmax[c] = int(a.max())
    print(f"[prep] ns ok {time.time()-t0:.0f}s", flush=True)

    # 21.9M × 50 = 1.1e9 个元素 > int32 offsets 上限 ⇒ 必须切片处理
    CH = 2_000_000
    seq_max = {}
    for name, c in zip(("cate", "brand", "btag"), HIS_COLS):
        mm = np.memmap(f"{args.out}/seq_{name}.i32", dtype=np.int32, mode="w+", shape=(n, args.seq_len))
        for s in range(0, n, CH):
            m = min(CH, n - s)
            mm[s:s + m] = his_to_array(tab[c][s:s + m], m, args.seq_len)
        seq_max[name] = int(mm.max())
        mm.flush()
        del mm
        print(f"[prep] seq_{name} ok max={seq_max[name]} {time.time()-t0:.0f}s", flush=True)

    np.save(f"{args.out}/ns_ids.i32.npy", ns)
    np.save(f"{args.out}/price.f32.npy", np.asarray(tab["price"].combine_chunks()))
    np.save(f"{args.out}/clk.i8.npy", np.asarray(tab["clk"].combine_chunks()).astype(np.int8))

    meta = {"rows": n, "seq_len": args.seq_len, "ns_cols": NS_COLS, "ns_cardinality": vmax,
            "seq_cardinality": seq_max,
            "ctr": float(np.asarray(tab["clk"].combine_chunks()).mean()),
            "files": {"ns_ids.i32.npy": [n, len(NS_COLS)], "price.f32.npy": [n],
                      "seq_cate.i32": [n, args.seq_len], "seq_brand.i32": [n, args.seq_len],
                      "seq_btag.i32": [n, args.seq_len], "clk.i8.npy": [n]}}
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("[prep] done", json.dumps(vmax), f"ctr={meta['ctr']:.4f}", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
