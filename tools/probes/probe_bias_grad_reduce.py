#!/usr/bin/env python3
"""probe_bias_grad_reduce.py — Linear 反向的 bias 梯度 (dy.sum(0)) 到底能不能更快?

背景（16 die profile, 3 稳态步, results/prof16_o1_kernel_details.csv）:
  aclnnReduceSum 每步 200 次 / 21.05 ms（3 步完全一致），且时长谱是双峰：
  p50 = 10.3 us，但 p75 = 138.8 us、max = 1173 us。
  邻居分析（同 stream 按 Start Time 排序）显示大号的都长这样：
      Mm_MatMulV3 → Mm_MatMulV3 → ReduceSum(0.46~1.17 ms) → Add → Cast(0.002)
  这是 F.linear 反向的标准形状: dW = dy^T@x (Mm), dx = dy@W (Mm), db = dy.sum(0) (ReduceSum), 再 Add 累加。
  ⇒ ReduceSum 是 **Linear 的 bias 梯度**，不是神秘的第三方来源。

本探针在真实形状上对比 3 种算法:
  A  现网: dy.sum(0)                          (vector ReduceSum)
  B  cube: ones(1,M) @ dy                     (mm, M=1)
  C  cube+累加: addmm_(db, ones, dy, beta=1)   (省掉后面的 Add)
另外给 D: dy.t().sum(1)? 不测，已知更差。
每个形状测 bf16 / fp32 两种 dtype（干路 bf16 下 dy 是 bf16，权重梯度累加区是 fp32）。
"""
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"

# 档 B: batch 1024/micro-batch, pyramid [1203,960,736,480,256,12], d=256
LS = [1203, 960, 736, 480, 256, 12]
B = 1024
D = 256


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    print("=" * 96)
    print("Linear bias 梯度：dy.sum(0) vs ones@dy vs addmm 累加")
    print("=" * 96)
    print(f"{'shape':>14s} {'dtype':>7s} {'A sum(0)':>10s} {'B ones@dy':>10s} {'C addmm':>10s} {'B/A':>7s} {'C/A':>7s} {'校验':>10s}")
    tot = {"A": 0.0, "B": 0.0, "C": 0.0}
    for L in LS:
        M = B * L
        for dt in (torch.bfloat16, torch.float32):
            dy = torch.randn(M, D, device=dev, dtype=dt).abs()
            ones = torch.ones(1, M, device=dev, dtype=dt)
            db = torch.zeros(1, D, device=dev, dtype=torch.float32)
            ref = dy.sum(0).float()

            a = bench(lambda: dy.sum(0))
            b = bench(lambda: torch.mm(ones, dy))
            got_b = torch.mm(ones, dy).float()
            # C: 原地累加到 fp32 缓冲（beta=1）
            def c_fn():
                torch.addmm(db, ones, dy, beta=1.0, alpha=1.0)
            c = bench(c_fn)
            torch.addmm(db, ones, dy, beta=1.0, alpha=1.0)
            got_c = db.clone().squeeze(0)

            err_b = (got_b - ref).abs().max().item()
            err_c = (got_c - ref).abs().max().item()
            ok = "OK" if max(err_b, err_c) < 2e-2 * ref.abs().max().item() else "DIFF"
            print(f"{f'{M}x{D}':>14s} {str(dt).split('.')[-1]:>7s} {a:10.3f} {b:10.3f} {c:10.3f} "
                  f"{a/b:6.2f}x {a/c:6.2f}x {ok:>10s}")
            print(f"{'':>14s} {'':>7s} 校验细节: B max|Δ|={err_b:.3e}  C max|Δ|={err_c:.3e}  ref absmax={ref.abs().max().item():.3e}")
            if dt is torch.bfloat16:
                tot["A"] += a
                tot["B"] += b
                tot["C"] += c
    print("-" * 96)
    print(f"按 6 个 L 求和（bf16，每个 L 计 1 次）: sum(0)={tot['A']:.2f} ms  ones@dy={tot['B']:.2f} ms  addmm={tot['C']:.2f} ms")
    print(f"⇒ 单步若 14 个带 bias 的 Linear（2 micro-batch × 7 block）全部替换，估算: "
          f"{(tot['A']-tot['C'])*14/6:.1f} ms 收益（当前 profile 实测 ReduceSum 合计 21.05 ms/步）")


if __name__ == "__main__":
    main()
