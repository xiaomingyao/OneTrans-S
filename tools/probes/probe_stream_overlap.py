#!/usr/bin/env python3
"""probe_stream_overlap.py — 关键问题：cube 与 vector 能不能**同时**跑？

背景：模型里每步 kernel 时长之和 636.1 ms，而设备"忙"的并集只有 605.5 ms ⇒ 只有 4.8% 重叠，
即 cube(348.4) 与 vector(~240) **几乎完全串行**。若能重叠，步时的下界从 636 掉到 max(348,240)=348。

同时顺手测 int8 GEMM 速度（fp8 路径已判死：torch._scaled_mm 0.3 TF/s、npu_quant_matmul(fp8) 161002）。
"""
import time

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
dev = "npu:0"


def bench(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


M, K, N = 1231872, 256, 1024
A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
X = torch.randn(M, 256, device=dev, dtype=torch.bfloat16)
Y = torch.randn(M, 256, device=dev, dtype=torch.bfloat16)
Z = torch.empty_like(X)

print("=" * 100)
print("[A] 单算子基准")
t_cube = bench(lambda: torch.mm(A, B))
t_vec = bench(lambda: torch.add(X, Y, out=Z))
print(f"  cube  mm(1.23M,256)@(256,1024)  {t_cube:7.3f} ms  ({2*M*K*N/t_cube/1e9:.0f} TF/s)")
print(f"  vector add(1.23M,256)           {t_vec:7.3f} ms  ({3*X.numel()*2/t_vec/1e9*1000:.0f} GB/s)")

print("\n[B] 串行 vs 双流并发")
t_seq = bench(lambda: (torch.mm(A, B), torch.add(X, Y, out=Z)))
print(f"  同流串行 (mm 后 add)            {t_seq:7.3f} ms")

if hasattr(torch.npu, "Stream"):
    s2 = torch.npu.Stream()
    def concurrent():
        ev = torch.npu.Event()
        cur = torch.npu.current_stream()
        ev.record(cur)
        with torch.npu.stream(s2):
            s2.wait_event(ev)
            torch.mm(A, B)
            ev2 = torch.npu.Event()
            ev2.record(s2)
        torch.add(X, Y, out=Z)
        cur.wait_event(ev2)
    try:
        t_con = bench(concurrent)
        ideal = max(t_cube, t_vec)
        print(f"  双流并发 (cube∥vector)          {t_con:7.3f} ms   理想下界 {ideal:7.3f} ms"
              f"   并发效率 {(t_seq - t_con) / (t_seq - ideal) * 100:.0f}%")
        print(f"  重叠掉的时间 {t_seq - t_con:.3f} ms（理论最多 {t_seq - ideal:.3f} ms）")
    except Exception as e:  # noqa: BLE001
        print(f"  双流并发 FAIL {type(e).__name__}: {str(e)[:150]}")
else:
    print("  本环境无 torch.npu.Stream")

print("\n[C] int8 GEMM 速度（fp8 已判死，int8 是唯一能跑的量化路径）")
Aq = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
Bq = torch.randint(-127, 127, (K, N), device=dev, dtype=torch.int8)
sc = torch.tensor([1.0], device=dev, dtype=torch.float32)
try:
    t_i8 = bench(lambda: torch_npu.npu_quant_matmul(Aq, Bq, sc))
    print(f"  npu_quant_matmul(int8)          {t_i8:7.3f} ms  ({2*M*K*N/t_i8/1e9:.0f} TF/s)  相对 bf16 {t_cube/t_i8:.2f}x")
except Exception as e:  # noqa: BLE001
    print(f"  int8 FAIL {type(e).__name__}: {str(e)[:150]}")
