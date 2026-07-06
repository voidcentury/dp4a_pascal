"""Benchmark baseline: dequant → fp32 → F.linear.

Measures:
  - Median wall-clock time (100 iters, warmup 10)
  - Peak VRAM usage (via torch.cuda.max_memory_allocated)
"""
import torch
import time
from common import load_mlp_weights, dequant_weight

torch.manual_seed(42)

def bench_layer(name, w_int8, scale, bias, M, num_iters=100, warmup=10):
    N, K = w_int8.shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = dequant_weight(w_int8, scale, "cuda")
    b = bias.to("cuda").float() if bias is not None else None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _ = torch.nn.functional.linear(x, w_fp32, b)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.nn.functional.linear(x, w_fp32, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / num_iters * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return avg_ms, peak_mb, x.shape, (N, K)

def main():
    w = load_mlp_weights()

    print(f"{'Layer':>10} {'M':>6} {'Shape(x)':>18} {'Shape(w)':>16} {'Time(ms)':>10} {'Peak(MB)':>10}")
    print("-" * 80)

    for name, w_int8, scale, bias in [
        ("ffn.0", w["ffn0_w_int8"], w["ffn0_scale"], w["ffn0_bias"]),
        ("ffn.2", w["ffn2_w_int8"], w["ffn2_scale"], w["ffn2_bias"]),
    ]:
        for M in [1, 4, 16, 64, 128, 256, 512]:
            avg_ms, peak_mb, x_shape, w_shape = bench_layer(
                name, w_int8, scale, bias, M
            )
            x_str = f"({M}, {x_shape[-1]})"
            w_str = f"({w_shape[0]}, {w_shape[1]})"
            print(f"{name:>10} {M:>6} {x_str:>18} {w_str:>16} {avg_ms:>10.3f} {peak_mb:>10.1f}")

if __name__ == "__main__":
    main()
