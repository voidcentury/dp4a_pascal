"""Benchmark DP4A kernel vs baseline (dequant → fp32 → F.linear).

Measures:
  - Median wall-clock time (100 iters, warmup 10)
  - Peak VRAM usage including persistent weight storage
"""
import torch
import time
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from common import load_mlp_weights

torch.manual_seed(42)

try:
    import dp4a_ext
except ImportError:
    print("dp4a_ext not built yet. Run: python setup.py build_ext --inplace")
    sys.exit(1)


def bench_baseline(x, w_int8, scale, bias, num_iters=100, warmup=10):
    # Persistent allocations (weight lives in VRAM as fp32)
    torch.cuda.reset_peak_memory_stats()
    w_fp32 = w_int8.to("cuda").float() * scale.to("cuda").item()
    b_fp32 = bias.to("cuda").float() if bias is not None else None
    peak_persistent = torch.cuda.max_memory_allocated()

    for _ in range(warmup):
        _ = torch.nn.functional.linear(x, w_fp32, b_fp32)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = torch.nn.functional.linear(x, w_fp32, b_fp32)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / num_iters * 1000
    peak_incr = torch.cuda.max_memory_allocated()
    total_mb = (peak_persistent + peak_incr) / (1024 * 1024)
    return avg_ms, total_mb


def bench_kernel(x, w_int8, scale, bias, num_iters=100, warmup=10):
    # Persistent allocations (weights stay in int8)
    torch.cuda.reset_peak_memory_stats()
    w_d = w_int8.to("cuda").contiguous()
    s_d = scale.to("cuda")
    b_d = bias.to("cuda").contiguous() if bias is not None else None
    peak_persistent = torch.cuda.max_memory_allocated()

    for _ in range(warmup):
        _ = dp4a_ext.int8_linear(x, w_d, s_d, b_d)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = dp4a_ext.int8_linear(x, w_d, s_d, b_d)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / num_iters * 1000
    peak_incr = torch.cuda.max_memory_allocated()
    total_mb = (peak_persistent + peak_incr) / (1024 * 1024)
    return avg_ms, total_mb


def main():
    w = load_mlp_weights()

    hdr = f"{'Layer':>10} {'M':>6} {'Time_base(ms)':>14} {'Time_kern(ms)':>14} "
    hdr += f"{'Speedup':>8} {'Peak_base(MB)':>14} {'Peak_kern(MB)':>14} {'VRAM_save':>10}"
    print(hdr)
    print("-" * 100)

    for name, wi, sc, bi in [
        ("ffn.0", w["ffn0_w_int8"], w["ffn0_scale"], w["ffn0_bias"]),
        ("ffn.2", w["ffn2_w_int8"], w["ffn2_scale"], w["ffn2_bias"]),
    ]:
        for M in [1, 4, 16, 64, 128, 256, 512]:
            N, K = wi.shape
            x = torch.randn(M, K, device="cuda", dtype=torch.float32)

            t_base, p_base = bench_baseline(x, wi, sc, bi)
            t_kern, p_kern = bench_kernel(x, wi, sc, bi)

            speedup = t_base / t_kern if t_kern > 0 else float("inf")
            vram_save = p_base - p_kern

            print(f"{name:>10} {M:>6} {t_base:>14.3f} {t_kern:>14.3f} "
                  f"{speedup:>7.2f}x {p_base:>14.1f} {p_kern:>14.1f} {vram_save:>+9.1f}")


if __name__ == "__main__":
    main()
