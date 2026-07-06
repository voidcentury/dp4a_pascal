"""Benchmark with realistic video generation token counts."""
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
    print("dp4a_ext not built yet")
    sys.exit(1)

TOKENS = [1, 128, 512, 880, 1024, 2048, 4096, 6144, 8192,
          12288, 16384, 24576, 27280, 28672, 32768]


def bench(fn_make, x, niters):
    """Run fn_make() niters times, return avg_ms + peak_mb."""
    fn = fn_make()
    for _ in range(min(3, niters)):
        fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(niters):
        fn()
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / niters * 1000
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return avg_ms, peak_mb


def main():
    w = load_mlp_weights()

    print(f"{'Layer':>22} {'Tok':>6} {'Base(ms)':>9} {'Kern(ms)':>9} {'Spd':>6} {'BaseMB':>7} {'KrnMB':>7} {'ΔMB':>6}")
    print("-" * 80)

    for name, wi, sc, bi in [
        ("ffn.0 [14336,3072]", w["ffn0_w_int8"], w["ffn0_scale"], w["ffn0_bias"]),
        ("ffn.2 [3072,14336]", w["ffn2_w_int8"], w["ffn2_scale"], w["ffn2_bias"]),
    ]:
        N, K = wi.shape
        w_d = wi.to("cuda").contiguous()
        s_d = sc.to("cuda")
        b_d = bi.to("cuda").contiguous() if bi is not None else None

        w_f = w_d.float() * sc.item()
        b_f = b_d.float() if b_d is not None else None

        for M in TOKENS:
            x = torch.randn(M, K, device="cuda", dtype=torch.float32)

            # Pick niters based on M: smaller M = more iters
            niters = max(10, min(500, int(10000 / max(M, 1))))

            t_base, p_base = bench(lambda: (lambda: torch.nn.functional.linear(x, w_f, b_f)), x, niters)
            t_kern, p_kern = bench(lambda: (lambda: dp4a_ext.int8_linear(x, w_d, s_d, b_d)), x, niters)

            spd = t_base / t_kern if t_kern > 0 else 0
            dmem = p_base - p_kern
            print(f"{name:>22} {M:>6} {t_base:>9.3f} {t_kern:>9.3f} {spd:>5.1f}x {p_base:>7.1f} {p_kern:>7.1f} {dmem:>+6.1f}", flush=True)


if __name__ == "__main__":
    main()
