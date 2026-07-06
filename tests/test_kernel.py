"""Compare DP4A kernel output against the dequant → fp32 → F.linear baseline."""
import torch
import sys
from pathlib import Path

# Add project root to path so dp4a_ext is found
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from common import load_mlp_weights, dequant_weight, baseline_linear

torch.manual_seed(42)

try:
    import dp4a_ext
except ImportError:
    print("dp4a_ext not built yet. Run: python setup.py build_ext --inplace")
    sys.exit(1)


def test_single(name, w_int8, scale, bias, M=1, rtol=0.01, atol=0.2):
    N, K = w_int8.shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)

    out_ref = baseline_linear(x, w_int8, scale, bias)
    out_kernel = dp4a_ext.int8_linear(x, w_int8.to("cuda"), scale.to("cuda"),
                                       bias.to("cuda") if bias is not None else None)

    err = (out_kernel - out_ref).abs()
    max_err = err.max().item()
    mean_err = err.mean().item()
    passed = torch.allclose(out_kernel, out_ref, rtol=rtol, atol=atol)
    cos_sim = torch.nn.functional.cosine_similarity(
        out_kernel.flatten(), out_ref.flatten(), dim=0).item()

    print(f"[{name} M={M:4d}] max_err={max_err:.4f}  mean_err={mean_err:.4f}  "
          f"cos={cos_sim:.6f}  {'✓' if passed else '✗ FAIL'}")
    assert cos_sim > 0.999, f"{name} M={M}: cos_sim={cos_sim:.6f} too low"


def test_conditions():
    """Verify gate conditions are enforced."""
    w = load_mlp_weights()

    # K % 4 != 0 should raise
    w_bad = torch.randint(-127, 127, (64, 5), dtype=torch.int8, device="cuda")
    s_bad = torch.tensor(0.01, device="cuda")
    try:
        dp4a_ext.int8_linear(torch.randn(1, 5, device="cuda"), w_bad, s_bad, None)
        print("[condition K%4==0] ✗ should have raised")
    except RuntimeError as e:
        print(f"[condition K%4==0] correctly rejected: {e}")


def main():
    w = load_mlp_weights()

    print("=== DP4A Kernel correctness ===")
    for name, wi, sc, bi in [
        ("ffn.0", w["ffn0_w_int8"], w["ffn0_scale"], w["ffn0_bias"]),
        ("ffn.2", w["ffn2_w_int8"], w["ffn2_scale"], w["ffn2_bias"]),
    ]:
        for M in [1, 4, 16, 64, 128, 512]:
            test_single(name, wi, sc, bi, M=M)

    print("\n=== 3D input ===")
    wi, sc, bi = w["ffn0_w_int8"], w["ffn0_scale"], w["ffn0_bias"]
    x = torch.randn(2, 64, 3072, device="cuda", dtype=torch.float32)
    out_ref = baseline_linear(x, wi, sc, bi)
    out_kernel = dp4a_ext.int8_linear(x, wi.to("cuda"), sc.to("cuda"), bi.to("cuda"))
    assert out_kernel.shape == out_ref.shape
    cos_sim = torch.nn.functional.cosine_similarity(
        out_kernel.flatten(), out_ref.flatten(), dim=0).item()
    assert cos_sim > 0.999, f"3D cos_sim={cos_sim:.6f}"
    print(f"[3d input] {list(x.shape)} -> {list(out_kernel.shape)} cos={cos_sim:.6f} ✓")

    # Test bias=None
    print("\n=== No bias ===")
    wi, sc, _ = w["ffn0_w_int8"], w["ffn0_scale"], w["ffn0_bias"]
    x = torch.randn(16, 3072, device="cuda", dtype=torch.float32)
    out_ref = baseline_linear(x, wi, sc, None)
    out_kernel = dp4a_ext.int8_linear(x, wi.to("cuda"), sc.to("cuda"), None)
    cos_sim = torch.nn.functional.cosine_similarity(
        out_kernel.flatten(), out_ref.flatten(), dim=0).item()
    assert cos_sim > 0.999
    print(f"[no bias] cos={cos_sim:.6f} ✓")

    print("\n=== Gate conditions ===")
    test_conditions()

    print("\nAll kernel tests passed ✓")


if __name__ == "__main__":
    main()
