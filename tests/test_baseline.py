"""Verify the dequant → fp32 → F.linear baseline produces correct outputs for various shapes."""
import torch
from common import load_mlp_weights, dequant_weight

torch.manual_seed(42)

def test_dequant_correctness():
    w = load_mlp_weights()
    # Verify dequant math: fp32 ≈ int8 * scale
    for name in ["ffn0", "ffn2"]:
        w_int8 = w[f"{name}_w_int8"]
        scale = w[f"{name}_scale"]
        w_fp32 = dequant_weight(w_int8, scale, "cpu")
        ref = w_int8.float() * scale.item()
        err = (w_fp32 - ref).abs().max().item()
        print(f"[{name}] dequant max error: {err:.2e} {'✓' if err < 1e-6 else '✗'}")
        assert err < 1e-6, f"dequant mismatch: {err:.2e}"

def run_baseline_forward(name, w_int8, scale, bias, M=1):
    N, K = w_int8.shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = dequant_weight(w_int8, scale, "cuda")
    b = bias.to("cuda").float() if bias is not None else None
    out = torch.nn.functional.linear(x, w_fp32, b)
    return x, out

def test_baseline_shapes():
    w = load_mlp_weights()
    for M in [1, 4, 16, 128, 512]:
        x, out = run_baseline_forward("ffn0", w["ffn0_w_int8"], w["ffn0_scale"],
                                       w["ffn0_bias"], M=M)
        print(f"[ffn0 M={M:4d}] x={list(x.shape)} out={list(out.shape)} ✓")
        assert out.shape == (M, 14336)

        x, out = run_baseline_forward("ffn2", w["ffn2_w_int8"], w["ffn2_scale"],
                                       w["ffn2_bias"], M=M)
        print(f"[ffn2 M={M:4d}] x={list(x.shape)} out={list(out.shape)} ✓")
        assert out.shape == (M, 3072)

def test_3d_input():
    """F.linear handles 3D natively; our kernel must too or we reshape."""
    w = load_mlp_weights()
    B, S, K = 2, 64, 3072
    x = torch.randn(B, S, K, device="cuda", dtype=torch.float32)
    w_fp32 = dequant_weight(w["ffn0_w_int8"], w["ffn0_scale"], "cuda")
    b = w["ffn0_bias"].to("cuda").float()
    out = torch.nn.functional.linear(x, w_fp32, b)
    print(f"[3d input] {list(x.shape)} -> {list(out.shape)} ✓")
    assert out.shape == (B, S, 14336)

if __name__ == "__main__":
    print("=== Test: dequant correctness ===")
    test_dequant_correctness()
    print("\n=== Test: baseline forward shapes ===")
    test_baseline_shapes()
    print("\n=== Test: 3D input ===")
    test_3d_input()
    print("\nAll baseline tests passed ✓")
