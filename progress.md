# DP4A INT8 GEMM — Progress

## Goal
Replace `torch.nn.functional.linear(x, w_fp32, bias)` with an INT8 DP4A kernel for **int8_tensorwise** quantized models on **Pascal GPUs** (sm_61, e.g. GTX 1050 Ti, 1080, Titan X), avoiding the dequantize → fp32 GEMM path.

## What was built

### 1. CUDA kernel (`dp4a_block.cu`)
Two kernels:

- **`quantize_x_int8_kernel`** — per-token activation quantization (fp32 → int8, one block per row, 256 threads)
- **`int8_gemm_dp4a_kernel`** — INT8×INT8 DP4A GEMM with fused dequant epilogue (`acc * scale_x[m] * scale_w + bias[n]`), where `scale_w` is a per-tensor scalar

Tile parameters (tuned for 2 blocks/SM on GP102):
- `BM=128, BN=64, BK=64, TM=8, TN=4, THREADS=256`
- `smem_x[2][128][17]` double-buffered + `smem_w[1][64][17]` single-buffered = 21,760 bytes
- Uses `__dp4a` (int8 dot-product-accumulate, sm_61+)
- 128 registers, 0 spills

### 2. PyTorch bindings (`dp4a_bindings.cpp`)
Exposes via `dp4a_ext` module:

- **`int8_linear(x, w_int8, scale_w, bias=None)`** — main op: quantizes activations → DP4A GEMM → dequant → returns fp32
  - `x: [..., K] fp32`, `w_int8: [N, K] int8`, `scale_w: scalar fp32`, `bias: [N] or None`
  - Requires `K % 4 == 0`, validates shapes/dtypes
  - Handles 3D inputs (reshape internally)
- **`quantize_weight_int8(w)`** — utility to fp32→int8 (per-tensor absmax)

### 3. Test weights (`test_mlp_block.safetensors`)
Extracted block 5 MLP from `wan2.2_ti2v_5B_int8.safetensors`:
- `ffn.0` (gate/up): `[14336, 3072] int8`, scalar weight_scale, `[14336] fp16` bias
- `ffn.2` (down): `[3072, 14336] int8`, scalar weight_scale, `[3072] fp16` bias
- Quantization format: `int8_tensorwise` (per-tensor scale, not per-channel)

### 4. Tests (`tests/`)
- `test_baseline.py` — verifies dequant → F.linear baseline (shapes, 3D)
- `test_kernel.py` — compares kernel vs baseline (cosine sim > 0.99995)
- `bench_baseline.py` — baseline timing + VRAM
- `bench_kernel.py` — kernel vs baseline timing + VRAM
- `bench_video.py` — timing across realistic WAN 2.2 token counts

## Results (GTX 1050 Ti, sm_61)

### Correctness
- Cosine similarity > **0.99995** vs fp32 baseline for all tested M (1→512)
- Mean abs error ~0.008 (ffn.0) / ~0.018 (ffn.2)
- K%4 and shape validation pass

### Performance
Baseline: dequant int8→fp32 once, then `F.linear` (cached fp32 weight).
Kernel: activation quantize-on-the-fly + DP4A GEMM (int8 weight stays int8).

| Tokens (M) | Baseline (ms) | Kernel (ms) | Speedup | Description |
|-----------|-------------|------------|---------|-------------|
| 1 | 1.7 | 2.2 | 0.8× | Single token (slow — quantize overhead) |
| 128 | 8.7 | 2.6 | 3.3× | Text tokens |
| 880 | 78.0 | 16.7 | 4.7× | 1 frame 720p |
| 4096 | 507.7 | 75.8 | 6.7× | 4 frames |
| 27280 | 3431.9 | 524.3 | 6.5× | Full 5s 720p video |

### VRAM savings (persistent weight storage)
- Baseline stores fp32 weights: **176 MB** per MLP layer
- Kernel stores int8 weights: **44 MB** per MLP layer
- Savings: **132 MB/layer**, ~**7.9 GB** for all 60 MLP layers in WAN 2.2 5B

## Key design decisions
1. **Per-tensor scalar scale** (not per-channel like the reference Forge project) — matches WAN's `int8_tensorwise` format
2. **Single-buffered W shared memory** — saves 4KB smem, enabling 2 blocks/SM on GP102
3. **fp32-only input/output** — matches WAN's fp32 path (can extend to fp16 output later)
4. **Activation quantization inside the op** — per-token dynamic, not persistent

## Next steps for the ComfyUI node

### What the node must do
1. Intercept `MixedPrecisionOps.Linear.forward()` (or `forward_comfy_cast_weights`) on modules where `layout_type == "TensorWiseINT8Layout"` and weight is `int8`
2. Check gate conditions: weight is `QuantizedTensor` with int8 data, input is fp32, `weight_function` is empty (bypass LoRA mode), `K % 4 == 0`
3. Extract raw `int8` data and scalar scale from the `QuantizedTensor` (`weight_qt.data` and `weight_qt._params.weight_scale`)
4. Route through `dp4a_ext.int8_linear(x, int8_data, scale_w, bias)`
5. Fall back to original forward on any error

### Reference integration pattern
See `guidelines.md` Q2 for the full ComfyUI patching approach. Key points:
- Override `forward_comfy_cast_weights`, not `forward` itself
- Use `model.clone()` before patching
- Use `cast_bias_weight` / `uncast_bias_weight` for proper offload support
- Check `weight_function` is empty (bypass LoRA mode guarantees this)
- The `QuantizedTensor` API: `.data` for raw int8, `._params.weight_scale` for scale

### Important caveats
- `QuantizedTensor` internals (`.data`, `._params`) are not public API — wrap extraction in try/except
- Bypass LoRA mode must be used (ensures `weight_function` is empty)
- Pascal GPU detection (sm_61) — only apply the patch when GPU supports `__dp4a`
