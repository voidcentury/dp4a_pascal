#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" {
    void launch_quantize_x_f32(
        const float*, int8_t*, float*, int, int, cudaStream_t);

    void launch_int8_gemm(
        const int8_t*, const int8_t*,
        const float*, float,
        const float*, float*,
        int, int, int, cudaStream_t);
}

// ---------------------------------------------------------------------------
// int8_linear
//
//   x        : [..., K]  fp32, CUDA
//   w_int8   : [N, K]    int8, CUDA (pre-quantized)
//   scale_w  : scalar    fp32 (per-tensor scale)
//   bias     : [N] or None, CUDA
//
//   Returns  : [..., N]  fp32
//
//   Internally:
//     1) Reshape x → 2D [M, K]
//     2) quantize_x: x → x_int8 [M, K] + scale_x [M]
//     3) int8_gemm: out = (x_int8 @ w_int8.T) * scale_x[m] * scale_w + bias[n]
// ---------------------------------------------------------------------------
torch::Tensor int8_linear_fwd(
    torch::Tensor x,
    torch::Tensor w_int8,
    torch::Tensor scale_w,
    c10::optional<torch::Tensor> bias_opt)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_int8.is_cuda(), "w_int8 must be CUDA");
    TORCH_CHECK(scale_w.is_cuda(), "scale_w must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(w_int8.dtype() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(scale_w.dtype() == torch::kFloat32, "scale_w must be float32");
    TORCH_CHECK(scale_w.numel() == 1, "scale_w must be a scalar tensor");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    TORCH_CHECK(w_int8.dim() == 2, "w_int8 must be 2D [N, K]");

    int K = x.size(-1);
    int N = w_int8.size(0);
    TORCH_CHECK(w_int8.size(1) == K, "w_int8.size(1) must equal x.size(-1)");
    TORCH_CHECK(K % 4 == 0, "K must be divisible by 4 for DP4A (got ", K, ")");

    auto out_shape = x.sizes().vec();
    out_shape.back() = N;

    auto x2d = x.reshape({-1, K}).contiguous();
    int M = x2d.size(0);

    w_int8 = w_int8.contiguous();
    float scale_w_val = scale_w.item<float>();

    auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    auto x_int8 = torch::empty({M, K}, opts_i8);
    auto scale_x = torch::empty({M}, opts_f32);
    auto out2d = torch::empty({M, N}, opts_f32);

    auto stream = c10::cuda::getCurrentCUDAStream();

    // Step 1: quantize activations
    launch_quantize_x_f32(
        x2d.data_ptr<float>(),
        x_int8.data_ptr<int8_t>(),
        scale_x.data_ptr<float>(),
        M, K, stream);

    // Step 2: prepare bias
    const float* bias_ptr = nullptr;
    torch::Tensor bias_f32;
    if (bias_opt.has_value()) {
        auto& b = bias_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.numel() == N, "bias numel must equal N (", N, "), got ", b.numel());
        bias_f32 = b.to(torch::kFloat32).contiguous();
        bias_ptr = bias_f32.data_ptr<float>();
    }

    // Step 3: DP4A GEMM
    launch_int8_gemm(
        x_int8.data_ptr<int8_t>(),
        w_int8.data_ptr<int8_t>(),
        scale_x.data_ptr<float>(),
        scale_w_val,
        bias_ptr,
        out2d.data_ptr<float>(),
        M, N, K, stream);

    return out2d.reshape(out_shape);
}

// ---------------------------------------------------------------------------
// quantize_weight_int8
//
//   w : [N, K] fp32/fp16, CUDA or CPU
//   Returns: (w_int8 [N, K] int8, scale_w scalar fp32)
//
//   Per-tensor absmax quantization.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> quantize_weight_int8(torch::Tensor w)
{
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N, K]");
    TORCH_CHECK(w.dtype() == torch::kFloat32 || w.dtype() == torch::kFloat16,
                "w must be float32 or float16");

    auto device = w.device().is_cuda() ? w.device() : torch::Device(torch::kCUDA, 0);
    auto w_fp32 = w.to(device, torch::kFloat32);

    // Per-tensor absmax
    auto absmax = w_fp32.abs().amax();
    auto scale = absmax / 127.0f;
    scale = torch::clamp_min(scale, 1e-12f);

    // Quantize
    auto w_scaled = w_fp32 / scale;
    auto w_int8 = w_scaled.round().clamp(-127.0f, 127.0f).to(torch::kInt8);

    return std::make_tuple(w_int8.contiguous(), scale.to(torch::kFloat32).contiguous());
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DP4A INT8 GEMM extension for Pascal GPUs";

    m.def("int8_linear", &int8_linear_fwd,
        "INT8 GEMM via DP4A. Quantizes activations on-the-fly (per-token).\n"
        "Args:\n"
        "  x:        [..., K] fp32\n"
        "  w_int8:   [N, K] int8 (precomputed)\n"
        "  scale_w:  scalar fp32 (per-tensor)\n"
        "  bias:     [N] or None\n"
        "Returns:\n"
        "  [..., N] fp32",
        py::arg("x"), py::arg("w_int8"), py::arg("scale_w"),
        py::arg("bias") = py::none());

    m.def("quantize_weight_int8", &quantize_weight_int8,
        "Pre-quantize a weight tensor for INT8 path.\n"
        "Args:\n"
        "  w: [N, K] fp32/fp16 weight\n"
        "Returns:\n"
        "  (w_int8 [N, K] int8, scale_w scalar fp32)",
        py::arg("w"));
}
