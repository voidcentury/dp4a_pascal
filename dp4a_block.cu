#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

// ===========================================================================
// Kernel 1: Quantize activations to INT8 (per-token absmax)
//
// One block per row. 256 threads, cooperative absmax + quantization.
// ===========================================================================

__global__ void quantize_x_int8_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ scale_x,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const float* x_row = x + row * K;
    int8_t* xq_row = x_q + row * K;
    int tid = threadIdx.x;
    const int BT = 256;

    // Pass 1: per-row absmax
    float my_max = 0.0f;
    for (int i = tid; i < K; i += BT) {
        float v = fabsf(x_row[i]);
        if (v > my_max) my_max = v;
    }

    my_max = warp_reduce_max(my_max);

    __shared__ float smem_max[8];   // 256 threads / 32 = 8 warps
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) smem_max[warp_id] = my_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < 8) ? smem_max[lane_id] : 0.0f;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            float s = v / 127.0f;
            if (s < 1e-12f) s = 1e-12f;
            smem_max[0] = s;
            scale_x[row] = s;
        }
    }
    __syncthreads();

    float inv_scale = 1.0f / smem_max[0];

    // Pass 2: quantize
    for (int i = tid; i < K; i += BT) {
        float v = x_row[i];
        int q = __float2int_rn(v * inv_scale);
        if (q > 127)  q = 127;
        if (q < -127) q = -127;
        xq_row[i] = (int8_t)q;
    }
}

// ===========================================================================
// Kernel 2: INT8 GEMM via DP4A
//
// Tile parameters (tuned for 2 blocks/SM on GP102 sm_61):
//   BM=128, BN=64, BK=64, TM=8, TN=4, THREADS=256
//   KDWORDS = BK/4 = 16 (four int8 packed into one int32)
//
// Shared memory layout:
//   smem_x [2][128][17]  int32 — double-buffered  = 17408 bytes
//   smem_w [1][ 64][17]  int32 — single-buffered  =  4352 bytes
//   Total: 21760 bytes  (<24576 → 2 blocks/SM on GP102 ✓)
//
// W single-buffered: prefetch travels through registers, committed to
// smem_w[0] after compute finishes. Saves 4352 bytes vs double-buffer.
//
// Scale_w is a per-tensor scalar (simplified from per-channel in reference).
// ===========================================================================

#define I8_BM  128
#define I8_BN  64
#define I8_BK  64
#define I8_TM  8
#define I8_TN  4

#define I8_TX          (I8_BN / I8_TN)            // 16
#define I8_TY          (I8_BM / I8_TM)            // 16
#define THREADS_PER_BLOCK (I8_TX * I8_TY)         // 256
#define KDWORDS        (I8_BK / 4)                // 16
#define LOADS_X        ((I8_BM * KDWORDS) / THREADS_PER_BLOCK) // 8
#define LOADS_W        ((I8_BN * KDWORDS) / THREADS_PER_BLOCK) // 4

__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void int8_gemm_dp4a_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ scale_x,
    float scale_w,                          // per-tensor scalar
    const float* __restrict__ bias,
    float* __restrict__ out,
    int M, int N, int K)
{
    __shared__ int32_t smem_x[2][I8_BM][KDWORDS + 1]; // 17408 bytes
    __shared__ int32_t smem_w[1][I8_BN][KDWORDS + 1]; //  4352 bytes

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * I8_TX + tx;

    const int blk_m = blockIdx.y * I8_BM;
    const int blk_n = blockIdx.x * I8_BN;

    int32_t acc[I8_TM][I8_TN];
    #pragma unroll
    for (int i = 0; i < I8_TM; i++)
        #pragma unroll
        for (int j = 0; j < I8_TN; j++)
            acc[i][j] = 0;

    const int num_tiles = (K + I8_BK - 1) / I8_BK;

    // ── Bootstrap: load tile 0 ───────────────────────────────────────────

    // smem_x[0]
    #pragma unroll
    for (int i = 0; i < LOADS_X; i++) {
        const int e = tid + i * THREADS_PER_BLOCK;
        const int mi = e / KDWORDS;
        const int ki = e % KDWORDS;
        const int gm = blk_m + mi;
        const int gk = ki * 4;
        int32_t v = 0;
        if (gm < M && gk < K)
            v = __ldg(reinterpret_cast<const int32_t*>(&x[gm * K + gk]));
        smem_x[0][mi][ki] = v;
    }

    // smem_w[0] (single buffer)
    #pragma unroll
    for (int i = 0; i < LOADS_W; i++) {
        const int e = tid + i * THREADS_PER_BLOCK;
        const int ni = e / KDWORDS;
        const int ki = e % KDWORDS;
        const int gn = blk_n + ni;
        const int gk = ki * 4;
        int32_t v = 0;
        if (gn < N && gk < K)
            v = __ldg(reinterpret_cast<const int32_t*>(&w[gn * K + gk]));
        smem_w[0][ni][ki] = v;
    }

    __syncthreads();

    int32_t prefetch_x[LOADS_X];
    int32_t prefetch_w[LOADS_W];

    // ── Main loop ────────────────────────────────────────────────────────
    for (int kt = 0; kt < num_tiles; kt++) {
        const int buf = kt & 1;
        const int nxt = 1 - buf;
        const int next_k0 = (kt + 1) * I8_BK;
        const bool has_next = (kt + 1 < num_tiles);

        // Prefetch next tile from global memory into registers
        if (has_next) {
            #pragma unroll
            for (int i = 0; i < LOADS_X; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int mi = e / KDWORDS;
                const int ki = e % KDWORDS;
                const int gm = blk_m + mi;
                const int gk = next_k0 + ki * 4;
                prefetch_x[i] = (gm < M && gk < K)
                    ? __ldg(reinterpret_cast<const int32_t*>(&x[gm * K + gk]))
                    : 0;
            }
            #pragma unroll
            for (int i = 0; i < LOADS_W; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int ni = e / KDWORDS;
                const int ki = e % KDWORDS;
                const int gn = blk_n + ni;
                const int gk = next_k0 + ki * 4;
                prefetch_w[i] = (gn < N && gk < K)
                    ? __ldg(reinterpret_cast<const int32_t*>(&w[gn * K + gk]))
                    : 0;
            }
        }

        // Compute on current smem buffers
        #pragma unroll
        for (int kk = 0; kk < KDWORDS; kk++) {
            int32_t x_reg[I8_TM];
            #pragma unroll
            for (int i = 0; i < I8_TM; i++)
                x_reg[i] = smem_x[buf][ty * I8_TM + i][kk];

            int32_t w_reg[I8_TN];
            #pragma unroll
            for (int j = 0; j < I8_TN; j++)
                w_reg[j] = smem_w[0][tx + j * I8_TX][kk];

            #pragma unroll
            for (int i = 0; i < I8_TM; i++)
                #pragma unroll
                for (int j = 0; j < I8_TN; j++)
                    acc[i][j] = __dp4a(x_reg[i], w_reg[j], acc[i][j]);
        }

        // Commit prefetched data to shared memory
        if (has_next) {
            #pragma unroll
            for (int i = 0; i < LOADS_X; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int mi = e / KDWORDS;
                const int ki = e % KDWORDS;
                smem_x[nxt][mi][ki] = prefetch_x[i];
            }
            #pragma unroll
            for (int i = 0; i < LOADS_W; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int ni = e / KDWORDS;
                const int ki = e % KDWORDS;
                smem_w[0][ni][ki] = prefetch_w[i];
            }
        }

        __syncthreads();
    }

    // ── Epilogue: int32 → fp32 dequant + bias + write ─────────────────────

    float scale_x_local[I8_TM];
    #pragma unroll
    for (int i = 0; i < I8_TM; i++) {
        const int m = blk_m + ty * I8_TM + i;
        scale_x_local[i] = (m < M) ? __ldg(&scale_x[m]) : 0.0f;
    }

    float bias_local[I8_TN];
    #pragma unroll
    for (int j = 0; j < I8_TN; j++) {
        const int n = blk_n + tx + j * I8_TX;
        bias_local[j] = (n < N && bias != nullptr) ? __ldg(&bias[n]) : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < I8_TM; i++) {
        const int m = blk_m + ty * I8_TM + i;
        if (m >= M) continue;
        const float sx = scale_x_local[i];
        #pragma unroll
        for (int j = 0; j < I8_TN; j++) {
            const int n = blk_n + tx + j * I8_TX;
            if (n < N) {
                out[m * N + n] = (float)acc[i][j] * sx * scale_w + bias_local[j];
            }
        }
    }
}

// ===========================================================================
// Host launchers
// ===========================================================================

extern "C" {

void launch_quantize_x_f32(
    const float* x, int8_t* x_q, float* scale_x,
    int M, int K, cudaStream_t stream)
{
    quantize_x_int8_kernel<<<M, 256, 0, stream>>>(x, x_q, scale_x, M, K);
}

void launch_int8_gemm(
    const int8_t* x_q, const int8_t* w_q,
    const float* scale_x, float scale_w,
    const float* bias, float* out,
    int M, int N, int K, cudaStream_t stream)
{
    dim3 block(I8_TX, I8_TY);
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    int8_gemm_dp4a_kernel<<<grid, block, 0, stream>>>(
        x_q, w_q, scale_x, scale_w, bias, out, M, N, K);
}

} // extern "C"
