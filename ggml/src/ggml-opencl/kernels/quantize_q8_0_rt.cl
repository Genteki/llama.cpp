// Activation-side quantization for the RT/dp8 path.
// Converts a row-major FP16 tensor (N x K) into:
//   Xq: (N, K)        uint8  = round(x / scale) + 128, per QK=32 block
//   Xd: (N, K/QK)     half   = block scale (max(|x|)/127)
//   Xs: (N, K/QK)     int    = sum of the QK uint8 values in each block
// Layout matches what mul_mm_q8_0_rt_f16_dp8 expects on its activation input.
//
// Launch geometry:
//   global = (K/QK, N)         (one work-group per (block_k, row))
//   local  = (QK,   1)         (32 work-items per group, one per element)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK 32

// Templated on src1 dtype — ggml-opencl mul_mat passes src1 as either fp32 or
// fp16. Two entry points share the body via SRC_DTYPE / SRC_TO_FLOAT macros.
#ifndef SRC_DTYPE
#define SRC_DTYPE float
#define SRC_TO_FLOAT(v) (v)
#endif

__attribute__((reqd_work_group_size(QK, 1, 1)))
__kernel void kernel_quantize_q8_0_rt(
    __global const uchar     * X_raw,  // raw bytes; view offset added below
    const ulong                X_off,  // byte offset into X_raw (ggml view semantics)
    __global       uchar     * Xq,     // (N, K)
    __global       half      * Xd,     // (N, K/QK)
    __global       int       * Xs,     // (N, K/QK)
    const int N, const int K
) {
    __global const SRC_DTYPE * X = (__global const SRC_DTYPE *)(X_raw + X_off);
    const int kb  = get_group_id(0);     // K block index (0 .. K/QK - 1)
    const int row = get_group_id(1);     // N row
    const int tid = get_local_id(0);     // 0 .. QK-1
    if (row >= N) return;

    const int k = kb * QK + tid;
    const float x = (k < K) ? SRC_TO_FLOAT(X[row * K + k]) : 0.0f;

    // ---- amax reduction across the 32-element block ----
    __local float amax_lds[QK];
    amax_lds[tid] = fabs(x);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = QK / 2; off > 0; off >>= 1) {
        if (tid < off) amax_lds[tid] = fmax(amax_lds[tid], amax_lds[tid + off]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float d  = amax_lds[0] * (1.0f / 127.0f);
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

    // ---- per-element symmetric Q8 then +128 shift ----
    int q = (int) round(x * id);
    q = clamp(q, -127, 127);
    const uchar u = (uchar)(q + 128);
    Xq[row * K + k] = u;

    // ---- sum of uint8 in this block (for the zero-point correction term
    //      consumed by the dp8 GEMM kernel) ----
    __local int sum_lds[QK];
    sum_lds[tid] = (int) u;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = QK / 2; off > 0; off >>= 1) {
        if (tid < off) sum_lds[tid] += sum_lds[tid + off];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        Xd[row * (K / QK) + kb] = (half) d;
        Xs[row * (K / QK) + kb] = sum_lds[0];
    }
}
