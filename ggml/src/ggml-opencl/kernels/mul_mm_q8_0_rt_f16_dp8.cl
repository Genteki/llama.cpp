// Q8_0 Row-Tile dp8 GEMM — TILED variant, v2 (reinstated for profiling probe).
//
// v2 config:
//   * A_q / B_q rows padded to 36 bytes (BK_PAD) to break the 32-byte row
//     stride that caused 16-way LDS bank conflict on Adreno (32-bank LDS).
//   * TM*TN = 16 outputs/WI (TM=4, TN=4) — cuts dot_acc[] + sums[] footprint.
//   * WG size (BM/TM) * (BN/TN) = 16 * 16 = 256.
//
// 3-buffer activation/weight layout (matches pi0_attach_rt_overlay):
//   Wq / Xq : (rows, K) uint8 row-major
//   Wd / Xd : (rows, K/QK) half
//   Ws / Xs : (rows, K/QK) int32

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable

#define QK     32
#define BM     64
#define BN     64
#define BK     32
#define BK_PAD 36           // LDS row stride to avoid 32-bank conflict
#define TM     4
#define TN     4
// WG size = (BM/TM) * (BN/TN) = 16 * 16 = 256

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_mul_mm_q8_0_rt_f16_dp8(
    __global const uchar4 * Wq,    // (M, K/4)
    __global const half   * Wd,    // (M, K/QK)
    __global const int    * Ws,    // (M, K/QK)
    __global const uchar4 * Xq,    // (N, K/4)
    __global const half   * Xd,    // (N, K/QK)
    __global const int    * Xs,    // (N, K/QK)
    __global       float  * Y,     // (M, N)
    const int M, const int N, const int K
) {
    const int wg_m = get_group_id(0);
    const int wg_n = get_group_id(1);
    const int tid  = get_local_id(0);
    const int th_r = tid % (BM / TM);        // 0..15
    const int th_c = tid / (BM / TM);        // 0..15

    // Padded LDS slabs to scatter across 32 banks instead of all hitting bank t.
    __local uchar A_q[BM * BK_PAD];          // 64 * 36 = 2304 B
    __local half  A_d[BM];                   //  128 B
    __local int   A_s[BM];                   //  256 B
    __local uchar B_q[BN * BK_PAD];          // 2304 B
    __local half  B_d[BN];                   //  128 B
    __local int   B_s[BN];                   //  256 B
    // Total LDS per WG: ~5.4 KB

    float sums[TM * TN];                     // 16 floats
    #pragma unroll
    for (int i = 0; i < TM * TN; ++i) sums[i] = 0.0f;

    const int n_blk_k = K / QK;
    const int K4      = K / 4;
    const int WG_SZ   = 256;

    for (int kb = 0; kb < n_blk_k; ++kb) {

        // ---- (1) Cooperatively load A slab (64 rows × 32 cols uint8, padded to 64 × 36) ----
        // 256 WIs share 64*32/4 = 512 uchar4 reads -> 2 per WI on average.
        for (int p = tid; p < BM * BK / 4; p += WG_SZ) {
            int r  = p / (BK / 4);
            int c4 = p % (BK / 4);
            int row_g = wg_m * BM + r;
            int k4_g  = kb * (BK / 4) + c4;
            uchar4 v = (row_g < M)
                ? Wq[row_g * K4 + k4_g]
                : (uchar4)(128, 128, 128, 128);
            // Write at padded position (row r, byte c4*4) within stride BK_PAD.
            *(__local uchar4 *)(A_q + r * BK_PAD + c4 * 4) = v;
        }
        for (int r = tid; r < BM; r += WG_SZ) {
            int row_g = wg_m * BM + r;
            if (row_g < M) {
                A_d[r] = Wd[row_g * n_blk_k + kb];
                A_s[r] = Ws[row_g * n_blk_k + kb];
            } else {
                A_d[r] = (half) 0.0f;
                A_s[r] = 128 * QK;
            }
        }

        // ---- (2) Cooperatively load B slab ----
        for (int p = tid; p < BN * BK / 4; p += WG_SZ) {
            int r  = p / (BK / 4);
            int c4 = p % (BK / 4);
            int row_g = wg_n * BN + r;
            int k4_g  = kb * (BK / 4) + c4;
            uchar4 v = (row_g < N)
                ? Xq[row_g * K4 + k4_g]
                : (uchar4)(128, 128, 128, 128);
            *(__local uchar4 *)(B_q + r * BK_PAD + c4 * 4) = v;
        }
        for (int r = tid; r < BN; r += WG_SZ) {
            int row_g = wg_n * BN + r;
            if (row_g < N) {
                B_d[r] = Xd[row_g * n_blk_k + kb];
                B_s[r] = Xs[row_g * n_blk_k + kb];
            } else {
                B_d[r] = (half) 0.0f;
                B_s[r] = 128 * QK;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- (3) Compute: 4×4 outputs per WI = 16 dp8 calls per (t, kb) ----
        uint dot_acc[TM * TN];
        #pragma unroll
        for (int i = 0; i < TM * TN; ++i) dot_acc[i] = 0u;

        const int row_base = th_r * TM;
        const int col_base = th_c * TN;

        #pragma unroll 8
        for (int t = 0; t < QK / 4; ++t) {
            uint a_pack[TM];
            uint b_pack[TN];
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                a_pack[r] = *(__local const uint *)(A_q + (row_base + r) * BK_PAD + t * 4);
            }
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                b_pack[c] = *(__local const uint *)(B_q + (col_base + c) * BK_PAD + t * 4);
            }
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    dot_acc[c * TM + r] = qcom_udot8_acc(a_pack[r], b_pack[c], dot_acc[c * TM + r]);
                }
            }
        }

        // ---- (4) Block-boundary dequant + zero-point correction ----
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int    sA = A_s[row_base + r];
            const float  dA = (float) A_d[row_base + r];
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                const int   sB = B_s[col_base + c];
                const float dB = (float) B_d[col_base + c];
                const int   ds = (int) dot_acc[c * TM + r]
                                 - 128 * (sA + sB) + 128 * 128 * QK;
                sums[c * TM + r] = mad(dA * dB, (float) ds, sums[c * TM + r]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- (5) Write 16 outputs per WI ----
    const int dr = wg_m * BM + th_r * TM;
    const int dc = wg_n * BN + th_c * TN;
    #pragma unroll
    for (int c = 0; c < TN; ++c) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int row = dr + r;
            const int col = dc + c;
            if (row < M && col < N) {
                Y[row * N + col] = sums[c * TM + r];
            }
        }
    }
}
