// src0_q, src0_d, src1 are transposed as a preprocessing step
// 4-bit weights are transposed in groups of 4 (unsigned short int)
// consider weights originally "next to each other", now "on top of each other"
// each fiber computes a 8x4 tile of output elements
// using unshuffled weights

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_mul_mat_Ab_Bi_8x4(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        __read_only image1d_buffer_t src1,  // B (1d image)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
    half8 B; // registers for activations
    half4 dequantized_weights; // registers for dequantized weights
    __global const ushort* weight_ptr = src0_q + gx_2; // pointer for weights
    __global const half* scale_ptr = src0_d + gx_2; // pointer for scales

    for(int i=0; i<k; i+=4){ //loop through K dimension

        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);

        // keep (i/4) and (i/32) in parenthesis, rounds down
        // load 4 consecutive groups of 4 weights
        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m)); // (i/4) because weights grouped in 4s

        // load 4 consecutive scales
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));// (i/32) because 1 scale per 32 elements

        // j=0
        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; //vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0; // dequantize a row of the 16 weights
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; // vector-scalar multiplication to accumulate
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;
    }

    int idx = (gy<<3)*m + (gx<<2); // vectorized store 16 elements

    // conditional check if store is to a valid location. Required when N is not a multiple of 8
    // if statements allow registers to be reused for each store
    // provides a performance boost due to reduced register footprint, which increases number of concurrent waves
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}

// ---- q4_K variant of kernel_mul_mat_Ab_Bi_8x4 (PI0 k-quant operating point) -----------------
// Same 8x4 image-B micro-tile as the q4_0 hero kernel, but with q4_K's two-level, asymmetric
// dequant -- co-designed for Adreno so the k-quant scale metadata costs the SAME memory traffic
// as q4_0's single scale (the q4_1 expansion to per-32 fp16 {scale,min} costs ~2x that = ~20%
// measured). Layout (all M-contiguous transposed, like the q4_0 path):
//   src0_q   [K/4][M] ushort : 4 nibbles/ushort, identical packing to q4_0 (the fast quant path)
//   src0_sub [K/32][M/4][8] uchar : INTERLEAVED per-32 sub-scale(s0..3)+sub-min(s4..7), the raw
//            6-bit q4_K values stored in uint8 -> ONE uchar8 vload gets 4 scales + 4 mins for the
//            fiber's 4 M-rows (1/2 the bytes AND 1/2 the loads of q4_1's two half4 scale/min reads;
//            total = 2.1MB = q4_0's scale-array size -> caches like q4_0).
//   src0_super [K/256][M/4][8] half : INTERLEAVED per-256 super-scale(s0..3)+super-min(s4..7),
//            one half8 vload per 256 K (read 1/64 as often -> ~free)
// Dequant: w = nibble * (sd*sub_scale) - (sdm*sub_min) = nibble*d_eff + m_eff, with d_eff/m_eff
// computed once per K-step (idle prefix ALU absorbs the convert+mul). Bit-budget ~4.6 b/w, lossless
// vs native q4_K. Needs K%256==0 and M%4==0 (caller pads); same grid as the q4_0 8x4.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_q4k(
        global const ushort * src0_q,       // quantized A (4 nibbles/ushort, [K/4][M])
        global const uchar  * src0_sub,     // interleaved 6-bit sub-scale+sub-min, [K/32][M/4][8]
        global const half   * src0_super,   // interleaved super-scale+super-min, [K/256][M/4][8]
        __read_only image1d_buffer_t src1,  // B (1d image, same Bi layout as q4_0)
        global float * dst,                 // C
        ulong offsetd,                      // byte offset into dst (matches q4_1 path)
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {
    dst = (global float *)((global char *)dst + offsetd);

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
    half8 B;                               // activations
    half4 dqw;                             // dequantized weights (4 M-rows)
    __global const ushort* weight_ptr = src0_q   + gx_2;     // quants for the 4 M-rows
    __global const uchar*  sub_ptr    = src0_sub   + (gx << 3); // 8 uchar (4 sc + 4 mn) for the 4 M-rows
    __global const half*   sup_ptr    = src0_super + (gx << 3); // 8 half  (4 sd + 4 sdm) for the 4 M-rows

    // Outer loop walks 32-block by 32-block: the effective {scale,min} (uint8->half convert +
    // super-mul) is computed ONCE per 32 K and reused across the 8 inner K-steps -- so the per-
    // element work matches the q4_1 kernel (just q*d_eff + m_eff) while the metadata bytes stay
    // halved. K%32==0 is guaranteed (q4_K -> K%256==0).
    for(int i=0; i<k; i+=32){
        uchar8 sub  = vload8(0, sub_ptr + (i >> 5) * (m_4 << 3)); // 4 sub-scale + 4 sub-min (6-bit)
        half8  sup  = vload8(0, sup_ptr + (i >> 8) * (m_4 << 3)); // 4 super-scale + 4 super-min
        half4  d_eff =   convert_half4(sub.s0123) * sup.s0123;    // effective per-element scale (per 32 K)
        half4  m_eff = -(convert_half4(sub.s4567) * sup.s4567);   // effective per-element min  (per 32 K)

        #pragma unroll
        for(int jj=0; jj<32; jj+=4){ // 8 inner K-steps (4 K each) reusing d_eff/m_eff
            int ii = i + jj;
            ushort4 bits4 = vload4(0, weight_ptr + (ii >> 2) * m); // 4 nibbles x 4 M-rows

            // j=0
            B.s0123 = read_imageh(src1, gy*2 + (ii)*(n_4));
            B.s4567 = read_imageh(src1, gy*2 + (ii)*(n_4)+1);
            dqw.s0 = (half)(bits4.s0 & 0x000F) * d_eff.s0 + m_eff.s0;
            dqw.s1 = (half)(bits4.s1 & 0x000F) * d_eff.s1 + m_eff.s1;
            dqw.s2 = (half)(bits4.s2 & 0x000F) * d_eff.s2 + m_eff.s2;
            dqw.s3 = (half)(bits4.s3 & 0x000F) * d_eff.s3 + m_eff.s3;
            c0 += B * dqw.s0; c1 += B * dqw.s1; c2 += B * dqw.s2; c3 += B * dqw.s3;

            // j=1
            B.s0123 = read_imageh(src1, gy*2 + (ii+1)*(n_4));
            B.s4567 = read_imageh(src1, gy*2 + (ii+1)*(n_4)+1);
            dqw.s0 = (half)((bits4.s0 & 0x00F0) >> 4) * d_eff.s0 + m_eff.s0;
            dqw.s1 = (half)((bits4.s1 & 0x00F0) >> 4) * d_eff.s1 + m_eff.s1;
            dqw.s2 = (half)((bits4.s2 & 0x00F0) >> 4) * d_eff.s2 + m_eff.s2;
            dqw.s3 = (half)((bits4.s3 & 0x00F0) >> 4) * d_eff.s3 + m_eff.s3;
            c0 += B * dqw.s0; c1 += B * dqw.s1; c2 += B * dqw.s2; c3 += B * dqw.s3;

            // j=2
            B.s0123 = read_imageh(src1, gy*2 + (ii+2)*(n_4));
            B.s4567 = read_imageh(src1, gy*2 + (ii+2)*(n_4)+1);
            dqw.s0 = (half)((bits4.s0 & 0x0F00) >> 8) * d_eff.s0 + m_eff.s0;
            dqw.s1 = (half)((bits4.s1 & 0x0F00) >> 8) * d_eff.s1 + m_eff.s1;
            dqw.s2 = (half)((bits4.s2 & 0x0F00) >> 8) * d_eff.s2 + m_eff.s2;
            dqw.s3 = (half)((bits4.s3 & 0x0F00) >> 8) * d_eff.s3 + m_eff.s3;
            c0 += B * dqw.s0; c1 += B * dqw.s1; c2 += B * dqw.s2; c3 += B * dqw.s3;

            // j=3
            B.s0123 = read_imageh(src1, gy*2 + (ii+3)*(n_4));
            B.s4567 = read_imageh(src1, gy*2 + (ii+3)*(n_4)+1);
            dqw.s0 = (half)((bits4.s0 & 0xF000) >> 12) * d_eff.s0 + m_eff.s0;
            dqw.s1 = (half)((bits4.s1 & 0xF000) >> 12) * d_eff.s1 + m_eff.s1;
            dqw.s2 = (half)((bits4.s2 & 0xF000) >> 12) * d_eff.s2 + m_eff.s2;
            dqw.s3 = (half)((bits4.s3 & 0xF000) >> 12) * d_eff.s3 + m_eff.s3;
            c0 += B * dqw.s0; c1 += B * dqw.s1; c2 += B * dqw.s2; c3 += B * dqw.s3;
        }
    }

    int idx = (gy<<3)*m + (gx<<2); // vectorized store 16 elements
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx); idx += m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx); }
}

// ---- f16-weight variant of kernel_mul_mat_Ab_Bi_8x4 ----------------------------------------
// Same access pattern (B/activations via image1d_buffer -> TP L1; weights via global/L2 with high
// cross-workgroup reuse; 8x4 micro-tile; half8 accum) but src0 (the "weights") is plain f16, not
// q4_0. Used for PI0 attention GEMMs where the "weight" is the f16 KV cache: QK = mul_mat(K, Q),
// AV = mul_mat(V_t, scores). Weight layout is M-contiguous transposed [K][M] (half at k*M + m),
// the f16 analog of the q4_0 [K/4][M] packing. NOTE: f16 weights are 4x the bytes of q4_0, so the
// L2 weight path carries 4x more traffic -- this kernel exists to MEASURE whether that unbalances
// the design (q4_0's 4x byte reduction is what balanced weights@L2 with B@TP-L1). K-tail handles
// K not a multiple of 4 (real n_kv = 774/825 for AV).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_f16(
        global const half * src0_w,         // f16 A (weights), [K][M] M-contiguous
        __read_only image1d_buffer_t src1,  // B (1d image, f16 Bi layout -- same as q4_0 path)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {
    int n_4 = n >> 2;

    // 1D-flattened launch: the (N-tile, M-tile) grid is packed into one dimension so only the final
    // workgroup is partial (<=127 idle fibers total), instead of wasting lanes per N-tile when
    // ceil(M/4) is not a multiple of the 128-lane workgroup (QK M=n_kv 774 -> 24% idle, AV M=256 ->
    // 50% idle with a 2D launch). gy = N-tile, gx = M-tile. gx_2 = gx*4 < m always (gx < ceil(M/4)),
    // so no M-overrun guard is needed; partial M-tiles (M%4!=0) are handled by the store guards and
    // the caller's weight-buffer padding.
    int m_tiles = (m + 3) >> 2;            // ceil(M/4)
    int n_tiles = (n_no_padding + 7) >> 3; // ceil(N/8)
    int flat = get_global_id(0);
    int gy = flat / m_tiles;
    if (gy >= n_tiles) { return; }         // tail fibers of the rounded-up launch
    int gx = flat - gy * m_tiles;
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
    half8 B; // registers for activations
    half4 w; // 4 f16 weights (the 4 M-rows handled by this fiber) at one K
    __global const half* weight_ptr = src0_w + gx_2; // pointer for weights

    int i = 0;
    for(; i+3 < k; i+=4){ // unrolled by 4 along K (matches the 4 B reads of the q4_0 kernel)
        // j=0
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        w = vload4(0, weight_ptr + (i)*(m));
        c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;
        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        w = vload4(0, weight_ptr + (i+1)*(m));
        c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;
        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        w = vload4(0, weight_ptr + (i+2)*(m));
        c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;
        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        w = vload4(0, weight_ptr + (i+3)*(m));
        c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;
    }
    for(; i < k; i++){ // K-tail (K not a multiple of 4)
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        w = vload4(0, weight_ptr + (i)*(m));
        c0 += B * w.s0; c1 += B * w.s1; c2 += B * w.s2; c3 += B * w.s3;
    }

    int n_row0 = gy << 3;
    int m_col0 = gx << 2;
    if (m_col0 + 3 < m && n_row0 + 7 < n_no_padding) {
        // interior: full 8x4 tile, all rows/cols valid (the common case, incl. all bench shapes)
        int idx = n_row0*m + m_col0;
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx); idx += m;
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx); idx += m;
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx); idx += m;
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx); idx += m;
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx); idx += m;
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx); idx += m;
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx); idx += m;
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    } else {
        // boundary: N-row and M-column guarded scalar stores (M%4!=0 partial tile, N%8!=0 tail)
        if (n_row0+0 < n_no_padding){ int b=(n_row0+0)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s0; if(m_col0+1<m)dst[b+1]=c1.s0; if(m_col0+2<m)dst[b+2]=c2.s0; if(m_col0+3<m)dst[b+3]=c3.s0; }
        if (n_row0+1 < n_no_padding){ int b=(n_row0+1)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s1; if(m_col0+1<m)dst[b+1]=c1.s1; if(m_col0+2<m)dst[b+2]=c2.s1; if(m_col0+3<m)dst[b+3]=c3.s1; }
        if (n_row0+2 < n_no_padding){ int b=(n_row0+2)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s2; if(m_col0+1<m)dst[b+1]=c1.s2; if(m_col0+2<m)dst[b+2]=c2.s2; if(m_col0+3<m)dst[b+3]=c3.s2; }
        if (n_row0+3 < n_no_padding){ int b=(n_row0+3)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s3; if(m_col0+1<m)dst[b+1]=c1.s3; if(m_col0+2<m)dst[b+2]=c2.s3; if(m_col0+3<m)dst[b+3]=c3.s3; }
        if (n_row0+4 < n_no_padding){ int b=(n_row0+4)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s4; if(m_col0+1<m)dst[b+1]=c1.s4; if(m_col0+2<m)dst[b+2]=c2.s4; if(m_col0+3<m)dst[b+3]=c3.s4; }
        if (n_row0+5 < n_no_padding){ int b=(n_row0+5)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s5; if(m_col0+1<m)dst[b+1]=c1.s5; if(m_col0+2<m)dst[b+2]=c2.s5; if(m_col0+3<m)dst[b+3]=c3.s5; }
        if (n_row0+6 < n_no_padding){ int b=(n_row0+6)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s6; if(m_col0+1<m)dst[b+1]=c1.s6; if(m_col0+2<m)dst[b+2]=c2.s6; if(m_col0+3<m)dst[b+3]=c3.s6; }
        if (n_row0+7 < n_no_padding){ int b=(n_row0+7)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s7; if(m_col0+1<m)dst[b+1]=c1.s7; if(m_col0+2<m)dst[b+2]=c2.s7; if(m_col0+3<m)dst[b+3]=c3.s7; }
    }
}

// ---- 8N x 2M variant of the f16-weight kernel (vs 8x4 above) ------------------------------------
// Each fiber computes a 2(M) x 8(N) tile -> M=head_dim=256 maps to exactly 128 fibers = a full
// 128-lane workgroup (AV). Tradeoff vs 8x4: B reused 2x/fiber instead of 4x (lower arithmetic
// intensity) BUT only 2 accumulators (less register pressure -> more concurrent waves) and a
// cleaner 128-lane B broadcast (one N-tile per WG). Same 1D-flatten launch (m_tiles = ceil(M/2)).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x2_f16(
        global const half * src0_w,
        __read_only image1d_buffer_t src1,
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int m_tiles = (m + 1) >> 1;            // ceil(M/2)
    int n_tiles = (n_no_padding + 7) >> 3; // ceil(N/8)
    int flat = get_global_id(0);
    int gy = flat / m_tiles;
    if (gy >= n_tiles) { return; }
    int gx = flat - gy * m_tiles;
    int gx_2 = gx << 1;                     // 2 M-rows per fiber

    half8 c0 = 0, c1 = 0; // 8x2 output elements
    half8 B;
    half2 w;
    __global const half* weight_ptr = src0_w + gx_2;

    int i = 0;
    for(; i+3 < k; i+=4){
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));   B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        w = vload2(0, weight_ptr + (i)*(m));   c0 += B * w.s0; c1 += B * w.s1;
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        w = vload2(0, weight_ptr + (i+1)*(m)); c0 += B * w.s0; c1 += B * w.s1;
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        w = vload2(0, weight_ptr + (i+2)*(m)); c0 += B * w.s0; c1 += B * w.s1;
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        w = vload2(0, weight_ptr + (i+3)*(m)); c0 += B * w.s0; c1 += B * w.s1;
    }
    for(; i < k; i++){
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        w = vload2(0, weight_ptr + (i)*(m)); c0 += B * w.s0; c1 += B * w.s1;
    }

    int n_row0 = gy << 3;
    int m_col0 = gx << 1;
    if (m_col0 + 1 < m && n_row0 + 7 < n_no_padding) {
        int idx = n_row0*m + m_col0;
        vstore2((float2)(c0.s0, c1.s0), 0, dst + idx); idx += m;
        vstore2((float2)(c0.s1, c1.s1), 0, dst + idx); idx += m;
        vstore2((float2)(c0.s2, c1.s2), 0, dst + idx); idx += m;
        vstore2((float2)(c0.s3, c1.s3), 0, dst + idx); idx += m;
        vstore2((float2)(c0.s4, c1.s4), 0, dst + idx); idx += m;
        vstore2((float2)(c0.s5, c1.s5), 0, dst + idx); idx += m;
        vstore2((float2)(c0.s6, c1.s6), 0, dst + idx); idx += m;
        vstore2((float2)(c0.s7, c1.s7), 0, dst + idx);
    } else {
        if (n_row0+0 < n_no_padding){ int b=(n_row0+0)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s0; if(m_col0+1<m)dst[b+1]=c1.s0; }
        if (n_row0+1 < n_no_padding){ int b=(n_row0+1)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s1; if(m_col0+1<m)dst[b+1]=c1.s1; }
        if (n_row0+2 < n_no_padding){ int b=(n_row0+2)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s2; if(m_col0+1<m)dst[b+1]=c1.s2; }
        if (n_row0+3 < n_no_padding){ int b=(n_row0+3)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s3; if(m_col0+1<m)dst[b+1]=c1.s3; }
        if (n_row0+4 < n_no_padding){ int b=(n_row0+4)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s4; if(m_col0+1<m)dst[b+1]=c1.s4; }
        if (n_row0+5 < n_no_padding){ int b=(n_row0+5)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s5; if(m_col0+1<m)dst[b+1]=c1.s5; }
        if (n_row0+6 < n_no_padding){ int b=(n_row0+6)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s6; if(m_col0+1<m)dst[b+1]=c1.s6; }
        if (n_row0+7 < n_no_padding){ int b=(n_row0+7)*m+m_col0; if(m_col0+0<m)dst[b]=c0.s7; if(m_col0+1<m)dst[b+1]=c1.s7; }
    }
}

// ---- Split-K variant of the 8x4 f16-weight kernel ----------------------------------------------
// For occupancy-starved shapes (diffusion AV: M=256,N=408 -> only ~26 workgroups < saturation ~50)
// split the K=n_kv reduction into k_split slices along a 2nd grid dim -> k_split x more workgroups.
// Each fiber reduces its K-slice and writes a partial plane to partials[ks*plane + n_row*M + m_col];
// kernel_splitk_reduce then sums the k_split planes into C. Unlike the q4_0 split-K, the f16 path
// has no scale-block alignment, so slice boundaries are arbitrary (last slice takes the remainder).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_f16_splitk(
        global const half * src0_w,
        __read_only image1d_buffer_t src1,
        global float * partials,
        int m, int n, int k, int n_no_padding, int k_split
) {
    int n_4 = n >> 2;
    int m_tiles = (m + 3) >> 2;
    int n_tiles = (n_no_padding + 7) >> 3;
    int flat = get_global_id(0);
    int ks   = get_global_id(1);           // K-slice index
    int gy = flat / m_tiles;
    if (gy >= n_tiles) { return; }
    int gx = flat - gy * m_tiles;
    int gx_2 = gx << 2;

    int k_per   = k / k_split;
    int k_start = ks * k_per;
    int k_end   = (ks == k_split - 1) ? k : (k_start + k_per);

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 w;
    __global const half* weight_ptr = src0_w + gx_2;

    int i = k_start;
    for(; i+3 < k_end; i+=4){
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));   B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        w = vload4(0, weight_ptr + (i)*(m));   c0 += B*w.s0; c1 += B*w.s1; c2 += B*w.s2; c3 += B*w.s3;
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        w = vload4(0, weight_ptr + (i+1)*(m)); c0 += B*w.s0; c1 += B*w.s1; c2 += B*w.s2; c3 += B*w.s3;
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        w = vload4(0, weight_ptr + (i+2)*(m)); c0 += B*w.s0; c1 += B*w.s1; c2 += B*w.s2; c3 += B*w.s3;
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        w = vload4(0, weight_ptr + (i+3)*(m)); c0 += B*w.s0; c1 += B*w.s1; c2 += B*w.s2; c3 += B*w.s3;
    }
    for(; i < k_end; i++){
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        w = vload4(0, weight_ptr + (i)*(m)); c0 += B*w.s0; c1 += B*w.s1; c2 += B*w.s2; c3 += B*w.s3;
    }

    int plane = m * n_no_padding;
    int base  = ks * plane;
    int n_row0 = gy << 3;
    int m_col0 = gx << 2;
    if (m_col0 + 3 < m && n_row0 + 7 < n_no_padding) {
        int idx = base + n_row0*m + m_col0;
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, partials + idx); idx += m;
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, partials + idx); idx += m;
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, partials + idx); idx += m;
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, partials + idx); idx += m;
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, partials + idx); idx += m;
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, partials + idx); idx += m;
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, partials + idx); idx += m;
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, partials + idx);
    } else {
        if (n_row0+0 < n_no_padding){ int b=base+(n_row0+0)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s0; if(m_col0+1<m)partials[b+1]=c1.s0; if(m_col0+2<m)partials[b+2]=c2.s0; if(m_col0+3<m)partials[b+3]=c3.s0; }
        if (n_row0+1 < n_no_padding){ int b=base+(n_row0+1)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s1; if(m_col0+1<m)partials[b+1]=c1.s1; if(m_col0+2<m)partials[b+2]=c2.s1; if(m_col0+3<m)partials[b+3]=c3.s1; }
        if (n_row0+2 < n_no_padding){ int b=base+(n_row0+2)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s2; if(m_col0+1<m)partials[b+1]=c1.s2; if(m_col0+2<m)partials[b+2]=c2.s2; if(m_col0+3<m)partials[b+3]=c3.s2; }
        if (n_row0+3 < n_no_padding){ int b=base+(n_row0+3)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s3; if(m_col0+1<m)partials[b+1]=c1.s3; if(m_col0+2<m)partials[b+2]=c2.s3; if(m_col0+3<m)partials[b+3]=c3.s3; }
        if (n_row0+4 < n_no_padding){ int b=base+(n_row0+4)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s4; if(m_col0+1<m)partials[b+1]=c1.s4; if(m_col0+2<m)partials[b+2]=c2.s4; if(m_col0+3<m)partials[b+3]=c3.s4; }
        if (n_row0+5 < n_no_padding){ int b=base+(n_row0+5)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s5; if(m_col0+1<m)partials[b+1]=c1.s5; if(m_col0+2<m)partials[b+2]=c2.s5; if(m_col0+3<m)partials[b+3]=c3.s5; }
        if (n_row0+6 < n_no_padding){ int b=base+(n_row0+6)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s6; if(m_col0+1<m)partials[b+1]=c1.s6; if(m_col0+2<m)partials[b+2]=c2.s6; if(m_col0+3<m)partials[b+3]=c3.s6; }
        if (n_row0+7 < n_no_padding){ int b=base+(n_row0+7)*m+m_col0; if(m_col0+0<m)partials[b]=c0.s7; if(m_col0+1<m)partials[b+1]=c1.s7; if(m_col0+2<m)partials[b+2]=c2.s7; if(m_col0+3<m)partials[b+3]=c3.s7; }
    }
}

// ---- Low-compute variant: IDENTICAL memory traffic to kernel_mul_mat_Ab_Bi_8x4 (all 4 B reads
// per K-step via image, full weight+scale vload4) but only ONE accumulator -> 1 of 4 dequant+FMA
// per j, i.e. ~1/4 the compute, memory unchanged. Tests compute- vs memory-bound: if abbi is
// compute-bound, cutting compute pulls the time DOWN toward the memory floor (~bonly_i); if
// memory-bound, the time is unchanged (~abbi). Compute kept alive (and the result stored) under a
// never-true guard to defeat DCE. Routed via PI0_GEMM_VARIANT=lowcompute.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_lowcompute(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        __read_only image1d_buffer_t src1,  // B (1d image)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {

    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0;   // ONLY ONE accumulator (vs 4) -> 1/4 the FMAs + dequants
    half8 B;
    half dq;
    __global const ushort* weight_ptr = src0_q + gx_2;
    __global const half* scale_ptr = src0_d + gx_2;

    for(int i=0; i<k; i+=4){

        // FULL weight + scale load (unchanged memory traffic)
        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));

        // FULL B reads (4 per step, unchanged memory) but only s0 dequant + only c0 FMA
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));   B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        dq = ((bits4.s0 & (0x000F)) - 8) * scale.s0;       c0 += B * dq;

        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dq = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0; c0 += B * dq;

        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dq = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0; c0 += B * dq;

        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dq = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0; c0 += B * dq;
    }

    // defeat DCE: collapse c0 to a scalar, store under a never-true guard (zero DRAM write)
    half acc = c0.s0+c0.s1+c0.s2+c0.s3+c0.s4+c0.s5+c0.s6+c0.s7;
    if (n_no_padding < 0) { dst[0] = acc; } // n_no_padding >= 0 always -> never executes
}

// ---- No-store variant: a clone of kernel_mul_mat_Ab_Bi_8x4 (image B, full compute) with the
// dst write-back REMOVED. The 8 vstore4 blocks are gone entirely. To stop the compiler from
// dead-code-eliminating the whole K loop (whose only observable effect would now be gone), the
// 32 accumulators are collapsed to one scalar and stored under a NEVER-TRUE guard
// (n_no_padding >= 0 always) -> all compute survives, zero bytes hit DRAM. abbi - nostore =
// the dst write-back cost. Routed via PI0_GEMM_VARIANT=abbi_nostore.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_nostore(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        __read_only image1d_buffer_t src1,  // B (1d image)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {

    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
    half8 B; // registers for activations
    half4 dequantized_weights; // registers for dequantized weights
    __global const ushort* weight_ptr = src0_q + gx_2; // pointer for weights
    __global const half* scale_ptr = src0_d + gx_2; // pointer for scales

    for(int i=0; i<k; i+=4){ //loop through K dimension

        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);

        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));

        // j=0
        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;
    }

    // NO write-back. Collapse the 32 results to one scalar and store it under a never-true guard
    // so the full K-loop compute survives DCE while zero bytes are written to DRAM.
    half acc = (c0.s0+c0.s1+c0.s2+c0.s3+c0.s4+c0.s5+c0.s6+c0.s7)
             + (c1.s0+c1.s1+c1.s2+c1.s3+c1.s4+c1.s5+c1.s6+c1.s7)
             + (c2.s0+c2.s1+c2.s2+c2.s3+c2.s4+c2.s5+c2.s6+c2.s7)
             + (c3.s0+c3.s1+c3.s2+c3.s3+c3.s4+c3.s5+c3.s6+c3.s7);
    if (n_no_padding < 0) { dst[0] = acc; } // n_no_padding >= 0 always -> never executes
}

// ---- Split-K variant: extra grid dim (global id 2) = K-slice index ----
// Each work-item reduces only over its 1/k_split of K and writes a PARTIAL plane
// to dst[ks*plane ...]; a separate reduce kernel then sums the k_split planes.
// Purpose: small-M GEMMs (e.g. M=1024) launch only 7*M/512 workgroups at N=51,
// starving the GPU; splitting K multiplies the workgroup count by k_split.
// Requires (k/k_split) % 32 == 0 (scale-block aligned) and % 4 == 0.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_splitk(
        global const ushort * src0_q,
        global const half  * src0_d,
        __read_only image1d_buffer_t src1,
        global float * dst,                 // partial buffer [k_split, n_no_padding, m]
        int m,
        int n,
        int k,
        int n_no_padding,
        int k_split
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int ks = get_global_id(2);
    int gx_2 = gx << 2;

    int k_per   = k / k_split;            // K elements per slice
    int i_start = ks * k_per;
    int i_end   = i_start + k_per;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;
    __global const ushort* weight_ptr = src0_q + gx_2;
    __global const half*   scale_ptr  = src0_d + gx_2;

    for(int i=i_start; i<i_end; i+=4){
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));

        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;

        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;

        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;

        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;
    }

    int plane = m * n_no_padding;
    __global float * out = dst + ks * plane;
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < plane){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,out+idx); idx += m; }
    if(idx+3 < plane){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,out+idx); idx += m; }
    if(idx+3 < plane){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,out+idx); idx += m; }
    if(idx+3 < plane){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,out+idx); idx += m; }
    if(idx+3 < plane){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,out+idx); idx += m; }
    if(idx+3 < plane){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,out+idx); idx += m; }
    if(idx+3 < plane){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,out+idx); idx += m; }
    if(idx+3 < plane){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,out+idx); }
}

// Sum k_split partial planes [k_split, plane] -> [plane].
kernel void kernel_splitk_reduce(
        global const float * partials,
        global float * dst,
        int plane,
        int k_split
) {
    int i = get_global_id(0);
    if (i >= plane) return;
    float s = 0.0f;
    for (int ks = 0; ks < k_split; ks++) {
        s += partials[ks * plane + i];
    }
    dst[i] = s;
}

// ---- Global-B variant: identical to kernel_mul_mat_Ab_Bi_8x4 except B (activations)
// is read through a plain global half* (vload8) instead of the image1d_buffer texture
// path. src1 holds the SAME transposed/Bi-laid-out f16 bytes as the image backing
// buffer. The original reads two adjacent half4 texels (read_imageh(src1,T) and T+1),
// which are 8 contiguous halfs, so the global equivalent is one vload8 at half-offset
// T*4 (T = gy*2 + j*n_4). Isolates the texture-cache (TP L1) contribution.
// NOTE: IDE clangd flags vload8(...,half*) as "ambiguous" — false positive; the Adreno
// driver (clBuildProgram at runtime) compiles it fine, as vload4(0,scale_ptr) proves.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_globalB(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        global const half  * src1,          // B (plain global buffer, same Bi layout)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {

    int m_4 = m >> 2;
    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
    half8 B; // registers for activations
    half4 dequantized_weights; // registers for dequantized weights
    __global const ushort* weight_ptr = src0_q + gx_2; // pointer for weights
    __global const half* scale_ptr = src0_d + gx_2; // pointer for scales
    __global const half* b_ptr = src1; // B base pointer (global-load path)

    for(int i=0; i<k; i+=4){ //loop through K dimension

        B = vload8(0, b_ptr + (gy*2 + (i)*(n_4))*4); // 2 adjacent texels = 8 contiguous halfs

        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));

        // j=0
        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B = vload8(0, b_ptr + (gy*2 + (i+1)*(n_4))*4);
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B = vload8(0, b_ptr + (gy*2 + (i+2)*(n_4))*4);
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B = vload8(0, b_ptr + (gy*2 + (i+3)*(n_4))*4);
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;
    }

    int idx = (gy<<3)*m + (gx<<2); // vectorized store 16 elements

    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}

// ---- Staggered-start variant of _globalB: each lane gx begins its K-walk at a DIFFERENT
// 4-K block (start block = gx % (k/4)) instead of all 128 lanes starting at i=0. The math is
// unchanged (each lane still sums the full K; addition commutes), but at any instant the 128
// lanes of a wave touch 128 DISTINCT B addresses instead of broadcasting one. Tests whether
// scattering the B reads (no coalescing, larger live B working set) beats the broadcast
// global path. B still comes from the plain global buffer (no texture). use_adreno_kernels
// guarantees k>=512 -> k/4>=128 > gx, so gx % (k/4) == gx (no wrap; 128 lanes all distinct).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_globalB_stagger(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        global const half  * src1,          // B (plain global buffer, same Bi layout)
        global float * dst,                 // C
        int m,                              // M
        int n,                              // N with padding
        int k,                              // K
        int n_no_padding                    // N without padding
) {

    int n_4 = n >> 2;

    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
    half8 B; // registers for activations
    half4 dequantized_weights; // registers for dequantized weights
    __global const ushort* weight_ptr = src0_q + gx_2; // pointer for weights
    __global const half* scale_ptr = src0_d + gx_2; // pointer for scales
    __global const half* b_ptr = src1; // B base pointer (global-load path)

    int n_blk = k >> 2;        // number of 4-K blocks along K
    int blk   = gx % n_blk;    // per-lane staggered start block (== gx when k>=512)

    for(int step=0; step<n_blk; step++){ //walk all K blocks, starting at a per-lane offset

        int i = blk << 2;      // K position of this block (covers i, i+1, i+2, i+3)

        B = vload8(0, b_ptr + (gy*2 + (i)*(n_4))*4); // 2 adjacent texels = 8 contiguous halfs

        ushort4 bits4 = vload4(0, weight_ptr + blk*(m)); // (i/4) == blk
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));

        // j=0
        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=1
        B = vload8(0, b_ptr + (gy*2 + (i+1)*(n_4))*4);
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=2
        B = vload8(0, b_ptr + (gy*2 + (i+2)*(n_4))*4);
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        // j=3
        B = vload8(0, b_ptr + (gy*2 + (i+3)*(n_4))*4);
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0;
        c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2;
        c3 += B * dequantized_weights.s3;

        blk++;
        if(blk >= n_blk) blk -= n_blk; // wrap (only triggers if k<512, which the gate forbids)
    }

    int idx = (gy<<3)*m + (gx<<2); // vectorized store 16 elements

    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
        idx += m;
    }
    if(idx+3 < m*n_no_padding){
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
    }
}

// ---- B-only probes: stream B at the EXACT same addresses/frequency as Ab_Bi, but with
// NO weights and NO dequant (pure B read -> accumulate -> store). Isolates the B read PATH
// with A entirely absent, to separate "dedicated TP L1 read path" from "L2 contention with
// A". src0_q/src0_d kept in the signature (uniform arg list) but unused. Output is not a
// real GEMM (timing-only). _global reads B via vload8, _image via the texture.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_Bonly_global(
        global const ushort * src0_q,       // unused
        global const half  * src0_d,        // unused
        global const half  * src1,          // B (plain global buffer)
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    __global const half* b_ptr = src1;
    for (int i = 0; i < k; i += 4) {
        // mirror the full kernel: each B is reused across all 4 accumulators (4 MACs/B there
        // -> 4 adds/B here) so compute intensity & latency-hiding match; B reads are unchanged.
        B = vload8(0, b_ptr + (gy*2 + (i)*(n_4))*4);   c0 += B; c1 += B; c2 += B; c3 += B;
        B = vload8(0, b_ptr + (gy*2 + (i+1)*(n_4))*4); c0 += B; c1 += B; c2 += B; c3 += B;
        B = vload8(0, b_ptr + (gy*2 + (i+2)*(n_4))*4); c0 += B; c1 += B; c2 += B; c3 += B;
        B = vload8(0, b_ptr + (gy*2 + (i+3)*(n_4))*4); c0 += B; c1 += B; c2 += B; c3 += B;
    }
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }
}

// ---- A-only probe: stream the WEIGHTS (A: q4_0 bits + scales) and dequant at the EXACT same
// addresses/frequency as the full kernel, but with NO B read and NO MAC-multiply (the
// dequantized weights are accumulated directly so the reads+dequant survive dead-code
// elimination). Isolates the weight read path (global/L2) + dequant cost with B entirely
// absent -> compare vs Bonly_image (B via TP L1) to see whether the full image kernel's two
// data paths (weights@L2 vs B@TP-L1) cost the same. src1 kept for a uniform arg list, unused.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_Aonly(
        global const ushort * src0_q,       // quantized A
        global const half  * src0_d,        // A scales
        global const half  * src1,          // unused
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half4 dequantized_weights;
    __global const ushort* weight_ptr = src0_q + gx_2;
    __global const half* scale_ptr = src0_d + gx_2;
    for (int i = 0; i < k; i += 4) {
        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));
        // mirror the full kernel's 4 dequant j-steps; accumulate dequantized weights (no B).
        // half8 += half is component-wise -> keeps the 4-accumulator shape, no MAC-multiply.
        // j=0
        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 += dequantized_weights.s0; c1 += dequantized_weights.s1; c2 += dequantized_weights.s2; c3 += dequantized_weights.s3;
        // j=1
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 += dequantized_weights.s0; c1 += dequantized_weights.s1; c2 += dequantized_weights.s2; c3 += dequantized_weights.s3;
        // j=2
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 += dequantized_weights.s0; c1 += dequantized_weights.s1; c2 += dequantized_weights.s2; c3 += dequantized_weights.s3;
        // j=3
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 += dequantized_weights.s0; c1 += dequantized_weights.s1; c2 += dequantized_weights.s2; c3 += dequantized_weights.s3;
    }
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_Bonly_image(
        global const ushort * src0_q,       // unused
        global const half  * src0_d,        // unused
        __read_only image1d_buffer_t src1,  // B (texture)
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    for (int i = 0; i < k; i += 4) {
        // mirror the full kernel: each B reused across all 4 accumulators (see Bonly_global).
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));   B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);   c0 += B; c1 += B; c2 += B; c3 += B;
        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1); c0 += B; c1 += B; c2 += B; c3 += B;
        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1); c0 += B; c1 += B; c2 += B; c3 += B;
        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4)); B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1); c0 += B; c1 += B; c2 += B; c3 += B;
    }
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }
}

// ============================================================================
// PI0 DIAGNOSTIC PROBES (E1, E2) -- timing-only, NOT on the production path.
// Selected via PI0_GEMM_VARIANT; share the standard 8-arg Ab_Bi signature so
// they slot into the existing N>1 dispatch (inject adds a 9th arg = inj count).
// ============================================================================

// E1: pure-FMA throughput microbench (calibrate the real ALU peak P_fma).
// No memory in the loop. 8 independent half8 accumulators = 64-way ILP, enough
// to hide FMA latency so we measure sustained THROUGHPUT, not latency. k is the
// inner trip count. Total FMAs = (threads) * k * 8(half8) * 8(lanes); host turns
// this into GFLOPS. b<1, a>0 -> fixed point ~1.0, stays in normal fp16 range.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_fma_peak(
        global const ushort * src0_q,       // unused
        global const half  * src0_d,        // unused
        __read_only image1d_buffer_t src1,  // unused
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    // distinct seeds (+ runtime gid term) so the 8 chains can't be CSE-collapsed to one
    half8 c0=(half8)(0.11f), c1=(half8)(0.23f), c2=(half8)(0.37f), c3=(half8)(0.51f);
    half8 c4=(half8)(0.67f), c5=(half8)(0.79f), c6=(half8)(0.91f), c7=(half8)(1.03f);
    c0 += (half8)((half)(gx & 7) * (half)0.001f);
    const half8 b = (half8)(0.9995f);
    const half8 a = (half8)(0.0005f);
    for (int i = 0; i < k; ++i) {
        // native MAD (matches the GEMM's `c += B*w`), NOT the fma() builtin -- the builtin
        // compiles to a correctly-rounded SOFTWARE routine on Adreno (~200x slower).
        c0 = c0*b + a; c1 = c1*b + a; c2 = c2*b + a; c3 = c3*b + a;
        c4 = c4*b + a; c5 = c5*b + a; c6 = c6*b + a; c7 = c7*b + a;
    }
    half8 cc = (c0+c1+c2+c3) + (c4+c5+c6+c7);
    half acc = cc.s0+cc.s1+cc.s2+cc.s3+cc.s4+cc.s5+cc.s6+cc.s7;
    if (n_no_padding < 0) { dst[gy + gx] = acc; } // n_no_padding=N>=0 -> never; defeats DCE
}

// E2: production 8x4 GEMM (with real store) + `inj` injected independent half8 FMAs
// per outer iteration. The injected FMAs touch only private regs (no memory) and are
// guarded by a single uniform `if (inj)` so the inj=0 baseline == production abbi.
// 4 dummy chains (d0..d3) reused every 4th slot => 4-way ILP so the injected work is
// not itself latency-bound and genuinely competes for issue slots. Real GFLOPS counts
// ONLY the real MACs (2*M*N*K); the inj curve shape is the result (flat-then-drop = free slots).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_inject(
        global const ushort * src0_q,
        global const half  * src0_d,
        __read_only image1d_buffer_t src1,
        global float * dst,
        int m, int n, int k, int n_no_padding,
        int inj
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;

    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    half4 dequantized_weights;
    __global const ushort* weight_ptr = src0_q + gx_2;
    __global const half* scale_ptr = src0_d + gx_2;

    half8 d0=(half8)(0.31f), d1=(half8)(0.53f), d2=(half8)(0.71f), d3=(half8)(0.97f);
    const half8 ib = (half8)(0.9995f);
    const half8 ia = (half8)(0.0005f);

    for(int i=0; i<k; i+=4){
        B.s0123 = read_imageh(src1, gy*2 + (i)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i)*(n_4)+1);
        ushort4 bits4 = vload4(0, weight_ptr + (i/4)*(m));
        half4 scale = vload4(0, scale_ptr + (i/32)*(m));

        dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
        dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
        dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
        dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;

        B.s0123 = read_imageh(src1, gy*2 + (i+1)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+1)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;

        B.s0123 = read_imageh(src1, gy*2 + (i+2)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+2)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;

        B.s0123 = read_imageh(src1, gy*2 + (i+3)*(n_4));
        B.s4567 = read_imageh(src1, gy*2 + (i+3)*(n_4)+1);
        dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
        dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
        dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
        dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
        c0 += B * dequantized_weights.s0; c1 += B * dequantized_weights.s1;
        c2 += B * dequantized_weights.s2; c3 += B * dequantized_weights.s3;

        // injected independent FMAs (uniform branch; inj=0 -> skipped entirely)
        if (inj) {
            if (inj >  0) d0 = d0*ib + ia;
            if (inj >  1) d1 = d1*ib + ia;
            if (inj >  2) d2 = d2*ib + ia;
            if (inj >  3) d3 = d3*ib + ia;
            if (inj >  4) d0 = d0*ib + ia;
            if (inj >  5) d1 = d1*ib + ia;
            if (inj >  6) d2 = d2*ib + ia;
            if (inj >  7) d3 = d3*ib + ia;
            if (inj >  8) d0 = d0*ib + ia;
            if (inj >  9) d1 = d1*ib + ia;
            if (inj > 10) d2 = d2*ib + ia;
            if (inj > 11) d3 = d3*ib + ia;
            if (inj > 12) d0 = d0*ib + ia;
            if (inj > 13) d1 = d1*ib + ia;
            if (inj > 14) d2 = d2*ib + ia;
            if (inj > 15) d3 = d3*ib + ia;
        }
    }

    // real store (identical to the production kernel)
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }

    // fold injected dummies into a never-true sink so they survive DCE
    half8 dd = (d0+d1) + (d2+d3);
    half ds = dd.s0+dd.s1+dd.s2+dd.s3+dd.s4+dd.s5+dd.s6+dd.s7;
    if (n_no_padding < 0) { dst[0] = ds; }
}

// E5: wider N-tile variants (16x4, 32x4) -- the weight-REUSE / bandwidth lever.
// The 8x4 kernel re-loads each M-block's weights once per N-tile-group = ceil(N/8) times.
// A 16-N tile halves that (ceil(N/16)); 32-N quarters it -> directly cuts weight DRAM traffic,
// the prefix bottleneck. Cost: 2x / 4x the half8 accumulators -> more registers -> lower
// occupancy. This sweep brackets reuse-vs-occupancy. Requires N % tile == 0 (bench uses N=768).
// Same 8-arg signature; dispatch sets global[0] = ceil(N/16 or /32).

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_16x4(
        global const ushort * src0_q, global const half * src0_d,
        __read_only image1d_buffer_t src1, global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);              // N-tile of 16
    int gx = get_global_id(1);
    int gx_2 = gx << 2;
    int bpx = gy << 2;                      // 16 N = 4 image pixels per WI
    half8 cA0=0,cA1=0,cA2=0,cA3=0;          // M0..3, N0..7
    half8 cB0=0,cB1=0,cB2=0,cB3=0;          // M0..3, N8..15
    half8 Ba, Bb; half4 w;
    __global const ushort* wp = src0_q + gx_2;
    __global const half*   sp = src0_d + gx_2;
    for (int i=0;i<k;i+=4){
        ushort4 bits4 = vload4(0, wp + (i/4)*m);
        half4 scale   = vload4(0, sp + (i/32)*m);
        #define J16(SH,MASK) \
            Ba.s0123=read_imageh(src1,bpx+(i+SH)*n_4+0); Ba.s4567=read_imageh(src1,bpx+(i+SH)*n_4+1); \
            Bb.s0123=read_imageh(src1,bpx+(i+SH)*n_4+2); Bb.s4567=read_imageh(src1,bpx+(i+SH)*n_4+3); \
            w.s0=((MASK(bits4.s0))-8)*scale.s0; w.s1=((MASK(bits4.s1))-8)*scale.s1; \
            w.s2=((MASK(bits4.s2))-8)*scale.s2; w.s3=((MASK(bits4.s3))-8)*scale.s3; \
            cA0+=Ba*w.s0; cB0+=Bb*w.s0; cA1+=Ba*w.s1; cB1+=Bb*w.s1; \
            cA2+=Ba*w.s2; cB2+=Bb*w.s2; cA3+=Ba*w.s3; cB3+=Bb*w.s3;
        #define M0(x) ((x)&0x000F)
        #define M1(x) (((x)&0x00F0)>>4)
        #define M2(x) (((x)&0x0F00)>>8)
        #define M3(x) (((x)&0xF000)>>12)
        J16(0,M0) J16(1,M1) J16(2,M2) J16(3,M3)
        #undef J16
        #undef M0
        #undef M1
        #undef M2
        #undef M3
    }
    int idx = (gy<<4)*m + (gx<<2);          // first N = gy*16
    // rows 0..7 from cA, 8..15 from cB
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s0,cA1.s0,cA2.s0,cA3.s0),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s1,cA1.s1,cA2.s1,cA3.s1),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s2,cA1.s2,cA2.s2,cA3.s2),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s3,cA1.s3,cA2.s3,cA3.s3),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s4,cA1.s4,cA2.s4,cA3.s4),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s5,cA1.s5,cA2.s5,cA3.s5),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s6,cA1.s6,cA2.s6,cA3.s6),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cA0.s7,cA1.s7,cA2.s7,cA3.s7),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s0,cB1.s0,cB2.s0,cB3.s0),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s1,cB1.s1,cB2.s1,cB3.s1),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s2,cB1.s2,cB2.s2,cB3.s2),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s3,cB1.s3,cB2.s3,cB3.s3),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s4,cB1.s4,cB2.s4,cB3.s4),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s5,cB1.s5,cB2.s5,cB3.s5),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s6,cB1.s6,cB2.s6,cB3.s6),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(cB0.s7,cB1.s7,cB2.s7,cB3.s7),0,dst+idx);}
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_32x4(
        global const ushort * src0_q, global const half * src0_d,
        __read_only image1d_buffer_t src1, global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);              // N-tile of 32
    int gx = get_global_id(1);
    int gx_2 = gx << 2;
    int bpx = gy << 3;                      // 32 N = 8 image pixels per WI
    half8 cA0=0,cA1=0,cA2=0,cA3=0;          // N0..7
    half8 cB0=0,cB1=0,cB2=0,cB3=0;          // N8..15
    half8 cC0=0,cC1=0,cC2=0,cC3=0;          // N16..23
    half8 cD0=0,cD1=0,cD2=0,cD3=0;          // N24..31
    half8 Ba,Bb,Bc,Bd; half4 w;
    __global const ushort* wp = src0_q + gx_2;
    __global const half*   sp = src0_d + gx_2;
    for (int i=0;i<k;i+=4){
        ushort4 bits4 = vload4(0, wp + (i/4)*m);
        half4 scale   = vload4(0, sp + (i/32)*m);
        #define J32(SH,MASK) \
            Ba.s0123=read_imageh(src1,bpx+(i+SH)*n_4+0); Ba.s4567=read_imageh(src1,bpx+(i+SH)*n_4+1); \
            Bb.s0123=read_imageh(src1,bpx+(i+SH)*n_4+2); Bb.s4567=read_imageh(src1,bpx+(i+SH)*n_4+3); \
            Bc.s0123=read_imageh(src1,bpx+(i+SH)*n_4+4); Bc.s4567=read_imageh(src1,bpx+(i+SH)*n_4+5); \
            Bd.s0123=read_imageh(src1,bpx+(i+SH)*n_4+6); Bd.s4567=read_imageh(src1,bpx+(i+SH)*n_4+7); \
            w.s0=((MASK(bits4.s0))-8)*scale.s0; w.s1=((MASK(bits4.s1))-8)*scale.s1; \
            w.s2=((MASK(bits4.s2))-8)*scale.s2; w.s3=((MASK(bits4.s3))-8)*scale.s3; \
            cA0+=Ba*w.s0;cB0+=Bb*w.s0;cC0+=Bc*w.s0;cD0+=Bd*w.s0; \
            cA1+=Ba*w.s1;cB1+=Bb*w.s1;cC1+=Bc*w.s1;cD1+=Bd*w.s1; \
            cA2+=Ba*w.s2;cB2+=Bb*w.s2;cC2+=Bc*w.s2;cD2+=Bd*w.s2; \
            cA3+=Ba*w.s3;cB3+=Bb*w.s3;cC3+=Bc*w.s3;cD3+=Bd*w.s3;
        #define M0(x) ((x)&0x000F)
        #define M1(x) (((x)&0x00F0)>>4)
        #define M2(x) (((x)&0x0F00)>>8)
        #define M3(x) (((x)&0xF000)>>12)
        J32(0,M0) J32(1,M1) J32(2,M2) J32(3,M3)
        #undef J32
        #undef M0
        #undef M1
        #undef M2
        #undef M3
    }
    int idx = (gy<<5)*m + (gx<<2);          // first N = gy*32
    #define ROW(V) if(idx+3<m*n_no_padding){vstore4((float4)V,0,dst+idx);idx+=m;}
    ROW((cA0.s0,cA1.s0,cA2.s0,cA3.s0)) ROW((cA0.s1,cA1.s1,cA2.s1,cA3.s1))
    ROW((cA0.s2,cA1.s2,cA2.s2,cA3.s2)) ROW((cA0.s3,cA1.s3,cA2.s3,cA3.s3))
    ROW((cA0.s4,cA1.s4,cA2.s4,cA3.s4)) ROW((cA0.s5,cA1.s5,cA2.s5,cA3.s5))
    ROW((cA0.s6,cA1.s6,cA2.s6,cA3.s6)) ROW((cA0.s7,cA1.s7,cA2.s7,cA3.s7))
    ROW((cB0.s0,cB1.s0,cB2.s0,cB3.s0)) ROW((cB0.s1,cB1.s1,cB2.s1,cB3.s1))
    ROW((cB0.s2,cB1.s2,cB2.s2,cB3.s2)) ROW((cB0.s3,cB1.s3,cB2.s3,cB3.s3))
    ROW((cB0.s4,cB1.s4,cB2.s4,cB3.s4)) ROW((cB0.s5,cB1.s5,cB2.s5,cB3.s5))
    ROW((cB0.s6,cB1.s6,cB2.s6,cB3.s6)) ROW((cB0.s7,cB1.s7,cB2.s7,cB3.s7))
    ROW((cC0.s0,cC1.s0,cC2.s0,cC3.s0)) ROW((cC0.s1,cC1.s1,cC2.s1,cC3.s1))
    ROW((cC0.s2,cC1.s2,cC2.s2,cC3.s2)) ROW((cC0.s3,cC1.s3,cC2.s3,cC3.s3))
    ROW((cC0.s4,cC1.s4,cC2.s4,cC3.s4)) ROW((cC0.s5,cC1.s5,cC2.s5,cC3.s5))
    ROW((cC0.s6,cC1.s6,cC2.s6,cC3.s6)) ROW((cC0.s7,cC1.s7,cC2.s7,cC3.s7))
    ROW((cD0.s0,cD1.s0,cD2.s0,cD3.s0)) ROW((cD0.s1,cD1.s1,cD2.s1,cD3.s1))
    ROW((cD0.s2,cD1.s2,cD2.s2,cD3.s2)) ROW((cD0.s3,cD1.s3,cD2.s3,cD3.s3))
    ROW((cD0.s4,cD1.s4,cD2.s4,cD3.s4)) ROW((cD0.s5,cD1.s5,cD2.s5,cD3.s5))
    ROW((cD0.s6,cD1.s6,cD2.s6,cD3.s6)) ROW((cD0.s7,cD1.s7,cD2.s7,cD3.s7))
    #undef ROW
}

// ============================================================================
// E6 PROBES (Q1/Q2/Q3) -- timing-only, default off.
// ============================================================================

// E6a (Q3): weights read via the TEXTURE pipe (image1d_buffer, read_imageui) instead of the
// global buffer (UCHE). arg0 is an image; everything else identical to the production 8x4.
// The A image is CL_R/CL_UNSIGNED_INT32 over extra0_q4_0->q: 1 texel = 1 uint32 = 2 ushorts.
// The production reads ushort4 (=2 uint32) per (i/4) step -> 2 read_imageui texels here.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_aimg(
        __read_only image1d_buffer_t src0_img,   // q4_0 weights as uint32 texels (texture pipe)
        global const half  * src0_d,
        __read_only image1d_buffer_t src1,       // B
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;
    half8 c0=0,c1=0,c2=0,c3=0; half8 B; half4 w;
    __global const half* scale_ptr = src0_d + gx_2;
    for(int i=0;i<k;i+=4){
        // 4 ushorts (4 M, 4 K-nibbles each) = 2 uint32 texels; ushort offset = gx_2 + (i/4)*m
        int bu = (gx_2 + (i/4)*m) >> 1;
        uint p0 = read_imageui(src0_img, bu).x;
        uint p1 = read_imageui(src0_img, bu+1).x;
        ushort4 bits4 = (ushort4)((ushort)(p0 & 0xFFFF),(ushort)(p0 >> 16),
                                  (ushort)(p1 & 0xFFFF),(ushort)(p1 >> 16));
        half4 scale = vload4(0, scale_ptr + (i/32)*m);
        B.s0123=read_imageh(src1,gy*2+(i)*n_4);   B.s4567=read_imageh(src1,gy*2+(i)*n_4+1);
        w.s0=((bits4.s0&0x000F)-8)*scale.s0; w.s1=((bits4.s1&0x000F)-8)*scale.s1;
        w.s2=((bits4.s2&0x000F)-8)*scale.s2; w.s3=((bits4.s3&0x000F)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+1)*n_4); B.s4567=read_imageh(src1,gy*2+(i+1)*n_4+1);
        w.s0=(((bits4.s0&0x00F0)>>4)-8)*scale.s0; w.s1=(((bits4.s1&0x00F0)>>4)-8)*scale.s1;
        w.s2=(((bits4.s2&0x00F0)>>4)-8)*scale.s2; w.s3=(((bits4.s3&0x00F0)>>4)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+2)*n_4); B.s4567=read_imageh(src1,gy*2+(i+2)*n_4+1);
        w.s0=(((bits4.s0&0x0F00)>>8)-8)*scale.s0; w.s1=(((bits4.s1&0x0F00)>>8)-8)*scale.s1;
        w.s2=(((bits4.s2&0x0F00)>>8)-8)*scale.s2; w.s3=(((bits4.s3&0x0F00)>>8)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+3)*n_4); B.s4567=read_imageh(src1,gy*2+(i+3)*n_4+1);
        w.s0=(((bits4.s0&0xF000)>>12)-8)*scale.s0; w.s1=(((bits4.s1&0xF000)>>12)-8)*scale.s1;
        w.s2=(((bits4.s2&0xF000)>>12)-8)*scale.s2; w.s3=(((bits4.s3&0xF000)>>12)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
    }
    int idx=(gy<<3)*m+(gx<<2);
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx);}
}

// E6b (Q1): production 8x4 with the store boundary checks REMOVED (unconditional vstore4).
// Tests the kernel's own claim that the guards reduce register footprint -> more waves.
// Requires N % 8 == 0 (all 8 N rows valid); bench uses N=768. Compile-time M/N/K folding is
// NOT tested separately: E2 showed address-calc lands in idle issue slots -> cannot help.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_noguard(
        global const ushort * src0_q, global const half * src0_d,
        __read_only image1d_buffer_t src1, global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;
    half8 c0=0,c1=0,c2=0,c3=0; half8 B; half4 w;
    __global const ushort* wp = src0_q + gx_2;
    __global const half*   sp = src0_d + gx_2;
    for(int i=0;i<k;i+=4){
        ushort4 bits4 = vload4(0, wp + (i/4)*m);
        half4 scale   = vload4(0, sp + (i/32)*m);
        B.s0123=read_imageh(src1,gy*2+(i)*n_4);   B.s4567=read_imageh(src1,gy*2+(i)*n_4+1);
        w.s0=((bits4.s0&0x000F)-8)*scale.s0;w.s1=((bits4.s1&0x000F)-8)*scale.s1;w.s2=((bits4.s2&0x000F)-8)*scale.s2;w.s3=((bits4.s3&0x000F)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+1)*n_4); B.s4567=read_imageh(src1,gy*2+(i+1)*n_4+1);
        w.s0=(((bits4.s0&0x00F0)>>4)-8)*scale.s0;w.s1=(((bits4.s1&0x00F0)>>4)-8)*scale.s1;w.s2=(((bits4.s2&0x00F0)>>4)-8)*scale.s2;w.s3=(((bits4.s3&0x00F0)>>4)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+2)*n_4); B.s4567=read_imageh(src1,gy*2+(i+2)*n_4+1);
        w.s0=(((bits4.s0&0x0F00)>>8)-8)*scale.s0;w.s1=(((bits4.s1&0x0F00)>>8)-8)*scale.s1;w.s2=(((bits4.s2&0x0F00)>>8)-8)*scale.s2;w.s3=(((bits4.s3&0x0F00)>>8)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+3)*n_4); B.s4567=read_imageh(src1,gy*2+(i+3)*n_4+1);
        w.s0=(((bits4.s0&0xF000)>>12)-8)*scale.s0;w.s1=(((bits4.s1&0xF000)>>12)-8)*scale.s1;w.s2=(((bits4.s2&0xF000)>>12)-8)*scale.s2;w.s3=(((bits4.s3&0xF000)>>12)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
    }
    int idx=(gy<<3)*m+(gx<<2);
    vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m;
    vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m;
    vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m;
    vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m;
    vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m;
    vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m;
    vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m;
    vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx);
}

// E6c (Q2): production 8x4 + #pragma unroll 2 on the K-loop (8 K per iter -> 2x live B/weight
// regs). Expectation: occupancy regression (E5). Same signature/layout as production.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_unroll(
        global const ushort * src0_q, global const half * src0_d,
        __read_only image1d_buffer_t src1, global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    int gx_2 = gx << 2;
    half8 c0=0,c1=0,c2=0,c3=0; half8 B; half4 w;
    __global const ushort* wp = src0_q + gx_2;
    __global const half*   sp = src0_d + gx_2;
    __attribute__((opencl_unroll_hint(2)))
    for(int i=0;i<k;i+=4){
        ushort4 bits4 = vload4(0, wp + (i/4)*m);
        half4 scale   = vload4(0, sp + (i/32)*m);
        B.s0123=read_imageh(src1,gy*2+(i)*n_4);   B.s4567=read_imageh(src1,gy*2+(i)*n_4+1);
        w.s0=((bits4.s0&0x000F)-8)*scale.s0;w.s1=((bits4.s1&0x000F)-8)*scale.s1;w.s2=((bits4.s2&0x000F)-8)*scale.s2;w.s3=((bits4.s3&0x000F)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+1)*n_4); B.s4567=read_imageh(src1,gy*2+(i+1)*n_4+1);
        w.s0=(((bits4.s0&0x00F0)>>4)-8)*scale.s0;w.s1=(((bits4.s1&0x00F0)>>4)-8)*scale.s1;w.s2=(((bits4.s2&0x00F0)>>4)-8)*scale.s2;w.s3=(((bits4.s3&0x00F0)>>4)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+2)*n_4); B.s4567=read_imageh(src1,gy*2+(i+2)*n_4+1);
        w.s0=(((bits4.s0&0x0F00)>>8)-8)*scale.s0;w.s1=(((bits4.s1&0x0F00)>>8)-8)*scale.s1;w.s2=(((bits4.s2&0x0F00)>>8)-8)*scale.s2;w.s3=(((bits4.s3&0x0F00)>>8)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
        B.s0123=read_imageh(src1,gy*2+(i+3)*n_4); B.s4567=read_imageh(src1,gy*2+(i+3)*n_4+1);
        w.s0=(((bits4.s0&0xF000)>>12)-8)*scale.s0;w.s1=(((bits4.s1&0xF000)>>12)-8)*scale.s1;w.s2=(((bits4.s2&0xF000)>>12)-8)*scale.s2;w.s3=(((bits4.s3&0xF000)>>12)-8)*scale.s3;
        c0+=B*w.s0;c1+=B*w.s1;c2+=B*w.s2;c3+=B*w.s3;
    }
    int idx=(gy<<3)*m+(gx<<2);
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx);idx+=m;}
    if(idx+3<m*n_no_padding){vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx);}
}

// Step-1 probe: B-only stream as INT8 (RGBA SIGNED_INT8) instead of f16. SAME broadcast access
// pattern and SAME number of texel reads as kernel_mul_mat_Ab_Bi_8x4_Bonly_image, but each texel
// is 4 bytes (4xint8) vs 8 bytes (4xhalf) -> HALF the B bytes. Tests whether B-read is
// byte-bandwidth-bound (time halves) or read-count/latency-bound (time unchanged). Timing only;
// the int8 image aliases the f16 B buffer (garbage data, fine for timing). convert_half4 mirrors
// the dequant a real q8-B kernel would do (the int8->fp16 convert, ALU ~free per E2).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_Bonly_image_i8(
        global const ushort * src0_q,       // unused
        global const half  * src0_d,        // unused
        __read_only image1d_buffer_t src1,  // B as int8 (RGBA SIGNED_INT8)
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_4 = n >> 2;
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    for (int i = 0; i < k; i += 4) {
        B.s0123 = convert_half4(read_imagei(src1, gy*2 + (i)*(n_4)));   B.s4567 = convert_half4(read_imagei(src1, gy*2 + (i)*(n_4)+1));   c0 += B; c1 += B; c2 += B; c3 += B;
        B.s0123 = convert_half4(read_imagei(src1, gy*2 + (i+1)*(n_4))); B.s4567 = convert_half4(read_imagei(src1, gy*2 + (i+1)*(n_4)+1)); c0 += B; c1 += B; c2 += B; c3 += B;
        B.s0123 = convert_half4(read_imagei(src1, gy*2 + (i+2)*(n_4))); B.s4567 = convert_half4(read_imagei(src1, gy*2 + (i+2)*(n_4)+1)); c0 += B; c1 += B; c2 += B; c3 += B;
        B.s0123 = convert_half4(read_imagei(src1, gy*2 + (i+3)*(n_4))); B.s4567 = convert_half4(read_imagei(src1, gy*2 + (i+3)*(n_4)+1)); c0 += B; c1 += B; c2 += B; c3 += B;
    }
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }
}

// Probe: B-only stream reading 128-bit texels (float4) -> 8 N per fetch, reinterpreted to 8 halfs.
// vs Bonly_image (half4 = 64-bit = 4 N/fetch): SAME bytes, SAME 8 N per k-substep, but HALF the
// read_image calls (1 float4 vs 2 half4). Directly measures the image->TP-L1 FETCH WIDTH:
// time ~0.5x of bonly_i -> TP fetch handles >=128-bit/instruction (fetch-bound, real lever);
// ~1.0x -> fetch width is 64-bit (128-bit read = 2 internal fetches, no win). Timing only.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_Bonly_image_f4(
        global const ushort * src0_q,       // unused
        global const half  * src0_d,        // unused
        __read_only image1d_buffer_t src1,  // B as RGBA float (128-bit texels, 8 halfs each)
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_8 = n >> 3;                       // float4 texels per k-row (each = 8 N)
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    float4 f;
    for (int i = 0; i < k; i += 4) {
        f = read_imagef(src1, gy + (i)*(n_8));   B = (half8)(as_half2(f.x), as_half2(f.y), as_half2(f.z), as_half2(f.w)); c0+=B; c1+=B; c2+=B; c3+=B;
        f = read_imagef(src1, gy + (i+1)*(n_8)); B = (half8)(as_half2(f.x), as_half2(f.y), as_half2(f.z), as_half2(f.w)); c0+=B; c1+=B; c2+=B; c3+=B;
        f = read_imagef(src1, gy + (i+2)*(n_8)); B = (half8)(as_half2(f.x), as_half2(f.y), as_half2(f.z), as_half2(f.w)); c0+=B; c1+=B; c2+=B; c3+=B;
        f = read_imagef(src1, gy + (i+3)*(n_8)); B = (half8)(as_half2(f.x), as_half2(f.y), as_half2(f.z), as_half2(f.w)); c0+=B; c1+=B; c2+=B; c3+=B;
    }
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }
}

// Disambiguator: same per-substep work as Bonly_image (4 half8 accumulates + loop + store) but
// B comes from a cheap per-i register value, ZERO image reads. B varies with i (convert int->half)
// so the loop can't be hoisted/strength-reduced. vs bonly_i: ~bonly_i -> the 2 image reads were
// free/hidden (kernel is accumulate/loop/occupancy-bound, NOT B-read-bound); << bonly_i -> the
// image reads ARE the cost (B-read-bound, fetch width 64-bit). Timing only.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_Bonly_noread(
        global const ushort * src0_q,       // unused
        global const half  * src0_d,        // unused
        __read_only image1d_buffer_t src1,  // unused (no reads)
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B;
    for (int i = 0; i < k; i += 4) {
        B = (half8)((half)(i      & 1023)); c0 += B; c1 += B; c2 += B; c3 += B;
        B = (half8)((half)((i+1)  & 1023)); c0 += B; c1 += B; c2 += B; c3 += B;
        B = (half8)((half)((i+2)  & 1023)); c0 += B; c1 += B; c2 += B; c3 += B;
        B = (half8)((half)((i+3)  & 1023)); c0 += B; c1 += B; c2 += B; c3 += B;
    }
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }
}

// Probe B (CORRECTED q8): B as RGBA-int16 (64-bit texel) packing 8 int8 -> 8 N per ONE 64-bit
// fetch (vs half4's 4 N/fetch) -> HALF the read_image calls of bonly_i for the same 8 N. This is
// the right packing (earlier bonly_i8 used 4-int8/32-bit = still 4 N/fetch; bonly_f4 used 8N/128b
// = 2 fetches). If TP fetch=64-bit-bound: ~0.5-0.7x of bonly_i. Unpack (8 int8 from 4 int16)
// mirrors the q8 dequant ALU. Timing only (int16 image aliases the f16 B buffer; garbage data).
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x4_Bonly_image_q8(
        global const ushort * src0_q,       // unused
        global const half  * src0_d,        // unused
        __read_only image1d_buffer_t src1,  // B as RGBA int16 (64-bit texel = 8 int8 = 8 N)
        global float * dst,
        int m, int n, int k, int n_no_padding
) {
    int n_8 = n >> 3;                       // int16x4 texels per k-row (each = 8 N)
    int gy = get_global_id(0);
    int gx = get_global_id(1);
    half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half8 B; int4 t;
    for (int i = 0; i < k; i += 4) {
        t = read_imagei(src1, gy + (i)*(n_8));   B = (half8)(convert_half(t.x&0xFF),convert_half((t.x>>8)&0xFF),convert_half(t.y&0xFF),convert_half((t.y>>8)&0xFF),convert_half(t.z&0xFF),convert_half((t.z>>8)&0xFF),convert_half(t.w&0xFF),convert_half((t.w>>8)&0xFF)); c0+=B;c1+=B;c2+=B;c3+=B;
        t = read_imagei(src1, gy + (i+1)*(n_8)); B = (half8)(convert_half(t.x&0xFF),convert_half((t.x>>8)&0xFF),convert_half(t.y&0xFF),convert_half((t.y>>8)&0xFF),convert_half(t.z&0xFF),convert_half((t.z>>8)&0xFF),convert_half(t.w&0xFF),convert_half((t.w>>8)&0xFF)); c0+=B;c1+=B;c2+=B;c3+=B;
        t = read_imagei(src1, gy + (i+2)*(n_8)); B = (half8)(convert_half(t.x&0xFF),convert_half((t.x>>8)&0xFF),convert_half(t.y&0xFF),convert_half((t.y>>8)&0xFF),convert_half(t.z&0xFF),convert_half((t.z>>8)&0xFF),convert_half(t.w&0xFF),convert_half((t.w>>8)&0xFF)); c0+=B;c1+=B;c2+=B;c3+=B;
        t = read_imagei(src1, gy + (i+3)*(n_8)); B = (half8)(convert_half(t.x&0xFF),convert_half((t.x>>8)&0xFF),convert_half(t.y&0xFF),convert_half((t.y>>8)&0xFF),convert_half(t.z&0xFF),convert_half((t.z>>8)&0xFF),convert_half(t.w&0xFF),convert_half((t.w>>8)&0xFF)); c0+=B;c1+=B;c2+=B;c3+=B;
    }
    int idx = (gy<<3)*m + (gx<<2);
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s0,c1.s0,c2.s0,c3.s0),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s1,c1.s1,c2.s1,c3.s1),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s2,c1.s2,c2.s2,c3.s2),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s3,c1.s3,c2.s3,c3.s3),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s4,c1.s4,c2.s4,c3.s4),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s5,c1.s5,c2.s5,c3.s5),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s6,c1.s6,c2.s6,c3.s6),0,dst+idx); idx+=m; }
    if(idx+3 < m*n_no_padding){ vstore4((float4)(c0.s7,c1.s7,c2.s7,c3.s7),0,dst+idx); }
}
