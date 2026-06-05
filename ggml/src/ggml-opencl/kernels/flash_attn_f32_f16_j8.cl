// flash_attn_f32_f16_j8.cl — J_STRIDE=8 variant of flash_attn_f32_f16.
//
// Identical interface and semantics to the original kernel, but the inner
// j-loop processes 8 K rows per iteration with 8 named-scalar dot_acc /
// score / p variables. This cuts the o_acc[] read-modify-write traffic
// 4× vs the original j+=2 implementation, dramatically reducing register
// spill pressure on Adreno's small per-WI register file (the original
// V product was ~40× slower than QK^T despite identical FLOP count;
// see attn_breakdown spike in dp8-spike/ for the diagnosis).
//
// Validated for PI0 shapes: DK=DV=128, BLOCK_M=32, BLOCK_N=32 → 4 outer
// j-iters per K-tile. BLOCK_N must be divisible by 8.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)

#ifndef J_STRIDE
#define J_STRIDE 8
#endif

#if J_STRIDE != 2 && J_STRIDE != 4 && J_STRIDE != 8 && J_STRIDE != 16
#error "J_STRIDE must be 2, 4, 8, or 16"
#endif
#if (BLOCK_N % J_STRIDE) != 0
#error "BLOCK_N must be divisible by J_STRIDE"
#endif

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) return 1.0f;
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
    return pow(base, exph);
}

// ---- Macros to unroll J_STRIDE lanes without making the body unreadable ----
#if J_STRIDE == 2
#define EACH_LANE(MACRO) MACRO(0) MACRO(1)
#elif J_STRIDE == 4
#define EACH_LANE(MACRO) MACRO(0) MACRO(1) MACRO(2) MACRO(3)
#elif J_STRIDE == 8
#define EACH_LANE(MACRO) MACRO(0) MACRO(1) MACRO(2) MACRO(3) MACRO(4) MACRO(5) MACRO(6) MACRO(7)
#elif J_STRIDE == 16
#define EACH_LANE(MACRO) MACRO(0) MACRO(1) MACRO(2) MACRO(3) MACRO(4) MACRO(5) MACRO(6) MACRO(7) MACRO(8) MACRO(9) MACRO(10) MACRO(11) MACRO(12) MACRO(13) MACRO(14) MACRO(15)
#endif
#define DOT_INIT(n)      ACC_TYPE4 dot_acc##n = (ACC_TYPE4)(0.0f);
#define DOT_MAD(n)       dot_acc##n = mad(qk, CONVERT_KV_ACC4(l_k[j+n][k]), dot_acc##n);
#define KROW(n)          const int k_row##n = k_start + j + n;
#define SCORE_DECL(n)    ACC_TYPE score##n = (dot_acc##n.s0 + dot_acc##n.s1 + dot_acc##n.s2 + dot_acc##n.s3) * scale;
#define CAUSAL_MASK(n)   if (k_row##n > (n_kv - n_q + my_query_row)) score##n = -INFINITY;
#define BOUND_MASK(n)    if (k_row##n >= n_kv) score##n = -INFINITY;
#define MASK_ADD(n)      if (k_row##n < n_kv) score##n += slope * (ACC_TYPE) mask_ptr[k_row##n];
#define SOFTCAP(n)       score##n = logit_softcap * tanh(score##n / logit_softcap);
#define P_DECL(n)        const ACC_TYPE p##n = exp(score##n - m_new);
#define V_ADD(n)         + p##n * CONVERT_KV_ACC4(l_v[j+n][i])
#define P_SUM(n)         + p##n
#define SCORE_MAX(n)     m_new = max(m_new, score##n);

__kernel void flash_attn_f32_f16_j8(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx  = head_batch_idx % n_head;

    const int gqa_ratio   = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx  = head_idx  % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    const float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row    = i / DK_VEC;
            const int col    = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) continue;

        // ===== J_STRIDE inner loop — NO outer-loop unroll (Adreno hates it) =====
        for (int j = 0; j < BLOCK_N; j += J_STRIDE) {
            EACH_LANE(KROW)

            // 8 parallel dot products into named-scalar accumulators
            EACH_LANE(DOT_INIT)
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                ACC_TYPE4 qk = q_priv[k];
                EACH_LANE(DOT_MAD)
            }
            EACH_LANE(SCORE_DECL)

            // ---- Masking ----
            if (is_causal) { EACH_LANE(CAUSAL_MASK) }
            EACH_LANE(BOUND_MASK)

            if (mask_base != NULL) {
                const global MASK_DATA_TYPE* mask_ptr =
                    (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                EACH_LANE(MASK_ADD)
            }

            if (logit_softcap > 0.0f) { EACH_LANE(SOFTCAP) }

            // ---- Softmax: m_new = max over m_i and all J_STRIDE scores ----
            ACC_TYPE m_new = m_i;
            EACH_LANE(SCORE_MAX)

            const ACC_TYPE scale_prev = exp(m_i - m_new);
            EACH_LANE(P_DECL)

            // ---- V product: ONE o_acc[i] update folds 8 contributions ----
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev
                         EACH_LANE(V_ADD);
            }

            const ACC_TYPE sum_p = (ACC_TYPE)(0) EACH_LANE(P_SUM);
            l_i = l_i * scale_prev + sum_p;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink  = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);
            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_acc[i] *= scale_o;
            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
        }
    }
}
