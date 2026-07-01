#include "pi0-common.h"

#include "log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opencl.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Small helpers
// ============================================================

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        LOG_ERR("Missing tensor: %s\n", name);
    }
    return t;
}

static void sincos_posemb(float t, int dim, float * out) {
    const int half = dim / 2;
    const float min_period = 4e-3f;
    const float max_period = 4.0f;
    for (int i = 0; i < half; i++) {
        float frac = (half > 1) ? (float)i / (half - 1) : 0.0f;
        float period = min_period * powf(max_period / min_period, frac);
        float x = t / period * 2.0f * (float)M_PI;
        out[i]        = sinf(x);
        out[half + i] = cosf(x);
    }
}

// Broadcast a 1-D bias [n] over a 2-D matmul output [n, m].
// Matches the explicit-repeat pattern used in the original code so we don't
// depend on ggml_add's broadcasting behavior changing across backends.
static ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * mat, ggml_tensor * bias) {
    ggml_tensor * b2d = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, mat, ggml_repeat(ctx, b2d, mat));
}

// ============================================================
// Public dequant helpers
// ============================================================

void embd_lookup_f32(ggml_tensor * t, int token_id, float * out) {
    const int64_t n_embd = t->ne[0];
    const size_t row_bytes = ggml_row_size(t->type, n_embd);

    // Row must be contiguous in memory; n_embd a multiple of the quant block size.
    if (row_bytes != (size_t) t->nb[1]) {
        GGML_ABORT("embd_lookup_f32: non-contiguous row (nb1=%zu, expected=%zu)",
                   (size_t) t->nb[1], row_bytes);
    }

    // The embedding table lives on the (OpenCL) device. A per-token ggml_backend_tensor_get of ONE
    // row is pathological on the Adreno q4_0 path (ggml-opencl get_tensor de-transposes + reads back
    // the ENTIRE ~262MB table every call ≈ 440ms). Read the whole table to a host copy ONCE, then
    // index it on CPU. The table is stable for the process, so a static per-tensor cache is safe.
    static const bool cache_disabled = [] { const char * e = getenv("PI0_EMBD_CACHE"); return e && e[0] == '0'; }();
    static std::map<const ggml_tensor *, std::vector<uint8_t>> host_cache;
    if (!cache_disabled) {
        auto it = host_cache.find(t);
        if (it == host_cache.end()) {
            std::vector<uint8_t> buf(ggml_nbytes(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
            it = host_cache.emplace(t, std::move(buf)).first;
        }
        const uint8_t * row = it->second.data() + (size_t) token_id * row_bytes;
        if (t->type == GGML_TYPE_F32) { memcpy(out, row, row_bytes); return; }
        if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * h = (const ggml_fp16_t *) row;
            for (int64_t j = 0; j < n_embd; j++) out[j] = ggml_fp16_to_fp32(h[j]);
            return;
        }
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        if (!tt || !tt->to_float) GGML_ABORT("embd_lookup_f32: type %s has no dequantizer", ggml_type_name(t->type));
        tt->to_float(row, out, n_embd);
        return;
    }

    // PI0_EMBD_CACHE=0: original per-row device read (slow, for A/B verification).
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out, (size_t) token_id * row_bytes, row_bytes);
        return;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n_embd);
        ggml_backend_tensor_get(t, tmp.data(), (size_t) token_id * row_bytes, row_bytes);
        for (int64_t j = 0; j < n_embd; j++) out[j] = ggml_fp16_to_fp32(tmp[j]);
        return;
    }
    const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
    if (!tt || !tt->to_float) GGML_ABORT("embd_lookup_f32: type %s has no dequantizer", ggml_type_name(t->type));
    std::vector<uint8_t> raw(row_bytes);
    ggml_backend_tensor_get(t, raw.data(), (size_t) token_id * row_bytes, row_bytes);
    tt->to_float(raw.data(), out, n_embd);
}

// ============================================================
// GGUF weight loading
// ============================================================

static bool load_gguf_weights(const char * path,
                              ggml_context ** out_ctx,
                              ggml_backend_buffer_t * out_buf,
                              ggml_backend_t backend) {
    struct ggml_context * meta = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gguf_ctx = gguf_init_from_file(path, params);
    if (!gguf_ctx) {
        LOG_ERR("Failed to load GGUF: %s\n", path);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    LOG_INF("Loading %d tensors from %s\n", n_tensors, path);

    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    ggml_init_params ctx_params = { ctx_size, nullptr, true };
    ggml_context * ctx_data = ggml_init(ctx_params);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * meta_tensor = ggml_get_tensor(meta, name);
        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data, meta_tensor);
        ggml_set_name(data_tensor, name);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
    if (!buf) {
        LOG_ERR("Failed to allocate buffer for %s\n", path);
        ggml_free(ctx_data);
        ggml_free(meta);
        gguf_free(gguf_ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path, "rb");
    if (!f) {
        LOG_ERR("Failed to open %s\n", path);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx_data);
        ggml_free(meta);
        gguf_free(gguf_ctx);
        return false;
    }

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
        size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, i);
        size_t nbytes = ggml_nbytes(cur);

        std::vector<uint8_t> read_buf(nbytes);
        fseek(f, offset, SEEK_SET);
        if (fread(read_buf.data(), 1, nbytes, f) != nbytes) {
            LOG_ERR("Failed to read tensor %s\n", name);
            fclose(f);
            return false;
        }
        ggml_backend_tensor_set(cur, read_buf.data(), 0, nbytes);
    }
    fclose(f);

    *out_ctx = ctx_data;
    *out_buf = buf;

    ggml_free(meta);
    gguf_free(gguf_ctx);
    return true;
}

bool load_pi0_model(pi0_model & m, const char * pali_path, const char * expert_path, ggml_backend_t backend) {
    if (!load_gguf_weights(pali_path, &m.ctx_pali, &m.buf_pali, backend)) {
        return false;
    }
    m.pali_embed      = get_tensor(m.ctx_pali, "token_embd.weight");
    m.pali_final_norm = get_tensor(m.ctx_pali, "output_norm.weight");
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[128];
        auto & l = m.pali_layers[il];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);   l.attn_norm = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);    l.ffn_norm  = get_tensor(m.ctx_pali, name);
        // Fused QKV (one [in, n_head*hd + 2*kv_dim] weight) if present; else separate q/k/v.
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", il);    l.qkv_proj  = ggml_get_tensor(m.ctx_pali, name);
        if (!l.qkv_proj) {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);  l.q_proj    = get_tensor(m.ctx_pali, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);  l.k_proj    = get_tensor(m.ctx_pali, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);  l.v_proj    = get_tensor(m.ctx_pali, name);
        }
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il); l.o_proj    = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);    l.gate_proj = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);      l.up_proj   = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);    l.down_proj = get_tensor(m.ctx_pali, name);
    }

    if (!load_gguf_weights(expert_path, &m.ctx_expert, &m.buf_expert, backend)) {
        return false;
    }
    m.expert_final_norm = get_tensor(m.ctx_expert, "output_norm.weight");
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[128];
        auto & l = m.expert_layers[il];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);   l.attn_norm = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);    l.ffn_norm  = get_tensor(m.ctx_expert, name);
        // Fused QKV (one [in, n_head*hd + 2*kv_dim] weight) if present; else separate q/k/v.
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", il);    l.qkv_proj  = ggml_get_tensor(m.ctx_expert, name);
        if (!l.qkv_proj) {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);  l.q_proj    = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);  l.k_proj    = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);  l.v_proj    = get_tensor(m.ctx_expert, name);
        }
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il); l.o_proj    = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);    l.gate_proj = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);      l.up_proj   = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);    l.down_proj = get_tensor(m.ctx_expert, name);
    }

    m.action_in_proj_w      = get_tensor(m.ctx_expert, "action_in_proj.weight");
    m.action_in_proj_b      = get_tensor(m.ctx_expert, "action_in_proj.bias");
    m.action_out_proj_w     = get_tensor(m.ctx_expert, "action_out_proj.weight");
    m.action_out_proj_b     = get_tensor(m.ctx_expert, "action_out_proj.bias");
    m.state_proj_w          = get_tensor(m.ctx_expert, "state_proj.weight");
    m.state_proj_b          = get_tensor(m.ctx_expert, "state_proj.bias");
    m.action_time_mlp_in_w  = get_tensor(m.ctx_expert, "action_time_mlp_in.weight");
    m.action_time_mlp_in_b  = get_tensor(m.ctx_expert, "action_time_mlp_in.bias");
    m.action_time_mlp_out_w = get_tensor(m.ctx_expert, "action_time_mlp_out.weight");
    m.action_time_mlp_out_b = get_tensor(m.ctx_expert, "action_time_mlp_out.bias");

    return true;
}

void free_pi0_model(pi0_model & m) {
    if (m.buf_pali)   ggml_backend_buffer_free(m.buf_pali);
    if (m.buf_expert) ggml_backend_buffer_free(m.buf_expert);
    if (m.ctx_pali)   ggml_free(m.ctx_pali);
    if (m.ctx_expert) ggml_free(m.ctx_expert);
    m.buf_pali = m.buf_expert = nullptr;
    m.ctx_pali = m.ctx_expert = nullptr;
}

// ============================================================
// Graph fragments
// ============================================================

static ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions,
                                int n_head, int head_dim) {
    ggml_tensor * reshaped = ggml_reshape_3d(ctx, x, head_dim, n_head, x->ne[1]);
    // Gemma uses NEOX-style RoPE (rotate split halves: x1,x2 = split(x); openpi
    // gemma._apply_rope), NOT the interleaved GGML_ROPE_TYPE_NORMAL.
    ggml_tensor * roped = ggml_rope_ext(ctx, reshaped, positions, nullptr,
        head_dim, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx, roped, n_head * head_dim, x->ne[1]);
}

// Gemma transformer layer with cache-resident K/V.
//
//   * New K/V (post-RoPE, F32) are written into `cache_k`/`cache_v` at row
//     range [write_offset .. write_offset + seq_len). ggml_cpy handles the
//     F32 → cache_type conversion on whichever backend you're running.
//   * Attention reads `cache_k`/`cache_v` at row range [0 .. attn_kv_len),
//     so the same function serves both the PaliGemma prefix pass
//     (write_offset=0, attn_kv_len=prefix_len ⇒ self-attention) and the
//     Action Expert (write_offset=prefix_len, attn_kv_len=prefix_len+51 ⇒
//     cross-attends to cached prefix KV plus the just-written suffix).
//
// The cpy ops are added to `graph` directly so they survive the topological
// expansion from the final output. ggml's allocator tracks the parent-view
// alias, so attention is correctly scheduled after the cpy writes.
static ggml_tensor * build_gemma_layer(ggml_context * ctx,
                                       ggml_cgraph * graph,
                                       ggml_tensor * input,
                                       const gemma_layer_weights & w,
                                       ggml_tensor * positions,
                                       ggml_tensor * attn_mask,
                                       int n_head, int n_kv_head, int head_dim,
                                       ggml_tensor * cache_k, ggml_tensor * cache_v,
                                       int write_offset, int attn_kv_len) {
    const int seq_len = input->ne[1];
    const int kv_dim  = n_kv_head * head_dim;

    ggml_tensor * x = ggml_rms_norm(ctx, input, 1e-6f);
    x = ggml_mul(ctx, x, w.attn_norm);

    ggml_tensor * q, * k, * v;
    if (w.qkv_proj) {
        // Fused QKV: one GEMM (large M=n_head*hd+2*kv_dim → more workgroups, one launch),
        // then split + cont (apply_rope/cpy need contiguous). q/k/v share the same x.
        const int q_dim = n_head * head_dim;
        ggml_tensor * qkv = ggml_mul_mat(ctx, w.qkv_proj, x);            // [q_dim+2*kv_dim, seq] f32
        q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim,  seq_len, qkv->nb[1], 0));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, seq_len, qkv->nb[1], (size_t) q_dim          * sizeof(float)));
        v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, seq_len, qkv->nb[1], (size_t)(q_dim + kv_dim) * sizeof(float)));
    } else {
        q = ggml_mul_mat(ctx, w.q_proj, x);
        k = ggml_mul_mat(ctx, w.k_proj, x);
        v = ggml_mul_mat(ctx, w.v_proj, x);
    }

    q = apply_rope(ctx, q, positions, n_head, head_dim);
    k = apply_rope(ctx, k, positions, n_kv_head, head_dim);

    // ---- Write new K/V into the cache (with potential F32 → kv_type conversion) ----
    const size_t row_bytes_k = cache_k->nb[1];
    const size_t row_bytes_v = cache_v->nb[1];

    ggml_tensor * k_dst = ggml_view_2d(ctx, cache_k, kv_dim, seq_len,
        row_bytes_k, (size_t) write_offset * row_bytes_k);
    ggml_tensor * v_dst = ggml_view_2d(ctx, cache_v, kv_dim, seq_len,
        row_bytes_v, (size_t) write_offset * row_bytes_v);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, k, k_dst));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v, v_dst));

    // ---- Attention reads cache[0..attn_kv_len) ----
    ggml_tensor * attn_k = ggml_view_2d(ctx, cache_k, kv_dim, attn_kv_len, row_bytes_k, 0);
    ggml_tensor * attn_v = ggml_view_2d(ctx, cache_v, kv_dim, attn_kv_len, row_bytes_v, 0);

    q      = ggml_reshape_3d(ctx, q,      head_dim, n_head,    seq_len);
    attn_k = ggml_reshape_3d(ctx, attn_k, head_dim, n_kv_head, attn_kv_len);
    attn_v = ggml_reshape_3d(ctx, attn_v, head_dim, n_kv_head, attn_kv_len);

    // ggml_flash_attn_ext expects k/v as [head_dim, kv_len, n_kv_head].
    attn_k = ggml_cont(ctx, ggml_permute(ctx, attn_k, 0, 2, 1, 3));
    attn_v = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));

    float scale = 1.0f / sqrtf((float)head_dim);

    // Env var PI0_NO_FLASH_ATTN=1 swaps in standard 3-step attention (mul_mat + soft_max + mul_mat).
    // Useful for comparing against flash_attn on small-context workloads where the
    // intermediate KQ matrix fits in cache and FlashAttention's o_acc spill hurts.
    static const bool use_flash_attn = []{
        const char * env = getenv("PI0_NO_FLASH_ATTN");
        if (env && env[0] == '1') { fprintf(stderr, "PI0: standard attention (no flash)\n"); return false; }
        return true;
    }();

    ggml_tensor * attn_out;
    if (use_flash_attn) {
        // Permute q from [head_dim, n_head, seq_len] (PI0's quirk) to the standard
        // [head_dim, seq_len, n_head] that flash_attn_ext expects. This is REQUIRED
        // for CORRECTNESS: without it flash_attn misreads ne[1]/ne[2] (n_q vs n_head
        // swap), producing wrong attention (PaliGemma prefix_out cos ~0.44 vs the
        // reference) as well as a ~3× GPU slowdown. Default ON; PI0_FIX_Q_LAYOUT=0
        // restores the old (broken) behavior for A/B testing.
        static const bool fix_q_layout = []{
            const char * env = getenv("PI0_FIX_Q_LAYOUT");
            if (env && env[0] == '0') { fprintf(stderr, "PI0: q layout fix DISABLED (flash_attn)\n"); return false; }
            return true;
        }();
        ggml_tensor * q_for_fa = fix_q_layout
            ? ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3))  // → [head_dim, seq_len, n_head]
            : q;
        attn_out = ggml_flash_attn_ext(ctx, q_for_fa, attn_k, attn_v, attn_mask, scale, 0.0f, 0.0f);
        if (fix_q_layout) {
            // flash_attn output with fixed q layout: [head_dim, n_head, seq_len] (standard).
            // Downstream reshape_2d expects exactly this. No further work needed.
        }
    } else {
        // Standard 3-step attention.
        // Layout:
        //   q_std:  [head_dim, seq_len,      n_head]
        //   attn_k: [head_dim, attn_kv_len,  n_kv_head]
        //   attn_v: [head_dim, attn_kv_len,  n_kv_head]
        ggml_tensor * q_std = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

        // V product. Reduction dim (n_kv) needs to be innermost in src0.
        //   v_t = transpose(attn_v) → [attn_kv_len, head_dim, n_kv_head]
        ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, attn_v));

        // PI0_ATTN_COLLAPSE=1: MQA shares ONE k/v head, so the broadcast batched
        // GEMM (r2=n_head) is identical to folding the n_head query heads into the
        // N dimension and running ONE big GEMM. Same math, but avoids the slow
        // batched/scalar kernels (native tiled kernel handles the large-N GEMM).
        // Only valid with no per-head mask (prefix is bidirectional, attn_mask==NULL)
        // and a single kv head; otherwise fall back to the batched path.
        static const bool use_collapse = []{
            const char * env = getenv("PI0_ATTN_COLLAPSE");
            if (env && env[0] == '1') { fprintf(stderr, "PI0: collapsed attention (MQA fold)\n"); return true; }
            return false;
        }();

        ggml_tensor * kqv;
        if (use_collapse && n_kv_head == 1) {
            ggml_tensor * q_coll = ggml_reshape_2d(ctx, q_std, head_dim, (int64_t) seq_len * n_head);
            ggml_tensor * kq = ggml_mul_mat(ctx, attn_k, q_coll);    // [n_kv, n_q*n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            // Diffusion suffix has a mask [n_kv, seq_len] shared across heads. The collapsed
            // kq is [n_kv, seq_len*n_head] with column j = (query j%seq_len, head j/seq_len),
            // so the mask must be tiled n_head times along N. ggml_repeat gives exactly
            // dest[:,j]=src[:,j%seq_len] — head-independent, numerically exact. (prefix: NULL.)
            ggml_tensor * mask_coll = attn_mask ? ggml_repeat(ctx, attn_mask, kq) : nullptr;
            kq = ggml_soft_max_ext(ctx, kq, mask_coll, scale, 0.0f);
            ggml_tensor * kqv_coll = ggml_mul_mat(ctx, v_t, kq);     // [head_dim, n_q*n_head]
            kqv = ggml_reshape_3d(ctx, kqv_coll, head_dim, seq_len, n_head);
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx, attn_k, q_std);     // [n_kv, n_q, n_head]
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            // NO causal mask — PaliGemma's prefix attention is BIDIRECTIONAL (image/text/
            // state/action tokens see each other fully). attn_mask is NULL for the prefix.
            kq = ggml_soft_max_ext(ctx, kq, attn_mask, scale, 0.0f);
            kqv = ggml_mul_mat(ctx, v_t, kq);                        // [head_dim, seq_len, n_head]
        }

        attn_out = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    }
    attn_out = ggml_reshape_2d(ctx, attn_out, n_head * head_dim, seq_len);

    ggml_tensor * attn_proj = ggml_mul_mat(ctx, w.o_proj, attn_out);
    input = ggml_add(ctx, input, attn_proj);

    x = ggml_rms_norm(ctx, input, 1e-6f);
    x = ggml_mul(ctx, x, w.ffn_norm);

    ggml_tensor * gate = ggml_mul_mat(ctx, w.gate_proj, x);
    ggml_tensor * up   = ggml_mul_mat(ctx, w.up_proj,   x);
    // Gemma FFN is GeGLU (gelu), NOT SwiGLU (silu). openpi uses nn.gelu /
    // HF "gelu_pytorch_tanh"; ggml_gelu is the matching tanh approximation.
    ggml_tensor * ffn  = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
    ffn = ggml_mul_mat(ctx, w.down_proj, ffn);

    input = ggml_add(ctx, input, ffn);
    return input;
}

// state token + action_time MLP, producing the suffix input [EXPERT_N_EMBD, 51].
// All matmuls go through ggml, so quantized expert projection weights work.
static ggml_tensor * build_suffix_tokens(ggml_context * ctx,
                                         pi0_model & m,
                                         ggml_tensor * state_in,       // [PI0_ACTION_DIM, 1]
                                         ggml_tensor * noisy_in,       // [PI0_ACTION_DIM, 50]
                                         ggml_tensor * time_in) {      // [EXPERT_N_EMBD,  50]
    // state token: [EXPERT_N_EMBD, 1]
    ggml_tensor * state_tok = ggml_mul_mat(ctx, m.state_proj_w, state_in);
    state_tok = add_bias(ctx, state_tok, m.state_proj_b);

    // action tokens (pre-MLP): [EXPERT_N_EMBD, 50]
    ggml_tensor * action_tok = ggml_mul_mat(ctx, m.action_in_proj_w, noisy_in);
    action_tok = add_bias(ctx, action_tok, m.action_in_proj_b);

    // concat with time embedding along feature dim 0 -> [2*EXPERT_N_EMBD, 50]
    ggml_tensor * concat = ggml_concat(ctx, action_tok, time_in, 0);

    // MLP: Linear -> SiLU -> Linear
    ggml_tensor * h = ggml_mul_mat(ctx, m.action_time_mlp_in_w, concat);
    h = add_bias(ctx, h, m.action_time_mlp_in_b);
    h = ggml_silu(ctx, h);

    ggml_tensor * action_emb = ggml_mul_mat(ctx, m.action_time_mlp_out_w, h);
    action_emb = add_bias(ctx, action_emb, m.action_time_mlp_out_b);

    // prepend the state token along the sequence dim -> [EXPERT_N_EMBD, 51]
    return ggml_concat(ctx, state_tok, action_emb, 1);
}

// ============================================================
// KV cache storage
// ============================================================

static void destroy_kv_storage(pi0_session & s) {
    if (s.kv_buf) { ggml_backend_buffer_free(s.kv_buf); s.kv_buf = nullptr; }
    if (s.kv_ctx) { ggml_free(s.kv_ctx);                s.kv_ctx = nullptr; }
    for (int il = 0; il < PI0_N_LAYER; il++) {
        s.cache_k[il] = nullptr;
        s.cache_v[il] = nullptr;
    }
    s.kv_alloc_prefix_len = 0;
    s.kv_alloc_type       = GGML_TYPE_F32;
}

// Allocate per-layer K/V cache of shape [kv_dim, prefix_len + suffix_len] in s.kv_type.
// No-op if existing storage already matches.
static bool ensure_kv_storage(pi0_session & s, ggml_backend_t backend, int prefix_len) {
    const int suffix_len = PI0_ACTION_HORIZON + 1;
    const int total_kv   = prefix_len + suffix_len;

    if (s.kv_buf
        && s.kv_alloc_prefix_len == prefix_len
        && s.kv_alloc_type       == s.kv_type) {
        return true;
    }

    destroy_kv_storage(s);

    const int kv_dim = PALI_N_KV_HEAD * PALI_HEAD_DIM; // = EXPERT_N_KV_HEAD * EXPERT_HEAD_DIM = 256

    size_t ctx_size = ggml_tensor_overhead() * (2 * PI0_N_LAYER + 8);
    ggml_init_params params = { ctx_size, nullptr, true };
    s.kv_ctx = ggml_init(params);

    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[64];
        s.cache_k[il] = ggml_new_tensor_2d(s.kv_ctx, s.kv_type, kv_dim, total_kv);
        snprintf(name, sizeof(name), "cache_k_%d", il);
        ggml_set_name(s.cache_k[il], name);

        s.cache_v[il] = ggml_new_tensor_2d(s.kv_ctx, s.kv_type, kv_dim, total_kv);
        snprintf(name, sizeof(name), "cache_v_%d", il);
        ggml_set_name(s.cache_v[il], name);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    s.kv_buf = ggml_backend_alloc_ctx_tensors_from_buft(s.kv_ctx, buft);
    if (!s.kv_buf) {
        LOG_ERR("Failed to allocate KV cache buffer (type=%s, prefix_len=%d)\n",
                ggml_type_name(s.kv_type), prefix_len);
        ggml_free(s.kv_ctx);
        s.kv_ctx = nullptr;
        return false;
    }

    s.kv_alloc_prefix_len = prefix_len;
    s.kv_alloc_type       = s.kv_type;
    LOG_INF("KV cache allocated: %d layers × [%d, %d] %s (%.2f MB)\n",
            PI0_N_LAYER, kv_dim, total_kv, ggml_type_name(s.kv_type),
            2.0 * PI0_N_LAYER * ggml_nbytes(s.cache_k[0]) / (1024.0 * 1024.0));
    return true;
}

// ============================================================
// Forward passes
// ============================================================

bool run_paligemma_prefix(pi0_model & m, pi0_session & s, ggml_backend_t backend,
                          const float * embeddings, int prefix_len) {
    if (!ensure_kv_storage(s, backend, prefix_len)) {
        return false;
    }
    s.prefix_len = prefix_len;

    size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PALI_N_EMBD, prefix_len);
    ggml_set_name(input, "prefix_input");
    ggml_set_input(input);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefix_len);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // NOTE: the Gemma sqrt(hidden) embedding scale must be applied to *text*
    // token embeddings only, NOT to SigLIP image features (PaliGemma convention).
    // It is therefore applied per-token during prefix assembly by the caller,
    // not here over the whole (image+text) prefix.
    ggml_tensor * cur = input;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);

    // Build each layer; K/V are written into cache_k/v[il][0..prefix_len)
    // and self-attention reads from those same rows.
    for (int il = 0; il < PI0_N_LAYER; il++) {
        cur = build_gemma_layer(ctx, graph, cur, m.pali_layers[il], positions, nullptr,
            PALI_N_HEAD, PALI_N_KV_HEAD, PALI_HEAD_DIM,
            s.cache_k[il], s.cache_v[il],
            /*write_offset=*/0, /*attn_kv_len=*/prefix_len);
    }

    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, m.pali_final_norm);
    ggml_set_name(cur, "prefix_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(graph, cur);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
        LOG_ERR("Failed to allocate prefix graph\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(input, embeddings, 0, sizeof(float) * PALI_N_EMBD * prefix_len);
    std::vector<int32_t> pos_data(prefix_len);
    for (int i = 0; i < prefix_len; i++) pos_data[i] = i;
    ggml_backend_tensor_set(positions, pos_data.data(), 0, sizeof(int32_t) * prefix_len);

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("Prefix compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // [debug/compare] dump the PaliGemma final hidden state (after final norm)
    // to isolate the PaliGemma transformer stage from the expert stage.
    if (const char * po = getenv("PI0_DUMP_PREFIX_OUT")) {
        FILE * pf = fopen(po, "wb");
        if (pf) {
            std::vector<float> buf((size_t) PALI_N_EMBD * prefix_len);
            ggml_backend_tensor_get(cur, buf.data(), 0, sizeof(float) * buf.size());
            fwrite(buf.data(), sizeof(float), buf.size(), pf);
            fclose(pf);
            LOG_INF("Dumped prefix_out to %s (%d tok x %d)\n", po, prefix_len, PALI_N_EMBD);
        }
    }

    // KV is now live in s.cache_k/v on the backend — no host roundtrip.
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// ---- Persistent expert graph (built once per (prefix_len, kv_type), reused across steps) ----

static void destroy_expert_graph(pi0_session & s) {
    if (s.graph_alloc) { ggml_gallocr_free(s.graph_alloc); s.graph_alloc = nullptr; }
    if (s.graph_ctx)   { ggml_free(s.graph_ctx);          s.graph_ctx   = nullptr; }
    s.graph = nullptr;
    s.graph_prefix_len = 0;
    s.graph_kv_type    = GGML_TYPE_F32;
    s.in_state = s.in_noisy = s.in_time = s.in_positions = nullptr;
    s.in_attn_mask = s.out_v_t = nullptr;
}

static bool build_expert_graph(pi0_model & m, pi0_session & s, ggml_backend_t backend, int prefix_len) {
    destroy_expert_graph(s);

    const int suffix_len = PI0_ACTION_HORIZON + 1;            // 51 = state + 50 actions
    const int total_kv   = prefix_len + suffix_len;

    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    s.graph_ctx = ggml_init(params);
    ggml_context * ctx = s.graph_ctx;

    // ---- Inputs ----
    s.in_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PI0_ACTION_DIM, 1);
    ggml_set_name(s.in_state, "state_in");
    ggml_set_input(s.in_state);

    // Whole batch of 50 noisy action tokens — one tensor, processed as a single batch
    // by ggml_mul_mat (never a per-token loop).
    s.in_noisy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PI0_ACTION_DIM, PI0_ACTION_HORIZON);
    ggml_set_name(s.in_noisy, "noisy_in");
    ggml_set_input(s.in_noisy);

    // Same scalar timestep, broadcast to all 50 positions on host so the time MLP
    // runs as one batched matmul.
    s.in_time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EXPERT_N_EMBD, PI0_ACTION_HORIZON);
    ggml_set_name(s.in_time, "time_in");
    ggml_set_input(s.in_time);

    s.in_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, suffix_len);
    ggml_set_name(s.in_positions, "suffix_positions");
    ggml_set_input(s.in_positions);

    s.in_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_kv, suffix_len);
    ggml_set_name(s.in_attn_mask, "attn_mask");
    ggml_set_input(s.in_attn_mask);

    s.graph = ggml_new_graph_custom(ctx, 16384, false);

    // ---- Suffix tokens (state_proj, action_in_proj, time MLP) ----
    ggml_tensor * cur = build_suffix_tokens(ctx, m, s.in_state, s.in_noisy, s.in_time);

    // ---- 18 expert layers; new K/V written to cache[prefix_len:total_kv]; attention reads cache[0:total_kv] ----
    for (int il = 0; il < PI0_N_LAYER; il++) {
        cur = build_gemma_layer(ctx, s.graph, cur, m.expert_layers[il],
            s.in_positions, s.in_attn_mask,
            EXPERT_N_HEAD, EXPERT_N_KV_HEAD, EXPERT_HEAD_DIM,
            s.cache_k[il], s.cache_v[il],
            /*write_offset=*/prefix_len, /*attn_kv_len=*/total_kv);
    }

    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, m.expert_final_norm);

    // Drop the leading state token; keep all 50 action positions for action_out_proj.
    ggml_tensor * action_hidden = ggml_view_2d(ctx, cur, EXPERT_N_EMBD, PI0_ACTION_HORIZON,
        cur->nb[1], cur->nb[1] * 1);

    s.out_v_t = ggml_mul_mat(ctx, m.action_out_proj_w, action_hidden);
    s.out_v_t = add_bias(ctx, s.out_v_t, m.action_out_proj_b);
    ggml_set_name(s.out_v_t, "v_t");
    ggml_set_output(s.out_v_t);

    ggml_build_forward_expand(s.graph, s.out_v_t);

    s.graph_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(s.graph_alloc, s.graph) || !ggml_gallocr_alloc_graph(s.graph_alloc, s.graph)) {
        LOG_ERR("Failed to allocate persistent expert graph\n");
        destroy_expert_graph(s);
        return false;
    }

    // ---- Upload constant-across-steps inputs once ----

    // positions: prefix_len .. prefix_len + suffix_len - 1
    std::vector<int32_t> pos_data(suffix_len);
    for (int i = 0; i < suffix_len; i++) pos_data[i] = prefix_len + i;
    ggml_backend_tensor_set(s.in_positions, pos_data.data(), 0, sizeof(int32_t) * suffix_len);

    // Attention mask. Prefix always visible; state attends to itself only;
    // actions attend to state + all actions.
    std::vector<float> mask_data(total_kv * suffix_len, 0.0f);
    const float NEG_INF = -INFINITY;
    for (int srow = 0; srow < suffix_len; srow++) {
        for (int kv = 0; kv < total_kv; kv++) {
            if (kv < prefix_len) {
                mask_data[srow * total_kv + kv] = 0.0f;
            } else {
                int suffix_kv = kv - prefix_len;
                if (srow == 0) {
                    mask_data[srow * total_kv + kv] = (suffix_kv == 0) ? 0.0f : NEG_INF;
                } else {
                    mask_data[srow * total_kv + kv] = 0.0f;
                }
            }
        }
    }
    ggml_backend_tensor_set(s.in_attn_mask, mask_data.data(), 0, sizeof(float) * total_kv * suffix_len);

    s.graph_prefix_len = prefix_len;
    s.graph_kv_type    = s.kv_type;
    return true;
}

bool run_expert_step(pi0_model & m, pi0_session & s, ggml_backend_t backend,
                     const float * state, const float * noisy_actions, float timestep,
                     float * v_t_out) {
    // (Re)build the graph if not present or if (prefix_len, kv_type) changed.
    if (!s.graph
        || s.graph_prefix_len != s.prefix_len
        || s.graph_kv_type    != s.kv_type) {
        if (!build_expert_graph(m, s, backend, s.prefix_len)) return false;
    }

    // Per-step inputs: state, noisy_actions, time. Prefix KV is already in
    // s.cache_k/v on the backend — no host roundtrip needed.
    ggml_backend_tensor_set(s.in_state, state, 0, sizeof(float) * PI0_ACTION_DIM);
    ggml_backend_tensor_set(s.in_noisy, noisy_actions, 0,
        sizeof(float) * PI0_ACTION_DIM * PI0_ACTION_HORIZON);

    // sincos(t) once, broadcast to all 50 positions, then upload as one tensor.
    std::vector<float> time_emb(EXPERT_N_EMBD);
    sincos_posemb(timestep, EXPERT_N_EMBD, time_emb.data());
    std::vector<float> time_emb_broadcast(EXPERT_N_EMBD * PI0_ACTION_HORIZON);
    for (int t = 0; t < PI0_ACTION_HORIZON; t++) {
        memcpy(&time_emb_broadcast[t * EXPERT_N_EMBD], time_emb.data(),
            sizeof(float) * EXPERT_N_EMBD);
    }
    ggml_backend_tensor_set(s.in_time, time_emb_broadcast.data(), 0,
        sizeof(float) * EXPERT_N_EMBD * PI0_ACTION_HORIZON);

    if (ggml_backend_graph_compute(backend, s.graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("Expert compute failed\n");
        return false;
    }

    ggml_backend_tensor_get(s.out_v_t, v_t_out, 0,
        sizeof(float) * PI0_ACTION_DIM * PI0_ACTION_HORIZON);
    return true;
}

void pi0_session_free(pi0_session & s) {
    destroy_expert_graph(s);
    destroy_kv_storage(s);
    s.prefix_len = 0;
}

// ============================================================
// Utility
// ============================================================

std::string resolve_model_dir(const std::string & path) {
    std::string dir = path;
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    if (dir.find(".gguf") != std::string::npos) {
        auto pos = dir.rfind('/');
        dir = (pos != std::string::npos) ? dir.substr(0, pos) : ".";
    }
    return dir;
}

// ============================================================
// Backend selection
// ============================================================

static const char * dev_type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "?";
    }
}

void pi0_dump_backends() {
    ggml_backend_load_all();
    const size_t n = ggml_backend_dev_count();
    LOG_INF("Available backend devices (%zu):\n", n);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        LOG_INF("  [%zu] %-12s %-5s %s\n",
                i,
                ggml_backend_dev_name(dev),
                dev_type_str(ggml_backend_dev_type(dev)),
                ggml_backend_dev_description(dev));
    }
}

ggml_backend_t pi0_init_backend(const std::string & name) {
    ggml_backend_load_all();

    if (name.empty() || name == "cpu" || name == "CPU") {
        LOG_INF("Backend: CPU\n");
        return ggml_backend_cpu_init();
    }

    if (name == "auto" || name == "gpu" || name == "GPU") {
        ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (b) {
            LOG_INF("Backend: %s (auto-selected GPU)\n", ggml_backend_name(b));
            return b;
        }
        LOG_WRN("No GPU backend available; falling back to CPU\n");
        return ggml_backend_cpu_init();
    }

    if (ggml_backend_t b = ggml_backend_init_by_name(name.c_str(), nullptr)) {
        LOG_INF("Backend: %s\n", ggml_backend_name(b));
        return b;
    }

    // Fallback: case-insensitive substring match against device names so e.g.
    // `--backend OpenCL` matches the Adreno device that registers as `GPUOpenCL`.
    auto lower = [](std::string s) {
        for (auto & c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
        return s;
    };
    const std::string needle = lower(name);
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (lower(ggml_backend_dev_name(dev)).find(needle) != std::string::npos) {
            if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr)) {
                LOG_INF("Backend: %s (matched '%s' by substring)\n",
                        ggml_backend_dev_name(dev), name.c_str());
                return b;
            }
        }
    }

    LOG_ERR("Backend '%s' not found.\n", name.c_str());
    pi0_dump_backends();
    return nullptr;
}

// ============================================================
// Tensor-type reporting
// ============================================================

static void dump_ctx_types(ggml_context * ctx, const char * label) {
    if (!ctx) return;
    std::map<int, int>    counts; // ggml_type -> count
    std::map<int, size_t> bytes;
    size_t total_bytes = 0;
    int    total_count = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        counts[(int) t->type]++;
        bytes[(int) t->type] += ggml_nbytes(t);
        total_bytes += ggml_nbytes(t);
        total_count++;
    }
    LOG_INF("[%s] %d tensors, %.2f MB total\n",
            label, total_count, total_bytes / (1024.0 * 1024.0));
    for (auto & kv : counts) {
        ggml_type ty = (ggml_type) kv.first;
        LOG_INF("    %-10s %4d tensors  %8.2f MB\n",
                ggml_type_name(ty), kv.second, bytes[kv.first] / (1024.0 * 1024.0));
    }
}

void pi0_dump_tensor_types(const pi0_model & m) {
    dump_ctx_types(m.ctx_pali,   "PaliGemma 2B");
    dump_ctx_types(m.ctx_expert, "Action Expert");
}

// ============================================================
// Shared CLI flag stripping
// ============================================================

pi0_cli pi0_strip_cli_args(int & argc, char ** argv) {
    pi0_cli c;
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            c.backend = argv[++i];
        } else if (strcmp(argv[i], "--kv-type") == 0 && i + 1 < argc) {
            c.kv_type = argv[++i];
        } else if (strcmp(argv[i], "--rt-bin") == 0 && i + 1 < argc) {
            c.rt_bin = argv[++i];
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return c;
}

ggml_type pi0_ggml_type_from_string(const std::string & s) {
    auto eq = [&s](const char * lit) {
        if (s.size() != strlen(lit)) return false;
        for (size_t i = 0; i < s.size(); i++) {
            char a = s[i];
            char b = lit[i];
            if (a >= 'A' && a <= 'Z') a = char(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = char(b - 'A' + 'a');
            if (a != b) return false;
        }
        return true;
    };
    if (eq("f32"))  return GGML_TYPE_F32;
    if (eq("f16"))  return GGML_TYPE_F16;
    if (eq("bf16")) return GGML_TYPE_BF16;
    if (eq("q8_0")) return GGML_TYPE_Q8_0;
    if (eq("q4_0")) return GGML_TYPE_Q4_0;
    if (eq("q4_1")) return GGML_TYPE_Q4_1;
    if (eq("q5_0")) return GGML_TYPE_Q5_0;
    if (eq("q5_1")) return GGML_TYPE_Q5_1;
    return GGML_TYPE_COUNT;
}

// ============================================================
// Phase 2D — Q8_0 Row-Tile overlay loader
// ============================================================
//
// File format (see dp8-spike/quantize_rt.py): little-endian
//   magic    [8] = "PI0RT\0\0\0"
//   version  u32 = 1
//   n_tens   u32
//   per tensor:
//     name_len u32, name bytes
//     rt_type  u32  (0 = Q8_0_RT)
//     M        i64
//     K        i64
//     TM       u32  (must be 4)
//     QK       u32  (must be 32)
//     payload  u64 bytes
//     payload  ... (M/TM)*(K/QK)*152 bytes
//
// Each 152-byte block covers 4 rows × 32 cols and contains
//   128 B uint8 (4 rows row-major) | 8 B fp16 scales[4] | 16 B i32 sums[4]
//
// We unpack into the three row-major host arrays the dp8 GEMM kernel expects:
//   Wq[M*K]   uint8     row-major
//   Wd[M*K32] fp16 bits (uint16_t for transport)
//   Ws[M*K32] int32

static constexpr int RT_TM_BLOCK = 4;
static constexpr int RT_QK_BLOCK = 32;
static constexpr int RT_BLOCK_BYTES = 128 + 8 + 16;   // 152

bool pi0_attach_rt_overlay(pi0_model & m, ggml_backend_t backend, const std::string & path) {
#ifndef GGML_USE_OPENCL
    // The RT overlay only targets the OpenCL backend (.rt.bin Adreno weights).
    // Guard the OpenCL symbol references so CPU-only / non-OpenCL builds link.
    // GGML_USE_OPENCL is propagated PUBLIC from ggml when the backend is built.
    (void) m; (void) backend; (void) path;
    LOG_INF("pi0_attach_rt_overlay: built without OpenCL backend; skipping\n");
    return false;
#else
    if (!ggml_backend_is_opencl(backend)) {
        LOG_INF("pi0_attach_rt_overlay: backend is not OpenCL; skipping\n");
        return false;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) { LOG_ERR("pi0_attach_rt_overlay: cannot open %s\n", path.c_str()); return false; }

    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "PI0RT\0\0\0", 8) != 0) {
        LOG_ERR("pi0_attach_rt_overlay: bad magic in %s\n", path.c_str());
        return false;
    }
    uint32_t version = 0, n_tens = 0;
    f.read((char*)&version, 4);
    f.read((char*)&n_tens, 4);
    if (!f || version != 1) {
        LOG_ERR("pi0_attach_rt_overlay: unsupported version %u\n", version);
        return false;
    }

    int n_attached = 0;
    int n_skipped  = 0;
    for (uint32_t i = 0; i < n_tens; ++i) {
        uint32_t name_len = 0;
        f.read((char*)&name_len, 4);
        if (!f) { LOG_ERR("pi0_attach_rt_overlay: truncated header\n"); return false; }
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint32_t rt_type = 0;
        int64_t  M = 0, K = 0;
        uint32_t TM_blk = 0, QK_blk = 0;
        uint64_t pb = 0;
        f.read((char*)&rt_type, 4);
        f.read((char*)&M,       8);
        f.read((char*)&K,       8);
        f.read((char*)&TM_blk,  4);
        f.read((char*)&QK_blk,  4);
        f.read((char*)&pb,      8);
        if (!f) { LOG_ERR("pi0_attach_rt_overlay: truncated entry %s\n", name.c_str()); return false; }

        if (rt_type != 0 || TM_blk != RT_TM_BLOCK || QK_blk != RT_QK_BLOCK) {
            LOG_WRN("pi0_attach_rt_overlay: skip %s (rt_type=%u TM=%u QK=%u)\n",
                    name.c_str(), rt_type, TM_blk, QK_blk);
            f.seekg((std::streamoff)pb, std::ios::cur);
            n_skipped++;
            continue;
        }

        // Look up the ggml tensor. PaliGemma backbone and Action Expert use
        // identical naming (blk.X.attn_q.weight) with different shapes, so
        // search both contexts and pick the one whose shape matches the .rt.bin
        // entry — ensures one --rt-bin can target either model.
        ggml_tensor * t = nullptr;
        {
            ggml_tensor * t_e = m.ctx_expert ? ggml_get_tensor(m.ctx_expert, name.c_str()) : nullptr;
            ggml_tensor * t_p = m.ctx_pali   ? ggml_get_tensor(m.ctx_pali,   name.c_str()) : nullptr;
            if (t_e && t_e->ne[0] == K && t_e->ne[1] == M) t = t_e;
            else if (t_p && t_p->ne[0] == K && t_p->ne[1] == M) t = t_p;
        }
        if (!t) {
            // Silently skip — overlay may include tensors not used by this build,
            // or no ggml tensor has the matching shape.
            f.seekg((std::streamoff)pb, std::ios::cur);
            n_skipped++;
            continue;
        }

        // Read payload and unpack into 3 row-major host arrays.
        std::vector<uint8_t> payload(pb);
        f.read((char*)payload.data(), pb);
        if (!f) { LOG_ERR("pi0_attach_rt_overlay: truncated payload for %s\n", name.c_str()); return false; }

        const int K32 = (int)(K / RT_QK_BLOCK);
        const int n_blk_m = (int)(M / RT_TM_BLOCK);
        const size_t expected_bytes = (size_t) n_blk_m * (size_t) K32 * RT_BLOCK_BYTES;
        if (pb != expected_bytes) {
            LOG_ERR("pi0_attach_rt_overlay: payload size mismatch on %s: got %llu expected %zu\n",
                    name.c_str(), (unsigned long long) pb, expected_bytes);
            return false;
        }

        std::vector<uint8_t>  Wq((size_t) M * (size_t) K);
        std::vector<uint16_t> Wd((size_t) M * (size_t) K32);
        std::vector<int32_t>  Ws((size_t) M * (size_t) K32);

        for (int mt = 0; mt < n_blk_m; ++mt) {
            for (int kb = 0; kb < K32; ++kb) {
                const uint8_t * blk = payload.data() + ((size_t) mt * K32 + kb) * RT_BLOCK_BYTES;
                for (int r = 0; r < RT_TM_BLOCK; ++r) {
                    const int row = mt * RT_TM_BLOCK + r;
                    // data: 32 uint8 per row → row-major position
                    std::memcpy(Wq.data() + (size_t) row * K + kb * RT_QK_BLOCK,
                                blk + r * RT_QK_BLOCK, RT_QK_BLOCK);
                    uint16_t s; std::memcpy(&s, blk + 128 + r * 2, 2);
                    int32_t  rs; std::memcpy(&rs, blk + 136 + r * 4, 4);
                    Wd[(size_t) row * K32 + kb] = s;
                    Ws[(size_t) row * K32 + kb] = rs;
                }
            }
        }

        const bool ok = ggml_backend_opencl_attach_rt_weights(
            backend, t,
            Wq.data(), Wd.data(), Ws.data(),
            (int) M, (int) K);
        if (!ok) {
            LOG_WRN("pi0_attach_rt_overlay: attach failed for %s\n", name.c_str());
            n_skipped++;
            continue;
        }
        n_attached++;
    }

    LOG_INF("pi0_attach_rt_overlay: attached %d/%u tensors from %s (skipped %d)\n",
            n_attached, n_tens, path.c_str(), n_skipped);
    return n_attached > 0;
#endif // GGML_USE_OPENCL
}
