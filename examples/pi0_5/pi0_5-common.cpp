#include "pi0_5-common.h"

#include "log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    if (!t) LOG_ERR("Missing tensor: %s\n", name);
    return t;
}

// sin-cos timestep embedding (openpi posemb_sincos, min_period 4e-3, max_period 4.0).
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
static ggml_tensor * add_bias(ggml_context * ctx, ggml_tensor * mat, ggml_tensor * bias) {
    ggml_tensor * b2d = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, mat, ggml_repeat(ctx, b2d, mat));
}

// Cast a matmul activation (src1) to the configured activation dtype. Default
// f16 — this is the lever that lets the OpenCL f16 GEMM kernels kick in. Pass
// --act-type f32 for exact CPU parity against openpi.
static ggml_tensor * act_cast(ggml_context * ctx, ggml_tensor * x, ggml_type act_type) {
    if (act_type == GGML_TYPE_F32 || x->type == act_type) return x;
    return ggml_cast(ctx, x, act_type);
}
// Activation-cast then matmul: out = W @ x  (W = weights/src0, x = activation/src1).
static ggml_tensor * mm_act(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_type act) {
    return ggml_mul_mat(ctx, w, act_cast(ctx, x, act));
}

// ============================================================
// Public dequant helper (embedding row lookup; identical to pi0)
// ============================================================

void embd_lookup_f32(ggml_tensor * t, int token_id, float * out) {
    const int64_t n_embd = t->ne[0];
    const size_t row_bytes = ggml_row_size(t->type, n_embd);
    if (row_bytes != (size_t) t->nb[1]) {
        GGML_ABORT("embd_lookup_f32: non-contiguous row (nb1=%zu, expected=%zu)",
                   (size_t) t->nb[1], row_bytes);
    }
    // Cache the whole table host-side once (per-row device reads are pathological
    // on the Adreno q4_0 path — see pi0 notes).
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
    if (!gguf_ctx) { LOG_ERR("Failed to load GGUF: %s\n", path); return false; }

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
        ggml_free(ctx_data); ggml_free(meta); gguf_free(gguf_ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path, "rb");
    if (!f) {
        LOG_ERR("Failed to open %s\n", path);
        ggml_backend_buffer_free(buf); ggml_free(ctx_data); ggml_free(meta); gguf_free(gguf_ctx);
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
            fclose(f); return false;
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

bool load_pi05_model(pi05_model & m, const char * pali_path, const char * expert_path, ggml_backend_t backend) {
    // ---- PaliGemma 2B (prefix): plain Gemma, identical to pi0 ----
    if (!load_gguf_weights(pali_path, &m.ctx_pali, &m.buf_pali, backend)) return false;
    m.pali_embed      = get_tensor(m.ctx_pali, "token_embd.weight");
    m.pali_final_norm = get_tensor(m.ctx_pali, "output_norm.weight");
    for (int il = 0; il < PI05_N_LAYER; il++) {
        char name[128];
        auto & l = m.pali_layers[il];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);   l.attn_norm = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);    l.ffn_norm  = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", il);    l.qkv_proj  = ggml_get_tensor(m.ctx_pali, name);
        if (!l.qkv_proj) {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);  l.q_proj = get_tensor(m.ctx_pali, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);  l.k_proj = get_tensor(m.ctx_pali, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);  l.v_proj = get_tensor(m.ctx_pali, name);
        }
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il); l.o_proj    = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);    l.gate_proj = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);      l.up_proj   = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);    l.down_proj = get_tensor(m.ctx_pali, name);
    }

    // ---- Action Expert 300M (suffix): adaRMS ----
    if (!load_gguf_weights(expert_path, &m.ctx_expert, &m.buf_expert, backend)) return false;
    m.expert_final_norm_dense_w = get_tensor(m.ctx_expert, "output_norm.dense.weight");
    m.expert_final_norm_dense_b = get_tensor(m.ctx_expert, "output_norm.dense.bias");
    if (!m.expert_final_norm_dense_w) {
        LOG_ERR("expert is missing output_norm.dense.* — not a pi0.5 (adaRMS) expert. "
                "Did you convert with --pi05?\n");
        return false;
    }
    for (int il = 0; il < PI05_N_LAYER; il++) {
        char name[128];
        auto & l = m.expert_layers[il];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.dense.weight", il); l.attn_norm_dense_w = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.attn_norm.dense.bias", il);   l.attn_norm_dense_b = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.dense.weight", il);  l.ffn_norm_dense_w  = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.dense.bias", il);    l.ffn_norm_dense_b  = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", il);        l.qkv_proj = ggml_get_tensor(m.ctx_expert, name);
        if (!l.qkv_proj) {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);      l.q_proj = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);      l.k_proj = get_tensor(m.ctx_expert, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);      l.v_proj = get_tensor(m.ctx_expert, name);
        }
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il);     l.o_proj    = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);        l.gate_proj = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);          l.up_proj   = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);        l.down_proj = get_tensor(m.ctx_expert, name);
    }

    m.action_in_proj_w  = get_tensor(m.ctx_expert, "action_in_proj.weight");
    m.action_in_proj_b  = get_tensor(m.ctx_expert, "action_in_proj.bias");
    m.action_out_proj_w = get_tensor(m.ctx_expert, "action_out_proj.weight");
    m.action_out_proj_b = get_tensor(m.ctx_expert, "action_out_proj.bias");
    m.time_mlp_in_w     = get_tensor(m.ctx_expert, "time_mlp_in.weight");
    m.time_mlp_in_b     = get_tensor(m.ctx_expert, "time_mlp_in.bias");
    m.time_mlp_out_w    = get_tensor(m.ctx_expert, "time_mlp_out.weight");
    m.time_mlp_out_b    = get_tensor(m.ctx_expert, "time_mlp_out.bias");
    return true;
}

void free_pi05_model(pi05_model & m) {
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
    // Gemma uses NEOX-style RoPE (split-half rotate), not interleaved.
    ggml_tensor * roped = ggml_rope_ext(ctx, reshaped, positions, nullptr,
        head_dim, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx, roped, n_head * head_dim, x->ne[1]);
}

// adaRMS modulation: cond [cond_dim,1] -> (scale,shift,gate) each [width,1] (contiguous).
struct adarms_mod { ggml_tensor * scale; ggml_tensor * shift; ggml_tensor * gate; };
static adarms_mod adarms_modulation(ggml_context * ctx, ggml_tensor * dense_w, ggml_tensor * dense_b,
                                    ggml_tensor * cond, int width, ggml_type act) {
    ggml_tensor * mod = mm_act(ctx, dense_w, cond, act);                 // [3*width, 1]
    mod = ggml_add(ctx, mod, ggml_reshape_2d(ctx, dense_b, ADARMS_MOD_DIM, 1));
    adarms_mod r;
    r.scale = ggml_cont(ctx, ggml_view_2d(ctx, mod, width, 1, mod->nb[1], (size_t) 0 * width * sizeof(float)));
    r.shift = ggml_cont(ctx, ggml_view_2d(ctx, mod, width, 1, mod->nb[1], (size_t) 1 * width * sizeof(float)));
    r.gate  = ggml_cont(ctx, ggml_view_2d(ctx, mod, width, 1, mod->nb[1], (size_t) 2 * width * sizeof(float)));
    return r;
}

// adaRMS norm: normed = rms(x)*(1+scale)+shift, computed as normed*scale + normed + shift.
// scale/shift are [width,1] broadcast over the seq dim.
static ggml_tensor * adarms_apply(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * scale, ggml_tensor * shift) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, 1e-6f);
    ggml_tensor * out = ggml_add(ctx, ggml_mul(ctx, n, scale), n);
    out = ggml_add(ctx, out, shift);
    return out;
}

// Gemma transformer layer with cache-resident K/V.
//   adarms_cond == null  -> plain RMSNorm (PaliGemma prefix). attn_mask is the
//                           bidirectional mask (NULL = full attention).
//   adarms_cond != null  -> pi0.5 action expert: adaRMS norms + gated residuals.
static ggml_tensor * build_gemma_layer(ggml_context * ctx, ggml_cgraph * graph,
                                       ggml_tensor * input,
                                       const gemma_layer_weights & w,
                                       ggml_tensor * positions, ggml_tensor * attn_mask,
                                       int n_head, int n_kv_head, int head_dim,
                                       ggml_tensor * cache_k, ggml_tensor * cache_v,
                                       int write_offset, int attn_kv_len,
                                       ggml_type act, ggml_tensor * adarms_cond) {
    const int seq_len = input->ne[1];
    const int kv_dim  = n_kv_head * head_dim;
    const int width   = input->ne[0];
    const bool ada    = (adarms_cond != nullptr);

    // ---- Pre-attention norm ----
    ggml_tensor * attn_gate = nullptr;
    ggml_tensor * x;
    if (ada) {
        adarms_mod md = adarms_modulation(ctx, w.attn_norm_dense_w, w.attn_norm_dense_b, adarms_cond, width, act);
        x = adarms_apply(ctx, input, md.scale, md.shift);
        attn_gate = md.gate;
    } else {
        x = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1e-6f), w.attn_norm);
    }

    ggml_tensor * q, * k, * v;
    if (w.qkv_proj) {
        const int q_dim = n_head * head_dim;
        ggml_tensor * qkv = mm_act(ctx, w.qkv_proj, x, act);
        q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, q_dim,  seq_len, qkv->nb[1], 0));
        k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, seq_len, qkv->nb[1], (size_t) q_dim          * sizeof(float)));
        v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kv_dim, seq_len, qkv->nb[1], (size_t)(q_dim + kv_dim) * sizeof(float)));
    } else {
        q = mm_act(ctx, w.q_proj, x, act);
        k = mm_act(ctx, w.k_proj, x, act);
        v = mm_act(ctx, w.v_proj, x, act);
    }

    q = apply_rope(ctx, q, positions, n_head, head_dim);
    k = apply_rope(ctx, k, positions, n_kv_head, head_dim);

    // Write new K/V into the cache (F32 -> kv_type via ggml_cpy), then read back
    // the full [0..attn_kv_len) range. Same path for prefix (self-attn) and the
    // expert (cross-attends to cached prefix KV + the just-written suffix). Note:
    // this uses ggml_cpy (which supports f16 KV on OpenCL); ggml_concat does NOT
    // (OpenCL concat is f32-only). The reused-graph hazard this could pose is
    // avoided by pinning cond + suffix tokens (see build_expert_graph).
    const size_t rbk = cache_k->nb[1], rbv = cache_v->nb[1];
    ggml_tensor * k_dst = ggml_view_2d(ctx, cache_k, kv_dim, seq_len, rbk, (size_t) write_offset * rbk);
    ggml_tensor * v_dst = ggml_view_2d(ctx, cache_v, kv_dim, seq_len, rbv, (size_t) write_offset * rbv);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, k, k_dst));
    ggml_build_forward_expand(graph, ggml_cpy(ctx, v, v_dst));
    ggml_tensor * attn_k = ggml_view_2d(ctx, cache_k, kv_dim, attn_kv_len, rbk, 0);
    ggml_tensor * attn_v = ggml_view_2d(ctx, cache_v, kv_dim, attn_kv_len, rbv, 0);

    q      = ggml_reshape_3d(ctx, q,      head_dim, n_head,    seq_len);
    attn_k = ggml_reshape_3d(ctx, attn_k, head_dim, n_kv_head, attn_kv_len);
    attn_v = ggml_reshape_3d(ctx, attn_v, head_dim, n_kv_head, attn_kv_len);
    attn_k = ggml_cont(ctx, ggml_permute(ctx, attn_k, 0, 2, 1, 3));
    attn_v = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));

    const float scale = 1.0f / sqrtf((float)head_dim);

    // Default = STANDARD (non-flash) attention. Two reasons: (1) it is bit-exact
    // with openpi (flash is 0.999994), and (2) it keeps QK / AV as ggml_mul_mat,
    // which is what the Adreno Ab_Bi f16 image-B attention kernels route through
    // (flash uses a fused kernel that bypasses them). Opt into flash with
    // PI0_FLASH_ATTN=1.
    static const bool use_flash_attn = []{
        const char * e = getenv("PI0_FLASH_ATTN");
        if (e && e[0] == '1') { fprintf(stderr, "PI05: flash attention (opt-in)\n"); return true; }
        return false;
    }();

    ggml_tensor * attn_out;
    if (use_flash_attn) {
        static const bool fix_q_layout = []{
            const char * e = getenv("PI0_FIX_Q_LAYOUT");
            if (e && e[0] == '0') { fprintf(stderr, "PI05: q layout fix DISABLED\n"); return false; }
            return true;
        }();
        ggml_tensor * q_for_fa = fix_q_layout
            ? ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3))  // -> [head_dim, seq_len, n_head]
            : q;
        // K/V keep their cache dtype (kv_type), matching the pi0 flash path.
        attn_out = ggml_flash_attn_ext(ctx, q_for_fa, attn_k, attn_v, attn_mask, scale, 0.0f, 0.0f);
    } else {
        ggml_tensor * q_std = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * v_t   = ggml_cont(ctx, ggml_transpose(ctx, attn_v));

        static const bool use_collapse = []{
            const char * e = getenv("PI0_ATTN_COLLAPSE");
            if (e && e[0] == '1') { fprintf(stderr, "PI05: collapsed attention (MQA fold)\n"); return true; }
            return false;
        }();

        ggml_tensor * kqv;
        if (use_collapse && n_kv_head == 1) {
            ggml_tensor * q_coll = ggml_reshape_2d(ctx, q_std, head_dim, (int64_t) seq_len * n_head);
            ggml_tensor * kq = ggml_mul_mat(ctx, attn_k, q_coll);
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            ggml_tensor * mask_coll = attn_mask ? ggml_repeat(ctx, attn_mask, kq) : nullptr;
            kq = ggml_soft_max_ext(ctx, kq, mask_coll, scale, 0.0f);
            ggml_tensor * kqv_coll = ggml_mul_mat(ctx, v_t, kq);
            kqv = ggml_reshape_3d(ctx, kqv_coll, head_dim, seq_len, n_head);
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx, attn_k, q_std);
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, attn_mask, scale, 0.0f);
            kqv = ggml_mul_mat(ctx, v_t, kq);
        }
        attn_out = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
    }
    attn_out = ggml_reshape_2d(ctx, attn_out, n_head * head_dim, seq_len);

    ggml_tensor * attn_proj = mm_act(ctx, w.o_proj, attn_out, act);
    // Gated residual (adaRMS) or plain residual (prefix).
    input = ada ? ggml_add(ctx, input, ggml_mul(ctx, attn_proj, attn_gate))
                : ggml_add(ctx, input, attn_proj);

    // ---- Pre-FFN norm + GeGLU ----
    ggml_tensor * ffn_gate_mod = nullptr;
    if (ada) {
        adarms_mod md = adarms_modulation(ctx, w.ffn_norm_dense_w, w.ffn_norm_dense_b, adarms_cond, width, act);
        x = adarms_apply(ctx, input, md.scale, md.shift);
        ffn_gate_mod = md.gate;
    } else {
        x = ggml_mul(ctx, ggml_rms_norm(ctx, input, 1e-6f), w.ffn_norm);
    }

    ggml_tensor * gate = mm_act(ctx, w.gate_proj, x, act);
    ggml_tensor * up   = mm_act(ctx, w.up_proj,   x, act);
    ggml_tensor * ffn  = ggml_mul(ctx, ggml_gelu(ctx, gate), up); // Gemma GeGLU (tanh gelu)
    ffn = mm_act(ctx, w.down_proj, ffn, act);

    input = ada ? ggml_add(ctx, input, ggml_mul(ctx, ffn, ffn_gate_mod))
                : ggml_add(ctx, input, ffn);
    return input;
}

// Suffix tokens (pi0.5): action_in_proj(noisy) only. Also computes the adaRMS
// conditioning vector  cond = swish(time_mlp_out(swish(time_mlp_in(sincos(t))))).
static ggml_tensor * build_suffix_tokens(ggml_context * ctx, pi05_model & m,
                                         ggml_tensor * noisy_in,   // [ACTION_DIM, HORIZON]
                                         ggml_tensor * time_in,    // [EXPERT_N_EMBD, 1] sincos(t)
                                         ggml_type act,
                                         ggml_tensor ** cond_out) {
    ggml_tensor * action_tok = mm_act(ctx, m.action_in_proj_w, noisy_in, act);
    action_tok = add_bias(ctx, action_tok, m.action_in_proj_b);

    ggml_tensor * h = mm_act(ctx, m.time_mlp_in_w, time_in, act);
    h = add_bias(ctx, h, m.time_mlp_in_b);
    h = ggml_silu(ctx, h);                                   // swish
    h = mm_act(ctx, m.time_mlp_out_w, h, act);
    h = add_bias(ctx, h, m.time_mlp_out_b);
    *cond_out = ggml_silu(ctx, h);                           // swish -> cond [EXPERT_N_EMBD,1]

    return action_tok; // [EXPERT_N_EMBD, HORIZON]
}

// ============================================================
// KV cache storage
// ============================================================

static void destroy_kv_storage(pi05_session & s) {
    if (s.kv_buf) { ggml_backend_buffer_free(s.kv_buf); s.kv_buf = nullptr; }
    if (s.kv_ctx) { ggml_free(s.kv_ctx);                s.kv_ctx = nullptr; }
    for (int il = 0; il < PI05_N_LAYER; il++) { s.cache_k[il] = nullptr; s.cache_v[il] = nullptr; }
    s.kv_alloc_prefix_len = 0;
    s.kv_alloc_type       = GGML_TYPE_F16;
}

static bool ensure_kv_storage(pi05_session & s, ggml_backend_t backend, int prefix_len) {
    const int total_kv = prefix_len + PI05_SUFFIX_LEN;
    if (s.kv_buf && s.kv_alloc_prefix_len == prefix_len && s.kv_alloc_type == s.kv_type) return true;
    destroy_kv_storage(s);

    const int kv_dim = PALI_N_KV_HEAD * PALI_HEAD_DIM; // = EXPERT_N_KV_HEAD * EXPERT_HEAD_DIM = 256
    size_t ctx_size = ggml_tensor_overhead() * (2 * PI05_N_LAYER + 8);
    ggml_init_params params = { ctx_size, nullptr, true };
    s.kv_ctx = ggml_init(params);

    for (int il = 0; il < PI05_N_LAYER; il++) {
        char name[64];
        s.cache_k[il] = ggml_new_tensor_2d(s.kv_ctx, s.kv_type, kv_dim, total_kv);
        snprintf(name, sizeof(name), "cache_k_%d", il); ggml_set_name(s.cache_k[il], name);
        s.cache_v[il] = ggml_new_tensor_2d(s.kv_ctx, s.kv_type, kv_dim, total_kv);
        snprintf(name, sizeof(name), "cache_v_%d", il); ggml_set_name(s.cache_v[il], name);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    s.kv_buf = ggml_backend_alloc_ctx_tensors_from_buft(s.kv_ctx, buft);
    if (!s.kv_buf) {
        LOG_ERR("Failed to allocate KV cache (type=%s, prefix_len=%d)\n", ggml_type_name(s.kv_type), prefix_len);
        ggml_free(s.kv_ctx); s.kv_ctx = nullptr; return false;
    }
    s.kv_alloc_prefix_len = prefix_len;
    s.kv_alloc_type       = s.kv_type;
    LOG_INF("KV cache allocated: %d layers x [%d, %d] %s (%.2f MB)\n",
            PI05_N_LAYER, kv_dim, total_kv, ggml_type_name(s.kv_type),
            2.0 * PI05_N_LAYER * ggml_nbytes(s.cache_k[0]) / (1024.0 * 1024.0));
    return true;
}

// ============================================================
// Forward passes
// ============================================================

bool run_paligemma_prefix(pi05_model & m, pi05_session & s, ggml_backend_t backend,
                          const float * embeddings, int prefix_len) {
    if (!ensure_kv_storage(s, backend, prefix_len)) return false;
    s.prefix_len = prefix_len;

    size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PALI_N_EMBD, prefix_len);
    ggml_set_name(input, "prefix_input"); ggml_set_input(input);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefix_len);
    ggml_set_name(positions, "positions"); ggml_set_input(positions);

    ggml_tensor * cur = input;
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);

    for (int il = 0; il < PI05_N_LAYER; il++) {
        cur = build_gemma_layer(ctx, graph, cur, m.pali_layers[il], positions, /*attn_mask=*/nullptr,
            PALI_N_HEAD, PALI_N_KV_HEAD, PALI_HEAD_DIM, s.cache_k[il], s.cache_v[il],
            /*write_offset=*/0, /*attn_kv_len=*/prefix_len, s.act_type, /*adarms_cond=*/nullptr);
    }
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, 1e-6f), m.pali_final_norm);
    ggml_set_name(cur, "prefix_out"); ggml_set_output(cur);
    ggml_build_forward_expand(graph, cur);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
        LOG_ERR("Failed to allocate prefix graph\n");
        ggml_gallocr_free(alloc); ggml_free(ctx); return false;
    }

    ggml_backend_tensor_set(input, embeddings, 0, sizeof(float) * PALI_N_EMBD * prefix_len);
    std::vector<int32_t> pos_data(prefix_len);
    for (int i = 0; i < prefix_len; i++) pos_data[i] = i;
    ggml_backend_tensor_set(positions, pos_data.data(), 0, sizeof(int32_t) * prefix_len);

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("Prefix compute failed\n");
        ggml_gallocr_free(alloc); ggml_free(ctx); return false;
    }
    if (const char * po = getenv("PI05_DUMP_PREFIX_OUT")) {
        FILE * pf = fopen(po, "wb");
        if (pf) {
            std::vector<float> buf((size_t) PALI_N_EMBD * prefix_len);
            ggml_backend_tensor_get(cur, buf.data(), 0, sizeof(float) * buf.size());
            fwrite(buf.data(), sizeof(float), buf.size(), pf); fclose(pf);
            LOG_INF("Dumped prefix_out to %s (%d tok x %d)\n", po, prefix_len, PALI_N_EMBD);
        }
    }
    ggml_gallocr_free(alloc); ggml_free(ctx);
    return true;
}

// ---- Persistent expert graph (built once per (prefix_len, kv_type, act_type)) ----

static void destroy_expert_graph(pi05_session & s) {
    if (s.graph_alloc) { ggml_gallocr_free(s.graph_alloc); s.graph_alloc = nullptr; }
    if (s.graph_ctx)   { ggml_free(s.graph_ctx);          s.graph_ctx   = nullptr; }
    s.graph = nullptr;
    s.graph_prefix_len = 0;
    s.graph_kv_type    = GGML_TYPE_F16;
    s.graph_act_type   = GGML_TYPE_F16;
    s.in_noisy = s.in_time = s.in_positions = s.in_attn_mask = s.out_v_t = nullptr;
}

static bool build_expert_graph(pi05_model & m, pi05_session & s, ggml_backend_t backend, int prefix_len) {
    destroy_expert_graph(s);
    const int total_kv = prefix_len + PI05_SUFFIX_LEN;

    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    s.graph_ctx = ggml_init(params);
    ggml_context * ctx = s.graph_ctx;

    s.in_noisy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PI05_ACTION_DIM, PI05_ACTION_HORIZON);
    ggml_set_name(s.in_noisy, "noisy_in"); ggml_set_input(s.in_noisy);
    s.in_time = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EXPERT_N_EMBD, 1);
    ggml_set_name(s.in_time, "time_in"); ggml_set_input(s.in_time);
    s.in_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, PI05_SUFFIX_LEN);
    ggml_set_name(s.in_positions, "suffix_positions"); ggml_set_input(s.in_positions);
    // pi0.5 suffix attention is fully visible (prefix + all action tokens), so no
    // additive mask is needed — full attention, matching openpi ar_mask=[1,0,...].
    s.in_attn_mask = nullptr;

    s.graph = ggml_new_graph_custom(ctx, 16384, false);

    ggml_tensor * cond = nullptr;
    ggml_tensor * cur  = build_suffix_tokens(ctx, m, s.in_noisy, s.in_time, s.act_type, &cond);

    // PIN the two "boundary" tensors of the reused subgraph. This expert graph is
    // persistent — built once and recomputed for all 10 denoise steps. The adaRMS
    // conditioning `cond` (consumed by 37 norms across the whole graph) and the
    // suffix-token stack input `cur` are long-lived; ggml_gallocr otherwise reuses
    // their buffers, which silently corrupts the result on steps 2..10 (step 1 is
    // clean because nothing has clobbered them yet — that is why the bug hides as a
    // "t != 1.0" / trajectory divergence). Marking them outputs pins their buffers
    // so the allocator never reuses them. Verified: restores exact per-step parity
    // with openpi (cos 1.000000 every step). Cost: two small persistent buffers.
    ggml_set_name(cond, "adarms_cond"); ggml_set_output(cond);
    ggml_set_name(cur,  "suffix_tok");  ggml_set_output(cur);

    for (int il = 0; il < PI05_N_LAYER; il++) {
        cur = build_gemma_layer(ctx, s.graph, cur, m.expert_layers[il], s.in_positions, /*attn_mask=*/nullptr,
            EXPERT_N_HEAD, EXPERT_N_KV_HEAD, EXPERT_HEAD_DIM, s.cache_k[il], s.cache_v[il],
            /*write_offset=*/prefix_len, /*attn_kv_len=*/total_kv, s.act_type, /*adarms_cond=*/cond);
    }

    // Final adaRMS norm (gate discarded), then action_out_proj over all 50 tokens.
    {
        adarms_mod md = adarms_modulation(ctx, m.expert_final_norm_dense_w, m.expert_final_norm_dense_b,
                                          cond, EXPERT_N_EMBD, s.act_type);
        cur = adarms_apply(ctx, cur, md.scale, md.shift);
    }
    s.out_v_t = mm_act(ctx, m.action_out_proj_w, cur, s.act_type);
    s.out_v_t = add_bias(ctx, s.out_v_t, m.action_out_proj_b);
    ggml_set_name(s.out_v_t, "v_t"); ggml_set_output(s.out_v_t);
    ggml_build_forward_expand(s.graph, s.out_v_t);

    s.graph_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(s.graph_alloc, s.graph) || !ggml_gallocr_alloc_graph(s.graph_alloc, s.graph)) {
        LOG_ERR("Failed to allocate persistent expert graph\n");
        destroy_expert_graph(s); return false;
    }

    // positions: prefix_len .. prefix_len + SUFFIX_LEN - 1
    std::vector<int32_t> pos_data(PI05_SUFFIX_LEN);
    for (int i = 0; i < PI05_SUFFIX_LEN; i++) pos_data[i] = prefix_len + i;
    ggml_backend_tensor_set(s.in_positions, pos_data.data(), 0, sizeof(int32_t) * PI05_SUFFIX_LEN);

    s.graph_prefix_len = prefix_len;
    s.graph_kv_type    = s.kv_type;
    s.graph_act_type   = s.act_type;
    return true;
}

bool run_expert_step(pi05_model & m, pi05_session & s, ggml_backend_t backend,
                     const float * noisy_actions, float timestep, float * v_t_out) {
    if (!s.graph || s.graph_prefix_len != s.prefix_len
        || s.graph_kv_type != s.kv_type || s.graph_act_type != s.act_type) {
        if (!build_expert_graph(m, s, backend, s.prefix_len)) return false;
    }

    ggml_backend_tensor_set(s.in_noisy, noisy_actions, 0,
        sizeof(float) * PI05_ACTION_DIM * PI05_ACTION_HORIZON);

    // sincos(t) -> [EXPERT_N_EMBD, 1]; the time MLP runs inside the graph.
    std::vector<float> time_emb(EXPERT_N_EMBD);
    sincos_posemb(timestep, EXPERT_N_EMBD, time_emb.data());
    ggml_backend_tensor_set(s.in_time, time_emb.data(), 0, sizeof(float) * EXPERT_N_EMBD);

    if (ggml_backend_graph_compute(backend, s.graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("Expert compute failed\n"); return false;
    }
    ggml_backend_tensor_get(s.out_v_t, v_t_out, 0,
        sizeof(float) * PI05_ACTION_DIM * PI05_ACTION_HORIZON);
    return true;
}

void pi05_session_free(pi05_session & s) {
    destroy_expert_graph(s);
    destroy_kv_storage(s);
    s.prefix_len = 0;
}

// ============================================================
// Utility / backend / reporting / CLI  (shared with the bench)
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

static const char * dev_type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "?";
    }
}

void pi05_dump_backends() {
    ggml_backend_load_all();
    const size_t n = ggml_backend_dev_count();
    LOG_INF("Available backend devices (%zu):\n", n);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        LOG_INF("  [%zu] %-12s %-5s %s\n", i, ggml_backend_dev_name(dev),
                dev_type_str(ggml_backend_dev_type(dev)), ggml_backend_dev_description(dev));
    }
}

ggml_backend_t pi05_init_backend(const std::string & name) {
    ggml_backend_load_all();
    if (name.empty() || name == "cpu" || name == "CPU") {
        LOG_INF("Backend: CPU\n"); return ggml_backend_cpu_init();
    }
    if (name == "auto" || name == "gpu" || name == "GPU") {
        ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (b) { LOG_INF("Backend: %s (auto GPU)\n", ggml_backend_name(b)); return b; }
        LOG_WRN("No GPU backend available; falling back to CPU\n");
        return ggml_backend_cpu_init();
    }
    if (ggml_backend_t b = ggml_backend_init_by_name(name.c_str(), nullptr)) {
        LOG_INF("Backend: %s\n", ggml_backend_name(b)); return b;
    }
    auto lower = [](std::string s){ for (auto & c : s) if (c>='A'&&c<='Z') c=char(c-'A'+'a'); return s; };
    const std::string needle = lower(name);
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (lower(ggml_backend_dev_name(dev)).find(needle) != std::string::npos) {
            if (ggml_backend_t b = ggml_backend_dev_init(dev, nullptr)) {
                LOG_INF("Backend: %s (matched '%s')\n", ggml_backend_dev_name(dev), name.c_str());
                return b;
            }
        }
    }
    LOG_ERR("Backend '%s' not found.\n", name.c_str());
    pi05_dump_backends();
    return nullptr;
}

static void dump_ctx_types(ggml_context * ctx, const char * label) {
    if (!ctx) return;
    std::map<int, int> counts; std::map<int, size_t> bytes;
    size_t total_bytes = 0; int total_count = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        counts[(int) t->type]++; bytes[(int) t->type] += ggml_nbytes(t);
        total_bytes += ggml_nbytes(t); total_count++;
    }
    LOG_INF("[%s] %d tensors, %.2f MB total\n", label, total_count, total_bytes / (1024.0 * 1024.0));
    for (auto & kv : counts)
        LOG_INF("    %-10s %4d tensors  %8.2f MB\n",
                ggml_type_name((ggml_type) kv.first), kv.second, bytes[kv.first] / (1024.0 * 1024.0));
}

void pi05_dump_tensor_types(const pi05_model & m) {
    dump_ctx_types(m.ctx_pali,   "PaliGemma 2B");
    dump_ctx_types(m.ctx_expert, "Action Expert (adaRMS)");
}

ggml_type pi05_ggml_type_from_string(const std::string & s) {
    auto eq = [&s](const char * lit) {
        if (s.size() != strlen(lit)) return false;
        for (size_t i = 0; i < s.size(); i++) {
            char a = s[i], b = lit[i];
            if (a>='A'&&a<='Z') a=char(a-'A'+'a');
            if (b>='A'&&b<='Z') b=char(b-'A'+'a');
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
    return GGML_TYPE_COUNT;
}

ggml_type pi05_resolve_act_type(const std::string & act_type, const std::string & backend) {
    // "auto" resolves to F32. A global f16 activation cast (mm_act) overflows the
    // action expert's residual stream on OpenCL -> NaN (the PaliGemma prefix
    // survives it, the expert does not). This mirrors pi0: the compute graph runs
    // f32 activations, and the f16 speed-up comes from the Adreno Ab_Bi kernels
    // that convert B to an f16 image *internally* (opt-in PI0_ATTN_IMG /
    // PI0_VIT_IMG), not from casting the whole graph. --act-type f16 is still
    // selectable for microbenchmarks but is not safe for full expert inference.
    (void) backend;
    if (act_type == "auto") return GGML_TYPE_F32;
    return pi05_ggml_type_from_string(act_type);
}

pi05_cli pi05_strip_cli_args(int & argc, char ** argv) {
    pi05_cli c;
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            c.backend = argv[++i];
        } else if (strcmp(argv[i], "--kv-type") == 0 && i + 1 < argc) {
            c.kv_type = argv[++i];
        } else if (strcmp(argv[i], "--act-type") == 0 && i + 1 < argc) {
            c.act_type = argv[++i];
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return c;
}
