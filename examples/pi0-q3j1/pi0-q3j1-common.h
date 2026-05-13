#pragma once

// PI0 shared code (Q3J1_TQ variant): real Q3J1 KV cache in HBM, like Q4_0/Q8_0.
//   - Prefix K/V tensors live in the backend buffer as GGML_TYPE_Q3J1_TQ
//     (3.83 MiB per inference, vs 27 MiB for fp32 — 7.1x smaller).
//   - Quantization (F32 → Q3J1_TQ) happens as a ggml_cpy node at the end
//     of the prefix-pass graph; downloaded as Q3J1_TQ bytes.
//   - Concatenation with the current diffusion step's K/V is done by
//     ggml_cpy-into-views over a pre-allocated Q3J1_TQ buffer (ggml_concat
//     is broken for quantized types — its concat_any path iterates per-element
//     with type_size, which is the block size in bytes for quantized types).
//   - ggml_flash_attn_ext consumes Q3J1_TQ K/V directly: dispatches via
//     type_traits_cpu->vec_dot for K (Q3J1_TQ vec_dot_type=F32) and
//     type_traits->to_float for V.

#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <memory>

// ---- PI0 constants ----

static constexpr int PI0_ACTION_DIM     = 32;
static constexpr int PI0_ACTION_HORIZON = 50;
static constexpr int PI0_N_LAYER        = 18;
static constexpr int PI0_NUM_STEPS      = 10;

// PaliGemma 2B config
static constexpr int PALI_N_EMBD    = 2048;
static constexpr int PALI_N_HEAD    = 8;
static constexpr int PALI_N_KV_HEAD = 1;
static constexpr int PALI_HEAD_DIM  = 256;
static constexpr int PALI_N_FF      = 16384;

// Action Expert 300M config
static constexpr int EXPERT_N_EMBD    = 1024;
static constexpr int EXPERT_N_HEAD    = 8;
static constexpr int EXPERT_N_KV_HEAD = 1;
static constexpr int EXPERT_HEAD_DIM  = 256;
static constexpr int EXPERT_N_FF      = 4096;

// ---- Weight structures ----

struct gemma_layer_weights {
    ggml_tensor * attn_norm;   // RMS norm weight (already +1)
    ggml_tensor * ffn_norm;    // RMS norm weight (already +1)
    ggml_tensor * q_proj;      // [n_head * head_dim, n_embd]
    ggml_tensor * k_proj;      // [n_kv_head * head_dim, n_embd]
    ggml_tensor * v_proj;      // [n_kv_head * head_dim, n_embd]
    ggml_tensor * o_proj;      // [n_embd, n_head * head_dim]
    ggml_tensor * gate_proj;   // [n_ff, n_embd]
    ggml_tensor * up_proj;     // [n_ff, n_embd]
    ggml_tensor * down_proj;   // [n_embd, n_ff]
};

struct pi0_model {
    // PaliGemma 2B
    ggml_tensor * pali_embed;  // [vocab, n_embd]
    gemma_layer_weights pali_layers[PI0_N_LAYER];
    ggml_tensor * pali_final_norm;

    // Action Expert 300M
    gemma_layer_weights expert_layers[PI0_N_LAYER];
    ggml_tensor * expert_final_norm;

    // Action projections
    ggml_tensor * action_in_proj_w;
    ggml_tensor * action_in_proj_b;
    ggml_tensor * action_out_proj_w;
    ggml_tensor * action_out_proj_b;
    ggml_tensor * state_proj_w;
    ggml_tensor * state_proj_b;
    ggml_tensor * action_time_mlp_in_w;
    ggml_tensor * action_time_mlp_in_b;
    ggml_tensor * action_time_mlp_out_w;
    ggml_tensor * action_time_mlp_out_b;

    // Backend resources
    ggml_context * ctx_pali   = nullptr;
    ggml_context * ctx_expert = nullptr;
    ggml_backend_buffer_t buf_pali   = nullptr;
    ggml_backend_buffer_t buf_expert = nullptr;
};

// ---- GGUF weight loading ----

static ggml_tensor * get_tensor(ggml_context * ctx, const char * name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t) {
        LOG_ERR("Missing tensor: %s\n", name);
    }
    return t;
}

static bool load_gguf_weights(
    const char * path,
    ggml_context ** out_ctx,
    ggml_backend_buffer_t * out_buf,
    ggml_backend_t backend
) {
    struct ggml_context * meta = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gguf_ctx = gguf_init_from_file(path, params);
    if (!gguf_ctx) {
        LOG_ERR("Failed to load GGUF: %s\n", path);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    LOG_INF("Loading %d tensors from %s\n", n_tensors, path);

    // Create data context with same tensors
    size_t ctx_size = ggml_tensor_overhead() * n_tensors;
    ggml_init_params ctx_params = { ctx_size, nullptr, true };
    ggml_context * ctx_data = ggml_init(ctx_params);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * meta_tensor = ggml_get_tensor(meta, name);
        ggml_tensor * data_tensor = ggml_dup_tensor(ctx_data, meta_tensor);
        ggml_set_name(data_tensor, name);
    }

    // Allocate backend buffer
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

    // Load tensor data from file
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

static bool load_pi0_model(pi0_model & m, const char * pali_path, const char * expert_path, ggml_backend_t backend) {
    // Load PaliGemma 2B
    if (!load_gguf_weights(pali_path, &m.ctx_pali, &m.buf_pali, backend)) {
        return false;
    }

    m.pali_embed = get_tensor(m.ctx_pali, "token_embd.weight");
    m.pali_final_norm = get_tensor(m.ctx_pali, "output_norm.weight");

    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[128];
        auto & l = m.pali_layers[il];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);   l.attn_norm = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);    l.ffn_norm  = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);      l.q_proj    = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);      l.k_proj    = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);      l.v_proj    = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il); l.o_proj    = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);    l.gate_proj = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);      l.up_proj   = get_tensor(m.ctx_pali, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);    l.down_proj = get_tensor(m.ctx_pali, name);
    }

    // Load Action Expert 300M
    if (!load_gguf_weights(expert_path, &m.ctx_expert, &m.buf_expert, backend)) {
        return false;
    }

    m.expert_final_norm = get_tensor(m.ctx_expert, "output_norm.weight");

    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[128];
        auto & l = m.expert_layers[il];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", il);   l.attn_norm = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", il);    l.ffn_norm  = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", il);      l.q_proj    = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", il);      l.k_proj    = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", il);      l.v_proj    = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", il); l.o_proj    = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", il);    l.gate_proj = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", il);      l.up_proj   = get_tensor(m.ctx_expert, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", il);    l.down_proj = get_tensor(m.ctx_expert, name);
    }

    // Action projections
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

// ---- Sinusoidal position embedding ----

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

// ---- Gemma RoPE ----

static ggml_tensor * apply_rope(
    ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions,
    int n_head, int head_dim
) {
    // x: [n_head * head_dim, seq] → reshape to [head_dim, n_head, seq]
    ggml_tensor * reshaped = ggml_reshape_3d(ctx, x, head_dim, n_head, x->ne[1]);
    // Gemma uses standard RoPE with base 10000
    ggml_tensor * roped = ggml_rope_ext(ctx, reshaped, positions, nullptr,
        head_dim, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx, roped, n_head * head_dim, x->ne[1]);
}

// ---- Gemma transformer layer ----

struct layer_kv {
    ggml_tensor * k; // [kv_dim, seq_len]
    ggml_tensor * v; // [kv_dim, seq_len]
};

// Build one Gemma transformer layer, optionally with prefix KV for cross-attention.
// gf is needed when prefix_kv is provided as Q3J1_TQ — the cpy nodes that populate
// the combined K/V buffer must be explicitly expanded into the graph (they target
// views of a pre-allocated tensor that downstream ops read directly).
static ggml_tensor * build_gemma_layer(
    ggml_context * ctx,
    ggml_cgraph  * gf,             // graph to expand cpy nodes into (may be null when prefix_kv is null)
    ggml_tensor * input,           // [n_embd, seq_len]
    const gemma_layer_weights & w,
    ggml_tensor * positions,       // [seq_len]
    ggml_tensor * attn_mask,       // [total_kv_len, seq_len] or nullptr
    int n_head, int n_kv_head, int head_dim,
    layer_kv * prefix_kv,          // if not null, prepend these KV
    layer_kv * out_kv              // if not null, store this layer's KV (F32, pre-quantization)
) {
    const int seq_len = input->ne[1];

    // Pre-attention RMS norm
    ggml_tensor * x = ggml_rms_norm(ctx, input, 1e-6f);
    x = ggml_mul(ctx, x, w.attn_norm);

    // Q, K, V projections
    ggml_tensor * q = ggml_mul_mat(ctx, w.q_proj, x);  // [n_head*head_dim, seq_len]
    ggml_tensor * k = ggml_mul_mat(ctx, w.k_proj, x);  // [kv_dim, seq_len]
    ggml_tensor * v = ggml_mul_mat(ctx, w.v_proj, x);  // [kv_dim, seq_len]

    // Apply RoPE
    q = apply_rope(ctx, q, positions, n_head, head_dim);
    k = apply_rope(ctx, k, positions, n_kv_head, head_dim);

    // Save KV if requested
    if (out_kv) {
        out_kv->k = k;
        out_kv->v = v;
    }

    // Prepend prefix KV for cross-attention.
    // For Q3J1_TQ K/V, ggml_concat is unusable (concat_any iterates per element
    // with type_size, which is the block size for quantized types). Instead we
    // pre-allocate a full [kv_dim, total_kv] Q3J1_TQ buffer and write prefix +
    // current K/V into it via ggml_cpy-into-views. The prefix slice is a same-type
    // byte memcpy; the current slice triggers F32 → Q3J1_TQ via from_float.
    if (prefix_kv) {
        const int prefix_len_l = prefix_kv->k->ne[1];
        const int total_kv_l   = prefix_len_l + seq_len;
        const int kv_dim       = n_kv_head * head_dim;

        if (prefix_kv->k->type == GGML_TYPE_Q3J1_TQ) {
            GGML_ASSERT(gf != nullptr && "build_gemma_layer: gf required for Q3J1_TQ prefix");

            ggml_tensor * full_k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3J1_TQ, kv_dim, total_kv_l);
            ggml_tensor * full_v = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3J1_TQ, kv_dim, total_kv_l);

            const size_t row_bytes = full_k->nb[1];
            ggml_tensor * k_prefix_view = ggml_view_2d(ctx, full_k, kv_dim, prefix_len_l, row_bytes, 0);
            ggml_tensor * k_curr_view   = ggml_view_2d(ctx, full_k, kv_dim, seq_len,      row_bytes, row_bytes * prefix_len_l);
            ggml_tensor * v_prefix_view = ggml_view_2d(ctx, full_v, kv_dim, prefix_len_l, row_bytes, 0);
            ggml_tensor * v_curr_view   = ggml_view_2d(ctx, full_v, kv_dim, seq_len,      row_bytes, row_bytes * prefix_len_l);

            // Same-type byte memcpy from input prefix tensors into the full buffer
            ggml_build_forward_expand(gf, ggml_cpy(ctx, prefix_kv->k, k_prefix_view));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, prefix_kv->v, v_prefix_view));
            // F32 → Q3J1_TQ via from_float (TurboQuant codebook + QJL projection)
            ggml_build_forward_expand(gf, ggml_cpy(ctx, k, k_curr_view));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, v, v_curr_view));

            k = full_k;
            v = full_v;
        } else {
            k = ggml_concat(ctx, prefix_kv->k, k, 1);
            v = ggml_concat(ctx, prefix_kv->v, v, 1);
        }
    }

    // Reshape for attention: Q=[head_dim, n_head, seq], K=[head_dim, n_kv_head, total_kv], V=[head_dim, n_kv_head, total_kv]
    int total_kv = k->ne[1];
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, seq_len);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv_head, total_kv);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv_head, total_kv);

    // Transpose K and V for flash attention: ggml_flash_attn_ext expects K=[head_dim, kv_len, n_kv_head]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [head_dim, total_kv, n_kv_head] -> not right
    // Actually ggml_flash_attn_ext expects: q=[head_dim, seq, n_head], k=[head_dim, kv_len, n_kv_head], v=[head_dim, kv_len, n_kv_head]
    // Our k is already [head_dim, n_kv_head, total_kv], need to permute to [head_dim, total_kv, n_kv_head]
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, k, head_dim * n_kv_head, total_kv)), head_dim, n_kv_head, total_kv), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_reshape_2d(ctx, v, head_dim * n_kv_head, total_kv)), head_dim, n_kv_head, total_kv), 0, 2, 1, 3));

    // Use ggml_flash_attn_ext for GQA-compatible attention
    float scale = 1.0f / sqrtf((float)head_dim);
    ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, q, k, v, attn_mask, scale, 0.0f, 0.0f);
    // attn_out: [head_dim, seq_len, n_head]
    attn_out = ggml_reshape_2d(ctx, attn_out, n_head * head_dim, seq_len);

    // Output projection
    ggml_tensor * attn_proj = ggml_mul_mat(ctx, w.o_proj, attn_out); // [n_embd, seq_len]

    // Residual
    input = ggml_add(ctx, input, attn_proj);

    // Post-attention RMS norm + FFN
    x = ggml_rms_norm(ctx, input, 1e-6f);
    x = ggml_mul(ctx, x, w.ffn_norm);

    // Gated FFN: SiLU(gate) * up → down
    ggml_tensor * gate = ggml_mul_mat(ctx, w.gate_proj, x); // [n_ff, seq_len]
    ggml_tensor * up   = ggml_mul_mat(ctx, w.up_proj, x);   // [n_ff, seq_len]
    ggml_tensor * ffn  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ffn = ggml_mul_mat(ctx, w.down_proj, ffn); // [n_embd, seq_len]

    // Residual
    input = ggml_add(ctx, input, ffn);

    return input;
}

// ---- PaliGemma prefix forward pass ----
// Runs the PaliGemma 2B prefix, caches per-layer K,V for cross-attention by the expert.

// Global KV data storage (set by prefix pass, used by expert).
// Q3J1_TQ byte buffers — sized via ggml_row_size(GGML_TYPE_Q3J1_TQ, kv_dim * prefix_len).
inline std::vector<std::vector<uint8_t>> g_prefix_k_q(PI0_N_LAYER);
inline std::vector<std::vector<uint8_t>> g_prefix_v_q(PI0_N_LAYER);
inline int g_prefix_len = 0;

static bool run_paligemma_prefix_v2(
    pi0_model & m,
    ggml_backend_t backend,
    const float * embeddings,
    int prefix_len
) {
    g_prefix_len = prefix_len;
    const int kv_dim = PALI_N_KV_HEAD * PALI_HEAD_DIM; // 256

    // 18 layers × ~30 tensors each + inputs/outputs
    size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PALI_N_EMBD, prefix_len);
    ggml_set_name(input, "prefix_input");
    ggml_set_input(input);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefix_len);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);

    // Scale embeddings by sqrt(n_embd)
    ggml_tensor * cur = ggml_scale(ctx, input, sqrtf((float)PALI_N_EMBD));

    // Build layers, collect F32 K/V
    layer_kv kvs[PI0_N_LAYER];
    for (int il = 0; il < PI0_N_LAYER; il++) {
        cur = build_gemma_layer(ctx, graph, cur, m.pali_layers[il], positions, nullptr,
            PALI_N_HEAD, PALI_N_KV_HEAD, PALI_HEAD_DIM,
            nullptr, &kvs[il]);
    }

    // Final norm (not strictly needed but ensures full forward pass)
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, m.pali_final_norm);
    ggml_set_name(cur, "prefix_out");
    ggml_set_output(cur);

    // Quantize F32 K/V → Q3J1_TQ as the final graph step. ggml_cast allocates
    // a new tensor of the target type and wires up a CPY op; the resulting
    // tensor is allocated by gallocr and downloadable via ggml_backend_tensor_get.
    ggml_tensor * kv_q[PI0_N_LAYER * 2]; // [il*2+0]=k, [il*2+1]=v
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[64];
        ggml_tensor * k_q = ggml_cast(ctx, kvs[il].k, GGML_TYPE_Q3J1_TQ);
        ggml_tensor * v_q = ggml_cast(ctx, kvs[il].v, GGML_TYPE_Q3J1_TQ);
        snprintf(name, sizeof(name), "prefix_k_q3j1_%d", il); ggml_set_name(k_q, name);
        snprintf(name, sizeof(name), "prefix_v_q3j1_%d", il); ggml_set_name(v_q, name);
        ggml_set_output(k_q);
        ggml_set_output(v_q);
        kv_q[il*2+0] = k_q;
        kv_q[il*2+1] = v_q;
    }

    ggml_build_forward_expand(graph, cur);
    for (int il = 0; il < PI0_N_LAYER; il++) {
        ggml_build_forward_expand(graph, kv_q[il*2+0]);
        ggml_build_forward_expand(graph, kv_q[il*2+1]);
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
        LOG_ERR("Failed to allocate prefix graph\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Set inputs
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

    // Download Q3J1_TQ bytes from the GPU buffer to host. 7.1x smaller than fp32.
    {
        const int64_t total_elems = (int64_t)kv_dim * prefix_len;
        const size_t  bytes_per_layer = ggml_row_size(GGML_TYPE_Q3J1_TQ, total_elems);
        for (int il = 0; il < PI0_N_LAYER; il++) {
            g_prefix_k_q[il].resize(bytes_per_layer);
            g_prefix_v_q[il].resize(bytes_per_layer);
            // kv_q[il*2+0] is the cpy node; its result tensor is the Q3J1_TQ output.
            ggml_backend_tensor_get(kv_q[il*2+0], g_prefix_k_q[il].data(), 0, bytes_per_layer);
            ggml_backend_tensor_get(kv_q[il*2+1], g_prefix_v_q[il].data(), 0, bytes_per_layer);
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// ---- Action Expert diffusion step ----

static bool run_expert_step(
    pi0_model & m,
    ggml_backend_t backend,
    const float * suffix_emb,   // [EXPERT_N_EMBD, 51] (state + action tokens)
    float * v_t_out             // [PI0_ACTION_DIM, PI0_ACTION_HORIZON] output
) {
    const int suffix_len = PI0_ACTION_HORIZON + 1; // 51
    const int prefix_len = g_prefix_len;
    const int total_kv = prefix_len + suffix_len;
    const int kv_dim = EXPERT_N_KV_HEAD * EXPERT_HEAD_DIM; // 256

    size_t ctx_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    // Input: suffix embeddings
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, EXPERT_N_EMBD, suffix_len);
    ggml_set_name(input, "suffix_input");
    ggml_set_input(input);

    // Positions for suffix tokens (start after prefix)
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, suffix_len);
    ggml_set_name(positions, "suffix_positions");
    ggml_set_input(positions);

    // Prefix KV tensors — Q3J1_TQ in HBM, like Q4_0/Q8_0 KV cache.
    // Sized in element count along axis 0 (kv_dim must be a multiple of QK_Q3J1_TQ=32).
    ggml_tensor * prefix_k_tensors[PI0_N_LAYER];
    ggml_tensor * prefix_v_tensors[PI0_N_LAYER];
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[64];
        prefix_k_tensors[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3J1_TQ, kv_dim, prefix_len);
        snprintf(name, sizeof(name), "prefix_k_%d", il);
        ggml_set_name(prefix_k_tensors[il], name);
        ggml_set_input(prefix_k_tensors[il]);

        prefix_v_tensors[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_Q3J1_TQ, kv_dim, prefix_len);
        snprintf(name, sizeof(name), "prefix_v_%d", il);
        ggml_set_name(prefix_v_tensors[il], name);
        ggml_set_input(prefix_v_tensors[il]);
    }

    // Attention mask: [total_kv, suffix_len]
    // suffix tokens can see: all prefix (1), state sees self only (1), actions see state + all actions (1)
    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, total_kv, suffix_len);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // Build transformer
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_tensor * cur = input;

    for (int il = 0; il < PI0_N_LAYER; il++) {
        layer_kv pkv;
        pkv.k = prefix_k_tensors[il];
        pkv.v = prefix_v_tensors[il];

        cur = build_gemma_layer(ctx, graph, cur, m.expert_layers[il], positions, attn_mask,
            EXPERT_N_HEAD, EXPERT_N_KV_HEAD, EXPERT_HEAD_DIM,
            &pkv, nullptr);
    }

    // Final norm
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, m.expert_final_norm);

    // Extract last action_horizon tokens (skip state token at position 0)
    ggml_tensor * action_hidden = ggml_view_2d(ctx, cur, EXPERT_N_EMBD, PI0_ACTION_HORIZON,
        cur->nb[1], cur->nb[1] * 1); // skip first token (state)

    // action_out_proj: [action_dim, n_embd] × [n_embd, action_horizon] → [action_dim, action_horizon]
    ggml_tensor * v_t = ggml_mul_mat(ctx, m.action_out_proj_w, action_hidden);
    v_t = ggml_add(ctx, v_t, ggml_repeat(ctx, ggml_reshape_2d(ctx, m.action_out_proj_b, PI0_ACTION_DIM, 1), v_t));
    ggml_set_name(v_t, "v_t");
    ggml_set_output(v_t);

    // Final output expansion (cpy nodes inside build_gemma_layer were already expanded)
    ggml_build_forward_expand(graph, v_t);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, graph) || !ggml_gallocr_alloc_graph(alloc, graph)) {
        LOG_ERR("Failed to allocate expert graph\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Set input data
    ggml_backend_tensor_set(input, suffix_emb, 0, sizeof(float) * EXPERT_N_EMBD * suffix_len);

    std::vector<int32_t> pos_data(suffix_len);
    for (int i = 0; i < suffix_len; i++) pos_data[i] = prefix_len + i;
    ggml_backend_tensor_set(positions, pos_data.data(), 0, sizeof(int32_t) * suffix_len);

    // Set prefix KV data — upload Q3J1_TQ bytes (7.1x smaller than fp32)
    for (int il = 0; il < PI0_N_LAYER; il++) {
        ggml_backend_tensor_set(prefix_k_tensors[il], g_prefix_k_q[il].data(), 0,
            g_prefix_k_q[il].size());
        ggml_backend_tensor_set(prefix_v_tensors[il], g_prefix_v_q[il].data(), 0,
            g_prefix_v_q[il].size());
    }

    // Build attention mask
    // Mask value: 0 = attend, -inf = don't attend
    std::vector<float> mask_data(total_kv * suffix_len, 0.0f);
    const float NEG_INF = -INFINITY;
    for (int s = 0; s < suffix_len; s++) {
        for (int kv = 0; kv < total_kv; kv++) {
            if (kv < prefix_len) {
                // All suffix tokens can attend to all prefix tokens
                mask_data[s * total_kv + kv] = 0.0f;
            } else {
                int suffix_kv = kv - prefix_len;
                if (s == 0) {
                    // State token: only attends to itself in suffix
                    mask_data[s * total_kv + kv] = (suffix_kv == 0) ? 0.0f : NEG_INF;
                } else {
                    // Action tokens: attend to state (0) and all action tokens (1..50)
                    // State is suffix_kv=0, actions are suffix_kv=1..50
                    if (suffix_kv == 0) {
                        mask_data[s * total_kv + kv] = 0.0f; // can see state
                    } else {
                        mask_data[s * total_kv + kv] = 0.0f; // can see all actions (bidirectional)
                    }
                }
            }
        }
    }
    ggml_backend_tensor_set(attn_mask, mask_data.data(), 0, sizeof(float) * total_kv * suffix_len);

    // Compute
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("Expert compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Extract output (use the named v_t tensor directly, not the last graph node
    // which may differ after ggml's topological sort)
    ggml_backend_tensor_get(v_t, v_t_out, 0, sizeof(float) * PI0_ACTION_DIM * PI0_ACTION_HORIZON);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

// ---- Prepare suffix embeddings ----

// Helper: read tensor data to f32 buffer, handling f16 conversion
static void tensor_to_f32(ggml_tensor * t, float * out) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out, 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(tmp[i]);
        }
    } else {
        GGML_ABORT("unsupported tensor type for f32 conversion");
    }
}

static void prepare_suffix(
    pi0_model & m,
    const float * state,       // [PI0_ACTION_DIM]
    const float * noisy_actions, // [PI0_ACTION_DIM * PI0_ACTION_HORIZON]
    float timestep,
    float * suffix_out          // [EXPERT_N_EMBD, 51] output
) {
    // This is done on CPU for simplicity

    // 1. State token: state_proj(state) = W @ state + b
    std::vector<float> state_w(EXPERT_N_EMBD * PI0_ACTION_DIM);
    std::vector<float> state_b(EXPERT_N_EMBD);
    tensor_to_f32(m.state_proj_w, state_w.data());
    tensor_to_f32(m.state_proj_b, state_b.data());

    // state_emb = W @ state + b  (W is [1024, 32], state is [32])
    for (int i = 0; i < EXPERT_N_EMBD; i++) {
        float sum = state_b[i];
        for (int j = 0; j < PI0_ACTION_DIM; j++) {
            sum += state_w[i * PI0_ACTION_DIM + j] * state[j];
        }
        suffix_out[i] = sum; // first token = state
    }

    // 2. Action tokens: action_in_proj(noisy_actions)
    std::vector<float> act_w(EXPERT_N_EMBD * PI0_ACTION_DIM);
    std::vector<float> act_b(EXPERT_N_EMBD);
    tensor_to_f32(m.action_in_proj_w, act_w.data());
    tensor_to_f32(m.action_in_proj_b, act_b.data());

    std::vector<float> action_emb(EXPERT_N_EMBD * PI0_ACTION_HORIZON);
    for (int t = 0; t < PI0_ACTION_HORIZON; t++) {
        for (int i = 0; i < EXPERT_N_EMBD; i++) {
            float sum = act_b[i];
            for (int j = 0; j < PI0_ACTION_DIM; j++) {
                sum += act_w[i * PI0_ACTION_DIM + j] * noisy_actions[t * PI0_ACTION_DIM + j];
            }
            action_emb[t * EXPERT_N_EMBD + i] = sum;
        }
    }

    // 3. Time embedding: sincos_posemb(timestep, EXPERT_N_EMBD)
    std::vector<float> time_emb(EXPERT_N_EMBD);
    sincos_posemb(timestep, EXPERT_N_EMBD, time_emb.data());

    // 4. Concat action_emb with time_emb, then MLP
    // action_time = [action_emb || time_emb_repeated] → [2 * EXPERT_N_EMBD, PI0_ACTION_HORIZON]
    // Then: mlp_in → swish → mlp_out
    std::vector<float> mlp_in_w(EXPERT_N_EMBD * 2 * EXPERT_N_EMBD);
    std::vector<float> mlp_in_b(EXPERT_N_EMBD);
    std::vector<float> mlp_out_w(EXPERT_N_EMBD * EXPERT_N_EMBD);
    std::vector<float> mlp_out_b(EXPERT_N_EMBD);
    tensor_to_f32(m.action_time_mlp_in_w, mlp_in_w.data());
    tensor_to_f32(m.action_time_mlp_in_b, mlp_in_b.data());
    tensor_to_f32(m.action_time_mlp_out_w, mlp_out_w.data());
    tensor_to_f32(m.action_time_mlp_out_b, mlp_out_b.data());

    for (int t = 0; t < PI0_ACTION_HORIZON; t++) {
        // Concat: [action_emb[t] || time_emb] → [2048]
        std::vector<float> concat_input(2 * EXPERT_N_EMBD);
        for (int i = 0; i < EXPERT_N_EMBD; i++) {
            concat_input[i] = action_emb[t * EXPERT_N_EMBD + i];
            concat_input[EXPERT_N_EMBD + i] = time_emb[i];
        }

        // MLP layer 1: W_in @ concat + b_in, then swish
        std::vector<float> hidden(EXPERT_N_EMBD);
        for (int i = 0; i < EXPERT_N_EMBD; i++) {
            float sum = mlp_in_b[i];
            for (int j = 0; j < 2 * EXPERT_N_EMBD; j++) {
                sum += mlp_in_w[i * 2 * EXPERT_N_EMBD + j] * concat_input[j];
            }
            // swish = x * sigmoid(x)
            hidden[i] = sum / (1.0f + expf(-sum));
        }

        // MLP layer 2: W_out @ hidden + b_out (linear output, no activation)
        // PI0 time MLP is: Linear → SiLU → Linear (see openpi pi0.py:173-176)
        float * out = &suffix_out[(t + 1) * EXPERT_N_EMBD]; // +1 to skip state token
        for (int i = 0; i < EXPERT_N_EMBD; i++) {
            float sum = mlp_out_b[i];
            for (int j = 0; j < EXPERT_N_EMBD; j++) {
                sum += mlp_out_w[i * EXPERT_N_EMBD + j] * hidden[j];
            }
            out[i] = sum;
        }
    }
}

// ---- Model directory path helper ----

static std::string resolve_model_dir(const std::string & path) {
    std::string dir = path;
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    if (dir.find(".gguf") != std::string::npos) {
        auto pos = dir.rfind('/');
        dir = (pos != std::string::npos) ? dir.substr(0, pos) : ".";
    }
    return dir;
}
