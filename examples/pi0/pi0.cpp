// PI0 action prediction example
//
// Uses flow matching diffusion to predict robot actions from images + text.
// Architecture: SigLIP (vision) → PaliGemma 2B (prefix) → Action Expert 300M (suffix)
//
// Usage:
//   llama-pi0 -m pi0-gemma-2b.gguf --mmproj pi0-mmproj.gguf
//       --expert pi0-action-expert.gguf
//       --image img1.jpg --image img2.jpg --image img3.jpg
//       -p "pick up the red block"

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

// PI0 constants
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

// Build one Gemma transformer layer, optionally with prefix KV for cross-attention
static ggml_tensor * build_gemma_layer(
    ggml_context * ctx,
    ggml_tensor * input,           // [n_embd, seq_len]
    const gemma_layer_weights & w,
    ggml_tensor * positions,       // [seq_len]
    ggml_tensor * attn_mask,       // [total_kv_len, seq_len] or nullptr
    int n_head, int n_kv_head, int head_dim,
    layer_kv * prefix_kv,          // if not null, prepend these KV
    layer_kv * out_kv              // if not null, store this layer's KV
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

    // Prepend prefix KV for cross-attention
    if (prefix_kv) {
        k = ggml_concat(ctx, prefix_kv->k, k, 1); // [kv_dim, prefix_len + seq_len]
        v = ggml_concat(ctx, prefix_kv->v, v, 1);
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
// Returns per-layer K,V for cross-attention

struct prefix_result {
    layer_kv kv[PI0_N_LAYER];
    int prefix_len;
};

static bool run_paligemma_prefix(
    pi0_model & m,
    ggml_backend_t backend,
    const float * embeddings,  // [PALI_N_EMBD, prefix_len] - image + text embeddings
    int prefix_len,
    prefix_result & result
) {
    result.prefix_len = prefix_len;

    // Allocate compute context
    size_t ctx_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead();
    ggml_init_params params = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    // Input tensor
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PALI_N_EMBD, prefix_len);
    ggml_set_name(input, "prefix_input");
    ggml_set_input(input);

    // Scale embeddings by sqrt(n_embd) - Gemma convention
    ggml_tensor * scale = ggml_new_f32(ctx, sqrtf((float)PALI_N_EMBD));
    ggml_tensor * cur = ggml_mul(ctx, input, scale);

    // Position tensor
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefix_len);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // No attention mask for prefix (bidirectional)

    // Build transformer layers
    layer_kv layer_kvs[PI0_N_LAYER];
    for (int il = 0; il < PI0_N_LAYER; il++) {
        cur = build_gemma_layer(ctx, cur, m.pali_layers[il], positions, nullptr,
            PALI_N_HEAD, PALI_N_KV_HEAD, PALI_HEAD_DIM,
            nullptr, &layer_kvs[il]);
    }

    // Final norm
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, m.pali_final_norm);
    ggml_set_name(cur, "prefix_output");
    ggml_set_output(cur);

    // Mark K,V as outputs so they get computed
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name_k[64], name_v[64];
        snprintf(name_k, sizeof(name_k), "pali_k_%d", il);
        snprintf(name_v, sizeof(name_v), "pali_v_%d", il);
        ggml_set_name(layer_kvs[il].k, name_k);
        ggml_set_name(layer_kvs[il].v, name_v);
        ggml_set_output(layer_kvs[il].k);
        ggml_set_output(layer_kvs[il].v);
    }

    // Build graph
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, cur);
    for (int il = 0; il < PI0_N_LAYER; il++) {
        ggml_build_forward_expand(graph, layer_kvs[il].k);
        ggml_build_forward_expand(graph, layer_kvs[il].v);
    }

    // Allocate and compute
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(alloc, graph)) {
        LOG_ERR("Failed to allocate prefix graph\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    ggml_gallocr_alloc_graph(alloc, graph);

    // Set input data
    ggml_backend_tensor_set(input, embeddings, 0, sizeof(float) * PALI_N_EMBD * prefix_len);

    std::vector<int32_t> pos_data(prefix_len);
    for (int i = 0; i < prefix_len; i++) pos_data[i] = i;
    ggml_backend_tensor_set(positions, pos_data.data(), 0, sizeof(int32_t) * prefix_len);

    // Compute
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        LOG_ERR("Failed to compute prefix graph\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    // Extract K,V data to CPU buffers
    for (int il = 0; il < PI0_N_LAYER; il++) {
        result.kv[il].k = layer_kvs[il].k;
        result.kv[il].v = layer_kvs[il].v;
    }

    // Note: we keep alloc and ctx alive since the KV data is in the allocated buffers
    // In a real implementation, copy to persistent CPU buffers
    // For now, we'll extract the data when building the expert graph

    // Actually, we need to copy the data since we'll free the allocator
    // Let's store raw data
    static std::vector<std::vector<float>> kv_data_k(PI0_N_LAYER);
    static std::vector<std::vector<float>> kv_data_v(PI0_N_LAYER);

    for (int il = 0; il < PI0_N_LAYER; il++) {
        size_t k_nelements = ggml_nelements(layer_kvs[il].k);
        size_t v_nelements = ggml_nelements(layer_kvs[il].v);
        kv_data_k[il].resize(k_nelements);
        kv_data_v[il].resize(v_nelements);
        ggml_backend_tensor_get(layer_kvs[il].k, kv_data_k[il].data(), 0, k_nelements * sizeof(float));
        ggml_backend_tensor_get(layer_kvs[il].v, kv_data_v[il].data(), 0, v_nelements * sizeof(float));
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    // Store pointers to the data (they persist in static vectors)
    for (int il = 0; il < PI0_N_LAYER; il++) {
        // We'll set these as tensor data in the expert graph
        result.kv[il].k = nullptr; // mark as raw data
        result.kv[il].v = nullptr;
    }

    // Store the raw data pointers globally for the expert to use
    // (ugly but functional - in production, use a proper data management scheme)
    return true;
}

// ---- Global KV data storage (set by prefix pass, used by expert) ----
static std::vector<std::vector<float>> g_prefix_k(PI0_N_LAYER);
static std::vector<std::vector<float>> g_prefix_v(PI0_N_LAYER);
static int g_prefix_len = 0;

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

    // Scale embeddings by sqrt(n_embd)
    ggml_tensor * cur = ggml_scale(ctx, input, sqrtf((float)PALI_N_EMBD));

    // Build layers, collect K,V
    layer_kv kvs[PI0_N_LAYER];
    for (int il = 0; il < PI0_N_LAYER; il++) {
        cur = build_gemma_layer(ctx, cur, m.pali_layers[il], positions, nullptr,
            PALI_N_HEAD, PALI_N_KV_HEAD, PALI_HEAD_DIM,
            nullptr, &kvs[il]);
    }

    // Final norm (not strictly needed but ensures full forward pass)
    cur = ggml_rms_norm(ctx, cur, 1e-6f);
    cur = ggml_mul(ctx, cur, m.pali_final_norm);
    ggml_set_name(cur, "prefix_out");
    ggml_set_output(cur);

    // Mark KVs as outputs
    for (int il = 0; il < PI0_N_LAYER; il++) {
        ggml_set_output(kvs[il].k);
        ggml_set_output(kvs[il].v);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(graph, cur);
    for (int il = 0; il < PI0_N_LAYER; il++) {
        ggml_build_forward_expand(graph, kvs[il].k);
        ggml_build_forward_expand(graph, kvs[il].v);
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

    // Extract KV to CPU
    for (int il = 0; il < PI0_N_LAYER; il++) {
        g_prefix_k[il].resize(kv_dim * prefix_len);
        g_prefix_v[il].resize(kv_dim * prefix_len);
        ggml_backend_tensor_get(kvs[il].k, g_prefix_k[il].data(), 0, sizeof(float) * kv_dim * prefix_len);
        ggml_backend_tensor_get(kvs[il].v, g_prefix_v[il].data(), 0, sizeof(float) * kv_dim * prefix_len);
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

    // Prefix KV tensors (input data from PaliGemma)
    ggml_tensor * prefix_k_tensors[PI0_N_LAYER];
    ggml_tensor * prefix_v_tensors[PI0_N_LAYER];
    for (int il = 0; il < PI0_N_LAYER; il++) {
        char name[64];
        prefix_k_tensors[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_dim, prefix_len);
        snprintf(name, sizeof(name), "prefix_k_%d", il);
        ggml_set_name(prefix_k_tensors[il], name);
        ggml_set_input(prefix_k_tensors[il]);

        prefix_v_tensors[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_dim, prefix_len);
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
    ggml_tensor * cur = input;

    for (int il = 0; il < PI0_N_LAYER; il++) {
        layer_kv pkv;
        pkv.k = prefix_k_tensors[il];
        pkv.v = prefix_v_tensors[il];

        cur = build_gemma_layer(ctx, cur, m.expert_layers[il], positions, attn_mask,
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

    // Build graph
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
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

    // Set prefix KV data
    for (int il = 0; il < PI0_N_LAYER; il++) {
        ggml_backend_tensor_set(prefix_k_tensors[il], g_prefix_k[il].data(), 0,
            sizeof(float) * kv_dim * prefix_len);
        ggml_backend_tensor_set(prefix_v_tensors[il], g_prefix_v[il].data(), 0,
            sizeof(float) * kv_dim * prefix_len);
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

    // Extract output
    ggml_backend_tensor_get(ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1),
        v_t_out, 0, sizeof(float) * PI0_ACTION_DIM * PI0_ACTION_HORIZON);

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

        // MLP layer 2: W_out @ hidden + b_out, then swish
        float * out = &suffix_out[(t + 1) * EXPERT_N_EMBD]; // +1 to skip state token
        for (int i = 0; i < EXPERT_N_EMBD; i++) {
            float sum = mlp_out_b[i];
            for (int j = 0; j < EXPERT_N_EMBD; j++) {
                sum += mlp_out_w[i * EXPERT_N_EMBD + j] * hidden[j];
            }
            out[i] = sum / (1.0f + expf(-sum));
        }
    }
}

// ---- Main ----

int main(int argc, char ** argv) {
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0 action prediction via flow matching\n\n");
        printf("Usage: %s -m <model_dir> \\\n", argv[0]);
        printf("         --image img1.jpg,img2.jpg,img3.jpg \\\n");
        printf("         -p <instruction>\n\n");
        printf("  -m <dir>   Model directory containing:\n");
        printf("               pi0-gemma-2b.gguf\n");
        printf("               pi0-mmproj.gguf\n");
        printf("               pi0-action-expert.gguf\n");
    };

    // Parse using common params (reuse --image for multiple images)
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    // Treat -m as model directory containing all 3 GGUF files
    std::string model_dir = params.model.path;
    // Strip trailing slash
    while (!model_dir.empty() && model_dir.back() == '/') model_dir.pop_back();

    // If user passed a file path, use its parent directory
    if (model_dir.find(".gguf") != std::string::npos) {
        auto pos = model_dir.rfind('/');
        if (pos != std::string::npos) {
            model_dir = model_dir.substr(0, pos);
        } else {
            model_dir = ".";
        }
    }

    std::string pali_path   = model_dir + "/pi0-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi0-mmproj.gguf";
    std::string expert_path = model_dir + "/pi0-action-expert.gguf";

    // Allow --mmproj override
    if (!params.mmproj.path.empty()) {
        mmproj_path = params.mmproj.path;
    }

    LOG_INF("Model directory: %s\n", model_dir.c_str());
    LOG_INF("  PaliGemma 2B:    %s\n", pali_path.c_str());
    LOG_INF("  Vision mmproj:   %s\n", mmproj_path.c_str());
    LOG_INF("  Action expert:   %s\n", expert_path.c_str());

    // Override params so llama loads the right model for tokenizer
    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    // Initialize backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG_ERR("Failed to init CPU backend\n");
        return 1;
    }

    // Load PI0 model weights
    pi0_model model;
    if (!load_pi0_model(model, pali_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0 model\n");
        return 1;
    }
    LOG_INF("Model loaded successfully\n");

    // Initialize LLM (needed for tokenizer and mtmd compatibility)
    auto llama_init = common_init_from_params(params);
    llama_model   * llm_model = llama_init->model();
    llama_context * llm_ctx   = llama_init->context();

    if (!llm_model || !llm_ctx) {
        LOG_ERR("Failed to load LLM for tokenizer\n");
        return 1;
    }

    // Initialize vision context
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) {
        LOG_ERR("Failed to load vision model\n");
        return 1;
    }

    // Load images (up to 3)
    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", img_path.c_str());
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
        LOG_INF("Loaded image: %s\n", img_path.c_str());
    }

    // Build prompt with image markers
    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) {
        prompt += mtmd_default_marker();
    }
    prompt += params.prompt;
    LOG_INF("Prompt: %s\n", prompt.c_str());

    // Tokenize
    mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &text,
                                bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Failed to tokenize, res = %d\n", res);
        return 1;
    }

    // Build prefix embeddings: encode images via mtmd, look up text tokens from embedding table
    // Load embedding table to CPU once (for text token lookup)
    std::vector<float> embd_table(ggml_nelements(model.pali_embed));
    tensor_to_f32(model.pali_embed, embd_table.data());

    const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    std::vector<float> prefix_embeddings;
    int actual_prefix_len = 0;
    const int n_embd_out = PALI_N_EMBD; // mmproj projects to this dim

    for (int ic = 0; ic < n_chunks; ic++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
        mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // Text tokens - look up embeddings from PaliGemma embedding table
            size_t n_tokens = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);

            for (size_t t = 0; t < n_tokens; t++) {
                llama_token token_id = tokens[t];
                const float * emb = &embd_table[(size_t)token_id * PALI_N_EMBD];
                prefix_embeddings.insert(prefix_embeddings.end(), emb, emb + PALI_N_EMBD);
                actual_prefix_len++;
            }
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // Image chunk - encode through vision model to get embeddings
            if (mtmd_encode_chunk(ctx_vision.get(), chunk) != 0) {
                LOG_ERR("Failed to encode image chunk %d\n", ic);
                return 1;
            }
            float * img_embd = mtmd_get_output_embd(ctx_vision.get());
            size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);

            prefix_embeddings.insert(prefix_embeddings.end(),
                img_embd, img_embd + n_img_tokens * n_embd_out);
            actual_prefix_len += (int)n_img_tokens;
        }
    }

    LOG_INF("Assembled prefix: %d tokens × %d dims\n", actual_prefix_len, PALI_N_EMBD);

    // Run PaliGemma prefix to get KV cache
    LOG_INF("Running PaliGemma prefix pass...\n");
    if (!run_paligemma_prefix_v2(model, backend, prefix_embeddings.data(), actual_prefix_len)) {
        LOG_ERR("Prefix pass failed\n");
        return 1;
    }
    LOG_INF("PaliGemma prefix KV cached (%d tokens)\n", actual_prefix_len);

    // Initialize noisy actions (pure noise at t=1)
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> x_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
    for (auto & v : x_t) v = normal(rng);

    // Robot state (zeros for demo - in production, read from robot)
    std::vector<float> robot_state(PI0_ACTION_DIM, 0.0f);

    // Diffusion loop: denoise from t=1 to t=0
    LOG_INF("\nRunning flow matching diffusion (%d steps)...\n", PI0_NUM_STEPS);
    const float dt = -1.0f / PI0_NUM_STEPS;

    for (int step = 0; step < PI0_NUM_STEPS; step++) {
        float t = 1.0f + step * dt;
        LOG_INF("  Step %d/%d (t=%.2f)\n", step + 1, PI0_NUM_STEPS, t);

        // Prepare suffix embeddings
        std::vector<float> suffix_emb(EXPERT_N_EMBD * (PI0_ACTION_HORIZON + 1));
        prepare_suffix(model, robot_state.data(), x_t.data(), t, suffix_emb.data());

        // Run action expert
        std::vector<float> v_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
        if (!run_expert_step(model, backend, suffix_emb.data(), v_t.data())) {
            LOG_ERR("Expert step %d failed\n", step);
            return 1;
        }

        // Update: x_t = x_t + dt * v_t
        for (int i = 0; i < PI0_ACTION_DIM * PI0_ACTION_HORIZON; i++) {
            x_t[i] += dt * v_t[i];
        }
    }

    // Output final actions
    printf("\n=== PI0 Action Predictions ===\n");
    printf("Instruction: %s\n", params.prompt.c_str());
    printf("Action horizon: %d steps, Action dim: %d\n\n", PI0_ACTION_HORIZON, PI0_ACTION_DIM);

    for (int t = 0; t < PI0_ACTION_HORIZON; t++) {
        printf("  step %2d: [", t);
        for (int d = 0; d < PI0_ACTION_DIM; d++) {
            printf("% .4f", x_t[t * PI0_ACTION_DIM + d]);
            if (d < PI0_ACTION_DIM - 1) printf(", ");
        }
        printf("]\n");
    }

    // Cleanup
    if (model.buf_pali)   ggml_backend_buffer_free(model.buf_pali);
    if (model.buf_expert) ggml_backend_buffer_free(model.buf_expert);
    if (model.ctx_pali)   ggml_free(model.ctx_pali);
    if (model.ctx_expert) ggml_free(model.ctx_expert);
    ggml_backend_free(backend);

    return 0;
}
