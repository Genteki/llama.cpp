#pragma once

// PI0.5 shared types and API.
// Implementation: pi0_5-common.cpp
//
// PI0.5 differs from PI0 in the *action expert* only (the SigLIP vision tower
// and the PaliGemma 2B prefix are byte-for-byte the pi0 path):
//   * No state token in the suffix — suffix = ACTION_HORIZON action tokens only
//     (state is folded into the discrete prefix text tokens upstream).
//   * Timestep is injected via adaptive RMSNorm (adaRMS): every expert norm
//     (input / post-attention / final) owns a Linear(cond_dim -> 3*width)
//     "dense" producing (scale, shift, gate); normed = rms(x)*(1+scale)+shift,
//     and the residual is gated as  x + y*gate.
//   * The conditioning vector is  time_mlp_out(swish(time_mlp_in(sincos(t)))),
//     swished — replacing pi0's action_time_mlp concat path.
//
// Default precision: f16 activations + f16 KV cache; weights per-tensor
// (q4_0 / q8_0 / f16) via the convert/quantize --tensor-type machinery.

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

// ---- Constants ----

static constexpr int PI05_ACTION_DIM     = 32;
static constexpr int PI05_ACTION_HORIZON = 50;
static constexpr int PI05_N_LAYER        = 18;
static constexpr int PI05_NUM_STEPS      = 10;

// Suffix is ACTION_HORIZON tokens — NO leading state token (the pi0.5 change).
static constexpr int PI05_SUFFIX_LEN     = PI05_ACTION_HORIZON; // 50

// PaliGemma 2B (prefix) — identical to pi0.
static constexpr int PALI_N_EMBD    = 2048;
static constexpr int PALI_N_HEAD    = 8;
static constexpr int PALI_N_KV_HEAD = 1;
static constexpr int PALI_HEAD_DIM  = 256;
static constexpr int PALI_N_FF      = 16384;

// Action Expert 300M (suffix).
static constexpr int EXPERT_N_EMBD    = 1024;
static constexpr int EXPERT_N_HEAD    = 8;
static constexpr int EXPERT_N_KV_HEAD = 1;
static constexpr int EXPERT_HEAD_DIM  = 256;
static constexpr int EXPERT_N_FF      = 4096;

// adaRMS: conditioning dim == expert width; modulation = 3*width (scale|shift|gate).
static constexpr int ADARMS_COND_DIM = EXPERT_N_EMBD;       // 1024
static constexpr int ADARMS_MOD_DIM  = 3 * EXPERT_N_EMBD;   // 3072

// ---- Weight structures ----
//
// One struct serves both experts. PaliGemma prefix layers populate the plain
// `attn_norm`/`ffn_norm` scale vectors (adaRMS dense pointers stay null). pi0.5
// action-expert layers populate the adaRMS `*_dense_*` pointers instead (the
// plain scale vectors stay null). build_gemma_layer dispatches on which is set.

struct gemma_layer_weights {
    // Plain Gemma RMSNorm scale (+1 baked at convert time). PaliGemma prefix only.
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * ffn_norm  = nullptr;

    // adaRMS modulation Dense (pi0.5 expert). dense_w: [cond_dim, 3*width].
    ggml_tensor * attn_norm_dense_w = nullptr;
    ggml_tensor * attn_norm_dense_b = nullptr;
    ggml_tensor * ffn_norm_dense_w  = nullptr;
    ggml_tensor * ffn_norm_dense_b  = nullptr;

    ggml_tensor * q_proj    = nullptr;
    ggml_tensor * k_proj    = nullptr;
    ggml_tensor * v_proj    = nullptr;
    ggml_tensor * qkv_proj  = nullptr; // fused [in, n_head*hd + 2*kv_dim]; if set, used instead of q/k/v
    ggml_tensor * o_proj    = nullptr;
    ggml_tensor * gate_proj = nullptr;
    ggml_tensor * up_proj   = nullptr;
    ggml_tensor * down_proj = nullptr;
};

struct pi05_model {
    // PaliGemma 2B (prefix)
    ggml_tensor * pali_embed = nullptr;
    gemma_layer_weights pali_layers[PI05_N_LAYER] = {};
    ggml_tensor * pali_final_norm = nullptr; // plain RMSNorm scale

    // Action Expert 300M (adaRMS)
    gemma_layer_weights expert_layers[PI05_N_LAYER] = {};
    ggml_tensor * expert_final_norm_dense_w = nullptr; // adaRMS final norm [cond_dim, 3*width]
    ggml_tensor * expert_final_norm_dense_b = nullptr;

    // Action + time projections.
    ggml_tensor * action_in_proj_w  = nullptr;
    ggml_tensor * action_in_proj_b  = nullptr;
    ggml_tensor * action_out_proj_w = nullptr;
    ggml_tensor * action_out_proj_b = nullptr;
    ggml_tensor * time_mlp_in_w     = nullptr;
    ggml_tensor * time_mlp_in_b     = nullptr;
    ggml_tensor * time_mlp_out_w    = nullptr;
    ggml_tensor * time_mlp_out_b    = nullptr;

    // Backend resources
    ggml_context * ctx_pali   = nullptr;
    ggml_context * ctx_expert = nullptr;
    ggml_backend_buffer_t buf_pali   = nullptr;
    ggml_backend_buffer_t buf_expert = nullptr;
};

// ---- Per-inference runtime state ----
//
// Owns the backend-resident per-layer K/V cache (configurable kv_type) and the
// persistent expert graph reused across the 10 diffusion steps. Lifecycle
// mirrors pi0: set kv_type/act_type, run_paligemma_prefix() fills the prefix KV,
// run_expert_step() writes the suffix KV and reads the full cache per step.

struct pi05_session {
    // Set BEFORE run_paligemma_prefix.
    ggml_type kv_type  = GGML_TYPE_F16; // KV cache element type (default f16)
    ggml_type act_type = GGML_TYPE_F16; // activation dtype fed into matmuls (default f16)

    int prefix_len = 0;

    // Persistent KV cache. cache_k/v[il] = [kv_dim, prefix_len + PI05_SUFFIX_LEN].
    int kv_alloc_prefix_len = 0;
    ggml_type kv_alloc_type = GGML_TYPE_F16;
    ggml_context        * kv_ctx               = nullptr;
    ggml_backend_buffer_t kv_buf               = nullptr;
    ggml_tensor         * cache_k[PI05_N_LAYER] = {};
    ggml_tensor         * cache_v[PI05_N_LAYER] = {};

    // Persistent expert graph (rebuilt if prefix_len / kv_type / act_type changes).
    int       graph_prefix_len = 0;
    ggml_type graph_kv_type    = GGML_TYPE_F16;
    ggml_type graph_act_type   = GGML_TYPE_F16;
    ggml_context  * graph_ctx   = nullptr;
    ggml_cgraph   * graph       = nullptr;
    ggml_gallocr_t  graph_alloc = nullptr;

    ggml_tensor * in_noisy     = nullptr; // [PI05_ACTION_DIM, PI05_ACTION_HORIZON]
    ggml_tensor * in_time      = nullptr; // [EXPERT_N_EMBD, 1]  (sincos(t), pre-MLP)
    ggml_tensor * in_positions = nullptr; // [PI05_SUFFIX_LEN]
    ggml_tensor * in_attn_mask = nullptr; // [prefix_len + PI05_SUFFIX_LEN, PI05_SUFFIX_LEN]
    ggml_tensor * out_v_t      = nullptr; // [PI05_ACTION_DIM, PI05_ACTION_HORIZON]
};

void pi05_session_free(pi05_session & s);

// Parse a ggml_type from a CLI string ("f32", "f16", "q8_0", "q4_0", ...).
// Returns GGML_TYPE_COUNT on unrecognized input.
ggml_type pi05_ggml_type_from_string(const std::string & s);

// ---- API ----

bool load_pi05_model(pi05_model & m,
                     const char * pali_path,
                     const char * expert_path,
                     ggml_backend_t backend);

void free_pi05_model(pi05_model & m);

// PaliGemma prefix pass — identical math to pi0 (plain RMSNorm, bidirectional).
// Allocates / resizes s.cache_k/v and writes prefix rows [0..prefix_len).
bool run_paligemma_prefix(pi05_model & m,
                          pi05_session & s,
                          ggml_backend_t backend,
                          const float * embeddings, // [PALI_N_EMBD * prefix_len]
                          int prefix_len);

// One Action Expert (adaRMS) denoising step. action_in_proj + time MLP + adaRMS
// all run inside the graph, so quantized expert weights work natively.
bool run_expert_step(pi05_model & m,
                     pi05_session & s,
                     ggml_backend_t backend,
                     const float * noisy_actions, // [PI05_ACTION_DIM * PI05_ACTION_HORIZON]
                     float timestep,
                     float * v_t_out);            // [PI05_ACTION_DIM * PI05_ACTION_HORIZON]

// Dequantize a single embedding row (token_id) into out_f32 (length t->ne[0]).
void embd_lookup_f32(ggml_tensor * t, int token_id, float * out_f32);

std::string resolve_model_dir(const std::string & path);

// ---- Backend selection ----

ggml_backend_t pi05_init_backend(const std::string & name);
void pi05_dump_backends();

// ---- Reporting ----

void pi05_dump_tensor_types(const pi05_model & m);

// ---- Shared CLI flag stripping ----
//
// Strips pi0.5-wide flags before common_params_parse sees them.

struct pi05_cli {
    std::string backend  = "cpu";
    std::string kv_type  = "f16";   // see pi05_ggml_type_from_string
    std::string act_type = "auto";  // "auto" => f16 on GPU, f32 on CPU
};

pi05_cli pi05_strip_cli_args(int & argc, char ** argv);

// Resolve the activation dtype, honoring "auto": f16 on a GPU backend (the
// deploy target / default precision) but f32 on CPU, where mul_mat cannot take
// f16 activations for quantized weights. Returns GGML_TYPE_COUNT if invalid.
ggml_type pi05_resolve_act_type(const std::string & act_type, const std::string & backend);
