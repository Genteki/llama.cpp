#pragma once

// PI0 shared types and API.
// Implementation: pi0-common.cpp

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

// ---- Constants ----

static constexpr int PI0_ACTION_DIM     = 32;
static constexpr int PI0_ACTION_HORIZON = 50;
static constexpr int PI0_N_LAYER        = 18;
static constexpr int PI0_NUM_STEPS      = 10;

// PaliGemma 2B
static constexpr int PALI_N_EMBD    = 2048;
static constexpr int PALI_N_HEAD    = 8;
static constexpr int PALI_N_KV_HEAD = 1;
static constexpr int PALI_HEAD_DIM  = 256;
static constexpr int PALI_N_FF      = 16384;

// Action Expert 300M
static constexpr int EXPERT_N_EMBD    = 1024;
static constexpr int EXPERT_N_HEAD    = 8;
static constexpr int EXPERT_N_KV_HEAD = 1;
static constexpr int EXPERT_HEAD_DIM  = 256;
static constexpr int EXPERT_N_FF      = 4096;

// ---- Weight structures ----

struct gemma_layer_weights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
    ggml_tensor * q_proj    = nullptr;
    ggml_tensor * k_proj    = nullptr;
    ggml_tensor * v_proj    = nullptr;
    ggml_tensor * qkv_proj  = nullptr; // fused [in, n_head*hd + 2*kv_dim]; if set, used instead of q/k/v
    ggml_tensor * o_proj    = nullptr;
    ggml_tensor * gate_proj = nullptr;
    ggml_tensor * up_proj   = nullptr;
    ggml_tensor * down_proj = nullptr;
};

struct pi0_model {
    // PaliGemma 2B
    ggml_tensor * pali_embed = nullptr;
    gemma_layer_weights pali_layers[PI0_N_LAYER] = {};
    ggml_tensor * pali_final_norm = nullptr;

    // Action Expert 300M
    gemma_layer_weights expert_layers[PI0_N_LAYER] = {};
    ggml_tensor * expert_final_norm = nullptr;

    // Action projections (may be quantized — read only via ggml_mul_mat in graphs)
    ggml_tensor * action_in_proj_w      = nullptr;
    ggml_tensor * action_in_proj_b      = nullptr;
    ggml_tensor * action_out_proj_w     = nullptr;
    ggml_tensor * action_out_proj_b     = nullptr;
    ggml_tensor * state_proj_w          = nullptr;
    ggml_tensor * state_proj_b          = nullptr;
    ggml_tensor * action_time_mlp_in_w  = nullptr;
    ggml_tensor * action_time_mlp_in_b  = nullptr;
    ggml_tensor * action_time_mlp_out_w = nullptr;
    ggml_tensor * action_time_mlp_out_b = nullptr;

    // Backend resources
    ggml_context * ctx_pali   = nullptr;
    ggml_context * ctx_expert = nullptr;
    ggml_backend_buffer_t buf_pali   = nullptr;
    ggml_backend_buffer_t buf_expert = nullptr;
};

// ---- Per-inference runtime state ----
//
// Owns:
//   * the per-layer K/V cache, backend-resident, in a configurable quant type
//     (kv_type) — this is the second benchmark sweep axis after --backend;
//   * the persistent expert ggml graph reused across the 10 diffusion steps.
//
// Lifecycle:
//   1. Set kv_type (default F32). Then default-construct (zero-init).
//   2. run_paligemma_prefix(): allocates / resizes the KV cache as needed and
//      writes the prefix portion of cache_k/v (positions 0..prefix_len-1).
//      F32→kv_type conversion happens inside the graph via ggml_cpy.
//   3. run_expert_step(): writes the suffix portion (positions prefix_len..
//      prefix_len+suffix_len-1) and reads the full cache for attention. Graph
//      is built lazily and rebuilt only if prefix_len or kv_type changes.
//   4. pi0_session_free() to release the cache buffer + graph.

struct pi0_session {
    // KV cache configuration. Set BEFORE run_paligemma_prefix.
    ggml_type kv_type = GGML_TYPE_F32;

    int prefix_len = 0; // length of the prefix written by run_paligemma_prefix

    // Persistent KV cache. Each cache_k/v[il] has shape [kv_dim, prefix_len + suffix_len]
    // in `kv_type`, where suffix_len = PI0_ACTION_HORIZON + 1. The prefix portion is
    // written once per inference; the suffix portion is overwritten every diffusion step.
    int kv_alloc_prefix_len = 0;
    ggml_type kv_alloc_type = GGML_TYPE_F32;
    ggml_context        * kv_ctx           = nullptr;
    ggml_backend_buffer_t kv_buf           = nullptr;
    ggml_tensor         * cache_k[PI0_N_LAYER] = {};
    ggml_tensor         * cache_v[PI0_N_LAYER] = {};

    // Persistent expert graph (rebuilt if prefix_len or kv_type changes).
    int       graph_prefix_len = 0;
    ggml_type graph_kv_type    = GGML_TYPE_F32;
    ggml_context  * graph_ctx   = nullptr;
    ggml_cgraph   * graph       = nullptr;
    ggml_gallocr_t  graph_alloc = nullptr;

    ggml_tensor * in_state     = nullptr;
    ggml_tensor * in_noisy     = nullptr;
    ggml_tensor * in_time      = nullptr;
    ggml_tensor * in_positions = nullptr;
    ggml_tensor * in_attn_mask = nullptr;
    ggml_tensor * out_v_t      = nullptr;
};

void pi0_session_free(pi0_session & s);

// Parse a ggml_type from a CLI string ("f32", "f16", "q8_0", "q4_0", ...).
// Returns GGML_TYPE_COUNT on unrecognized input.
ggml_type pi0_ggml_type_from_string(const std::string & s);

// ---- API ----

bool load_pi0_model(pi0_model & m,
                    const char * pali_path,
                    const char * expert_path,
                    ggml_backend_t backend);

void free_pi0_model(pi0_model & m);

// PaliGemma prefix pass. Allocates / resizes s.cache_k/v as needed and
// writes the prefix portion (positions 0..prefix_len-1) on the backend.
bool run_paligemma_prefix(pi0_model & m,
                          pi0_session & s,
                          ggml_backend_t backend,
                          const float * embeddings, // [PALI_N_EMBD * prefix_len]
                          int prefix_len);

// One Action Expert denoising step. Suffix prep (state_proj, action_in_proj,
// time MLP) runs inside the graph, so quantized expert weights work natively.
bool run_expert_step(pi0_model & m,
                     pi0_session & s,
                     ggml_backend_t backend,
                     const float * state,         // [PI0_ACTION_DIM]
                     const float * noisy_actions, // [PI0_ACTION_DIM * PI0_ACTION_HORIZON]
                     float timestep,
                     float * v_t_out);            // [PI0_ACTION_DIM * PI0_ACTION_HORIZON]

// Dequantize a single embedding row (token_id) into out_f32 (length t->ne[0]).
// Works for F32, F16, and any quantized type registered with ggml_get_type_traits.
void embd_lookup_f32(ggml_tensor * t, int token_id, float * out_f32);

// ---- PI0 Q8_0 Row-Tile overlay (Phase 2D) ----
//
// Load a .rt.bin overlay produced by dp8-spike/quantize_rt.py and attach the
// repacked weights to the OpenCL backend so subsequent ggml_mul_mat goes
// through the dp8 path. No-op (returns false) if backend is not OpenCL or the
// device lacks cl_qcom_dot_product8.
//
// Tensor lookup uses gguf names (e.g. "blk.7.attn_q.weight") against
// m.ctx_expert, so only Action Expert weights are overlaid.
bool pi0_attach_rt_overlay(pi0_model & m, ggml_backend_t backend, const std::string & path);

std::string resolve_model_dir(const std::string & path);

// ---- Backend selection ----
//
// Resolve a backend by name: "cpu", "auto"/"gpu" (first available GPU), or a
// concrete name like "OpenCL", "Vulkan", "Metal", "CUDA". Returns nullptr if
// the requested backend is unavailable (use pi0_dump_backends() to see what is).

ggml_backend_t pi0_init_backend(const std::string & name);

// List available backend devices to stderr.
void pi0_dump_backends();

// ---- Reporting ----
//
// Print a per-context histogram of tensor types (F16, Q4_0, Q4_K, ...) and the
// total weight footprint. Useful at the top of a benchmark run to label which
// quantization is actually being measured.

void pi0_dump_tensor_types(const pi0_model & m);

// ---- Shared CLI flag stripping ----
//
// Strips pi0-wide flags ("--backend NAME") from argv before common_params_parse
// sees them. Mirrors the bench's existing strip_bench_args helper.

struct pi0_cli {
    std::string backend = "cpu";
    std::string kv_type = "f32"; // see pi0_ggml_type_from_string
    std::string rt_bin  = "";    // optional Phase-2D Q8_0_RT overlay (.rt.bin)
};

pi0_cli pi0_strip_cli_args(int & argc, char ** argv);
