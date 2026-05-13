// PI0 unified benchmark — measures per-phase timing across KV-cache modes.
//
// Modes (--kv-mode):
//   fp16             F32 K/V baseline (uncompressed, matches existing pi0-bench)
//   q4_0             Q4_0 K/V (4.5 bpw, standard llama.cpp KV compression)
//   q3j1_trivial     Q3J1_TQ K/V, codebook-only dequant (env Q3J1_TQ_SKIP_QJL=1)
//   q3j1_qjl_scalar  Q3J1_TQ K/V, full QJL via scalar vec_dot (env Q3J1_TQ_DISABLE_SIMD=1)
//   q3j1_simd        Q3J1_TQ K/V, full QJL via AVX2 vec_dot (same workload as qjl_scalar)
//   q3j1_fa          Q3J1_TQ K/V, true-QJL FA path (env Q3J1_TQ_USE_FA=1)
//   q3j1_predeq      Q3J1_TQ K/V, pre-dequant K and V to fp32 once per thread chunk,
//                    then plain fp32 inner dot (env Q3J1_TQ_PRE_DEQUANT_KV=1).
//                    Removes the 408× per-Q-row redundant K/V dequant in _one_chunk.
//
// Pairs to read off the comparison table:
//   q3j1_qjl_scalar vs q3j1_simd    → pure SIMD speedup on identical workload
//   q3j1_qjl_scalar vs q3j1_fa      → algorithmic speedup from the QJL identity
//   q3j1_qjl_scalar vs q3j1_trivial → cost of the 1-bit residual reconstruction
//   q3j1_simd vs q3j1_predeq        → cost of redundant per-Q-row K/V dequant
//
// Usage:
//   llama-pi0-bench-all -m <model_dir> --image img.jpg -p "instruction"
//                       --kv-mode <mode> [-n 100] [--warmup 5]

#include "pi0-bench-all-common.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>

// ---- KV-mode dispatch ----

enum kv_mode {
    KV_MODE_FP16,
    KV_MODE_Q4_0,
    KV_MODE_Q3J1_TRIVIAL,
    KV_MODE_Q3J1_QJL_SCALAR,
    KV_MODE_Q3J1_SIMD,
    KV_MODE_Q3J1_FA,
    KV_MODE_Q3J1_PREDEQ,
};

static const char * kv_mode_name(kv_mode m) {
    switch (m) {
        case KV_MODE_FP16:            return "fp16";
        case KV_MODE_Q4_0:            return "q4_0";
        case KV_MODE_Q3J1_TRIVIAL:    return "q3j1_trivial";
        case KV_MODE_Q3J1_QJL_SCALAR: return "q3j1_qjl_scalar";
        case KV_MODE_Q3J1_SIMD:       return "q3j1_simd";
        case KV_MODE_Q3J1_FA:         return "q3j1_fa";
        case KV_MODE_Q3J1_PREDEQ:     return "q3j1_predeq";
    }
    return "?";
}

static ggml_type kv_mode_to_ggml_type(kv_mode m) {
    switch (m) {
        case KV_MODE_FP16:            return GGML_TYPE_F32; // F32 K/V baseline (label retained for parity with existing README)
        case KV_MODE_Q4_0:            return GGML_TYPE_Q4_0;
        case KV_MODE_Q3J1_TRIVIAL:
        case KV_MODE_Q3J1_QJL_SCALAR:
        case KV_MODE_Q3J1_SIMD:
        case KV_MODE_Q3J1_FA:
        case KV_MODE_Q3J1_PREDEQ:     return GGML_TYPE_Q3J1_TQ;
    }
    return GGML_TYPE_F32;
}

static void apply_kv_mode_env(kv_mode m) {
    // Clear any prior toggles before applying current mode
    unsetenv("Q3J1_TQ_SKIP_QJL");
    unsetenv("Q3J1_TQ_USE_FA");
    unsetenv("Q3J1_TQ_DISABLE_SIMD");
    unsetenv("Q3J1_TQ_PRE_DEQUANT_KV");
    switch (m) {
        case KV_MODE_Q3J1_TRIVIAL:    setenv("Q3J1_TQ_SKIP_QJL",        "1", 1); break;
        case KV_MODE_Q3J1_QJL_SCALAR: setenv("Q3J1_TQ_DISABLE_SIMD",    "1", 1); break;
        case KV_MODE_Q3J1_FA:         setenv("Q3J1_TQ_USE_FA",          "1", 1); break;
        case KV_MODE_Q3J1_PREDEQ:     setenv("Q3J1_TQ_PRE_DEQUANT_KV",  "1", 1); break;
        default: break;
    }
}

static bool parse_kv_mode(const char * s, kv_mode & out) {
    if (strcmp(s, "fp16")            == 0) { out = KV_MODE_FP16;            return true; }
    if (strcmp(s, "q4_0")            == 0) { out = KV_MODE_Q4_0;            return true; }
    if (strcmp(s, "q3j1_trivial")    == 0) { out = KV_MODE_Q3J1_TRIVIAL;    return true; }
    if (strcmp(s, "q3j1_qjl_scalar") == 0) { out = KV_MODE_Q3J1_QJL_SCALAR; return true; }
    if (strcmp(s, "q3j1_simd")       == 0) { out = KV_MODE_Q3J1_SIMD;       return true; }
    if (strcmp(s, "q3j1_fa")         == 0) { out = KV_MODE_Q3J1_FA;         return true; }
    if (strcmp(s, "q3j1_predeq")     == 0) { out = KV_MODE_Q3J1_PREDEQ;     return true; }
    return false;
}

// ---- Timing infrastructure ----

enum Phase {
    PHASE_VISION_ENCODE,
    PHASE_PREFIX_PASS,
    PHASE_SUFFIX_PREP,
    PHASE_EXPERT_STEP,
    PHASE_DIFFUSION_TOTAL,
    PHASE_INFERENCE_TOTAL,
    PHASE_COUNT
};

static const char * phase_names[PHASE_COUNT] = {
    "Vision Encode",
    "Prefix Pass",
    "Suffix Prep (x10)",
    "Expert Step (x10)",
    "Diffusion Total",
    "Inference Total",
};

struct PhaseStats {
    std::vector<double> samples_ms;
    double total() const { return std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0); }
    double avg()   const { return samples_ms.empty() ? 0.0 : total() / (double)samples_ms.size(); }
    double min_val() const { return samples_ms.empty() ? 0.0 : *std::min_element(samples_ms.begin(), samples_ms.end()); }
    double max_val() const { return samples_ms.empty() ? 0.0 : *std::max_element(samples_ms.begin(), samples_ms.end()); }
    double stddev() const {
        if (samples_ms.size() < 2) return 0.0;
        double m = avg(), sum_sq = 0.0;
        for (double v : samples_ms) sum_sq += (v - m) * (v - m);
        return std::sqrt(sum_sq / (double)(samples_ms.size() - 1));
    }
};

static void print_table(PhaseStats stats[PHASE_COUNT], kv_mode mode, int n_iter, int n_warmup) {
    double total_inference = stats[PHASE_INFERENCE_TOTAL].total();

    printf("\n=== PI0 Benchmark [kv-mode=%s] (%d iterations, %d warmup) ===\n\n",
           kv_mode_name(mode), n_iter, n_warmup);
    printf("%-20s | %10s | %8s | %8s | %8s | %8s | %7s\n",
           "Phase", "Total ms", "Avg ms", "Min ms", "Max ms", "Std ms", "% Total");
    printf("---------------------+------------+----------+----------+----------+----------+--------\n");

    for (int i = 0; i < PHASE_COUNT; i++) {
        if (i == PHASE_DIFFUSION_TOTAL) {
            printf("---------------------+------------+----------+----------+----------+----------+--------\n");
        }
        double pct = (total_inference > 0) ? 100.0 * stats[i].total() / total_inference : 0.0;
        printf("%-20s | %10.1f | %8.2f | %8.2f | %8.2f | %8.2f | %5.1f%%\n",
               phase_names[i], stats[i].total(), stats[i].avg(),
               stats[i].min_val(), stats[i].max_val(), stats[i].stddev(), pct);
    }

    printf("\n");
    if (total_inference > 0) {
        printf("Throughput: %.2f inferences/sec\n", n_iter / (total_inference / 1000.0));
    }

    // Single-line summary tag for wrapper-script aggregation
    printf("BENCH_RESULT mode=%s avg_ms=%.2f expert_step_avg_ms=%.2f\n",
           kv_mode_name(mode),
           stats[PHASE_INFERENCE_TOTAL].avg(),
           stats[PHASE_EXPERT_STEP].avg());
}

// ---- Custom arg parsing ----

struct bench_params {
    int     n_iter   = 100;
    int     n_warmup = 5;
    kv_mode mode     = KV_MODE_FP16;
};

static bench_params strip_bench_args(int & argc, char ** argv) {
    bench_params bp;
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--bench-n") == 0) && i + 1 < argc) {
            bp.n_iter = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            bp.n_warmup = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kv-mode") == 0 && i + 1 < argc) {
            const char * mode_str = argv[++i];
            if (!parse_kv_mode(mode_str, bp.mode)) {
                fprintf(stderr, "Unknown --kv-mode: %s (valid: fp16, q4_0, q3j1_trivial, q3j1_qjl_scalar, q3j1_simd, q3j1_fa, q3j1_predeq)\n", mode_str);
                exit(1);
            }
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return bp;
}

// ---- Main ----

int main(int argc, char ** argv) {
    bench_params bp = strip_bench_args(argc, argv);
    common_params params;

    // Apply env-var toggles before any ggml init so the quant fns pick them up.
    apply_kv_mode_env(bp.mode);
    const ggml_type kv_type = kv_mode_to_ggml_type(bp.mode);

    auto show_help = [](int, char ** argv) {
        printf("PI0 unified benchmark — KV-cache mode comparison\n\n");
        printf("Usage: %s -m <model_dir> --image img1.jpg,img2.jpg,img3.jpg \\\n", argv[0]);
        printf("         -p <instruction> --kv-mode <mode> [-n 100] [--warmup 5]\n\n");
        printf("  --kv-mode <m>  fp16 | q4_0 | q3j1_trivial | q3j1_qjl_scalar | q3j1_simd | q3j1_fa | q3j1_predeq\n");
        printf("  -n <int>       Number of benchmark iterations (default: 100)\n");
        printf("  --warmup <int> Warmup iterations (default: 5)\n");
        printf("\nNote: use comma-separated paths for --image (not repeated flags)\n");
    };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    std::string model_dir = resolve_model_dir(params.model.path);
    std::string pali_path   = model_dir + "/pi0-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi0-mmproj.gguf";
    std::string expert_path = model_dir + "/pi0-action-expert.gguf";

    if (!params.mmproj.path.empty()) mmproj_path = params.mmproj.path;
    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    LOG_INF("Model directory: %s\n", model_dir.c_str());
    LOG_INF("KV mode: %s (ggml_type=%s)\n", kv_mode_name(bp.mode), ggml_type_name(kv_type));
    LOG_INF("Benchmark: %d iterations + %d warmup\n", bp.n_iter, bp.n_warmup);

    // ---- Initialize (one-time) ----

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { LOG_ERR("Failed to init CPU backend\n"); return 1; }

    pi0_model model;
    if (!load_pi0_model(model, pali_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0 model\n");
        return 1;
    }

    auto llama_init = common_init_from_params(params);
    llama_model   * llm_model = llama_init->model();
    llama_context * llm_ctx   = llama_init->context();
    if (!llm_model || !llm_ctx) { LOG_ERR("Failed to load LLM for tokenizer\n"); return 1; }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) { LOG_ERR("Failed to load vision model\n"); return 1; }

    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) { LOG_ERR("Failed to load image: %s\n", img_path.c_str()); return 1; }
        bitmaps.entries.push_back(std::move(bmp));
    }

    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) prompt += mtmd_default_marker();
    prompt += params.prompt;

    mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    if (mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &text,
                      bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        LOG_ERR("Failed to tokenize\n");
        return 1;
    }

    std::vector<float> embd_table(ggml_nelements(model.pali_embed));
    tensor_to_f32(model.pali_embed, embd_table.data());

    const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    std::vector<float> robot_state(PI0_ACTION_DIM, 0.0f);
    const float dt = -1.0f / PI0_NUM_STEPS;

    LOG_INF("Starting benchmark...\n");

    // ---- Benchmark loop ----

    PhaseStats stats[PHASE_COUNT];
    const int total_runs = bp.n_warmup + bp.n_iter;

    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < bp.n_warmup);
        if (!is_warmup && run == bp.n_warmup) {
            LOG_INF("Warmup done. Measuring %d iterations...\n", bp.n_iter);
        }

        int64_t t_iter_start = ggml_time_us();
        int64_t t0 = ggml_time_us();

        std::vector<float> prefix_embeddings;
        int actual_prefix_len = 0;

        for (int ic = 0; ic < n_chunks; ic++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
            mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                for (size_t t = 0; t < n_tokens; t++) {
                    const float * emb = &embd_table[(size_t)tokens[t] * PALI_N_EMBD];
                    prefix_embeddings.insert(prefix_embeddings.end(), emb, emb + PALI_N_EMBD);
                    actual_prefix_len++;
                }
            } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (mtmd_encode_chunk(ctx_vision.get(), chunk) != 0) {
                    LOG_ERR("Vision encode failed at run %d\n", run); return 1;
                }
                float * img_embd = mtmd_get_output_embd(ctx_vision.get());
                size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);
                prefix_embeddings.insert(prefix_embeddings.end(),
                    img_embd, img_embd + n_img_tokens * PALI_N_EMBD);
                actual_prefix_len += (int)n_img_tokens;
            }
        }

        int64_t t1 = ggml_time_us();

        if (!run_paligemma_prefix_v2(model, backend, prefix_embeddings.data(), actual_prefix_len, kv_type)) {
            LOG_ERR("Prefix pass failed at run %d\n", run); return 1;
        }
        int64_t t2 = ggml_time_us();

        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> x_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
        for (auto & v : x_t) v = normal(rng);

        double suffix_prep_ms = 0.0;
        double expert_step_ms = 0.0;

        for (int step = 0; step < PI0_NUM_STEPS; step++) {
            float t = 1.0f + step * dt;

            int64_t ts0 = ggml_time_us();
            std::vector<float> suffix_emb(EXPERT_N_EMBD * (PI0_ACTION_HORIZON + 1));
            prepare_suffix(model, robot_state.data(), x_t.data(), t, suffix_emb.data());
            int64_t ts1 = ggml_time_us();
            suffix_prep_ms += (ts1 - ts0) / 1000.0;

            std::vector<float> v_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
            if (!run_expert_step(model, backend, suffix_emb.data(), v_t.data())) {
                LOG_ERR("Expert step failed at run %d step %d\n", run, step); return 1;
            }
            int64_t ts2 = ggml_time_us();
            expert_step_ms += (ts2 - ts1) / 1000.0;

            for (int i = 0; i < PI0_ACTION_DIM * PI0_ACTION_HORIZON; i++) {
                x_t[i] += dt * v_t[i];
            }
        }

        int64_t t3 = ggml_time_us();

        if (!is_warmup) {
            double vision_ms = (t1 - t0) / 1000.0;
            double prefix_ms = (t2 - t1) / 1000.0;
            double diff_ms   = (t3 - t2) / 1000.0;
            double total_ms  = (t3 - t_iter_start) / 1000.0;

            stats[PHASE_VISION_ENCODE].samples_ms.push_back(vision_ms);
            stats[PHASE_PREFIX_PASS].samples_ms.push_back(prefix_ms);
            stats[PHASE_SUFFIX_PREP].samples_ms.push_back(suffix_prep_ms);
            stats[PHASE_EXPERT_STEP].samples_ms.push_back(expert_step_ms);
            stats[PHASE_DIFFUSION_TOTAL].samples_ms.push_back(diff_ms);
            stats[PHASE_INFERENCE_TOTAL].samples_ms.push_back(total_ms);

            if ((run - bp.n_warmup + 1) % 10 == 0 || run == total_runs - 1) {
                LOG_INF("  [%d/%d] iter %.1f ms (vision %.1f, prefix %.1f, diffusion %.1f)\n",
                    run - bp.n_warmup + 1, bp.n_iter,
                    total_ms, vision_ms, prefix_ms, diff_ms);
            }
        }
    }

    print_table(stats, bp.mode, bp.n_iter, bp.n_warmup);

    if (model.buf_pali)   ggml_backend_buffer_free(model.buf_pali);
    if (model.buf_expert) ggml_backend_buffer_free(model.buf_expert);
    if (model.ctx_pali)   ggml_free(model.ctx_pali);
    if (model.ctx_expert) ggml_free(model.ctx_expert);
    ggml_backend_free(backend);

    return 0;
}
