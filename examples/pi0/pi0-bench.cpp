// PI0 benchmark — measures per-phase timing across multiple inference iterations.
//
// Usage:
//   llama-pi0-bench -m <model_dir> --image img.jpg -p "instruction" [-n 100] [--warmup 5]

#include "pi0-common.h"

#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---- Timing infrastructure ----

enum Phase {
    PHASE_VISION_ENCODE,
    PHASE_PREFIX_PASS,
    PHASE_EXPERT_STEP,
    PHASE_DIFFUSION_TOTAL,
    PHASE_INFERENCE_TOTAL,
    PHASE_COUNT
};

static const char * phase_names[PHASE_COUNT] = {
    "Vision Encode",
    "Prefix Pass",
    "Expert Step (x10)", // suffix prep is folded into the expert graph
    "Diffusion Total",
    "Inference Total",
};

struct PhaseStats {
    std::vector<double> samples_ms;

    double total() const {
        return std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    }
    double avg() const {
        return samples_ms.empty() ? 0.0 : total() / (double)samples_ms.size();
    }
    double min_val() const {
        return samples_ms.empty() ? 0.0 : *std::min_element(samples_ms.begin(), samples_ms.end());
    }
    double max_val() const {
        return samples_ms.empty() ? 0.0 : *std::max_element(samples_ms.begin(), samples_ms.end());
    }
    double stddev() const {
        if (samples_ms.size() < 2) return 0.0;
        double m = avg();
        double sum_sq = 0.0;
        for (double v : samples_ms) sum_sq += (v - m) * (v - m);
        return std::sqrt(sum_sq / (double)(samples_ms.size() - 1));
    }
};

static void print_table(PhaseStats stats[PHASE_COUNT], int n_iter, int n_warmup) {
    double total_inference = stats[PHASE_INFERENCE_TOTAL].total();

    printf("\n=== PI0 Benchmark (%d iterations, %d warmup) ===\n\n", n_iter, n_warmup);
    printf("%-20s | %10s | %8s | %8s | %8s | %8s | %7s\n",
           "Phase", "Total ms", "Avg ms", "Min ms", "Max ms", "Std ms", "% Total");
    printf("---------------------+------------+----------+----------+----------+----------+--------\n");

    for (int i = 0; i < PHASE_COUNT; i++) {
        if (i == PHASE_DIFFUSION_TOTAL) {
            printf("---------------------+------------+----------+----------+----------+----------+--------\n");
        }
        double pct = (total_inference > 0) ? 100.0 * stats[i].total() / total_inference : 0.0;
        printf("%-20s | %10.1f | %8.2f | %8.2f | %8.2f | %8.2f | %5.1f%%\n",
               phase_names[i],
               stats[i].total(),
               stats[i].avg(),
               stats[i].min_val(),
               stats[i].max_val(),
               stats[i].stddev(),
               pct);
    }

    printf("\n");
    if (total_inference > 0) {
        printf("Throughput: %.2f inferences/sec\n", n_iter / (total_inference / 1000.0));
    }
}

// ---- Custom arg parsing ----
// Strip bench-specific args (-n, --warmup) from argv before passing to common_params_parse,
// which would reject unknown arguments.

struct bench_params {
    int  n_iter      = 100;
    int  n_warmup    = 5;
    bool prefix_only = false; // skip the diffusion loop so the OpenCL per-kernel
                              // summary isolates the prefix attention GEMMs
};

static bench_params strip_bench_args(int & argc, char ** argv) {
    bench_params bp;
    int out = 1; // keep argv[0]
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--bench-n") == 0) && i + 1 < argc) {
            bp.n_iter = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            bp.n_warmup = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefix-only") == 0) {
            bp.prefix_only = true;
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return bp;
}

// ---- Main ----

int main(int argc, char ** argv) {
    // Strip shared pi0 flags (--backend) first, then bench-specific ones.
    pi0_cli      cli = pi0_strip_cli_args(argc, argv);
    bench_params bp  = strip_bench_args(argc, argv);
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0 benchmark — per-phase timing\n\n");
        printf("Usage: %s -m <model_dir> --image img1.jpg,img2.jpg,img3.jpg \\\n", argv[0]);
        printf("         -p <instruction> [-n 100] [--warmup 5] [--backend cpu|auto|<name>]\n");
        printf("         [--kv-type f32|f16|q8_0|q4_0]\n\n");
        printf("  -n <int>       Number of benchmark iterations (default: 100)\n");
        printf("  --warmup <int> Warmup iterations (default: 5)\n");
        printf("  --backend      Backend: 'cpu' (default), 'auto'/'gpu', or a concrete name.\n");
        printf("  --kv-type      KV cache element type: f32 (default), f16, q8_0, q4_0, ...\n");
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

    // Fused-QKV PaliGemma variant feeds only the pi0 loader; llama.cpp's gemma
    // loader (tokenizer) still needs the unfused pi0-gemma-2b.gguf.
    std::string pali_pi0_path = model_dir + "/pi0-gemma-2b-fused.gguf";
    if (FILE * f = fopen(pali_pi0_path.c_str(), "rb")) { fclose(f); } else { pali_pi0_path = pali_path; }

    if (!params.mmproj.path.empty()) {
        mmproj_path = params.mmproj.path;
    }

    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    // Mirror --backend cpu onto llama's own loader so it doesn't auto-offload
    // the tokenizer model to a GPU that can't hold it (Adreno OpenCL alloc cap).
    if (cli.backend == "cpu" || cli.backend == "CPU") {
        params.n_gpu_layers   = 0;
        params.mmproj_use_gpu = false;
    }

    LOG_INF("Model directory: %s\n", model_dir.c_str());
    LOG_INF("Benchmark: %d iterations + %d warmup\n", bp.n_iter, bp.n_warmup);

    // ---- Initialize (one-time) ----

    ggml_backend_t backend = pi0_init_backend(cli.backend);
    if (!backend) {
        LOG_ERR("Failed to init backend '%s'\n", cli.backend.c_str());
        return 1;
    }

    pi0_model model;
    if (!load_pi0_model(model, pali_pi0_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0 model\n");
        return 1;
    }
    pi0_dump_tensor_types(model);

    // Phase-2D: optional Q8_0 Row-Tile overlay(s) for the Adreno dp8 path.
    // Accepts comma-separated paths so one invocation can cover both
    // Action Expert and PaliGemma backbone (separate .rt.bin files).
    if (!cli.rt_bin.empty()) {
        size_t start = 0;
        while (start < cli.rt_bin.size()) {
            size_t comma = cli.rt_bin.find(',', start);
            std::string path = cli.rt_bin.substr(start, comma - start);
            start = (comma == std::string::npos) ? cli.rt_bin.size() : comma + 1;
            if (path.empty()) continue;
            LOG_INF("Attaching Q8_0_RT overlay from %s\n", path.c_str());
            if (!pi0_attach_rt_overlay(model, backend, path)) {
                LOG_WRN("RT overlay attach failed for %s; continuing\n", path.c_str());
            }
        }
    }

    auto llama_init = common_init_from_params(params);
    llama_model   * llm_model = llama_init->model();
    llama_context * llm_ctx   = llama_init->context();
    if (!llm_model || !llm_ctx) {
        LOG_ERR("Failed to load LLM for tokenizer\n");
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) {
        LOG_ERR("Failed to load vision model\n");
        return 1;
    }

    // Load images
    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", img_path.c_str());
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
    }

    // Build prompt and tokenize
    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) {
        prompt += mtmd_default_marker();
    }
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

    // Per-row dequant via embd_lookup_f32 — no full-table dequant needed.
    const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    std::vector<float> emb_row(PALI_N_EMBD);

    // Per-inference state (KV cache lives on the backend; reused across iterations).
    pi0_session session;
    session.kv_type = pi0_ggml_type_from_string(cli.kv_type);
    if (session.kv_type == GGML_TYPE_COUNT) {
        LOG_ERR("Unknown --kv-type '%s' (try f32, f16, q8_0, q4_0)\n", cli.kv_type.c_str());
        return 1;
    }
    LOG_INF("KV cache type: %s\n", ggml_type_name(session.kv_type));

    // Robot state (zeros)
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

        // ---- Phase: Vision Encode ----
        int64_t t0 = ggml_time_us();

        std::vector<float> prefix_embeddings;
        int actual_prefix_len = 0;

        for (int ic = 0; ic < n_chunks; ic++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
            mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                // Gemma scales text token embeddings by sqrt(hidden); image
                // features from SigLIP stay unscaled (PaliGemma convention).
                const float embd_scale = sqrtf((float) PALI_N_EMBD);
                for (size_t t = 0; t < n_tokens; t++) {
                    embd_lookup_f32(model.pali_embed, tokens[t], emb_row.data());
                    for (float & e : emb_row) e *= embd_scale;
                    prefix_embeddings.insert(prefix_embeddings.end(), emb_row.begin(), emb_row.end());
                    actual_prefix_len++;
                }
            } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (mtmd_encode_chunk(ctx_vision.get(), chunk) != 0) {
                    LOG_ERR("Vision encode failed at run %d\n", run);
                    return 1;
                }
                float * img_embd = mtmd_get_output_embd(ctx_vision.get());
                size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);
                prefix_embeddings.insert(prefix_embeddings.end(),
                    img_embd, img_embd + n_img_tokens * PALI_N_EMBD);
                actual_prefix_len += (int)n_img_tokens;
            }
        }

        int64_t t1 = ggml_time_us();

        // ---- Phase: Prefix Pass ----
        if (!run_paligemma_prefix(model, session, backend, prefix_embeddings.data(), actual_prefix_len)) {
            LOG_ERR("Prefix pass failed at run %d\n", run);
            return 1;
        }
        int64_t t2 = ggml_time_us();

        // ---- Phase: Diffusion Loop ----
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        std::vector<float> x_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
        for (auto & v : x_t) v = normal(rng);

        double expert_step_ms = 0.0;
        std::vector<float> v_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);

        // --prefix-only skips diffusion so the OpenCL per-kernel summary contains
        // only the prefix attention GEMMs (no expert/diffusion contamination).
        if (!bp.prefix_only) {
            for (int step = 0; step < PI0_NUM_STEPS; step++) {
                float t = 1.0f + step * dt;

                int64_t ts0 = ggml_time_us();
                if (!run_expert_step(model, session, backend,
                                     robot_state.data(), x_t.data(), t, v_t.data())) {
                    LOG_ERR("Expert step failed at run %d step %d\n", run, step);
                    return 1;
                }
                int64_t ts1 = ggml_time_us();
                expert_step_ms += (ts1 - ts0) / 1000.0;

                for (int i = 0; i < PI0_ACTION_DIM * PI0_ACTION_HORIZON; i++) {
                    x_t[i] += dt * v_t[i];
                }
            }
        }

        int64_t t3 = ggml_time_us();

        // ---- Dump final action for numerical-correctness A/B testing ----
        // Only first measured iter (avoids noisy multi-iter output).
        if (!is_warmup && run == bp.n_warmup) {
            fprintf(stderr, "ACTION_DUMP: x_t[0:7] = ");
            for (int i = 0; i < PI0_ACTION_DIM && i < 7; i++) {
                fprintf(stderr, "%.6f ", x_t[i]);
            }
            fprintf(stderr, "\n");
            double sum = 0, sumsq = 0;
            for (int i = 0; i < PI0_ACTION_DIM * PI0_ACTION_HORIZON; i++) {
                sum += x_t[i]; sumsq += (double)x_t[i] * x_t[i];
            }
            fprintf(stderr, "ACTION_DUMP: full mean=%.6f rms=%.6f n=%d\n",
                sum / (PI0_ACTION_DIM * PI0_ACTION_HORIZON),
                std::sqrt(sumsq / (PI0_ACTION_DIM * PI0_ACTION_HORIZON)),
                PI0_ACTION_DIM * PI0_ACTION_HORIZON);
        }

        // ---- Record stats ----
        if (!is_warmup) {
            double vision_ms  = (t1 - t0) / 1000.0;
            double prefix_ms  = (t2 - t1) / 1000.0;
            double diff_ms    = (t3 - t2) / 1000.0;
            double total_ms   = (t3 - t_iter_start) / 1000.0;

            stats[PHASE_VISION_ENCODE].samples_ms.push_back(vision_ms);
            stats[PHASE_PREFIX_PASS].samples_ms.push_back(prefix_ms);
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

    // ---- Print results ----
    print_table(stats, bp.n_iter, bp.n_warmup);

    // Cleanup
    pi0_session_free(session);
    free_pi0_model(model);
    ggml_backend_free(backend);

    return 0;
}
