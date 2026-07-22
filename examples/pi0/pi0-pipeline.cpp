// PI0 heterogeneous phase-pipeline scheduling test.
//
// VLA inference (unlike a sequential LLM) is async-friendly: the next inference
// can start while the robot is still acting on the current action chunk
// (PI-RTC, arXiv:2506.07339). The Snapdragon 8 Elite is heterogeneous — Adreno
// GPU (OpenCL) + Hexagon NPU (HTP) — and PI0 already splits cleanly onto them:
//   * GPU stage = Vision (SigLIP), via mtmd with use_gpu -> OpenCL
//   * NPU stage = Prefix + Flow-Matching, via --backend HTP0 (shared KV cache)
//
// Running the phases of consecutive inferences as a PIPELINE lets frame N+1's
// Vision (GPU) overlap frame N's Prefix+FM (NPU), so steady-state throughput is
// bounded by the busiest resource, not the sum:
//
//   Image - ViT(GPU) - Prefix(NPU) - FM(NPU) - Acting
//                    Image - ViT(GPU) - Prefix(NPU) - FM(NPU) - Acting
//                                     Image - ViT(GPU) - Prefix(NPU) - ...
//
// This tool measures three things on-device:
//   sequential  baseline per-frame latency (one thread, phases in series)
//   probe       Vision(GPU) || Prefix+FM(NPU) run concurrently -> the shared-DRAM
//               contention factor the pipeline's overlap actually pays
//   pipeline    two worker threads + a queue over a stream of frames -> steady
//               initiation interval (II), throughput and speedup vs sequential
//
// Usage:
//   llama-pi0-pipeline -m <model_dir> --image img.jpg -p "instruction" \
//       --backend HTP0 [--frames 20] [--mode all|sequential|probe|pipeline]
//
// Run the NON-profiling build: the per-phase profiling setters touch shared
// globals and would race across the two worker threads.

#include "pi0-common.h"

#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ---- Small helpers ----

static inline double us_to_ms(int64_t us) { return us / 1000.0; }

static void action_stats(const std::vector<float> & x, double & mean, double & rms) {
    double sum = 0, sumsq = 0;
    for (float v : x) { sum += v; sumsq += (double) v * v; }
    mean = sum / (double) x.size();
    rms  = std::sqrt(sumsq / (double) x.size());
}

// One vision "frame plan" segment, precomputed once on the main thread so the
// GPU worker never touches the NPU-resident embedding table (embd_lookup_f32
// reads model.pali_embed, which lives in the HTP buffer). Text segments are
// constant across frames; image segments are re-encoded on the GPU each frame.
struct chunk_seg {
    bool is_image = false;
    const mtmd_input_chunk * chunk = nullptr; // valid iff is_image
    std::vector<float> text_embd;             // valid iff !is_image (already scaled)
    int n_tokens = 0;
};

// Bounded blocking queue for the ViT->Prefix handoff (a host float buffer).
struct frame_item {
    int frame_id = 0;                // -1 = end sentinel
    std::vector<float> embd;         // [PALI_N_EMBD * prefix_len]
    int prefix_len = 0;
};

class frame_queue {
    std::mutex m;
    std::condition_variable cv_not_full, cv_not_empty;
    std::queue<frame_item> q;
    size_t cap;
public:
    int64_t consumer_wait_us = 0; // total time the consumer spent blocked (stall)
    explicit frame_queue(size_t c) : cap(c) {}

    void push(frame_item it) {
        std::unique_lock<std::mutex> lk(m);
        cv_not_full.wait(lk, [&] { return q.size() < cap; });
        q.push(std::move(it));
        cv_not_empty.notify_one();
    }
    frame_item pop() {
        std::unique_lock<std::mutex> lk(m);
        int64_t w0 = ggml_time_us();
        cv_not_empty.wait(lk, [&] { return !q.empty(); });
        consumer_wait_us += ggml_time_us() - w0;
        frame_item it = std::move(q.front());
        q.pop();
        cv_not_full.notify_one();
        return it;
    }
};

// ---- CLI ----

struct pipe_params {
    int  frames     = 20;
    int  probe_reps = 5;
    std::string mode = "all"; // all | sequential | probe | pipeline
};

static pipe_params strip_pipe_args(int & argc, char ** argv) {
    pipe_params pp;
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            pp.frames = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            pp.mode = argv[++i];
        } else if (strcmp(argv[i], "--probe-reps") == 0 && i + 1 < argc) {
            pp.probe_reps = std::atoi(argv[++i]);
        } else {
            argv[out++] = argv[i];
        }
    }
    argc = out;
    return pp;
}

int main(int argc, char ** argv) {
    pi0_cli     cli = pi0_strip_cli_args(argc, argv);
    pipe_params pp  = strip_pipe_args(argc, argv);
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0 heterogeneous phase-pipeline scheduling test\n\n");
        printf("Usage: %s -m <model_dir> --image img1.jpg,... -p <instruction> \\\n", argv[0]);
        printf("         --backend HTP0 [--frames 20] [--mode all|sequential|probe|pipeline]\n\n");
        printf("  --backend      pi0 backend for Prefix+FM (use HTP0 for the NPU).\n");
        printf("  --frames <int> stream length for the pipeline test (default: 20)\n");
        printf("  --mode <str>   which test(s) to run (default: all)\n");
        printf("  --probe-reps   concurrent-probe repetitions (default: 5)\n");
        printf("\nVision runs on the GPU (OpenCL) via mtmd; Prefix+FM on the pi0 backend.\n");
    };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    std::string model_dir   = resolve_model_dir(params.model.path);
    std::string pali_path   = model_dir + "/pi0-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi0-mmproj.gguf";
    std::string expert_path = model_dir + "/pi0-action-expert.gguf";

    std::string pali_pi0_path = model_dir + "/pi0-gemma-2b-fused.gguf";
    if (FILE * f = fopen(pali_pi0_path.c_str(), "rb")) { fclose(f); } else { pali_pi0_path = pali_path; }

    if (!params.mmproj.path.empty()) {
        mmproj_path = params.mmproj.path;
    }
    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    if (cli.backend == "cpu" || cli.backend == "CPU") {
        params.n_gpu_layers   = 0;
        params.mmproj_use_gpu = false;
    }

    // ---- One-time init (main thread only) ----

    ggml_backend_t backend = pi0_init_backend(cli.backend); // NPU stage: Prefix + FM
    if (!backend) {
        LOG_ERR("Failed to init backend '%s'\n", cli.backend.c_str());
        return 1;
    }
    LOG_INF("pi0 backend (Prefix+FM): %s | Vision(mtmd) use_gpu=%d\n",
            cli.backend.c_str(), (int) params.mmproj_use_gpu);

    pi0_model model;
    if (!load_pi0_model(model, pali_pi0_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0 model\n");
        return 1;
    }
    pi0_dump_tensor_types(model);

    auto llama_init = common_init_from_params(params);
    llama_model * llm_model = llama_init->model();
    if (!llm_model) {
        LOG_ERR("Failed to load LLM for tokenizer\n");
        return 1;
    }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu; // GPU stage lives here (OpenCL)
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) {
        LOG_ERR("Failed to load vision model\n");
        return 1;
    }

    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", img_path.c_str());
            return 1;
        }
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

    // Build the frame plan. Text embeddings (constant across frames) are looked
    // up ONCE here on the main thread so the GPU worker stays OpenCL-only.
    std::vector<chunk_seg> plan;
    int prefix_len = 0;
    {
        const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
        std::vector<float> emb_row(PALI_N_EMBD);
        const float embd_scale = sqrtf((float) PALI_N_EMBD);
        for (int ic = 0; ic < n_chunks; ic++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
            chunk_seg seg;
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n_tokens = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                seg.is_image = false;
                seg.n_tokens = (int) n_tokens;
                seg.text_embd.reserve(n_tokens * PALI_N_EMBD);
                for (size_t t = 0; t < n_tokens; t++) {
                    embd_lookup_f32(model.pali_embed, tokens[t], emb_row.data());
                    for (float & e : emb_row) e *= embd_scale;
                    seg.text_embd.insert(seg.text_embd.end(), emb_row.begin(), emb_row.end());
                }
            } else {
                seg.is_image = true;
                seg.chunk    = chunk;
                seg.n_tokens = (int) mtmd_input_chunk_get_n_tokens(chunk);
            }
            prefix_len += seg.n_tokens;
            plan.push_back(std::move(seg));
        }
    }
    LOG_INF("prefix_len = %d tokens\n", prefix_len);

    pi0_session session;
    session.kv_type = pi0_ggml_type_from_string(cli.kv_type);
    if (session.kv_type == GGML_TYPE_COUNT) {
        LOG_ERR("Unknown --kv-type '%s'\n", cli.kv_type.c_str());
        return 1;
    }

    std::vector<float> robot_state(PI0_ACTION_DIM, 0.0f);
    const float dt = -1.0f / PI0_NUM_STEPS;

    // GPU stage: encode one frame's images (OpenCL) + stitch in the precomputed
    // text embeddings. Touches ONLY ctx_vision. Returns prefix_len.
    auto encode_frame = [&](std::vector<float> & out) -> int {
        out.clear();
        out.reserve((size_t) prefix_len * PALI_N_EMBD);
        for (const chunk_seg & seg : plan) {
            if (seg.is_image) {
                if (mtmd_encode_chunk(ctx_vision.get(), seg.chunk) != 0) {
                    LOG_ERR("Vision encode failed\n");
                    return -1;
                }
                float * img = mtmd_get_output_embd(ctx_vision.get());
                out.insert(out.end(), img, img + (size_t) seg.n_tokens * PALI_N_EMBD);
            } else {
                out.insert(out.end(), seg.text_embd.begin(), seg.text_embd.end());
            }
        }
        return prefix_len;
    };

    // NPU stage: Prefix pass + PI0_NUM_STEPS Euler flow-matching steps. Touches
    // ONLY the pi0 backend + session. Returns the final action mean/rms.
    auto run_npu = [&](const float * embd, int plen, double & mean, double & rms) -> bool {
        if (!run_paligemma_prefix(model, session, backend, embd, plen)) return false;
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> x_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
        for (auto & v : x_t) v = normal(rng);
        std::vector<float> v_t(x_t.size());
        for (int step = 0; step < PI0_NUM_STEPS; step++) {
            float t = 1.0f + step * dt;
            if (!run_expert_step(model, session, backend, robot_state.data(), x_t.data(), t, v_t.data())) {
                return false;
            }
            for (size_t i = 0; i < x_t.size(); i++) x_t[i] += dt * v_t[i];
        }
        action_stats(x_t, mean, rms);
        return true;
    };

    const bool do_seq   = pp.mode == "all" || pp.mode == "sequential";
    const bool do_probe = pp.mode == "all" || pp.mode == "probe";
    const bool do_pipe  = pp.mode == "all" || pp.mode == "pipeline";
    const int  N        = std::max(4, pp.frames);

    // Warm up once (JIT / cache fill) — encode + full NPU pass.
    {
        std::vector<float> e;
        double mn, rs;
        if (encode_frame(e) < 0 || !run_npu(e.data(), prefix_len, mn, rs)) {
            LOG_ERR("Warmup failed\n");
            return 1;
        }
        LOG_INF("Warmup ok (action mean=%.6f rms=%.6f)\n", mn, rs);
    }

    printf("\n========================================================================\n");
    printf(" PI0 heterogeneous phase-pipeline test — GPU=Vision(OpenCL), NPU=Prefix+FM(%s)\n", cli.backend.c_str());
    printf(" frames=%d  diffusion steps=%d  prefix_len=%d\n", N, PI0_NUM_STEPS, prefix_len);
    printf("========================================================================\n");

    // Measured single-resource stage times (filled by the sequential pass; used
    // as the analytical ceiling for the pipeline / probe).
    double seq_vision_ms = 0, seq_npu_ms = 0, seq_latency_ms = 0;
    double ref_mean = 0, ref_rms = 0;

    // ---- Sequential baseline ----
    if (do_seq) {
        double sum_vision = 0, sum_npu = 0, sum_total = 0;
        std::vector<float> e;
        for (int f = 0; f < N; f++) {
            int64_t t0 = ggml_time_us();
            if (encode_frame(e) < 0) return 1;
            int64_t t1 = ggml_time_us();
            double mn, rs;
            if (!run_npu(e.data(), prefix_len, mn, rs)) return 1;
            int64_t t2 = ggml_time_us();
            sum_vision += us_to_ms(t1 - t0);
            sum_npu    += us_to_ms(t2 - t1);
            sum_total  += us_to_ms(t2 - t0);
            if (f == 0) { ref_mean = mn; ref_rms = rs; }
        }
        seq_vision_ms  = sum_vision / N;
        seq_npu_ms     = sum_npu    / N;
        seq_latency_ms = sum_total  / N;
        printf("\n[sequential] per-frame latency %.1f ms  (%.2f Hz)\n", seq_latency_ms, 1000.0 / seq_latency_ms);
        printf("             Vision(GPU) %.1f ms | Prefix+FM(NPU) %.1f ms\n", seq_vision_ms, seq_npu_ms);
        printf("             action mean=%.6f rms=%.6f\n", ref_mean, ref_rms);
    }

    // ---- Concurrent probe: isolate shared-DRAM contention ----
    if (do_probe) {
        std::vector<float> embd0;
        if (encode_frame(embd0) < 0) return 1; // fixed NPU input

        double sum_gpu = 0, sum_npu = 0, sum_conc = 0;
        for (int r = 0; r < pp.probe_reps; r++) {
            std::vector<float> scratch;
            double mn, rs;

            int64_t g0 = ggml_time_us();
            if (encode_frame(scratch) < 0) return 1;
            double t_gpu = us_to_ms(ggml_time_us() - g0);

            int64_t n0 = ggml_time_us();
            if (!run_npu(embd0.data(), prefix_len, mn, rs)) return 1;
            double t_npu = us_to_ms(ggml_time_us() - n0);

            // Both engines at once, on independent inputs (no data dependency).
            std::atomic<bool> gpu_ok{true}, npu_ok{true};
            int64_t c0 = ggml_time_us();
            std::thread ta([&] { std::vector<float> s; if (encode_frame(s) < 0) gpu_ok = false; });
            std::thread tb([&] { double m2, r2; if (!run_npu(embd0.data(), prefix_len, m2, r2)) npu_ok = false; });
            ta.join(); tb.join();
            double t_conc = us_to_ms(ggml_time_us() - c0);
            if (!gpu_ok || !npu_ok) { LOG_ERR("probe concurrent stage failed\n"); return 1; }

            sum_gpu += t_gpu; sum_npu += t_npu; sum_conc += t_conc;
        }
        double a_gpu = sum_gpu / pp.probe_reps;
        double a_npu = sum_npu / pp.probe_reps;
        double a_conc = sum_conc / pp.probe_reps;
        double ideal = std::max(a_gpu, a_npu);                 // perfect overlap
        double serial = a_gpu + a_npu;                         // zero overlap
        double efficiency = serial / a_conc;                   // 1.0 = serialized, serial/ideal = perfect
        double best_eff   = serial / ideal;
        double penalty = a_conc - ideal;

        printf("\n[probe] solo Vision(GPU) %.1f ms | solo Prefix+FM(NPU) %.1f ms\n", a_gpu, a_npu);
        printf("        concurrent %.1f ms  (serial %.1f, ideal-overlap %.1f)\n", a_conc, serial, ideal);
        printf("        overlap efficiency %.2fx of serial (%.2fx = perfect)\n", efficiency, best_eff);
        printf("        contention penalty %.1f ms above ideal overlap (%.0f%% of ideal)\n",
               penalty, 100.0 * penalty / ideal);
    }

    // ---- Full 2-thread pipeline over a stream of frames ----
    if (do_pipe) {
        frame_queue q(2); // double-buffer
        std::vector<double> vision_ms(N, 0), npu_ms(N, 0);
        std::vector<int64_t> complete_us(N, 0);
        std::atomic<bool> ok{true};
        double pipe_mean = 0, pipe_rms = 0;

        int64_t wall0 = ggml_time_us();

        std::thread gpu_worker([&] {
            for (int f = 0; f < N && ok; f++) {
                frame_item it;
                it.frame_id = f;
                int64_t t0 = ggml_time_us();
                it.prefix_len = encode_frame(it.embd);
                vision_ms[f] = us_to_ms(ggml_time_us() - t0);
                if (it.prefix_len < 0) { ok = false; break; }
                q.push(std::move(it));
            }
            q.push(frame_item{ -1, {}, 0 }); // sentinel
        });

        std::thread npu_worker([&] {
            for (;;) {
                frame_item it = q.pop();
                if (it.frame_id < 0) break;
                int64_t t0 = ggml_time_us();
                double mn, rs;
                if (!run_npu(it.embd.data(), it.prefix_len, mn, rs)) { ok = false; break; }
                int64_t t1 = ggml_time_us();
                npu_ms[it.frame_id]     = us_to_ms(t1 - t0);
                complete_us[it.frame_id] = t1;
                if (it.frame_id == N - 1) { pipe_mean = mn; pipe_rms = rs; }
            }
        });

        gpu_worker.join();
        npu_worker.join();
        int64_t wall1 = ggml_time_us();
        if (!ok) { LOG_ERR("pipeline stage failed\n"); return 1; }

        double wall_ms = us_to_ms(wall1 - wall0);
        double sum_vision = 0, sum_npu = 0;
        for (int f = 0; f < N; f++) { sum_vision += vision_ms[f]; sum_npu += npu_ms[f]; }

        // Steady-state initiation interval: inter-completion gap, skipping the
        // first 2 completions (pipeline fill).
        int k = std::min(2, N - 1);
        double II_ms = us_to_ms(complete_us[N - 1] - complete_us[k]) / (double) (N - 1 - k);
        double gpu_busy = 100.0 * sum_vision / wall_ms;
        double npu_busy = 100.0 * sum_npu    / wall_ms;
        double stall_ms = us_to_ms(q.consumer_wait_us);

        printf("\n[pipeline] %d frames in %.1f ms  ->  %.2f Hz effective\n", N, wall_ms, N * 1000.0 / wall_ms);
        printf("           steady-state II %.1f ms  (%.2f Hz)\n", II_ms, 1000.0 / II_ms);
        printf("           GPU busy %.0f%% | NPU busy %.0f%% | NPU stall %.1f ms\n", gpu_busy, npu_busy, stall_ms);
        printf("           action mean=%.6f rms=%.6f\n", pipe_mean, pipe_rms);

        if (do_seq) {
            double stage_ceiling = std::max(seq_vision_ms, seq_npu_ms);
            printf("\n== verdict ==\n");
            printf("  sequential latency  %.1f ms  (%.2f Hz)\n", seq_latency_ms, 1000.0 / seq_latency_ms);
            printf("  pipeline II         %.1f ms  (%.2f Hz)  -> %.2fx throughput\n",
                   II_ms, 1000.0 / II_ms, seq_latency_ms / II_ms);
            printf("  busiest stage (NPU=%s) ceiling %.1f ms; contention excess %.1f ms\n",
                   (seq_npu_ms >= seq_vision_ms ? "yes" : "no"), stage_ceiling, II_ms - stage_ceiling);
            double dm = std::fabs(pipe_mean - ref_mean), dr = std::fabs(pipe_rms - ref_rms);
            printf("  correctness: |dmean|=%.6f |drms|=%.6f  (%s)\n",
                   dm, dr, (dm < 1e-4 && dr < 1e-4) ? "MATCHES sequential" : "DIVERGED — check");
        }
    }

    printf("\n");

    pi0_session_free(session);
    free_pi0_model(model);
    ggml_backend_free(backend);
    return 0;
}
