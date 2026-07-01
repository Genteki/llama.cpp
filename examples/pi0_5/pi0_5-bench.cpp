// PI0.5 benchmark + profiler — one tool, four modes (--mode).
//
//   --mode phase    per-phase wall-clock (vision / prefix / diffusion) + the
//                   analytical GFLOPS each phase sustains. (default)
//   --mode op       targeted operator baseline-vs-optimized micro-benchmarks at
//                   the real pi0.5 shapes: prefix QK^T (batched vs MQA-collapse)
//                   and the action-expert qkv_proj GEMM (f16 vs q8_0 vs q4_0).
//                   Self-contained — needs only --backend (no -m/-p/--image).
//   --mode quant    full inference latency + weight footprint for the loaded
//                   weight set, sweeping the activation precision (A16 vs A32);
//                   run it once per -m dir (f16/q8/q4) to compare the W axis.
//   --mode profile  one clean inference for per-kernel timing. On an OpenCL build
//                   configured with -DGGML_OPENCL_PROFILING=ON the backend writes
//                   cl_profiling.csv (per-kernel ms + estimated GFLOPS).
//
// Shared flags: --backend, --kv-type, --act-type, -n, --warmup.

#include "pi0_5-common.h"

#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ============================================================
// Analytical FLOP model (2 FLOP per MAC)
// ============================================================

static inline double gemm_flops(double K, double N, double M) { return 2.0 * K * N * M; }

// PaliGemma prefix, one full pass over P tokens.
static double prefix_flops(int P) {
    double f = 0;
    for (int l = 0; l < PI05_N_LAYER; l++) {
        f += gemm_flops(PALI_N_EMBD, PALI_N_HEAD * PALI_HEAD_DIM, P);     // q
        f += gemm_flops(PALI_N_EMBD, PALI_N_KV_HEAD * PALI_HEAD_DIM, P);  // k
        f += gemm_flops(PALI_N_EMBD, PALI_N_KV_HEAD * PALI_HEAD_DIM, P);  // v
        f += gemm_flops(PALI_N_HEAD * PALI_HEAD_DIM, PALI_N_EMBD, P);     // o
        f += gemm_flops(PALI_N_EMBD, PALI_N_FF, P) * 2;                   // gate + up
        f += gemm_flops(PALI_N_FF, PALI_N_EMBD, P);                       // down
        f += 2.0 * PALI_HEAD_DIM * (double) P * P * PALI_N_HEAD * 2;      // KQ + AV
    }
    return f;
}

// Action expert, one denoising step (kv = P + suffix).
static double expert_step_flops(int P) {
    const int M  = PI05_ACTION_HORIZON;          // 50
    const int kv = P + PI05_SUFFIX_LEN;
    double f = 0;
    f += gemm_flops(PI05_ACTION_DIM, EXPERT_N_EMBD, M);                   // action_in_proj
    f += gemm_flops(EXPERT_N_EMBD, EXPERT_N_EMBD, 1) * 2;                 // time_mlp in/out
    for (int l = 0; l < PI05_N_LAYER; l++) {
        f += gemm_flops(ADARMS_COND_DIM, ADARMS_MOD_DIM, 1) * 2;          // adaRMS dense (attn+ffn)
        f += gemm_flops(EXPERT_N_EMBD, EXPERT_N_HEAD * EXPERT_HEAD_DIM, M);
        f += gemm_flops(EXPERT_N_EMBD, EXPERT_N_KV_HEAD * EXPERT_HEAD_DIM, M) * 2;
        f += gemm_flops(EXPERT_N_HEAD * EXPERT_HEAD_DIM, EXPERT_N_EMBD, M);
        f += gemm_flops(EXPERT_N_EMBD, EXPERT_N_FF, M) * 2;
        f += gemm_flops(EXPERT_N_FF, EXPERT_N_EMBD, M);
        f += 2.0 * EXPERT_HEAD_DIM * (double) M * kv * EXPERT_N_HEAD * 2; // KQ + AV
    }
    f += gemm_flops(ADARMS_COND_DIM, ADARMS_MOD_DIM, 1);                  // final adaRMS dense
    f += gemm_flops(EXPERT_N_EMBD, PI05_ACTION_DIM, M);                  // action_out_proj
    return f;
}

// ============================================================
// Generic GEMM micro-bench (for --mode op)
// ============================================================

// Times out = W[K,N] @ x[K,M] with weights stored as `wtype` and activations
// cast to `act`. Weights/activations are random bytes — values are irrelevant to
// timing. Returns average ms over `iters` (after `warmup`).
static double bench_gemm(ggml_backend_t backend, int K, int N, int M,
                         ggml_type wtype, ggml_type act, int iters, int warmup, bool set_f32_prec) {
    size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params p = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(p);

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_input(W); ggml_set_input(x);
    ggml_tensor * xa  = (act == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, act);
    ggml_tensor * out = ggml_mul_mat(ctx, W, xa);
    if (set_f32_prec) ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    ggml_set_output(out);

    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, g)) { ggml_gallocr_free(alloc); ggml_free(ctx); return -1; }

    std::vector<uint8_t> wbuf(ggml_nbytes(W));
    std::vector<uint8_t> xbuf(ggml_nbytes(x));
    std::mt19937 rng(1234);
    for (auto & b : wbuf) b = (uint8_t) rng();
    std::vector<float> xf(K * M);
    for (auto & v : xf) v = std::normal_distribution<float>(0,1)(rng);
    ggml_backend_tensor_set(W, wbuf.data(), 0, wbuf.size());
    ggml_backend_tensor_set(x, xf.data(),   0, xbuf.size());

    for (int i = 0; i < warmup; i++) ggml_backend_graph_compute(backend, g);
    ggml_backend_synchronize(backend);
    int64_t t0 = ggml_time_us();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, g);
    ggml_backend_synchronize(backend);
    int64_t t1 = ggml_time_us();

    ggml_gallocr_free(alloc); ggml_free(ctx);
    return (t1 - t0) / 1000.0 / iters;
}

static void report_gemm(const char * label, ggml_backend_t backend, int K, int N, int M,
                        ggml_type wtype, ggml_type act, int iters, int warmup) {
    // CPU mul_mat needs f32 activations for quantized weights; OpenCL Adreno
    // kernels take f16 activations even for q4_0/q8_0 (the fast path we want).
    const bool is_cpu = strstr(ggml_backend_name(backend), "CPU") != nullptr;
    const bool wquant = (wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_F32);
    ggml_type eff_act = (is_cpu && wquant) ? GGML_TYPE_F32 : act;
    double ms = bench_gemm(backend, K, N, M, wtype, eff_act, iters, warmup, false);
    double gf = gemm_flops(K, N, M) / 1e9 / (ms / 1000.0);
    double mb = ggml_row_size(wtype, K) * (double) N / (1024.0 * 1024.0);
    printf("  %-28s K=%-5d N=%-5d M=%-3d  W=%-5s A=%-4s | %8.3f ms | %8.1f GFLOPS | W=%6.1f MB\n",
           label, K, N, M, ggml_type_name(wtype), ggml_type_name(eff_act), ms, gf, mb);
}

static int run_op_bench(ggml_backend_t backend, ggml_type act, int iters, int warmup,
                        int prefix_len) {
    printf("\n=== PI0.5 op micro-bench (act=%s, %d iters / %d warmup) ===\n",
           ggml_type_name(act), iters, warmup);

    // ---- Action-expert qkv_proj GEMM: baseline f16 vs optimized q8_0/q4_0 ----
    // Fused qkv out = n_head*hd + 2*kv_dim = 2048 + 512 = 2560; per-token M=50.
    const int qkv_out = EXPERT_N_HEAD * EXPERT_HEAD_DIM + 2 * EXPERT_N_KV_HEAD * EXPERT_HEAD_DIM;
    printf("\n[expert qkv_proj]  in=%d out=%d M=%d\n", EXPERT_N_EMBD, qkv_out, PI05_ACTION_HORIZON);
    report_gemm("baseline W=f16",  backend, EXPERT_N_EMBD, qkv_out, PI05_ACTION_HORIZON, GGML_TYPE_F16,  act, iters, warmup);
    report_gemm("opt      W=q8_0", backend, EXPERT_N_EMBD, qkv_out, PI05_ACTION_HORIZON, GGML_TYPE_Q8_0, act, iters, warmup);
    report_gemm("opt      W=q4_0", backend, EXPERT_N_EMBD, qkv_out, PI05_ACTION_HORIZON, GGML_TYPE_Q4_0, act, iters, warmup);

    // ---- Action-expert FFN gate GEMM (the run-10x bandwidth lever) ----
    printf("\n[expert ffn_gate]  in=%d out=%d M=%d\n", EXPERT_N_EMBD, EXPERT_N_FF, PI05_ACTION_HORIZON);
    report_gemm("baseline W=f16",  backend, EXPERT_N_EMBD, EXPERT_N_FF, PI05_ACTION_HORIZON, GGML_TYPE_F16,  act, iters, warmup);
    report_gemm("opt      W=q4_0", backend, EXPERT_N_EMBD, EXPERT_N_FF, PI05_ACTION_HORIZON, GGML_TYPE_Q4_0, act, iters, warmup);

    // ---- Prefix QK^T: baseline (batched per-head) vs optimized (MQA-collapse) ----
    // n_kv_head=1, so the 8 query heads can fold into the N dimension of one GEMM.
    // baseline: 8 separate [kv x P] GEMMs (modelled by M=P, batched B=n_head);
    // optimized: one [kv x P*n_head] GEMM.
    printf("\n[prefix QK^T]  head_dim=%d kv=%d P=%d n_head=%d (n_kv_head=1)\n",
           PALI_HEAD_DIM, prefix_len, prefix_len, PALI_N_HEAD);
    {
        // baseline: per-head KQ, summed (approx via n_head separate GEMMs)
        double ms_base = 0;
        for (int h = 0; h < PALI_N_HEAD; h++)
            ms_base += bench_gemm(backend, PALI_HEAD_DIM, prefix_len, prefix_len, GGML_TYPE_F16, act, iters, warmup, true);
        double fl = 2.0 * PALI_HEAD_DIM * (double) prefix_len * prefix_len * PALI_N_HEAD;
        printf("  %-28s | %8.3f ms | %8.1f GFLOPS\n", "baseline batched(8x)", ms_base, fl / 1e9 / (ms_base / 1000.0));
        // optimized: one collapsed GEMM with N = P*n_head
        double ms_coll = bench_gemm(backend, PALI_HEAD_DIM, prefix_len, prefix_len * PALI_N_HEAD, GGML_TYPE_F16, act, iters, warmup, true);
        printf("  %-28s | %8.3f ms | %8.1f GFLOPS  (speedup %.2fx)\n", "opt MQA-collapse(1x)",
               ms_coll, fl / 1e9 / (ms_coll / 1000.0), ms_base / ms_coll);
    }
    return 0;
}

// ============================================================
// Shared full-inference setup (phase / quant / profile)
// ============================================================

struct loaded {
    pi05_model model;
    std::unique_ptr<common_init_result> llama_init;
    mtmd::context_ptr ctx_vision;
    mtmd::input_chunks chunks{mtmd_input_chunks_init()};
    mtmd::bitmaps bitmaps;
    int n_chunks = 0;
};

// Assemble prefix embeddings (vision encode + text). Returns prefix length.
static int assemble_prefix(loaded & L, std::vector<float> & prefix, const std::string & prompt_text) {
    prefix.clear();
    int P = 0;
    std::vector<float> row(PALI_N_EMBD);
    const float scale = sqrtf((float) PALI_N_EMBD);
    for (int ic = 0; ic < L.n_chunks; ic++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(L.chunks.ptr.get(), ic);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_encode_chunk(L.ctx_vision.get(), chunk) != 0) return -1;
        float * e = mtmd_get_output_embd(L.ctx_vision.get());
        size_t n = mtmd_input_chunk_get_n_tokens(chunk);
        prefix.insert(prefix.end(), e, e + n * PALI_N_EMBD);
        P += (int) n;
    }
    const llama_vocab * vocab = llama_model_get_vocab(L.llama_init->model());
    std::vector<llama_token> toks;
    toks.push_back(llama_vocab_bos(vocab));
    std::vector<llama_token> body = common_tokenize(L.llama_init->context(), prompt_text, false, false);
    toks.insert(toks.end(), body.begin(), body.end());
    std::vector<llama_token> nl = common_tokenize(L.llama_init->context(), "\n", false, false);
    toks.insert(toks.end(), nl.begin(), nl.end());
    for (llama_token t : toks) {
        embd_lookup_f32(L.model.pali_embed, t, row.data());
        for (float & v : row) v *= scale;
        prefix.insert(prefix.end(), row.begin(), row.end());
        P++;
    }
    return P;
}

// ============================================================
// Arg stripping
// ============================================================

struct bench_params {
    int n_iter = 20;
    int n_warmup = 3;
    std::string mode = "phase"; // phase | op | quant | profile
};

static bench_params strip_bench_args(int & argc, char ** argv) {
    bench_params bp;
    int out = 1;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-n") == 0) && i + 1 < argc)            bp.n_iter   = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)   bp.n_warmup = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)     bp.mode     = argv[++i];
        else argv[out++] = argv[i];
    }
    argc = out;
    return bp;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char ** argv) {
    pi05_cli     cli = pi05_strip_cli_args(argc, argv);
    bench_params bp  = strip_bench_args(argc, argv);
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0.5 benchmark/profiler\n\n");
        printf("Usage: %s -m <model_dir> --image a.jpg,b.jpg,c.jpg -p <instr> \\\n", argv[0]);
        printf("         --mode {phase|op|quant|profile} [-n 20] [--warmup 3] \\\n");
        printf("         [--backend ...] [--kv-type f16] [--act-type f16]\n");
        printf("  --mode op needs only --backend (no model/images).\n");
    };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) return 1;
    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    ggml_type kv  = pi05_ggml_type_from_string(cli.kv_type);
    ggml_type act = pi05_resolve_act_type(cli.act_type, cli.backend);
    if (kv == GGML_TYPE_COUNT || act == GGML_TYPE_COUNT) {
        LOG_ERR("Bad --kv-type/--act-type\n"); return 1;
    }

    if (cli.backend == "cpu" || cli.backend == "CPU") {
        params.n_gpu_layers = 0; params.mmproj_use_gpu = false;
    }
    ggml_backend_t backend = pi05_init_backend(cli.backend);
    if (!backend) { LOG_ERR("Failed to init backend '%s'\n", cli.backend.c_str()); return 1; }

    // ---- op mode: standalone, no model needed ----
    if (bp.mode == "op") {
        int op_prefix_len = 816; // typical pi0.5 prefix: 3x256 image + ~48 text
        int rc = run_op_bench(backend, act, bp.n_iter, bp.n_warmup, op_prefix_len);
        ggml_backend_free(backend);
        return rc;
    }

    // ---- full-inference modes need the model + vision ----
    std::string model_dir   = resolve_model_dir(params.model.path);
    std::string pali_path   = model_dir + "/pi05-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi05-mmproj.gguf";
    std::string expert_path = model_dir + "/pi05-action-expert.gguf";
    std::string pali_fused  = model_dir + "/pi05-gemma-2b-fused.gguf";
    if (FILE * f = fopen(pali_fused.c_str(), "rb")) { fclose(f); } else { pali_fused = pali_path; }
    if (!params.mmproj.path.empty()) mmproj_path = params.mmproj.path;
    params.model.path = pali_path; params.mmproj.path = mmproj_path;

    loaded L;
    if (!load_pi05_model(L.model, pali_fused.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0.5 model\n"); return 1;
    }
    pi05_dump_tensor_types(L.model);

    L.llama_init = common_init_from_params(params);
    if (!L.llama_init->model() || !L.llama_init->context()) { LOG_ERR("tokenizer load failed\n"); return 1; }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu; mparams.n_threads = params.cpuparams.n_threads; mparams.warmup = params.warmup;
    L.ctx_vision.reset(mtmd_init_from_file(params.mmproj.path.c_str(), L.llama_init->model(), mparams));
    if (!L.ctx_vision.get()) { LOG_ERR("vision load failed\n"); return 1; }

    for (const auto & ip : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(L.ctx_vision.get(), ip.c_str()));
        if (!bmp.ptr) { LOG_ERR("Failed to load image %s\n", ip.c_str()); return 1; }
        L.bitmaps.entries.push_back(std::move(bmp));
    }
    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) prompt += mtmd_default_marker();
    prompt += params.prompt;
    mtmd_input_text text{prompt.c_str(), true, true};
    auto bc = L.bitmaps.c_ptr();
    if (mtmd_tokenize(L.ctx_vision.get(), L.chunks.ptr.get(), &text, bc.data(), bc.size()) != 0) {
        LOG_ERR("tokenize failed\n"); return 1;
    }
    L.n_chunks = mtmd_input_chunks_size(L.chunks.ptr.get());

    // Helper that runs one full inference and fills per-phase ms.
    auto one_inference = [&](pi05_session & s, double & vision_ms, double & prefix_ms,
                             double & diff_ms, int & P) -> bool {
        std::vector<float> prefix;
        int64_t t0 = ggml_time_us();
        P = assemble_prefix(L, prefix, params.prompt);
        if (P < 0) return false;
        int64_t t1 = ggml_time_us();
        if (!run_paligemma_prefix(L.model, s, backend, prefix.data(), P)) return false;
        int64_t t2 = ggml_time_us();
        std::mt19937 rng(42);
        std::vector<float> x_t(PI05_ACTION_DIM * PI05_ACTION_HORIZON), v_t(x_t.size());
        for (auto & v : x_t) v = std::normal_distribution<float>(0,1)(rng);
        const float dt = -1.0f / PI05_NUM_STEPS;
        for (int step = 0; step < PI05_NUM_STEPS; step++) {
            float t = 1.0f + step * dt;
            if (!run_expert_step(L.model, s, backend, x_t.data(), t, v_t.data())) return false;
            for (size_t i = 0; i < x_t.size(); i++) x_t[i] += dt * v_t[i];
        }
        int64_t t3 = ggml_time_us();
        vision_ms = (t1-t0)/1000.0; prefix_ms = (t2-t1)/1000.0; diff_ms = (t3-t2)/1000.0;
        return true;
    };

    auto print_phase = [&](const char * tag, double v, double pre, double d, int P) {
        double dstep = d / PI05_NUM_STEPS;
        printf("\n=== %s (P=%d, kv=%s, act=%s) ===\n", tag, P, ggml_type_name(kv), ggml_type_name(act));
        printf("  %-18s %9s %12s\n", "phase", "ms", "GFLOPS");
        printf("  %-18s %9.1f %12.1f\n", "vision encode", v, 0.0);
        printf("  %-18s %9.1f %12.1f\n", "prefix (1x)", pre, prefix_flops(P)/1e9/(pre/1000.0));
        printf("  %-18s %9.1f %12.1f\n", "expert step (1x)", dstep, expert_step_flops(P)/1e9/(dstep/1000.0));
        printf("  %-18s %9.1f %12.1f\n", "diffusion (10x)", d, expert_step_flops(P)*PI05_NUM_STEPS/1e9/(d/1000.0));
        printf("  %-18s %9.1f\n", "TOTAL", v+pre+d);
    };

    if (bp.mode == "profile") {
        pi05_session s; s.kv_type = kv; s.act_type = act;
        double v=0,pre=0,d=0; int P=0;
        if (!one_inference(s, v, pre, d, P)) { LOG_ERR("inference failed\n"); return 1; }
        print_phase("profile (1 iter)", v, pre, d, P);
        printf("\nOpenCL builds with -DGGML_OPENCL_PROFILING=ON write per-kernel\n"
               "timing + estimated GFLOPS to ./cl_profiling.csv on backend teardown.\n");
        pi05_session_free(s);
    } else if (bp.mode == "quant") {
        // Sweep activation precision at the loaded weight set; report W/A label.
        // f16 activations are only valid for quantized weights on GPU backends,
        // so on CPU we report A32 only (see README).
        printf("\n=== PI0.5 quant sweep (weights from -m dir; run per dir for the W axis) ===\n");
        const bool is_cpu = strstr(ggml_backend_name(backend), "CPU") != nullptr;
        std::vector<ggml_type> acts = is_cpu ? std::vector<ggml_type>{ GGML_TYPE_F32 }
                                             : std::vector<ggml_type>{ GGML_TYPE_F16, GGML_TYPE_F32 };
        for (ggml_type a : acts) {
            pi05_session s; s.kv_type = kv; s.act_type = a;
            std::vector<double> tot;
            double v=0,pre=0,d=0; int P=0;
            for (int r = 0; r < bp.n_warmup; r++) one_inference(s, v, pre, d, P);
            for (int r = 0; r < bp.n_iter; r++) { if (!one_inference(s, v, pre, d, P)) return 1; tot.push_back(v+pre+d); }
            double avg = std::accumulate(tot.begin(), tot.end(), 0.0) / tot.size();
            const char * abits = (a == GGML_TYPE_F16) ? "A16" : "A32";
            printf("  act=%-4s (%s) | avg total %8.1f ms | prefix %.1f | diffusion %.1f\n",
                   ggml_type_name(a), abits, avg, pre, d);
            pi05_session_free(s);
        }
        printf("  (weight bits = the expert/prefix gguf tensor types reported above)\n");
    } else { // phase (default)
        pi05_session s; s.kv_type = kv; s.act_type = act;
        double v=0,pre=0,d=0; int P=0;
        std::vector<double> vt, pt, dt_, tt;
        for (int r = 0; r < bp.n_warmup; r++) one_inference(s, v, pre, d, P);
        for (int r = 0; r < bp.n_iter; r++) {
            if (!one_inference(s, v, pre, d, P)) { LOG_ERR("inference failed\n"); return 1; }
            vt.push_back(v); pt.push_back(pre); dt_.push_back(d); tt.push_back(v+pre+d);
        }
        auto avg = [](std::vector<double>&x){ return std::accumulate(x.begin(),x.end(),0.0)/x.size(); };
        v=avg(vt); pre=avg(pt); d=avg(dt_);
        print_phase("phase timing (avg)", v, pre, d, P);
        printf("\nThroughput: %.2f inferences/sec\n", 1000.0 / avg(tt));
        pi05_session_free(s);
    }

    free_pi05_model(L.model);
    ggml_backend_free(backend);
    return 0;
}
