// pi0-gemm-bench.cpp
//
// Microbenchmark to settle one question: for the PI0 Adreno q4_0 GEMM, how much
// of kernel_mul_mat_Ab_Bi_8x4's speed comes from reading B (activations) through
// the image1d_buffer texture path (TP L1) vs a plain global load?
//
// It times a single MxNxK GEMM (default 4096^3) routed four ways, all on the SAME
// quantized weights / activations (only the kernel / B-load path differ):
//
//   abbi     q4_0 x f32  kernel_mul_mat_Ab_Bi_8x4          B via image (texture)   <- the hero
//   abbi_gb  q4_0 x f32  kernel_mul_mat_Ab_Bi_8x4_globalB  B via global half*      <- same kernel, no texture
//   l4lm     q4_0 x f32  mul_mm_q4_0_f32_l4_lm             LDS-tiled, B via global
//   f16      f16  x f32  mul_mm_f16_f32_l4_lm              LDS-tiled f16 reference
//
// abbi vs abbi_gb is the decisive A/B: identical kernel, identical B bytes/layout,
// the ONLY difference is texture-read vs global-read of B. The gap (if any) is the
// texture-cache contribution.
//
// Routing is selected via the PI0_GEMM_VARIANT env var, read inside the OpenCL
// backend (ggml-opencl.cpp). The q4_0 weights for the l4lm case are uploaded with
// the env set so they stay in the standard (non-transposed) layout the LDS kernel
// expects; the abbi/abbi_gb weights use the default Ab_Bi-transposed layout.
//
// A correctness pass compares every config's output against the abbi result, so a
// layout/routing mistake can't masquerade as a fast (wrong) kernel.

#include "pi0-common.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// One ready-to-run GEMM: c = mul_mat(a[K,M], b[K,N]) -> c[M,N].
struct gemm_case {
    ggml_context * ctx   = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph  * gf    = nullptr;
    ggml_tensor  * a     = nullptr; // weights [K, M]
    ggml_tensor  * b     = nullptr; // activations f32 [K, N]
    ggml_tensor  * c     = nullptr; // output f32 [M, N]
};

static gemm_case build_case(ggml_backend_t backend, ggml_type a_type, int M, int N, int K) {
    gemm_case g;
    ggml_init_params ip = {
        /*.mem_size  =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 1024,
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    g.ctx = ggml_init(ip);
    g.a = ggml_new_tensor_2d(g.ctx, a_type,        K, M);
    g.b = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, N);
    ggml_set_input(g.a);
    ggml_set_input(g.b);
    g.c = ggml_mul_mat(g.ctx, g.a, g.b);
    ggml_set_output(g.c);

    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.c);

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.alloc, g.gf)) {
        fprintf(stderr, "alloc FAILED (M=%d N=%d K=%d)\n", M, N, K);
        exit(1);
    }
    return g;
}

// Deterministic, bounded source values (magnitudes irrelevant to timing; matter for the
// correctness check, where all configs see the same logical A and B).
static float src_val(int64_t i) { return (float)((i % 17) - 8) * 0.03125f; }

static void upload_weights(ggml_tensor * a, const std::vector<uint8_t> & q4,
                           const std::vector<ggml_fp16_t> & f16) {
    if (a->type == GGML_TYPE_Q4_0) {
        ggml_backend_tensor_set(a, q4.data(), 0, q4.size());
    } else { // F16
        ggml_backend_tensor_set(a, f16.data(), 0, f16.size() * sizeof(ggml_fp16_t));
    }
}

static void upload_b(ggml_tensor * b) {
    const int64_t n = ggml_nelements(b);
    std::vector<float> f(n);
    for (int64_t i = 0; i < n; ++i) f[i] = src_val(i);
    ggml_backend_tensor_set(b, f.data(), 0, n * sizeof(float));
}

// Run gf `iters` times under PI0_GEMM_VARIANT=variant, return min ms. Optionally capture C.
static double time_case(ggml_backend_t backend, const gemm_case & g, const char * variant,
                        int iters, int warmup, std::vector<float> * out_c) {
    if (variant) setenv("PI0_GEMM_VARIANT", variant, 1);
    else         unsetenv("PI0_GEMM_VARIANT");

    for (int i = 0; i < warmup; ++i) ggml_backend_graph_compute(backend, g.gf);
    ggml_backend_synchronize(backend);

    double sum_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, g.gf);
        ggml_backend_synchronize(backend);
        sum_ms += (ggml_time_us() - t0) / 1000.0;
    }
    if (out_c) {
        out_c->resize(ggml_nelements(g.c));
        ggml_backend_tensor_get(g.c, out_c->data(), 0, ggml_nbytes(g.c));
    }
    return sum_ms / iters; // mean over the timed iters (same statistic as the device summary)
}

// max relative difference between two C buffers (||a-b|| / (|a|+eps), worst element).
static double max_rel_diff(const std::vector<float> & x, const std::vector<float> & y) {
    double m = 0.0;
    const size_t n = x.size() < y.size() ? x.size() : y.size();
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)x[i] - (double)y[i]);
        const double s = std::fabs((double)x[i]) + 1e-3;
        if (d / s > m) m = d / s;
    }
    return m;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string backend_name = "OpenCL";
    int M = 4096, N = 4096, K = 4096;
    int iters = 30, warmup = 5;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](int & v) { if (i + 1 < argc) v = atoi(argv[++i]); };
        if      (a == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else if (a == "-M") take(M);
        else if (a == "-N") take(N);
        else if (a == "-K") take(K);
        else if (a == "-n") take(iters);
        else if (a == "--warmup") take(warmup);
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    pi0_dump_backends();
    ggml_backend_t backend = pi0_init_backend(backend_name);
    if (!backend) { fprintf(stderr, "backend '%s' unavailable\n", backend_name.c_str()); return 1; }

    const double flop = 2.0 * (double)M * N * K;

    printf("\nPI0 q4_0 GEMM B-load-path bench   backend=%s\n", backend_name.c_str());
    printf("shape: M=%d  N=%d  K=%d   (FLOP=2MNK=%.2f G)   iters=%d warmup=%d\n\n",
           M, N, K, flop / 1e9, iters, warmup);

    // Quantize one f32 weight matrix [K,M] -> standard q4_0 bytes; also keep an f16 copy.
    const int64_t na = (int64_t)M * K;
    std::vector<float> srcA(na);
    for (int64_t i = 0; i < na; ++i) srcA[i] = src_val(i);
    std::vector<uint8_t>    q4(ggml_row_size(GGML_TYPE_Q4_0, K) * M);
    std::vector<ggml_fp16_t> hA(na);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, srcA.data(), q4.data(), 0, M, K, nullptr);
    ggml_fp32_to_fp16_row(srcA.data(), hA.data(), na);

    // Build the cases. a1 (Ab_Bi layout) serves abbi + abbi_gb; a2 (standard layout) serves l4lm.
    gemm_case q4_adreno = build_case(backend, GGML_TYPE_Q4_0, M, N, K);
    gemm_case q4_std    = build_case(backend, GGML_TYPE_Q4_0, M, N, K);
    gemm_case f16_case  = build_case(backend, GGML_TYPE_F16,  M, N, K);
    gemm_case f16_abbi  = build_case(backend, GGML_TYPE_F16,  M, N, K); // f16 weights, Ab_Bi (image B)

    // Upload weights. CRITICAL: set the env BEFORE uploading so the q4_0 layout matches
    // how each case will be run (l4lm -> standard/untransposed; abbi -> Ab_Bi-transposed).
    setenv("PI0_GEMM_VARIANT", "l4lm", 1);
    upload_weights(q4_std.a, q4, hA);
    unsetenv("PI0_GEMM_VARIANT");
    upload_weights(q4_adreno.a, q4, hA);
    upload_weights(f16_case.a, q4, hA);
    // abbi_f16 reads weights in the M-contiguous transposed [K][M] layout (half at k*M+m), the f16
    // analog of the q4_0 Ab_Bi packing. hA is standard [M][K] (hA[m*K+k]); transpose into hT.
    std::vector<ggml_fp16_t> hT(na);
    for (int64_t mm = 0; mm < M; ++mm)
        for (int64_t kk = 0; kk < K; ++kk)
            hT[kk * M + mm] = hA[mm * K + kk];
    ggml_backend_tensor_set(f16_abbi.a, hT.data(), 0, hT.size() * sizeof(ggml_fp16_t));
    upload_b(q4_adreno.b);
    upload_b(q4_std.b);
    upload_b(f16_case.b);
    upload_b(f16_abbi.b);

    // Time each config (min over iters). Capture C for the correctness pass.
    // The q4_0 Ab_Bi grid needs M a multiple of 512 (gws[1]=M/4 % 128 == 0); skip it otherwise
    // (e.g. M=256 for AV) so the f16 variants can still be benchmarked.
    const bool q4_ok = (M % 512 == 0);
    std::vector<float> c_abbi, c_gb, c_l4lm, c_f16, c_abbi_f16;
    const double ms_abbi     = q4_ok ? time_case(backend, q4_adreno, "abbi",     iters, warmup, &c_abbi) : 0.0;
    const double ms_gb       = q4_ok ? time_case(backend, q4_adreno, "abbi_gb",  iters, warmup, &c_gb)   : 0.0;
    const double ms_l4lm     = q4_ok ? time_case(backend, q4_std,    "l4lm",     iters, warmup, &c_l4lm) : 0.0;
    const double ms_f16      = time_case(backend, f16_case,  nullptr,    iters, warmup, &c_f16);
    const double ms_abbi_f16 = time_case(backend, f16_abbi,  "abbi_f16", iters, warmup, &c_abbi_f16);

    auto row = [&](const char * label, const char * path, double ms) {
        printf("  %-9s %-26s  mean %8.3f ms   %8.1f GFLOPS\n", label, path, ms, flop / (ms * 1e6));
    };
    printf("kernel / B-load path:\n");
    row("abbi",     "q4_0 Ab_Bi (B=image)",     ms_abbi);
    row("abbi_f16", "f16  Ab_Bi (B=image)",     ms_abbi_f16);
    row("abbi_gb",  "q4_0 Ab_Bi (B=global)",    ms_gb);
    row("l4lm",     "q4_0 l4_lm (LDS, B=glob)", ms_l4lm);
    row("f16",      "f16  l4_lm (LDS ref)",     ms_f16);

    // The two decisive numbers for the attention-GEMM port:
    printf("\n  f16-weight cost (4x bytes vs q4_0):  abbi_f16 vs abbi = %.2fx\n", ms_abbi_f16 / ms_abbi);
    printf("  win over today's attention GEMM:     abbi_f16 vs f16  = %.2fx\n", ms_f16 / ms_abbi_f16);

    // Decisive number: how much the texture path buys on the otherwise-identical kernel.
    printf("\n  texture(image) speedup on Ab_Bi:  abbi vs abbi_gb = %.2fx  (%.1f%% %s)\n",
           ms_gb / ms_abbi, std::fabs(1.0 - ms_abbi / ms_gb) * 100.0,
           ms_abbi < ms_gb ? "faster with image" : "no image benefit");

    // ---- q4_K operating point: the Adreno-optimal q4_K GEMM is the existing 8x4 image-B
    // kernel kernel_gemm_noshuffle_q4_1_f32 (auto-routed for Q4_1 src0). It is the q4_0
    // Ab_Bi_8x4 kernel plus a per-32 fp16 `min` array: dequant `nibble*scale + min` instead
    // of `(nibble-8)*scale`. q4_K's GPU-hostile 6-bit super-block scales/mins are decoded on
    // the host into effective per-32 fp16 {scale,min} == exactly the q4_1 SoA layout, so the
    // 4-bit quants stay on q4_0's fast path and the ONLY extra weight traffic is one half4 min
    // vload per 32-block (+0.5 bit/weight on the small, cached scale path). Values are real
    // q4_K (quantize Q4_K -> dequant -> repack Q4_1, ~lossless on the affine 16-grid), so this
    // measures q4_K accuracy at the q4_1-layout speed. Opt-in PI0_GEMM_Q4K so the default run's
    // per-kernel DEVICE summary stays clean. Needs M%512 (same 8x4 grid as abbi) and K%256.
    if (q4_ok && getenv("PI0_GEMM_Q4K") && (K % 256 == 0)) {
        // W=q8_0 (A=f16): fills the q8_0 weight row of the quant-sensitivity table. Auto-routes to
        // the q8_0 Adreno 8x4 image-B path (enable_adreno_trans_weight); no variant needed.
        {
            unsetenv("PI0_GEMM_VARIANT");
            std::vector<uint8_t> q8(ggml_row_size(GGML_TYPE_Q8_0, K) * M);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, srcA.data(), q8.data(), 0, M, K, nullptr);
            gemm_case q8_case = build_case(backend, GGML_TYPE_Q8_0, M, N, K);
            ggml_backend_tensor_set(q8_case.a, q8.data(), 0, q8.size());
            upload_b(q8_case.b);
            std::vector<float> c_q8;
            const double ms_q8 = time_case(backend, q8_case, nullptr, iters, warmup, &c_q8);
            printf("\nW-precision sweep (A=f16, Ab_Bi 8x4)  [M=%d N=%d K=%d]:\n", M, N, K);
            printf("  q8_0   W=q8_0  8x4 (B=image)   mean %8.3f ms   %8.1f GFLOPS\n", ms_q8, flop/(ms_q8*1e6));
            ggml_gallocr_free(q8_case.alloc); ggml_free(q8_case.ctx);
        }

        std::vector<uint8_t> q4k(ggml_row_size(GGML_TYPE_Q4_K, K) * M);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, srcA.data(), q4k.data(), 0, M, K, nullptr);
        std::vector<float> Wk(na);
        ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(q4k.data(), Wk.data(), na); // exact q4_K reconstruction
        std::vector<uint8_t> q41(ggml_row_size(GGML_TYPE_Q4_1, K) * M);
        ggml_quantize_chunk(GGML_TYPE_Q4_1, Wk.data(), q41.data(), 0, M, K, nullptr); // q4_K values in q4_1 format

        gemm_case q4k_case = build_case(backend, GGML_TYPE_Q4_1, M, N, K);
        ggml_backend_tensor_set(q4k_case.a, q41.data(), 0, q41.size()); // -> SoA {q,d,m} convert+transpose
        upload_b(q4k_case.b);

        std::vector<float> c_q4k;
        const double ms_q4k = time_case(backend, q4k_case, nullptr, iters, warmup, &c_q4k); // q4_1 auto-routes

        printf("\nq4_K (q4_1-layout) Ab_Bi_8x4 GEMM  [M=%d N=%d K=%d]:\n", M, N, K);
        printf("  q4k    q4_K via q4_1 8x4 (B=image)  mean %8.3f ms   %8.1f GFLOPS\n", ms_q4k, flop/(ms_q4k*1e6));
        printf("  abbi   q4_0       8x4 (B=image)     mean %8.3f ms   %8.1f GFLOPS\n", ms_abbi, flop/(ms_abbi*1e6));
        printf("  min-array cost (q4_K 5b vs q4_0 4.5b):  %.2fx slower  (%.1f%%)\n",
               ms_abbi > 0 ? ms_q4k/ms_abbi : 0.0, ms_abbi > 0 ? (ms_q4k/ms_abbi - 1.0)*100.0 : 0.0);
        printf("  accuracy (max rel diff vs f16 ref):  q4_K %.2e   q4_0 %.2e\n",
               max_rel_diff(c_q4k, c_f16), max_rel_diff(c_abbi, c_f16));

        // --- native q4_K via the new uint8-sub-scale kernel (kernel_mul_mat_Ab_Bi_8x4_q4k) ---
        // SAME q4_K values (uploaded as native q4_K bytes), but the Adreno-co-designed layout:
        // interleaved uint8 sub-scale+min + interleaved fp16 super -> k-quant metadata traffic ==
        // q4_0's single scale array (vs the q4_1 expansion's 2x). Env set BEFORE upload so set_tensor
        // runs the host q4_K decode (mirrors the l4lm upload-time gating).
        setenv("PI0_GEMM_VARIANT", "q4k", 1);
        gemm_case q4kn_case = build_case(backend, GGML_TYPE_Q4_K, M, N, K);
        ggml_backend_tensor_set(q4kn_case.a, q4k.data(), 0, q4k.size()); // native q4_K bytes
        upload_b(q4kn_case.b);
        std::vector<float> c_q4kn;
        const double ms_q4kn = time_case(backend, q4kn_case, "q4k", iters, warmup, &c_q4kn);
        unsetenv("PI0_GEMM_VARIANT");

        printf("  q4kn   q4_K native 8x4 (uint8 sub)  mean %8.3f ms   %8.1f GFLOPS  (vs q4_0 %.2fx, %+.1f%%)\n",
               ms_q4kn, flop/(ms_q4kn*1e6),
               ms_abbi  > 0 ? ms_q4kn/ms_abbi : 0.0, ms_abbi > 0 ? (ms_q4kn/ms_abbi - 1.0)*100.0 : 0.0);
        printf("  q4kn vs q4_1-layout speed: %.2fx   correctness vs q4_1-layout (same q4_K values): %.2e\n",
               ms_q4k > 0 ? ms_q4kn/ms_q4k : 0.0, max_rel_diff(c_q4kn, c_q4k));
        ggml_gallocr_free(q4kn_case.alloc); ggml_free(q4kn_case.ctx);

        ggml_gallocr_free(q4k_case.alloc); ggml_free(q4k_case.ctx);
    }

    // Extended probes (hot-bank sweep + B-only). Opt-in via PI0_GEMM_FULL, because their
    // extra dispatches pollute the per-kernel DEVICE summary for abbi/abbi_gb (more calls).
    // Default run = just the 4 configs above, so host AND device timers stay clean.
    if (getenv("PI0_GEMM_FULL")) {
    // Hot-bank probe: reshape the 128-lane wave so it spans LWS0 distinct gy -> LWS0
    // distinct B addresses (128/LWS0 lanes share each). If globalB's slowness is from all
    // 128 lanes hammering ONE B address/bank, spreading them should speed globalB up; abbi
    // (texture broadcast) should be ~flat. Same math throughout (verified above).
    printf("\nhot-bank probe: vary B-address sharing per wave (workgroup = 128 lanes)\n");
    printf("  LWS0  lanes/B-addr      abbi GFLOPS   abbi_gb GFLOPS   gb speedup vs LWS0=1\n");
    double gb_base = 0.0;
    for (int l0 : {1, 2, 4, 8, 16}) {
        char buf[8]; snprintf(buf, sizeof(buf), "%d", l0);
        setenv("PI0_GEMM_LWS0", buf, 1);
        const double a = time_case(backend, q4_adreno, "abbi",    iters, warmup, nullptr);
        const double g = time_case(backend, q4_adreno, "abbi_gb", iters, warmup, nullptr);
        if (l0 == 1) gb_base = g;
        printf("  %-5d %-15d  %11.1f   %14.1f   %.2fx\n",
               l0, 128 / l0, flop / (a * 1e6), flop / (g * 1e6), gb_base / g);
    }
    unsetenv("PI0_GEMM_LWS0");
    printf("  (hot-bank hypothesis: abbi_gb GFLOPS rises as lanes/B-addr falls; abbi ~flat.\n");
    printf("   the abbi_gb plateau / LWS0=1 ratio approximates the lost bank parallelism.)\n");
    } // PI0_GEMM_FULL (hot-bank sweep)

    // B-only probe: opt-in via PI0_GEMM_BONLY (or PI0_GEMM_FULL). Skips the sweep, so it's a
    // SHORT run (no thermal build-up) and the per-kernel device summary stays clean for
    // abbi/abbi_gb (35 calls each) -> same-run, same-thermal-state decomposition below.
    if (getenv("PI0_GEMM_FULL") || getenv("PI0_GEMM_BONLY")) {
    // Positive test: stream B ONLY (no weights, A entirely absent) via image vs global, at
    // the exact same B addresses. If image is still ~4x faster with A gone, the bottleneck
    // is the B read path itself (dedicated TP L1), not L2 contention with the weights A.
    // If they equalize, A-contention was the cause. (Output is garbage; timing-only.)
    const double ms_bi = time_case(backend, q4_adreno, "bonly_i", iters, warmup, nullptr);
    const double ms_bg = time_case(backend, q4_adreno, "bonly_g", iters, warmup, nullptr);
    const double ms_ao = time_case(backend, q4_adreno, "aonly",   iters, warmup, nullptr);
    const double ms_bi8 = time_case(backend, q4_adreno, "bonly_i8", iters, warmup, nullptr);
    const double ms_bf4 = time_case(backend, q4_adreno, "bonly_f4", iters, warmup, nullptr);
    const double ms_bnr = time_case(backend, q4_adreno, "bonly_noread", iters, warmup, nullptr);
    const double ms_bq8 = time_case(backend, q4_adreno, "bonly_q8", iters, warmup, nullptr);
    printf("\nB-only stream (no weights -> A absent), same B addresses:\n");
    printf("  bonly_i   B=image half4 (64b,4N/fetch)   mean %8.3f ms\n", ms_bi);
    printf("  bonly_i8  B=image int8  (32b,4N/fetch)   mean %8.3f ms     i8/f16 = %.2fx  (byte test: 0.5x=byte-bound)\n", ms_bi8, ms_bi8 / ms_bi);
    printf("  bonly_f4  B=image float4(128b,8N/fetch)  mean %8.3f ms     f4/f16 = %.2fx  (FETCH-WIDTH test: 0.5x=fetch>=128b)\n", ms_bf4, ms_bf4 / ms_bi);
    printf("  bonly_nr  accumulates only, NO reads     mean %8.3f ms     nr/f16 = %.2fx  (read-bottleneck test: ~1.0x=reads free)\n", ms_bnr, ms_bnr / ms_bi);
    printf("  bonly_q8  B=RGBA-int16 (8 int8/64b fetch)mean %8.3f ms     q8/f16 = %.2fx  (CORRECT q8: 0.5-0.7x=2x fewer fetches)\n", ms_bq8, ms_bq8 / ms_bi);
    printf("  bonly_g   B=global                       mean %8.3f ms     image/global = %.2fx\n", ms_bg, ms_bg / ms_bi);
    printf("  (bonly_q8 = HALF the 64-bit fetches of bonly_i for same 8N. <bonly_i -> q8-packed B is the real B-read lever.)\n");

    // Probe A: weight-path scaling. aonly with gy-grid divided (emulates larger N_tile at neutral
    // occupancy). ~0.5x per halving -> weight-path is traffic/re-read bound -> N_tile up halves it.
    printf("\nProbe A: weight-path vs N_tile (aonly, gy-grid / GYDIV; raw ms, each does 1/GYDIV the N):\n");
    double a_base = 0.0;
    for (int d : {1, 2, 4}) {
        char buf[8]; snprintf(buf, sizeof(buf), "%d", d);
        setenv("PI0_GEMM_GYDIV", buf, 1);
        const double a = time_case(backend, q4_adreno, "aonly", iters, warmup, nullptr);
        if (d == 1) a_base = a;
        printf("  GYDIV=%d (N_tile x%d)   mean %8.3f ms   vs GYDIV1 = %.2fx  (ideal if traffic-bound: %.2fx)\n",
               d, d, a, a_base > 0 ? a / a_base : 0.0, 1.0 / d);
    }
    unsetenv("PI0_GEMM_GYDIV");
    printf("  (~1/GYDIV -> weight-path traffic-bound -> 16x2's N_tile=16 halves weight-path. flat -> cached, won't help.)\n");

    // A-only vs B-only: the weight (L2) + dequant path vs the B read paths, each with the
    // OTHER operand absent. Tests whether the full image kernel's two data streams are balanced.
    printf("\nA-only stream (weights q4_0 + dequant, no B) vs B-only paths:\n");
    printf("  aonly    A=global (q4+dequant)  mean %8.3f ms\n", ms_ao);
    printf("  bonly_i  B=image  (TP L1)       mean %8.3f ms     aonly/bonly_i = %.2fx\n", ms_bi, ms_bi > 0 ? ms_ao / ms_bi : 0.0);
    printf("  bonly_g  B=global (L2)          mean %8.3f ms     aonly/bonly_g = %.2fx\n", ms_bg, ms_bg > 0 ? ms_ao / ms_bg : 0.0);
    printf("  (aonly ~= bonly_i -> weight(L2)+dequant ~ B(TP L1): full image kernel is balanced\n");
    printf("   across its two data paths. aonly << bonly_i -> weights cheap, B@TP-L1 dominates.\n");
    printf("   aonly >> bonly_i -> the weight stream dominates.)\n");

    // Store (dst write-back) cost: abbi_nostore is a separate clone of the full image kernel with
    // the 8 vstore4 blocks removed (compute kept alive via a scalar collapse under a never-true
    // guard) -> identical compute, zero DRAM write-back. Re-time abbi adjacently (same thermal).
    const double ms_abbi2   = time_case(backend, q4_adreno, "abbi",         iters, warmup, nullptr);
    const double ms_nostore = time_case(backend, q4_adreno, "abbi_nostore", iters, warmup, nullptr);
    printf("\nstore (dst write-back) cost, full image kernel:\n");
    printf("  abbi          full (+store)  mean %8.3f ms\n", ms_abbi2);
    printf("  abbi_nostore  store skipped  mean %8.3f ms\n", ms_nostore);
    printf("  -> dst write-back ~= %.1f ms (%.1f%% of full); C = M*N*4 = %.0f MB streamed to DRAM\n",
           ms_abbi2 - ms_nostore, ms_abbi2 > 0 ? (ms_abbi2 - ms_nostore) / ms_abbi2 * 100.0 : 0.0,
           (double)M * N * 4.0 / (1024.0*1024.0));

    // Compute- vs memory-bound test: lowcompute keeps FULL memory (all B reads + weight vload4)
    // but only 1 of 4 dequant+FMA per j (single accumulator) -> ~1/4 compute, memory unchanged.
    // compute-bound -> time drops toward the memory floor (~bonly_i); memory-bound -> unchanged.
    const double ms_lc = time_case(backend, q4_adreno, "lowcompute", iters, warmup, nullptr);
    printf("\ncompute- vs memory-bound (full memory, ~1/4 compute):\n");
    printf("  abbi        full compute       mean %8.3f ms\n", ms_abbi2);
    printf("  lowcompute  ~1/4 FMA+dequant   mean %8.3f ms   (dropped %.1f ms, %.1f%%)\n",
           ms_lc, ms_abbi2 - ms_lc, ms_abbi2 > 0 ? (ms_abbi2 - ms_lc) / ms_abbi2 * 100.0 : 0.0);
    printf("  bonly_i (B-only memory floor)  mean %8.3f ms\n", ms_bi);
    printf("  -> lowcompute ~= memory floor -> COMPUTE-bound (abbi's ~9ms over bonly_i was FMA);\n");
    printf("     lowcompute ~= abbi          -> MEMORY-bound (cutting compute didn't help).\n");

    // Same-run, same-thermal-state decomposition (host mean). Removing A drops weight loads
    // + dequant + the MAC multiply. For image, B sits in TP L1 so A's L2 traffic never
    // touched it -> the image saving is just (compute + weight-load), exposed. For global, B
    // shares L2 with A -> the EXTRA saving beyond image's is the A<->B L2 contention.
    const double save_img = ms_abbi - ms_bi;
    const double save_glb = ms_gb   - ms_bg;
    printf("\nremoving A (full -> B-only), same run:\n");
    printf("  image   %7.1f -> %7.1f ms   saved %6.1f ms\n", ms_abbi, ms_bi, save_img);
    printf("  global  %7.1f -> %7.1f ms   saved %6.1f ms\n", ms_gb,   ms_bg, save_glb);
    printf("  -> image saving ~= exposed compute+weight-load; global's EXTRA saving =\n");
    printf("     A<->B L2 contention ~= %.1f ms (%.1fx of image's)\n",
           save_glb - save_img, save_img > 0 ? save_glb / save_img : 0.0);
    } // B-only probe

    // Staggered-start probe: opt-in via PI0_GEMM_STAGGER. Same global-B GEMM but each lane
    // starts its K-walk at a different 4-K block, so the 128 lanes of a wave read 128 DISTINCT
    // B addresses per step instead of broadcasting one. Tests whether scattering the global B
    // reads beats the broadcast global path -- and how far it stays from the image/TP-L1 path.
    if (getenv("PI0_GEMM_FULL") || getenv("PI0_GEMM_STAGGER")) {
    std::vector<float> c_stag;
    const double ms_stag = time_case(backend, q4_adreno, "abbi_gb_stag", iters, warmup, &c_stag);
    printf("\nstaggered-start global B (each lane starts at a distinct 4-K block):\n");
    row("abbi",         "Ab_Bi_8x4 (B=image)",          ms_abbi);
    row("abbi_gb",      "Ab_Bi_8x4 (B=global,bcast)",   ms_gb);
    row("abbi_gb_stag", "Ab_Bi_8x4 (B=global,stagger)", ms_stag);
    printf("  stagger vs broadcast (global): %.2fx  (%+.1f%% time vs bcast)\n",
           ms_gb / ms_stag, (ms_stag / ms_gb - 1.0) * 100.0);
    printf("  stagger vs image:              %.2fx slower than image\n", ms_stag / ms_abbi);
    printf("  correctness vs abbi (reordered per-lane K-sum -> small fp diff expected): %.2e\n",
           max_rel_diff(c_abbi, c_stag));
    printf("  (stagger FASTER than bcast -> broadcast had a hidden same-address serialization;\n");
    printf("   stagger SLOWER/EQUAL -> broadcast coalescing was already optimal,\n");
    printf("   so the image win is the dedicated TP L1 read path, not avoiding same-addr contention.)\n");
    } // staggered probe

    // ===== E1: FMA-peak microbench (calibrate the real ALU peak P_fma) =====
    // Pure-FMA kernel, no memory; 8 half8 accumulators (64-way ILP) -> THROUGHPUT not latency.
    // Each thread does k iters * 8 half8 FMAs; threads = ceil(N/8)*(M/4). Compares the measured
    // peak against abbi's achieved GFLOPS -> the true utilization %. Opt-in: PI0_GEMM_E1.
    if (q4_ok && (getenv("PI0_GEMM_E1") || getenv("PI0_GEMM_FULL"))) {
        const double ms_fp = time_case(backend, q4_adreno, "fma_peak", iters, warmup, nullptr);
        const double threads = (double)((N + 7) / 8) * (double)(M / 4);
        const double fmas    = threads * (double)K * 8.0 /*half8 accs*/ * 8.0 /*lanes/half8*/;
        const double p_fma   = 2.0 * fmas / (ms_fp * 1e6); // GFLOPS
        const double achieved = flop / (ms_abbi * 1e6);
        printf("\nE1  FMA-peak microbench (no memory, 64-way ILP):\n");
        printf("  fma_peak   mean %8.3f ms   P_fma = %8.1f GFLOPS  (%.2f TF)\n", ms_fp, p_fma, p_fma/1e3);
        printf("  abbi achieved %8.1f GFLOPS  ->  utilization = %.1f%% of measured P_fma\n",
               achieved, 100.0 * achieved / p_fma);
        printf("  (P_fma ~= abbi -> GEMM already at the real FMA ceiling = issue/throughput bound;\n");
        printf("   P_fma >> abbi -> the GEMM's loss is non-FMA (memory/mix/occupancy) -> E2..E5.)\n");
    }

    // ===== E2: dummy-FMA injection (the bubble / free-decode test) =====
    // Production GEMM + N injected independent FMAs/outer-iter. Real GFLOPS counts ONLY real MACs.
    // Flat-then-drop -> there are free issue slots (free-decode budget = the knee). Immediate
    // proportional drop -> issue/mix-bound, no spare slots. Opt-in: PI0_GEMM_E2.
    if (q4_ok && (getenv("PI0_GEMM_E2") || getenv("PI0_GEMM_FULL"))) {
        printf("\nE2  dummy-FMA injection (real GFLOPS = real MACs only):\n");
        printf("  inj   mean ms     real GFLOPS    vs inj=0\n");
        double base = 0.0;
        for (int inj : {0, 2, 4, 8, 12, 16}) {
            char buf[8]; snprintf(buf, sizeof(buf), "%d", inj);
            setenv("PI0_GEMM_INJECT", buf, 1);
            const double ms = time_case(backend, q4_adreno, "inject", iters, warmup, nullptr);
            const double g  = flop / (ms * 1e6);
            if (inj == 0) base = g;
            printf("  %-4d  %8.3f    %10.1f    %.3fx\n", inj, ms, g, base > 0 ? g / base : 0.0);
        }
        unsetenv("PI0_GEMM_INJECT");
        printf("  (flat until inj=k then drops -> k free FMA slots/iter (free-decode budget);\n");
        printf("   drops from inj=2 -> issue/mix-bound, no spare slots.)\n");
    }

    // ===== E5: N-tile sweep (weight-reuse / bandwidth lever) =====
    // Wider N-tile re-reads each M-block's weights fewer times (8x4: ceil(N/8)x; 16x4: ceil(N/16)x;
    // 32x4: ceil(N/32)x) -> less weight DRAM traffic (the prefix bottleneck), at the cost of 2x/4x
    // accumulator registers (lower occupancy). Needs N%16/ N%32 == 0 (use -N 768). Opt-in PI0_GEMM_E5.
    if (q4_ok && (getenv("PI0_GEMM_E5") || getenv("PI0_GEMM_FULL"))) {
        printf("\nE5  N-tile sweep (weight reuse vs occupancy), shape M=%d N=%d K=%d:\n", M, N, K);
        if (N % 32 != 0) {
            printf("  (skipped: N=%d not a multiple of 32; rerun with -N 768)\n", N);
        } else {
            std::vector<float> c_n16, c_n32;
            const double ms_8  = time_case(backend, q4_adreno, "abbi", iters, warmup, nullptr);
            const double ms_16 = time_case(backend, q4_adreno, "n16",  iters, warmup, &c_n16);
            const double ms_32 = time_case(backend, q4_adreno, "n32",  iters, warmup, &c_n32);
            printf("  tile   weight re-reads   mean ms     GFLOPS    speedup   maxreldiff vs abbi\n");
            printf("  8x4    ceil(N/8)=%-4d    %8.3f   %8.1f    1.00x     --\n",      (N+7)/8,  ms_8,  flop/(ms_8 *1e6));
            printf("  16x4   ceil(N/16)=%-4d   %8.3f   %8.1f    %.2fx     %.2e\n", (N+15)/16, ms_16, flop/(ms_16*1e6), ms_8/ms_16, max_rel_diff(c_abbi, c_n16));
            printf("  32x4   ceil(N/32)=%-4d   %8.3f   %8.1f    %.2fx     %.2e\n", (N+31)/32, ms_32, flop/(ms_32*1e6), ms_8/ms_32, max_rel_diff(c_abbi, c_n32));
            printf("  (speedup>1 -> weight-reuse win dominates; plateau/regression -> occupancy drop\n");
            printf("   from 2x/4x registers cancels it. maxreldiff ~0 confirms correctness.)\n");
        }
    }

    // ===== E6: kernel-level micro-optimizations =====
    // aimg = weights via texture (image1d_buffer) instead of global/UCHE (Q3, primary);
    // noguard = store boundary checks removed (Q1); unroll = K-loop unroll 2 (Q2).
    // All vs abbi at the current shape. Opt-in PI0_GEMM_E6 (noguard/unroll need N%8==0).
    if (q4_ok && (getenv("PI0_GEMM_E6") || getenv("PI0_GEMM_FULL"))) {
        printf("\nE6  kernel micro-opts vs abbi, shape M=%d N=%d K=%d:\n", M, N, K);
        std::vector<float> c_ai, c_ng, c_ur;
        const double ms_b  = time_case(backend, q4_adreno, "abbi",    iters, warmup, nullptr);
        const double ms_ai = time_case(backend, q4_adreno, "aimg",    iters, warmup, &c_ai);
        const double ms_ng = time_case(backend, q4_adreno, "noguard", iters, warmup, &c_ng);
        const double ms_ur = time_case(backend, q4_adreno, "unroll",  iters, warmup, &c_ur);
        printf("  variant   what                       mean ms    GFLOPS   speedup   maxreldiff\n");
        printf("  abbi      q4_0 weights via UCHE       %8.3f  %8.1f   1.00x     --\n",      ms_b,  flop/(ms_b *1e6));
        printf("  aimg      q4_0 weights via TEXTURE    %8.3f  %8.1f   %.2fx     %.2e\n", ms_ai, flop/(ms_ai*1e6), ms_b/ms_ai, max_rel_diff(c_abbi, c_ai));
        printf("  noguard   store guards removed        %8.3f  %8.1f   %.2fx     %.2e\n", ms_ng, flop/(ms_ng*1e6), ms_b/ms_ng, max_rel_diff(c_abbi, c_ng));
        printf("  unroll    K-loop unroll 2             %8.3f  %8.1f   %.2fx     %.2e\n", ms_ur, flop/(ms_ur*1e6), ms_b/ms_ur, max_rel_diff(c_abbi, c_ur));
        printf("  (aimg>1 -> texture cache helps weight delivery (stacks with quant); ~1 -> texture\n");
        printf("   win was B's broadcast, weights don't share it. noguard/unroll<=1 -> occupancy (E5).)\n");
    }

    // Correctness: every config must agree with abbi (else its layout/routing is wrong).
    printf("\ncorrectness (max rel diff vs abbi):\n");
    printf("  abbi_gb %.2e   l4lm %.2e   f16 %.2e   abbi_f16 %.2e\n",
           max_rel_diff(c_abbi, c_gb), max_rel_diff(c_abbi, c_l4lm), max_rel_diff(c_abbi, c_f16),
           max_rel_diff(c_abbi, c_abbi_f16));
    printf("  (abbi_gb ~0: identical q4_0 math, only B path differs -> layout/routing sanity.\n");
    printf("   l4lm small: same q4_0 weights, different accumulation order.\n");
    printf("   f16 larger: f16 weights vs q4_0-quantized weights -> quantization error, expected.)\n");

    ggml_gallocr_free(q4_adreno.alloc); ggml_free(q4_adreno.ctx);
    ggml_gallocr_free(q4_std.alloc);    ggml_free(q4_std.ctx);
    ggml_gallocr_free(f16_case.alloc);  ggml_free(f16_case.ctx);
    ggml_gallocr_free(f16_abbi.alloc);  ggml_free(f16_abbi.ctx);
    ggml_backend_free(backend);
    return 0;
}
