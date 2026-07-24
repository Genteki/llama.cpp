// PI0 operator-level co-execution (contention) benchmark.
//
// Produces the g_ij / h_ij matrices for the heterogeneous-scheduling problem:
//   g_ij = GFLOPS of matmul type i on the GPU (OpenCL) while the NPU (HTP) runs type j
//   h_ij = GFLOPS of matmul type i on the NPU (HTP)  while the GPU runs type j
// Column j=0 = the other device idle (solo speed).
//
// Only the matmul operator types are covered here (batch 1). Weights are
// synthesized at the precision each op really uses in PI0 (prefix = q4_0,
// vision & expert-FFN = f16), allocated exactly like the model loader
// (ggml_backend_get_default_buffer_type) so q4_0 lands on the HTP repack path
// and hits HMX. Per-tensor buffers stay < 1 GB (no full-model load, no OpenCL
// alloc-cap issue). Timing is data-independent so weight/input values are dummy.
//
// Concurrency model mirrors pi0-pipeline: the foreground op runs on its backend
// on the main thread while a background thread keeps the OTHER backend busy
// looping type j. Each backend is touched by exactly one thread at a time.
//
// Usage: llama-pi0-opmat-bench [--iters 25]   (needs both OpenCL + HTP0)

#include "pi0-common.h"

#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

struct OpDef { const char * name; int K, M, N; ggml_type wt; };

// matmul: weight [K,M], input [K,N] -> out [M,N]; c = 2*M*K*N FLOP
static OpDef OPS[] = {
    { "vis_proj",  1152,  1152, 256, GGML_TYPE_F16  }, // SigLIP q/k/v/o proj
    { "vis_ff_up", 1152,  4304, 256, GGML_TYPE_F16  }, // SigLIP FFN up
    { "vis_ff_dn", 4304,  1152, 256, GGML_TYPE_F16  }, // SigLIP FFN down
    { "pre_qo",    2048,  2048, 773, GGML_TYPE_Q4_0 }, // PaliGemma q/o proj
    { "pre_ff_up", 2048, 16384, 773, GGML_TYPE_Q4_0 }, // PaliGemma FFN up/gate
    { "pre_ff_dn",16384,  2048, 773, GGML_TYPE_Q4_0 }, // PaliGemma FFN down
    { "fm_ff_up",  1024,  4096,  51, GGML_TYPE_F16  }, // Expert FFN up
    { "fm_ff_dn",  4096,  1024,  51, GGML_TYPE_F16  }, // Expert FFN down
};
static const int NOP = (int) (sizeof(OPS) / sizeof(OPS[0]));

// One backend's synthesized weight set (all NOP weights in one buffer).
struct Weights {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         w[NOP] = {};
};

static Weights make_weights(ggml_backend_t be) {
    Weights W;
    W.ctx = ggml_init({ ggml_tensor_overhead() * (NOP + 2), nullptr, true });
    for (int i = 0; i < NOP; i++) {
        W.w[i] = ggml_new_tensor_2d(W.ctx, OPS[i].wt, OPS[i].K, OPS[i].M);
        ggml_set_name(W.w[i], OPS[i].name);
    }
    // Same allocation call the pi0 model loader uses -> q4_0 lands on the HTP
    // repack path (HMX-eligible), f16 stays plain.
    W.buf = ggml_backend_alloc_ctx_tensors_from_buft(W.ctx, ggml_backend_get_default_buffer_type(be));
    if (!W.buf) { LOG_ERR("weight alloc failed\n"); exit(1); }

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> uni(-0.1f, 0.1f);
    for (int i = 0; i < NOP; i++) {
        const int64_t K = OPS[i].K, M = OPS[i].M;
        if (OPS[i].wt == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> d((size_t) K * M);
            for (auto & x : d) x = ggml_fp32_to_fp16(uni(rng));
            ggml_backend_tensor_set(W.w[i], d.data(), 0, d.size() * sizeof(ggml_fp16_t));
        } else { // q4_0
            std::vector<float> f((size_t) K * M);
            for (auto & x : f) x = uni(rng);
            const size_t qsz = ggml_row_size(OPS[i].wt, K) * M;
            std::vector<uint8_t> q(qsz);
            ggml_quantize_chunk(OPS[i].wt, f.data(), q.data(), 0, M, K, nullptr);
            ggml_backend_tensor_set(W.w[i], q.data(), 0, qsz);
        }
    }
    return W;
}

// A single-op runnable graph (weight * input -> out) on one backend.
struct OpG {
    ggml_context *        ctx = nullptr;
    ggml_cgraph *         gf  = nullptr;
    ggml_backend_t        be  = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    double                c   = 0.0; // GFlop per call
};

static OpG make_graph(ggml_backend_t be, ggml_tensor * w, const OpDef & op) {
    OpG g;
    g.be = be;
    g.ctx = ggml_init({ ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true });
    ggml_tensor * inp = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, op.K, op.N);
    ggml_set_input(inp);
    ggml_tensor * out = ggml_mul_mat(g.ctx, w, inp);
    ggml_set_output(out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be); // allocates inp + out (w already allocated)
    if (!g.buf) { LOG_ERR("graph alloc failed for %s\n", op.name); exit(1); }
    std::vector<float> d((size_t) op.K * op.N, 0.01f);
    ggml_backend_tensor_set(inp, d.data(), 0, d.size() * sizeof(float));
    g.c = 2.0 * op.M * op.K * op.N / 1e9;
    return g;
}

// Foreground op GFLOPS while (optional) bg op keeps the other backend busy.
static double measure(OpG & fg, OpG * bg, int iters) {
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(fg.be, fg.gf); // warmup
    ggml_backend_synchronize(fg.be);

    std::atomic<bool> stop{ false };
    std::thread bgt;
    if (bg) {
        bgt = std::thread([&] {
            while (!stop.load(std::memory_order_relaxed)) ggml_backend_graph_compute(bg->be, bg->gf);
        });
    }
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < iters; i++) ggml_backend_graph_compute(fg.be, fg.gf);
    ggml_backend_synchronize(fg.be);
    const int64_t t1 = ggml_time_us();

    stop.store(true);
    if (bg) bgt.join();

    const double sec = (t1 - t0) / 1e6;
    return sec > 0 ? fg.c * iters / sec : 0.0;
}

static void print_matrix(const char * title, double m[NOP][NOP + 1]) {
    printf("\n=== %s  (GFLOPS) ===\n", title);
    printf("%-10s %8s", "fg \\ bg", "idle");
    for (int j = 0; j < NOP; j++) printf(" %8s", OPS[j].name);
    printf("\n");
    for (int i = 0; i < NOP; i++) {
        printf("%-10s", OPS[i].name);
        for (int j = 0; j <= NOP; j++) printf(" %8.0f", m[i][j]);
        printf("\n");
    }
}

int main(int argc, char ** argv) {
    int iters = 25;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    }

    ggml_backend_load_all();
    ggml_backend_t be_ocl = pi0_init_backend("OpenCL");
    ggml_backend_t be_htp = pi0_init_backend("HTP0");
    if (!be_ocl || !be_htp) {
        LOG_ERR("need BOTH OpenCL and HTP0 backends (got ocl=%p htp=%p)\n", (void *) be_ocl, (void *) be_htp);
        return 1;
    }
    LOG_INF("OpenCL + HTP0 up; synthesizing %d matmul types, iters=%d\n", NOP, iters);

    Weights W_ocl = make_weights(be_ocl);
    Weights W_htp = make_weights(be_htp);

    OpG G[NOP][2]; // [op][0=OpenCL, 1=HTP]
    for (int i = 0; i < NOP; i++) {
        G[i][0] = make_graph(be_ocl, W_ocl.w[i], OPS[i]);
        G[i][1] = make_graph(be_htp, W_htp.w[i], OPS[i]);
    }

    printf("\n=== operator types (matmul) ===\n");
    printf("%-10s %-22s %8s\n", "type", "dims (K->M, N=seq)", "c GFlop");
    for (int i = 0; i < NOP; i++) {
        char dims[48];
        snprintf(dims, sizeof(dims), "%d->%d, %d", OPS[i].K, OPS[i].M, OPS[i].N);
        printf("%-10s %-22s %8.2f  [%s]\n", OPS[i].name, dims, G[i][0].c,
               OPS[i].wt == GGML_TYPE_Q4_0 ? "q4_0" : "f16");
    }

    double g[NOP][NOP + 1], h[NOP][NOP + 1];

    // g_ij: GPU foreground, NPU background
    for (int i = 0; i < NOP; i++)
        for (int j = 0; j <= NOP; j++)
            g[i][j] = measure(G[i][0], j == 0 ? nullptr : &G[j - 1][1], iters);

    // h_ij: NPU foreground, GPU background
    for (int i = 0; i < NOP; i++)
        for (int j = 0; j <= NOP; j++)
            h[i][j] = measure(G[i][1], j == 0 ? nullptr : &G[j - 1][0], iters);

    print_matrix("g_ij : GPU(OpenCL) op i while NPU runs op j", g);
    print_matrix("h_ij : NPU(HTP) op i while GPU runs op j", h);

    printf("\n(col 'idle' = other device idle; batch 1 = matmul types only)\n");
    return 0;
}
