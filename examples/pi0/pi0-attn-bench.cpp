// pi0-attn-bench.cpp
//
// Microbenchmark for the PI0 PaliGemma prefix attention GEMMs, at the real
// prefix shapes (head_dim=256, n_q=n_kv=769, n_head=8, n_kv_head=1).
//
// PI0 uses MQA: the 8 query heads share ONE k/v head. The score (Q·Kᵀ) and value
// (scores·V) products are therefore a broadcast batched GEMM (r2 = n_head = 8).
// Because the single k/v is reused across all 8 query heads, the batch can be
// rewritten as ONE large GEMM by folding the 8 heads into the N dimension
// ("collapsed"): same total FLOPs, but one launch with large N instead of 8
// small batched GEMMs.
//
// This bench measures batched-vs-collapsed for both GEMMs so the efficiency of
// each formulation can be compared directly. The K dimension (=head_dim=256 for
// the score GEMM) is the arithmetic-intensity bound; collapsing changes launch /
// occupancy overhead, not that bound.
//
// Routing note (OpenCL backend): the CLBlast path engages only for the *batched*
// score GEMM (it requires ne12>1 && r2>1). Set PI0_ATTN_CLBLAST=1 and
// PI0_ATTN_CLBLAST_MINQ<=769 to route it. The collapsed form has ne12==1, so it
// always uses the native kernel — i.e. with CLBlast on, "batched" measures
// CLBlast and "collapsed" measures native.

#include "pi0-common.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void fill_tensor(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (int64_t i = 0; i < n; ++i) {
        f[i] = (float)((i % 17) - 8) * 0.03125f; // small, bounded — values irrelevant for timing
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
    } else { // GGML_TYPE_F16
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(f.data(), h.data(), n);
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
    }
}

// op: 0 = score (Q·Kᵀ, K=head_dim), 1 = value (scores·V, K=n_kv)
// atype = type of the shared k/v operand (F16 or F32); b (q/scores) is always F32.
static void bench_one(ggml_backend_t backend, const char * label,
                      int op, bool collapsed,
                      int D, int NQ, int NKV, int H,
                      int iters, int warmup, ggml_type atype) {
    ggml_init_params ip = {
        /*.mem_size  =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 1024,
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    // mul_mat(a, b): contraction over ne[0]; a is broadcast over b in dim 2.
    ggml_tensor * a;
    ggml_tensor * b;
    if (op == 0) {                 // score:  k[D,NKV,1] · q[D,NQ,H] -> [NKV,NQ,H]
        a = ggml_new_tensor_3d(ctx, atype, D, NKV, 1);
        b = collapsed ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ * H, 1)
                      : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ, H);
    } else {                       // value:  v[NKV,D,1] · kq[NKV,NQ,H] -> [D,NQ,H]
        a = ggml_new_tensor_3d(ctx, atype, NKV, D, 1);
        b = collapsed ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NKV, NQ * H, 1)
                      : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NKV, NQ, H);
    }
    ggml_set_input(a);
    ggml_set_input(b);

    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_set_output(c);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        printf("  %-26s  alloc FAILED\n", label);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return;
    }

    fill_tensor(a);
    fill_tensor(b);

    for (int i = 0; i < warmup; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    double min_ms = 1e30;
    for (int i = 0; i < iters; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const double ms = (ggml_time_us() - t0) / 1000.0;
        if (ms < min_ms) min_ms = ms;
    }

    // Same total FLOPs for batched and collapsed: 2 * NKV * NQ * D * H.
    const double flop   = 2.0 * (double)NKV * NQ * D * H;
    const double gflops = flop / (min_ms * 1e6);
    printf("  %-26s  min %8.3f ms   %8.1f GFLOPS\n", label, min_ms, gflops);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string backend_name = "OpenCL";
    int iters = 30, warmup = 5;
    int D = PALI_HEAD_DIM, NQ = 769, H = PALI_N_HEAD, NKV = -1; // NKV<0 -> NKV=NQ
    ggml_type atype = GGML_TYPE_F16; // type of the shared k/v operand

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto take = [&](int & v) { if (i + 1 < argc) v = atoi(argv[++i]); };
        if      (a == "--backend" && i + 1 < argc) backend_name = argv[++i];
        else if (a == "-n")        take(iters);
        else if (a == "--warmup")  take(warmup);
        else if (a == "--prefix")  take(NQ);
        else if (a == "--nkv")     take(NKV);
        else if (a == "-D")        take(D);
        else if (a == "-H")        take(H);
        else if (a == "--a-type" && i + 1 < argc) {
            const std::string t = argv[++i];
            atype = (t == "f32") ? GGML_TYPE_F32 : GGML_TYPE_F16;
        }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (NKV < 0) NKV = NQ;

    pi0_dump_backends();
    ggml_backend_t backend = pi0_init_backend(backend_name);
    if (!backend) {
        fprintf(stderr, "backend '%s' unavailable\n", backend_name.c_str());
        return 1;
    }

    const char * clb  = getenv("PI0_ATTN_CLBLAST");
    const char * minq = getenv("PI0_ATTN_CLBLAST_MINQ");
    printf("\nPI0 prefix attention GEMM bench   backend=%s\n", backend_name.c_str());
    printf("shapes: head_dim=%d  n_q=%d  n_kv=%d  n_head=%d  n_kv_head=1  (MQA, r2=%d)\n",
           D, NQ, NKV, H, H);
    const char * tuned = getenv("PI0_CLBLAST_TUNED");
    printf("iters=%d  warmup=%d  a-type=%s  CLBLAST=%s  MINQ=%s  TUNED=%s  (batched uses CLBlast; collapsed always native)\n\n",
           iters, warmup, atype == GGML_TYPE_F32 ? "f32" : "f16",
           clb ? clb : "0", minq ? minq : "-", tuned ? tuned : "0");

    printf("Score  Q·Kᵀ    (M=n_kv=%d  N=n_q*[1|%d]  K=head_dim=%d):\n", NKV, H, D);
    bench_one(backend, "batched   (8 x small GEMM)", 0, false, D, NQ, NKV, H, iters, warmup, atype);
    bench_one(backend, "collapsed (1 x N*8 GEMM)",   0, true,  D, NQ, NKV, H, iters, warmup, atype);

    printf("\nValue  scores·V (M=head_dim=%d  N=n_q*[1|%d]  K=n_kv=%d):\n", D, H, NKV);
    bench_one(backend, "batched   (8 x small GEMM)", 1, false, D, NQ, NKV, H, iters, warmup, atype);
    bench_one(backend, "collapsed (1 x N*8 GEMM)",   1, true,  D, NQ, NKV, H, iters, warmup, atype);

    ggml_backend_free(backend);
    return 0;
}
