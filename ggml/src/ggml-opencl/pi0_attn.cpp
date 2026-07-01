// pi0_attn.cpp — PI0 attention GEMM acceleration for the Adreno OpenCL backend.
//
// Routes PI0's two collapsed-MQA attention GEMMs through the f16-weight image-B kernel
// (kernel_mul_mat_Ab_Bi_8x4_f16, the f16 analog of the q4_0 Ab_Bi GEMM):
//     QK = mul_mat(K_cache, Q)     -> scores [n_kv, n_q*n_head]
//     AV = mul_mat(V_t,    scores) -> out    [head_dim, n_q*n_head]
// The "weights" are the f16 KV cache (read via L2, high cross-workgroup reuse); the
// activations (Q / scores) are read via image1d_buffer -> texture-processor L1. Measured
// 1839-2246 GFLOPS device on attention shapes, 3.6-4.3x over the f16 l4_lm path.
//
// This file is #included into ggml-opencl.cpp (it uses that TU's context struct, kernel
// handles, and helpers); CMake does not compile it standalone. Opt-in via PI0_ATTN_IMG=1;
// default leaves every path byte-for-byte unchanged.
//
// Layout note: the Ab_Bi kernel wants src0 weights M-contiguous [K][M] and src1 in the
// transposed f16 "Bi" image. Both KV cache (src0) and Q/scores (src1) are naturally
// [head_dim, seq] (reduction-contiguous), so BOTH operands are transposed per call:
//     src0 (f16) -> kernel_transpose_16_buf -> A_T buffer [K][M] (k*M + m)
//     src1 (f32) -> kernel_transpose_32_16  -> B image (f16, Bi, N zero-padded to mult of 8)

// PI0_ATTN_IMG: 0/unset = off, 1 = QK+AV, 2 = QK only, 3 = AV only (isolation/debug).
static int pi0_attn_img_mode() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("PI0_ATTN_IMG"); v = e ? atoi(e) : 0; }
    return v;
}

// Eligible = a PI0 collapsed attention GEMM. f16 src0, f32 src1, 2D (no batch), contiguous,
// and one of:
//   QK: ne00 (reduction = head_dim) == 256, ne01 (M = n_kv)     >= 64
//   AV: ne01 (M = head_dim)         == 256, ne00 (reduction=n_kv) >= 64
// plus N = dst->ne[1] >= 32. The exact head_dim==256 + 2D gate keeps ViT (batched, ne2>1)
// and FFN/projection (q4_0 weights, or non-256 dims) matmuls out.
static bool pi0_attn_img_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const int mode = pi0_attn_img_mode();
    if (mode == 0) return false;
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) dst->ne[1];
    if (N < 32) return false;
    const bool qk = (K == 256 && M >= 64);   // reduce over head_dim
    const bool av = (M == 256 && K >= 64);   // M = head_dim
    if (qk) return (mode == 1 || mode == 2);
    if (av) return (mode == 1 || mode == 3);
    return false;
}

// PI0_VIT_IMG: route ViT (SigLIP) projection/FFN GEMMs through the SAME f16 image-B kernel.
// Vision weights stay f16 (no quantization -> zero parity risk); the 1D-flatten launch handles any
// M and the kernel's internal Kt padding handles any K, so no M%512 / K%32 padding is needed (unlike
// the q4_0 Ab_Bi path). 0/unset = off, 1 = on. Default off -> every path byte-for-byte unchanged.
static int pi0_vit_img_mode() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("PI0_VIT_IMG"); v = e ? atoi(e) : 0; }
    return v;
}

// Eligible = a ViT proj/FFN GEMM: f16 src0 (weight), f32 src1 (activations), 2D (no batch),
// contiguous, N = dst->ne[1] >= 32. M and K bounded to [512, 8192] keeps this to the SigLIP
// width/MLP GEMMs (1152/4304): excludes attention (head_dim=72 reductions < 512) and the huge
// lm_head (M = vocab > 8192). LM/expert FFN weights are q4_0, not f16, so they're excluded by type.
static bool pi0_vit_img_eligible(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (pi0_vit_img_mode() == 0) return false;
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int N = (int) dst->ne[1];
    return M >= 512 && M <= 8192 && K >= 512 && K <= 8192 && N >= 32;
}

// NOTE: caching the transposed weight (A_T) across passes was tried and REVERTED — the ~800MB of
// persistent device buffers cost far more in clCreateBuffer stalls (+27s on ViT) than the per-call
// transpose saves. Transposing every call with a small, freed-and-reallocated buffer is cheaper
// (driver recycles the pages), and still nets ViT Vision Encode -43% over the l4_lm baseline.
static void ggml_cl_mul_mat_pi0_attn_img(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    cl_context context = backend_ctx->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;
    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offset1 = extra1->offset + src1->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int M = (int) src0->ne[1]; // output rows  (QK: n_kv ; AV: head_dim)
    const int K = (int) src0->ne[0]; // reduction    (QK: head_dim ; AV: n_kv)
    const int N = (int) dst->ne[1];  // output cols  (n_q * n_head)

    cl_int           status;
    cl_buffer_region region;

    // --- src0 (f16 KV) -> A_T [K][M] M-contiguous, via kernel_transpose_16_buf ---
    // out[x*ldo + y] = in[y*ldi + x], x in [0,K), y in [0,M); ldi=K, ldo=M -> A_T[k*M + m].
    // +pad bytes so a partial M-tile (M%4 != 0, e.g. n_kv=774) can over-read up to 3 halfs.
    region.origin = offset0;
    region.size   = (size_t) K * M * sizeof(ggml_fp16_t);
    cl_mem src0_sub = clCreateSubBuffer(extra0->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    // A_T = reused scratch (grow-once), NOT a fresh per-call clCreateBuffer. The per-call 2.6MB device
    // allocation was the bulk of the ViT host gap (~1ms × ~486 GEMMs/pass ≈ 500ms). In-order queue
    // serializes transpose→GEMM→next-transpose, so one buffer is safe to reuse (same as prealloc_act_trans).
    backend_ctx->prealloc_pi0_at.allocate(context, region.size + 64);
    cl_mem A_T = backend_ctx->prealloc_pi0_at.buffer;
    {
        cl_kernel tk = backend_ctx->kernel_transpose_16_buf;
        int ldi = K, ldo = M;
        CL_CHECK(clSetKernelArg(tk, 0, sizeof(cl_mem), &src0_sub));
        CL_CHECK(clSetKernelArg(tk, 1, sizeof(cl_mem), &A_T));
        CL_CHECK(clSetKernelArg(tk, 2, sizeof(int),    &ldi));
        CL_CHECK(clSetKernelArg(tk, 3, sizeof(int),    &ldo));
        size_t tg[3] = { (size_t) K, (size_t) M, 1 };
        backend_ctx->enqueue_ndrange_kernel(tk, 2, tg, NULL, dst);
    }

    // --- src1 (f32 Q/scores) -> B image (f16 Bi, N zero-padded), via kernel_transpose_32_16 ---
    // The B image packs the reduction dim in RGBA texels (4 reductions / texel), so the input's
    // reduction width must be a multiple of 4. QK reduces over head_dim=256 (ok). AV reduces over
    // n_kv (774/825, NOT %4) -> the input rows misalign to texels -> garbage. Fix: copy src1 into a
    // K-padded temp ([N][Kt], Kt = ceil(K/4)*4) so texels align. The GEMM still reduces over real K
    // (k arg below), so the pad columns are never read and need not be zeroed.
    const int extra_elements = N % 8;
    const int padding = extra_elements > 0 ? 8 - extra_elements : 0;
    const int Kt = (K + 3) & ~3; // reduction width rounded up to a multiple of 4 (for the texels)
    region.origin = offset1;
    region.size   = (size_t) K * N * sizeof(float);
    cl_mem B_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    cl_mem B_in_buf = B_sub;   // buffer the input image views (B_sub, or the K-padded temp)
    cl_mem B_pad    = NULL;
    if (Kt != K) {
        B_pad = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) Kt * N * sizeof(float), NULL, &status);
        CL_CHECK(status);
        // strided copy: N rows of K floats from src1 [N][K] into temp [N][Kt] (pad cols unread).
        size_t src_origin[3] = {0,0,0}, dst_origin[3] = {0,0,0};
        size_t rgn[3] = { (size_t) K * sizeof(float), (size_t) N, 1 };
        CL_CHECK(clEnqueueCopyBufferRect(backend_ctx->queue, B_sub, B_pad, src_origin, dst_origin, rgn,
            (size_t) K * sizeof(float), 0, (size_t) Kt * sizeof(float), 0, 0, NULL, NULL));
        B_in_buf = B_pad;
    }
    region.origin = 0;
    region.size   = (size_t) Kt * (N + padding) * sizeof(float) / 2; // /2 for FP16
    backend_ctx->prealloc_act_trans.allocate(context, region.size);
    cl_mem B_d = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    cl_image_format fmt_in  = { CL_RGBA, CL_FLOAT };
    cl_image_desc   desc_in = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)(Kt * N / 4), 0,0,0,0,0,0,0, { B_in_buf } };
    cl_mem B_in_img = clCreateImage(context, 0, &fmt_in, &desc_in, NULL, &status);
    CL_CHECK(status);
    cl_image_format fmt_out  = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc   desc_out = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)(Kt * (N + padding) / 4), 0,0,0,0,0,0,0, { B_d } };
    cl_mem B_image1d = clCreateImage(context, 0, &fmt_out, &desc_out, NULL, &status);
    CL_CHECK(status);
    {
        int height_B = N / 4; if (height_B == 0) { height_B = 1; }
        int width_B  = Kt / 4;
        int padded_height_B = (N + padding) / 4;
        cl_kernel tk = backend_ctx->kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(tk, 0, sizeof(cl_mem), &B_in_img));
        CL_CHECK(clSetKernelArg(tk, 1, sizeof(cl_mem), &B_image1d));
        CL_CHECK(clSetKernelArg(tk, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(tk, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(tk, 4, sizeof(int),    &padded_height_B));
        size_t tl[2] = { 1, 16 };
        size_t tg[2] = { (size_t) width_B, (size_t) padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(tk, 2, tg, tl, dst);
    }

    // --- C output sub-buffer ---
    region.origin = offsetd;
    region.size   = (size_t) M * N * sizeof(float);
    cl_mem C_d = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);

    // --- GEMM ---
    // 1D-flattened launch: total fibers = ceil(N/8) * ceil(M/4), rounded up to lws=128. Only the
    // last workgroup is partial; the kernel maps flat id -> (gy, gx) and early-returns gy>=n_tiles.
    const int padded_N = N + padding;
    const size_t lws1    = 128;
    const size_t m_tiles = (size_t) ((M + 3) / 4);
    const size_t n_tiles = (size_t) ((N + 7) / 8);
    const size_t total   = m_tiles * n_tiles;
    const size_t wgs     = (total + lws1 - 1) / lws1;

    // Split-K for occupancy-starved shapes (diffusion AV: ~26 WGs < saturation ~50). Slice the K
    // reduction k_split ways along a 2nd grid dim -> k_split x more WGs, then sum the partials.
    // Gate: too few WGs AND K long enough to split. Prefix AV (387 WGs) / diffusion QK (82) skip it.
    int k_split = 1;
    if (wgs < 48 && K >= 256) {
        k_split = 2; // 2 beats 4 here: k=4 over-splits, the reduce pass outgrows the occupancy gain
        const char * e = getenv("PI0_ATTN_SPLITK"); // optional override for tuning
        if (e) { k_split = atoi(e); }
        if (k_split < 1) { k_split = 1; }
    }

    if (k_split > 1) {
        const int plane = M * N;
        cl_mem partials = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t) k_split * plane * sizeof(float), NULL, &status);
        CL_CHECK(status);
        cl_kernel sk = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f16_splitk;
        CL_CHECK(clSetKernelArg(sk, 0, sizeof(cl_mem), &A_T));
        CL_CHECK(clSetKernelArg(sk, 1, sizeof(cl_mem), &B_image1d));
        CL_CHECK(clSetKernelArg(sk, 2, sizeof(cl_mem), &partials));
        CL_CHECK(clSetKernelArg(sk, 3, sizeof(int),    &M));
        CL_CHECK(clSetKernelArg(sk, 4, sizeof(int),    &padded_N));
        CL_CHECK(clSetKernelArg(sk, 5, sizeof(int),    &K));
        CL_CHECK(clSetKernelArg(sk, 6, sizeof(int),    &N));
        CL_CHECK(clSetKernelArg(sk, 7, sizeof(int),    &k_split));
        size_t sk_gws[2] = { ((total + lws1 - 1) / lws1) * lws1, (size_t) k_split };
        size_t sk_lws[2] = { lws1, 1 };
        backend_ctx->enqueue_ndrange_kernel(sk, 2, sk_gws, sk_lws, dst);

        cl_kernel rk = backend_ctx->kernel_splitk_reduce;
        CL_CHECK(clSetKernelArg(rk, 0, sizeof(cl_mem), &partials));
        CL_CHECK(clSetKernelArg(rk, 1, sizeof(cl_mem), &C_d));
        CL_CHECK(clSetKernelArg(rk, 2, sizeof(int),    &plane));
        CL_CHECK(clSetKernelArg(rk, 3, sizeof(int),    &k_split));
        size_t rd_gws[1] = { (size_t)(((plane + 63) / 64) * 64) };
        size_t rd_lws[1] = { 64 };
        backend_ctx->enqueue_ndrange_kernel(rk, 1, rd_gws, rd_lws, dst);
        CL_CHECK(clReleaseMemObject(partials));
    } else {
        cl_kernel fk = backend_ctx->CL_mul_mat_Ab_Bi_8x4_f16;
        CL_CHECK(clSetKernelArg(fk, 0, sizeof(cl_mem), &A_T));
        CL_CHECK(clSetKernelArg(fk, 1, sizeof(cl_mem), &B_image1d));
        CL_CHECK(clSetKernelArg(fk, 2, sizeof(cl_mem), &C_d));
        CL_CHECK(clSetKernelArg(fk, 3, sizeof(int),    &M));
        CL_CHECK(clSetKernelArg(fk, 4, sizeof(int),    &padded_N));
        CL_CHECK(clSetKernelArg(fk, 5, sizeof(int),    &K));
        CL_CHECK(clSetKernelArg(fk, 6, sizeof(int),    &N));
        size_t gws[1] = { ((total + lws1 - 1) / lws1) * lws1 };
        size_t lws[1] = { lws1 };
        backend_ctx->enqueue_ndrange_kernel(fk, 1, gws, lws, dst);
    }

    CL_CHECK(clReleaseMemObject(src0_sub));
    // A_T is the reused prealloc_pi0_at scratch — do NOT release it.
    CL_CHECK(clReleaseMemObject(B_sub));
    if (B_pad) { CL_CHECK(clReleaseMemObject(B_pad)); }
    CL_CHECK(clReleaseMemObject(B_d));
    CL_CHECK(clReleaseMemObject(B_in_img));
    CL_CHECK(clReleaseMemObject(B_image1d));
    CL_CHECK(clReleaseMemObject(C_d));
}
