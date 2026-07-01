// pi0-udot8-bench: pure-OpenCL int8-vs-fp16 probes for the Adreno 830.
//
// Modes:
//   (default)  ALU throughput micro-probe: fp16 mad vs int8 qcom_udot8_acc -> r = udot8/fp16.
//   gemm M N K Ab_Bi-style tiled GEMM (B via image1d_buffer -> TP-L1, weights via L2):
//              f16 (half4 dot), W8A8 (int8 udot8), and a W4A8 TILE SWEEP (q4 weight unpacked
//              to int8 + udot8) over (TM x TN) per-fiber tiles. Weight (L2) traffic/MAC ~ 1/TN,
//              activation (image) traffic/MAC ~ 1/TM; the sweep finds the best work-per-fiber +
//              L2/image read balance. Baseline to beat: production q4 f16 Ab_Bi ~2591 GF (gate+up).
//
// Standalone (no ggml). Build target: llama-pi0-udot8-bench.

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

#define CHK(x) do { cl_int _e = (x); if (_e != CL_SUCCESS) { \
    fprintf(stderr, "CL error %d at %s:%d\n", _e, __FILE__, __LINE__); exit(1); } } while (0)

static const char * KSRC = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable

__kernel void bench_fp16(__global float * out, const int iters) {
    const int gid = get_global_id(0);
    const half8 a = (half8)((half)0.999f);
    const half8 b = (half8)((half)(0.5f + 0.001f * (float)(gid & 7)));
    half8 c0=(half8)((half)0.011f), c1=(half8)((half)0.012f), c2=(half8)((half)0.013f), c3=(half8)((half)0.014f),
          c4=(half8)((half)0.015f), c5=(half8)((half)0.016f), c6=(half8)((half)0.017f), c7=(half8)((half)0.018f);
    for (int i = 0; i < iters; ++i) {
        c0 = mad(a, c0, b); c1 = mad(a, c1, b); c2 = mad(a, c2, b); c3 = mad(a, c3, b);
        c4 = mad(a, c4, b); c5 = mad(a, c5, b); c6 = mad(a, c6, b); c7 = mad(a, c7, b);
    }
    half8 s = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7));
    out[gid] = (float)(s.s0+s.s1+s.s2+s.s3+s.s4+s.s5+s.s6+s.s7);
}
__kernel void bench_fp16_h4(__global float * out, const int iters) {
    const int gid = get_global_id(0);
    const half4 a = (half4)((half)0.999f);
    const half4 b = (half4)((half)(0.5f + 0.001f * (float)(gid & 7)));
    half4 c0=(half4)((half)0.011f), c1=(half4)((half)0.012f), c2=(half4)((half)0.013f), c3=(half4)((half)0.014f),
          c4=(half4)((half)0.015f), c5=(half4)((half)0.016f), c6=(half4)((half)0.017f), c7=(half4)((half)0.018f);
    for (int i = 0; i < iters; ++i) {
        c0 = mad(a, c0, b); c1 = mad(a, c1, b); c2 = mad(a, c2, b); c3 = mad(a, c3, b);
        c4 = mad(a, c4, b); c5 = mad(a, c5, b); c6 = mad(a, c6, b); c7 = mad(a, c7, b);
    }
    half4 s = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7));
    out[gid] = (float)(s.s0+s.s1+s.s2+s.s3);
}
__kernel void bench_fp16_h1(__global float * out, const int iters) {
    const int gid = get_global_id(0);
    const half a = (half)0.999f;
    const half b = (half)(0.5f + 0.001f * (float)(gid & 7));
    half c0=(half)0.011f, c1=(half)0.012f, c2=(half)0.013f, c3=(half)0.014f,
         c4=(half)0.015f, c5=(half)0.016f, c6=(half)0.017f, c7=(half)0.018f;
    for (int i = 0; i < iters; ++i) {
        c0 = mad(a, c0, b); c1 = mad(a, c1, b); c2 = mad(a, c2, b); c3 = mad(a, c3, b);
        c4 = mad(a, c4, b); c5 = mad(a, c5, b); c6 = mad(a, c6, b); c7 = mad(a, c7, b);
    }
    half s = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7));
    out[gid] = (float)s;
}
__kernel void bench_udot8(__global int * out, const int iters) {
    const int  gid = get_global_id(0);
    const uint b   = 0x01010101u + (uint)(gid & 7);
    int c0=gid, c1=gid+1, c2=gid+2, c3=gid+3, c4=gid+4, c5=gid+5, c6=gid+6, c7=gid+7;
    for (int i = 0; i < iters; ++i) {
        c0 = qcom_udot8_acc((uint)c0, b, c0); c1 = qcom_udot8_acc((uint)c1, b, c1);
        c2 = qcom_udot8_acc((uint)c2, b, c2); c3 = qcom_udot8_acc((uint)c3, b, c3);
        c4 = qcom_udot8_acc((uint)c4, b, c4); c5 = qcom_udot8_acc((uint)c5, b, c5);
        c6 = qcom_udot8_acc((uint)c6, b, c6); c7 = qcom_udot8_acc((uint)c7, b, c7);
    }
    out[gid] = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7));
}

// fixed 4M x 8N baselines (B image -> TP-L1, weights L2, 4-K packed per read unit)
__kernel void gemm_f16(__global const half4 * A, __read_only image1d_buffer_t B,
                       __global float * C, const int M, const int N, const int K, const float scale) {
    const int kbc = K >> 2;
    const int m_tiles = (M + 3) >> 2;
    const int n_tiles = (N + 7) >> 3;
    const int flat = get_global_id(0);
    const int gy = flat / m_tiles; if (gy >= n_tiles) return;
    const int gx = flat - gy * m_tiles;
    const int m0 = gx << 2, n0 = gy << 3;
    float acc[32];
    #pragma unroll
    for (int t = 0; t < 32; ++t) acc[t] = 0.0f;
    for (int kb = 0; kb < kbc; ++kb) {
        half4 w0=A[kb*M+m0], w1=A[kb*M+m0+1], w2=A[kb*M+m0+2], w3=A[kb*M+m0+3];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            half4 b = read_imageh(B, (n0 + i) * kbc + kb);
            half4 p0=w0*b, p1=w1*b, p2=w2*b, p3=w3*b;
            acc[0*8+i]+=(float)(p0.s0+p0.s1+p0.s2+p0.s3);
            acc[1*8+i]+=(float)(p1.s0+p1.s1+p1.s2+p1.s3);
            acc[2*8+i]+=(float)(p2.s0+p2.s1+p2.s2+p2.s3);
            acc[3*8+i]+=(float)(p3.s0+p3.s1+p3.s2+p3.s3);
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) { int m = m0 + j; if (m >= M) continue;
        for (int i = 0; i < 8; ++i) { int n = n0 + i; if (n >= N) continue;
            C[m*N + n] = acc[j*8+i] * scale; } }
}
__kernel void gemm_i8(__global const uint * A, __read_only image1d_buffer_t B,
                      __global float * C, const int M, const int N, const int K, const float scale) {
    const int kbc = K >> 2;
    const int m_tiles = (M + 3) >> 2;
    const int n_tiles = (N + 7) >> 3;
    const int flat = get_global_id(0);
    const int gy = flat / m_tiles; if (gy >= n_tiles) return;
    const int gx = flat - gy * m_tiles;
    const int m0 = gx << 2, n0 = gy << 3;
    int acc[32];
    #pragma unroll
    for (int t = 0; t < 32; ++t) acc[t] = 0;
    for (int kb = 0; kb < kbc; ++kb) {
        uint w0=A[kb*M+m0], w1=A[kb*M+m0+1], w2=A[kb*M+m0+2], w3=A[kb*M+m0+3];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint b = read_imageui(B, (n0 + i) * kbc + kb).x;
            acc[0*8+i]=qcom_udot8_acc(w0,b,acc[0*8+i]);
            acc[1*8+i]=qcom_udot8_acc(w1,b,acc[1*8+i]);
            acc[2*8+i]=qcom_udot8_acc(w2,b,acc[2*8+i]);
            acc[3*8+i]=qcom_udot8_acc(w3,b,acc[3*8+i]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) { int m = m0 + j; if (m >= M) continue;
        for (int i = 0; i < 8; ++i) { int n = n0 + i; if (n >= N) continue;
            C[m*N + n] = (float)acc[j*8+i] * scale; } }
}
)CLC";

// Parameterized W4A8 (TM x TN per-fiber tile, set via -DTM/-DTN). Requires M % TM == 0.
static const char * KSRC_W4A8 = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 8
#endif
#ifdef ACC16
#define ACCT half
#else
#define ACCT int
#endif
inline uint unpack_q4(ushort p) {
    return ((uint)(p&0xF)) | (((uint)((p>>4)&0xF))<<8) | (((uint)((p>>8)&0xF))<<16) | (((uint)((p>>12)&0xF))<<24);
}
__kernel void gemm_w4a8(__global const ushort * A, __read_only image1d_buffer_t B,
                        __global float * C, const int M, const int N, const int K, const float scale) {
    const int kbc = K >> 2;
    const int m_tiles = (M + TM - 1) / TM;
    const int n_tiles = (N + TN - 1) / TN;
    const int flat = get_global_id(0);
    const int gy = flat / m_tiles; if (gy >= n_tiles) return;
    const int gx = flat - gy * m_tiles;
    const int m0 = gx * TM, n0 = gy * TN;
    ACCT acc[TM*TN];
    #pragma unroll
    for (int t = 0; t < TM*TN; ++t) acc[t] = (ACCT)0;
    for (int kb = 0; kb < kbc; ++kb) {
        uint w[TM];
        #pragma unroll
        for (int j = 0; j < TM; ++j) w[j] = unpack_q4(A[kb*M + m0 + j]);
        #pragma unroll
        for (int i = 0; i < TN; ++i) {
            uint b = read_imageui(B, (n0 + i) * kbc + kb).x;
#ifdef ACC16
            #pragma unroll
            for (int j = 0; j < TM; ++j) acc[j*TN + i] += (half)qcom_udot8_acc(w[j], b, 0);
#else
            #pragma unroll
            for (int j = 0; j < TM; ++j) acc[j*TN + i] = qcom_udot8_acc(w[j], b, acc[j*TN + i]);
#endif
        }
    }
    #pragma unroll
    for (int j = 0; j < TM; ++j) { int m = m0 + j; if (m >= M) continue;
        for (int i = 0; i < TN; ++i) { int n = n0 + i; if (n >= N) continue;
            C[m*N + n] = (float)acc[j*TN + i] * scale; } }
}
)CLC";

// W4A8 with RGBA-uint32 image: read_imageui returns uint4 = 16 int8 (4 K-blocks) per read.
// Weights packed ushort4 = 16 nibbles per read. 4x fewer image AND weight read instructions.
static const char * KSRC_W4A8_RGBA = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable
#ifndef TM
#define TM 4
#endif
#ifndef TN
#define TN 8
#endif
#ifdef ACC16
#define ACCT half
#else
#define ACCT int
#endif
inline uint unpack_q4(ushort p) {
    return ((uint)(p&0xF)) | (((uint)((p>>4)&0xF))<<8) | (((uint)((p>>8)&0xF))<<16) | (((uint)((p>>12)&0xF))<<24);
}
__kernel void gemm_w4a8(__global const ushort4 * A, __read_only image1d_buffer_t B,
                        __global float * C, const int M, const int N, const int K, const float scale) {
    const int kb16 = K >> 4;
    const int m_tiles = (M + TM - 1) / TM;
    const int n_tiles = (N + TN - 1) / TN;
    const int flat = get_global_id(0);
    const int gy = flat / m_tiles; if (gy >= n_tiles) return;
    const int gx = flat - gy * m_tiles;
    const int m0 = gx * TM, n0 = gy * TN;
    ACCT acc[TM*TN];
    #pragma unroll
    for (int t = 0; t < TM*TN; ++t) acc[t] = (ACCT)0;
    for (int kbb = 0; kbb < kb16; ++kbb) {
        uint w[TM][4];
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            ushort4 p = A[kbb*M + m0 + j];
            w[j][0]=unpack_q4(p.s0); w[j][1]=unpack_q4(p.s1); w[j][2]=unpack_q4(p.s2); w[j][3]=unpack_q4(p.s3);
        }
        #pragma unroll
        for (int i = 0; i < TN; ++i) {
            uint4 b = read_imageui(B, (n0 + i) * kb16 + kbb);
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                int d = qcom_udot8_acc(w[j][0], b.x, qcom_udot8_acc(w[j][1], b.y,
                        qcom_udot8_acc(w[j][2], b.z, qcom_udot8_acc(w[j][3], b.w, 0))));
                acc[j*TN + i] += (ACCT)d;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < TM; ++j) { int m = m0 + j; if (m >= M) continue;
        for (int i = 0; i < TN; ++i) { int n = n0 + i; if (n >= N) continue;
            C[m*N + n] = (float)acc[j*TN + i] * scale; } }
}
)CLC";

// Production-style N-vectorized f16 (4M x 8N, half8 per M-row, weight broadcast, K-serial).
// -DDENSE switches B-read: 2x read_imageh (4 half/read, RGBA-half) -> 1x read_imageui (8 half/read, RGBA-uint32).
static const char * KSRC_F16NV = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemm_f16nv(__global const half * A, __read_only image1d_buffer_t B,
                         __global float * C, const int M, const int N, const int K, const float scale) {
    const int m_tiles = (M + 3) >> 2;
    const int n_tiles = (N + 7) >> 3;
    const int flat = get_global_id(0);
    const int gy = flat / m_tiles; if (gy >= n_tiles) return;
    const int gx = flat - gy * m_tiles;
    const int m0 = gx << 2, n0 = gy << 3;
    half8 c0=(half8)((half)0), c1=(half8)((half)0), c2=(half8)((half)0), c3=(half8)((half)0);
    for (int k = 0; k < K; ++k) {
        half4 w = vload4(0, A + (size_t)k*M + m0);
        half8 bb;
#ifdef DENSE
        uint4 p = read_imageui(B, gy*K + k);
        bb = as_half8(p);
#else
        bb.s0123 = read_imageh(B, (gy*K + k)*2);
        bb.s4567 = read_imageh(B, (gy*K + k)*2 + 1);
#endif
        c0 += bb * w.s0; c1 += bb * w.s1; c2 += bb * w.s2; c3 += bb * w.s3;
    }
#define STORE8(CC, MM) { int m=(MM); if(m<M){ int b=m*N+n0; \
    if(n0+0<N)C[b+0]=(float)CC.s0*scale; if(n0+1<N)C[b+1]=(float)CC.s1*scale; \
    if(n0+2<N)C[b+2]=(float)CC.s2*scale; if(n0+3<N)C[b+3]=(float)CC.s3*scale; \
    if(n0+4<N)C[b+4]=(float)CC.s4*scale; if(n0+5<N)C[b+5]=(float)CC.s5*scale; \
    if(n0+6<N)C[b+6]=(float)CC.s6*scale; if(n0+7<N)C[b+7]=(float)CC.s7*scale; } }
    STORE8(c0, m0+0) STORE8(c1, m0+1) STORE8(c2, m0+2) STORE8(c3, m0+3)
#undef STORE8
}
)CLC";

static cl_half f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e    = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m    = x & 0x7fffffu;
    if (e <= 0)  return (cl_half)sign;
    if (e >= 31) return (cl_half)(sign | 0x7c00u);
    return (cl_half)(sign | ((uint32_t)e << 10) | (m >> 13));
}

static cl_program build_prog(cl_context ctx, cl_device_id dev, const char * src, const char * extra) {
    cl_int err; cl_program p = clCreateProgramWithSource(ctx, 1, &src, nullptr, &err); CHK(err);
    char clcver[128] = {0}; clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, sizeof(clcver), clcver, nullptr);
    int vmaj = 3, vmin = 0;
    { const char * z = strstr(clcver, "OpenCL C "); if (z) sscanf(z + 9, "%d.%d", &vmaj, &vmin); }
    char opts[256];
    snprintf(opts, sizeof(opts),
             "-cl-std=CL%d.%d -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math %s",
             vmaj, vmin, extra ? extra : "");
    if (clBuildProgram(p, 1, &dev, opts, nullptr, nullptr) != CL_SUCCESS) {
        size_t n = 0; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::vector<char> log(n + 1, 0); clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        fprintf(stderr, "build failed:\n%s\n", log.data()); exit(1);
    }
    return p;
}

static double run_alu(cl_command_queue q, cl_kernel k, cl_mem out, int iters, size_t gws, size_t lws, int reps) {
    CHK(clSetKernelArg(k, 0, sizeof(cl_mem), &out)); CHK(clSetKernelArg(k, 1, sizeof(int), &iters));
    for (int w = 0; w < 3; ++w) CHK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr));
    CHK(clFinish(q));
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        cl_event ev; CHK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, &ev));
        CHK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1;
        CHK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
        CHK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr));
        clReleaseEvent(ev); double ns = (double)(t1 - t0); if (ns < best) best = ns;
    }
    return best;
}

static double time_gemm(cl_command_queue q, cl_kernel k, cl_mem A, cl_mem Bimg, cl_mem C,
                        int M, int N, int K, float scale, size_t gws, size_t lws, int reps) {
    CHK(clSetKernelArg(k, 0, sizeof(cl_mem), &A));   CHK(clSetKernelArg(k, 1, sizeof(cl_mem), &Bimg));
    CHK(clSetKernelArg(k, 2, sizeof(cl_mem), &C));   CHK(clSetKernelArg(k, 3, sizeof(int), &M));
    CHK(clSetKernelArg(k, 4, sizeof(int), &N));      CHK(clSetKernelArg(k, 5, sizeof(int), &K));
    CHK(clSetKernelArg(k, 6, sizeof(float), &scale));
    for (int w = 0; w < 3; ++w) CHK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, nullptr));
    CHK(clFinish(q));
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        cl_event ev; CHK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &gws, &lws, 0, nullptr, &ev));
        CHK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1;
        CHK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
        CHK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr));
        clReleaseEvent(ev); double ns = (double)(t1 - t0); if (ns < best) best = ns;
    }
    return best;
}

static double relL2(cl_command_queue q, cl_mem C, const std::vector<double> & ref, int M, int N, double rn) {
    std::vector<float> h((size_t)M*N);
    CHK(clEnqueueReadBuffer(q, C, CL_TRUE, 0, h.size()*sizeof(float), h.data(), 0, nullptr, nullptr));
    double e = 0; for (size_t i = 0; i < h.size(); ++i) { double d = h[i] - ref[i]; e += d*d; }
    return sqrt(e / rn);
}

static cl_device_id pick_device(cl_platform_id * out) {
    cl_uint np = 0; CHK(clGetPlatformIDs(0, nullptr, &np));
    std::vector<cl_platform_id> ps(np); CHK(clGetPlatformIDs(np, ps.data(), nullptr));
    cl_device_id dev = nullptr;
    for (cl_platform_id p : ps) {
        cl_uint nd = 0; if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || !nd) continue;
        std::vector<cl_device_id> ds(nd); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        char pn[256] = {0}; clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(pn), pn, nullptr);
        dev = ds[0]; *out = p; if (strstr(pn, "QUALCOMM") || strstr(pn, "Qualcomm")) break;
    }
    return dev;
}

static void gemm_run(cl_context ctx, cl_device_id dev, cl_command_queue q,
                     cl_program base, int M, int N, int K, int reps, bool check) {
    if (M % 8 || K % 4) { fprintf(stderr, "need M%%8==0, K%%4==0\n"); return; }
    const int kbc  = K / 4;
    const int Npad = ((N + 31) / 32) * 32;   // covers TN up to 32, zero-padded

    std::vector<float> W((size_t)M*K), X((size_t)N*K);
    uint32_t s = 12345u; auto rnd = [&]() { s = s*1664525u + 1013904223u; return (float)((s>>8)&0xffff)/65536.0f; };
    float wmax = 0, xmax = 0;
    for (auto & v : W) { v = rnd(); if (v > wmax) wmax = v; }
    for (auto & v : X) { v = rnd(); if (v > xmax) xmax = v; }
    const float dW = wmax/255.0f, dX = xmax/255.0f, dW4 = wmax/15.0f;

    std::vector<cl_half>  Ah((size_t)kbc*M*4);
    std::vector<uint32_t> Ai((size_t)kbc*M);
    std::vector<uint16_t> Aq4((size_t)kbc*M);
    for (int kb = 0; kb < kbc; ++kb) for (int m = 0; m < M; ++m) {
        uint32_t pk = 0; uint16_t pk4 = 0;
        for (int t = 0; t < 4; ++t) {
            float w = W[(size_t)m*K + kb*4 + t];
            Ah[((size_t)kb*M + m)*4 + t] = f2h(w);
            uint32_t qv = (uint32_t)(w/dW + 0.5f); if (qv > 255) qv = 255; pk |= qv << (8*t);
            uint32_t q4 = (uint32_t)(w/dW4 + 0.5f); if (q4 > 15) q4 = 15; pk4 |= (uint16_t)(q4 << (4*t));
        }
        Ai[(size_t)kb*M + m] = pk; Aq4[(size_t)kb*M + m] = pk4;
    }
    std::vector<cl_half>  Bh((size_t)Npad*kbc*4, 0);
    std::vector<uint32_t> Bi((size_t)Npad*kbc, 0);
    for (int n = 0; n < N; ++n) for (int kb = 0; kb < kbc; ++kb) {
        uint32_t pk = 0;
        for (int t = 0; t < 4; ++t) {
            float x = X[(size_t)n*K + kb*4 + t];
            Bh[((size_t)n*kbc + kb)*4 + t] = f2h(x);
            uint32_t qv = (uint32_t)(x/dX + 0.5f); if (qv > 255) qv = 255; pk |= qv << (8*t);
        }
        Bi[(size_t)n*kbc + kb] = pk;
    }

    cl_int err;
    cl_mem Ahb  = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Ah.size()*sizeof(cl_half), Ah.data(), &err); CHK(err);
    cl_mem Aib  = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Ai.size()*sizeof(uint32_t), Ai.data(), &err); CHK(err);
    cl_mem Aq4b = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Aq4.size()*sizeof(uint16_t), Aq4.data(), &err); CHK(err);
    cl_mem Bhb  = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Bh.size()*sizeof(cl_half), Bh.data(), &err); CHK(err);
    cl_mem Bib  = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Bi.size()*sizeof(uint32_t), Bi.data(), &err); CHK(err);
    cl_image_format fh = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc   dh = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)Npad*kbc, 0,0,0,0,0,0,0, { Bhb } };
    cl_mem Bhimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fh, &dh, nullptr, &err); CHK(err);
    cl_image_format fi = { CL_R, CL_UNSIGNED_INT32 };
    cl_image_desc   di = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)Npad*kbc, 0,0,0,0,0,0,0, { Bib } };
    cl_mem Biimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fi, &di, nullptr, &err); CHK(err);
    cl_mem Cbuf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t)M*N*sizeof(float), nullptr, &err); CHK(err);

    const double flop = 2.0 * M * N * K;
    const size_t lws = 128;
    auto grid = [&](int tm, int tn) {
        size_t total = (size_t)((M + tm - 1)/tm) * ((N + tn - 1)/tn);
        return ((total + lws - 1) / lws) * lws;
    };

    // reference (only when checking; small shapes)
    std::vector<double> ref; double rn = 0;
    if (check) {
        ref.assign((size_t)M*N, 0.0);
        for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
            double acc = 0; for (int k = 0; k < K; ++k) acc += (double)W[(size_t)m*K+k]*X[(size_t)n*K+k];
            ref[(size_t)m*N+n] = acc; rn += acc*acc;
        }
    }

    printf("GEMM  M=%d N=%d K=%d  (FLOP=%.2fG)  reps=%d\n", M, N, K, flop/1e9, reps);

    cl_kernel kf = clCreateKernel(base, "gemm_f16", &err); CHK(err);
    cl_kernel ki = clCreateKernel(base, "gemm_i8",  &err); CHK(err);
    { size_t g = grid(4,8);
      double t = time_gemm(q, kf, Ahb, Bhimg, Cbuf, M,N,K, 1.0f, g, lws, reps);
      printf("  f16  4x8   : %8.3f ms  %8.1f GFLOP/s%s\n", t*1e-6, flop/t,
             check ? "" : ""); if (check) printf("              relL2 %.2e\n", relL2(q,Cbuf,ref,M,N,rn)); }
    { size_t g = grid(4,8);
      double t = time_gemm(q, ki, Aib, Biimg, Cbuf, M,N,K, dW*dX, g, lws, reps);
      printf("  W8A8 4x8   : %8.3f ms  %8.1f GFLOP/s\n", t*1e-6, flop/t);
      if (check) printf("              relL2 %.2e\n", relL2(q,Cbuf,ref,M,N,rn)); }
    clReleaseKernel(kf); clReleaseKernel(ki);

    // W4A8 tile sweep
    const int cfgs[][2] = { {4,8},{2,16},{8,4},{8,8},{4,16},{2,32},{1,32},{1,64},{4,4} };
    const char * accs[][2] = { {"int ", ""}, {"f16a", "-DACC16"} };  // f16a ignores overflow
    printf("  --- W4A8 tile sweep (q4 wt, int8 act, udot8) ---\n");
    for (auto & c : cfgs) {
        int tm = c[0], tn = c[1];
        for (auto & a : accs) {
            char def[96]; snprintf(def, sizeof(def), "-DTM=%d -DTN=%d %s", tm, tn, a[1]);
            cl_program pw = build_prog(ctx, dev, KSRC_W4A8, def);
            cl_kernel  kw = clCreateKernel(pw, "gemm_w4a8", &err); CHK(err);
            size_t g = grid(tm, tn);
            double t = time_gemm(q, kw, Aq4b, Biimg, Cbuf, M,N,K, dW4*dX, g, lws, reps);
            printf("  W4A8 %dx%-2d %s: %8.3f ms  %8.1f GFLOP/s  (acc=%d)", tm, tn, a[0], t*1e-6, flop/t, tm*tn);
            if (check) printf("  relL2 %.2e", relL2(q,Cbuf,ref,M,N,rn));
            printf("\n");
            clReleaseKernel(kw); clReleaseProgram(pw);
        }
    }

    // --- RGBA-uint32 image variant: 16 int8 per read (4x fewer image + weight read instr) ---
    if (K % 16 == 0) {
        const int KB16 = K / 16;
        std::vector<uint16_t> Arg((size_t)KB16*M*4);
        std::vector<uint32_t> Brg((size_t)Npad*KB16*4, 0);
        for (int kbb = 0; kbb < KB16; ++kbb) for (int m = 0; m < M; ++m)
            for (int sb = 0; sb < 4; ++sb) {
                uint16_t pk4 = 0;
                for (int t = 0; t < 4; ++t) {
                    float w = W[(size_t)m*K + kbb*16 + sb*4 + t];
                    uint32_t q4 = (uint32_t)(w/dW4 + 0.5f); if (q4 > 15) q4 = 15;
                    pk4 |= (uint16_t)(q4 << (4*t));
                }
                Arg[((size_t)kbb*M + m)*4 + sb] = pk4;
            }
        for (int n = 0; n < N; ++n) for (int kbb = 0; kbb < KB16; ++kbb)
            for (int c = 0; c < 4; ++c) {
                uint32_t pk = 0;
                for (int t = 0; t < 4; ++t) {
                    float x = X[(size_t)n*K + kbb*16 + c*4 + t];
                    uint32_t qv = (uint32_t)(x/dX + 0.5f); if (qv > 255) qv = 255;
                    pk |= qv << (8*t);
                }
                Brg[((size_t)n*KB16 + kbb)*4 + c] = pk;
            }
        cl_mem Argb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Arg.size()*sizeof(uint16_t), Arg.data(), &err); CHK(err);
        cl_mem Brgb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Brg.size()*sizeof(uint32_t), Brg.data(), &err); CHK(err);
        cl_image_format frg = { CL_RGBA, CL_UNSIGNED_INT32 };
        cl_image_desc   drg = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)Npad*KB16, 0,0,0,0,0,0,0, { Brgb } };
        cl_mem Brgimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &frg, &drg, nullptr, &err); CHK(err);
        printf("  --- W4A8 RGBA image (16 int8/read) ---\n");
        for (auto & c : cfgs) {
            int tm = c[0], tn = c[1];
            for (auto & a : accs) {
                char def[96]; snprintf(def, sizeof(def), "-DTM=%d -DTN=%d %s", tm, tn, a[1]);
                cl_program pw = build_prog(ctx, dev, KSRC_W4A8_RGBA, def);
                cl_kernel  kw = clCreateKernel(pw, "gemm_w4a8", &err); CHK(err);
                size_t g = grid(tm, tn);
                double t = time_gemm(q, kw, Argb, Brgimg, Cbuf, M,N,K, dW4*dX, g, lws, reps);
                printf("  W4A8rgba %dx%-2d %s: %8.3f ms  %8.1f GFLOP/s  (acc=%d)", tm, tn, a[0], t*1e-6, flop/t, tm*tn);
                if (check) printf("  relL2 %.2e", relL2(q,Cbuf,ref,M,N,rn));
                printf("\n");
                clReleaseKernel(kw); clReleaseProgram(pw);
            }
        }
        clReleaseMemObject(Argb); clReleaseMemObject(Brgb); clReleaseMemObject(Brgimg);
    }

    // --- f16 production-style N-vectorized: sparse (2x half4) vs dense (uint4=8 half) B-read ---
    {
        const int ntg = Npad / 8;
        std::vector<cl_half>  Anv((size_t)K*M);
        for (int k = 0; k < K; ++k) for (int m = 0; m < M; ++m) Anv[(size_t)k*M + m] = f2h(W[(size_t)m*K + k]);
        std::vector<cl_half>  Bsp((size_t)ntg*K*8, 0);   // RGBA-half: 8 half per (ng,k) as 2 texels
        std::vector<uint32_t> Bds((size_t)ntg*K*4, 0);   // RGBA-uint32: 8 half per (ng,k) as 4 uint32
        for (int ng = 0; ng < ntg; ++ng) for (int k = 0; k < K; ++k) {
            cl_half lane[8];
            for (int l = 0; l < 8; ++l) { int n = ng*8 + l; lane[l] = (n < N) ? f2h(X[(size_t)n*K + k]) : (cl_half)0; }
            for (int l = 0; l < 8; ++l) Bsp[((size_t)ng*K + k)*8 + l] = lane[l];
            for (int c = 0; c < 4; ++c) Bds[((size_t)ng*K + k)*4 + c] = (uint32_t)lane[2*c] | ((uint32_t)lane[2*c+1] << 16);
        }
        cl_mem Anvb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Anv.size()*sizeof(cl_half), Anv.data(), &err); CHK(err);
        cl_mem Bspb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Bsp.size()*sizeof(cl_half), Bsp.data(), &err); CHK(err);
        cl_mem Bdsb = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, Bds.size()*sizeof(uint32_t), Bds.data(), &err); CHK(err);
        cl_image_format fsp = { CL_RGBA, CL_HALF_FLOAT };
        cl_image_desc   dsp = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)ntg*K*2, 0,0,0,0,0,0,0, { Bspb } };
        cl_mem Bspimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fsp, &dsp, nullptr, &err); CHK(err);
        cl_image_format fds = { CL_RGBA, CL_UNSIGNED_INT32 };
        cl_image_desc   dds = { CL_MEM_OBJECT_IMAGE1D_BUFFER, (size_t)ntg*K, 0,0,0,0,0,0,0, { Bdsb } };
        cl_mem Bdsimg = clCreateImage(ctx, CL_MEM_READ_ONLY, &fds, &dds, nullptr, &err); CHK(err);
        printf("  --- f16 N-vectorized (production-style) 4x8: sparse vs dense B-read ---\n");
        size_t g = grid(4, 8);
        { cl_program pp = build_prog(ctx, dev, KSRC_F16NV, nullptr); cl_kernel kk = clCreateKernel(pp, "gemm_f16nv", &err); CHK(err);
          double t = time_gemm(q, kk, Anvb, Bspimg, Cbuf, M,N,K, 1.0f, g, lws, reps);
          printf("  f16nv sparse: %8.3f ms  %8.1f GFLOP/s", t*1e-6, flop/t);
          if (check) printf("  relL2 %.2e", relL2(q,Cbuf,ref,M,N,rn)); printf("\n");
          clReleaseKernel(kk); clReleaseProgram(pp); }
        { cl_program pp = build_prog(ctx, dev, KSRC_F16NV, "-DDENSE"); cl_kernel kk = clCreateKernel(pp, "gemm_f16nv", &err); CHK(err);
          double t = time_gemm(q, kk, Anvb, Bdsimg, Cbuf, M,N,K, 1.0f, g, lws, reps);
          printf("  f16nv dense : %8.3f ms  %8.1f GFLOP/s", t*1e-6, flop/t);
          if (check) printf("  relL2 %.2e", relL2(q,Cbuf,ref,M,N,rn)); printf("\n");
          clReleaseKernel(kk); clReleaseProgram(pp); }
        clReleaseMemObject(Anvb); clReleaseMemObject(Bspb); clReleaseMemObject(Bdsb);
        clReleaseMemObject(Bspimg); clReleaseMemObject(Bdsimg);
    }

    clReleaseMemObject(Ahb); clReleaseMemObject(Aib); clReleaseMemObject(Aq4b);
    clReleaseMemObject(Bhb); clReleaseMemObject(Bib);
    clReleaseMemObject(Bhimg); clReleaseMemObject(Biimg); clReleaseMemObject(Cbuf);
}

int main(int argc, char ** argv) {
    const bool gemm_mode = (argc > 1 && strcmp(argv[1], "gemm") == 0);
    cl_platform_id plat = nullptr; cl_device_id dev = pick_device(&plat);
    if (!dev) { fprintf(stderr, "no GPU OpenCL device\n"); return 1; }
    char dn[256] = {0}; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
    std::vector<char> ext(8192, 0); clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext.size(), ext.data(), nullptr);
    bool dp8 = strstr(ext.data(), "cl_qcom_dot_product8") != nullptr;
    printf("device: %s   cl_qcom_dot_product8: %s\n", dn, dp8 ? "yes" : "NO");
    if (!dp8) return 1;

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); CHK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err); CHK(err);
    cl_program base = build_prog(ctx, dev, KSRC, nullptr);

    if (gemm_mode) {
        int M = (argc > 2) ? atoi(argv[2]) : 16384;
        int N = (argc > 3) ? atoi(argv[3]) : 769;
        int K = (argc > 4) ? atoi(argv[4]) : 2048;
        int reps = (argc > 5) ? atoi(argv[5]) : 30;
        printf("\n--- correctness (small) ---\n");
        gemm_run(ctx, dev, q, base, 256, 264, 512, 5, true);
        printf("\n--- timing (requested) ---\n");
        gemm_run(ctx, dev, q, base, M, N, K, reps, false);
    } else {
        size_t gws = (argc > 1) ? (size_t)strtoull(argv[1], nullptr, 10) : (1u << 18);
        int iters = (argc > 2) ? atoi(argv[2]) : 8192;
        int reps  = (argc > 3) ? atoi(argv[3]) : 20;
        const size_t lws = 256; gws = (gws + lws - 1)/lws*lws;
        printf("ALU probe: gws=%zu iters=%d reps=%d\n\n", gws, iters, reps);
        cl_kernel kf8 = clCreateKernel(base, "bench_fp16",    &err); CHK(err);
        cl_kernel kf4 = clCreateKernel(base, "bench_fp16_h4", &err); CHK(err);
        cl_kernel kf1 = clCreateKernel(base, "bench_fp16_h1", &err); CHK(err);
        cl_kernel ki  = clCreateKernel(base, "bench_udot8",   &err); CHK(err);
        cl_mem of = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, gws*sizeof(float), nullptr, &err); CHK(err);
        cl_mem oi = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, gws*sizeof(int),   nullptr, &err); CHK(err);
        double n8 = run_alu(q, kf8, of, iters, gws, lws, reps);
        double n4 = run_alu(q, kf4, of, iters, gws, lws, reps);
        double n1 = run_alu(q, kf1, of, iters, gws, lws, reps);
        double ni = run_alu(q, ki,  oi, iters, gws, lws, reps);
        double base_mac = (double)gws * iters * 8.0;
        double g8 = base_mac*8.0/n8, g4 = base_mac*4.0/n4, g1 = base_mac*1.0/n1, gi = base_mac*4.0/ni;
        double gb = g8; if (g4>gb) gb=g4; if (g1>gb) gb=g1;
        printf("fp16 half8 %8.1f GMAC/s\nfp16 half4 %8.1f\nfp16 half1 %8.1f\nint8 udot8 %8.1f\n", g8,g4,g1,gi);
        printf("\n  r = udot8 / fp16_peak = %.3fx\n", gi/gb);
        clReleaseKernel(kf8); clReleaseKernel(kf4); clReleaseKernel(kf1); clReleaseKernel(ki);
        clReleaseMemObject(of); clReleaseMemObject(oi);
    }
    clReleaseProgram(base); clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 0;
}
