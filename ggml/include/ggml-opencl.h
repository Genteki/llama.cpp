#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

// PI0 / Q8_0 Row-Tile dp8 path -------------------------------------------------
//
// Associate a weight tensor with three pre-uploaded cl_mem buffers describing
// it in Row-Tile (RT) format consumed by the cl_qcom_dot_product8 kernel:
//   cl_mem_Wq : uchar (M, K)         per-element q8 + 128 (uchar4 view in kernel)
//   cl_mem_Wd : half  (M, K/QK=32)   per-block fp16 scale
//   cl_mem_Ws : int   (M, K/QK=32)   per-block sum of uint8 (for zero-pt correction)
//
// While registered, any ggml_mul_mat that hits this tensor as src0 dispatches
// through the dp8 path (quantize_q8_0_rt + mul_mm_q8_0_rt_f16_dp8) instead of
// the FP-dequant kernel. Caller owns the cl_mem buffers and must keep them
// alive at least as long as the ggml_tensor.
//
// Parameter types are void* so the public header can stay free of cl.h.
GGML_BACKEND_API void ggml_backend_opencl_register_rt_tensor(
    const struct ggml_tensor * tensor,
    void * cl_mem_Wq,
    void * cl_mem_Wd,
    void * cl_mem_Ws,
    int M, int K);

// Drop a previously registered RT tensor.
GGML_BACKEND_API void ggml_backend_opencl_unregister_rt_tensor(
    const struct ggml_tensor * tensor);

// High-level convenience: allocate the three cl_mem buffers from `backend`'s
// OpenCL context, upload `Wq_bytes`/`Wd_bytes`/`Ws_bytes`, then register the
// tensor in one call. Returns true on success. Buffer ownership stays with the
// OpenCL backend until ggml_backend_opencl_unregister_rt_tensor is called or
// the backend is freed (caller does NOT free the cl_mem buffers directly).
//
// Expected byte layouts (row-major):
//   Wq_bytes : M * K            (uint8 = q+128)
//   Wd_bytes : M * (K/32) * 2   (fp16 scale)
//   Ws_bytes : M * (K/32) * 4   (int32 sum)
//
// Returns false if backend is not OpenCL, dp8 extension is missing on the
// device, or sizes don't match.
GGML_BACKEND_API bool ggml_backend_opencl_attach_rt_weights(
    ggml_backend_t backend,
    const struct ggml_tensor * tensor,
    const void * Wq_bytes,
    const void * Wd_bytes,
    const void * Ws_bytes,
    int M, int K);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
