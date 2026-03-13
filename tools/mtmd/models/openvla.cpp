#include "models.h"

/*
 * OpenVLA-7B vision graph: fused DINOv2 ViT-L/14 + SigLIP ViT-So400M/14
 *
 * Architecture:
 *   1. DINOv2 encoder (24 layers, penultimate extraction → 23 layers)
 *      - CLS token + 4 register tokens + 256 patch tokens = 261 tokens
 *      - Output: extract 256 patch tokens → [n_embd=1024, n_patches=256]
 *      - Has LayerScale
 *
 *   2. SigLIP encoder (27 layers, penultimate extraction → 26 layers)
 *      - 256 patch tokens (no CLS, no registers)
 *      - Output: [fused_n_embd=1152, n_patches=256]
 *
 *   3. Concatenate along embedding dim: [2176, 256]
 *
 *   4. 3-layer MLP projector: fc1 → GELU → fc2 → GELU → fc3
 *      [2176] → [4096] → [4096] → [4096]
 */

ggml_cgraph * clip_graph_openvla::build() {
    const auto & fhp = hparams; // fused hparams accessible via model.hparams

    const int fused_n_embd  = fhp.fused_n_embd;
    const int fused_n_head  = fhp.fused_n_head;
    const int fused_n_layer = fhp.fused_n_layer;
    const int fused_d_head  = fused_n_embd / fused_n_head;
    const float fused_eps   = fhp.fused_eps;
    const float fused_kq_scale = 1.0f / sqrtf((float)fused_d_head);
    const int n_reg = 4; // DINOv2 register tokens

    // Both encoders extract from the penultimate layer
    const int dinov2_run_layers = n_layer - 1;
    const int siglip_run_layers = fused_n_layer - 1;

    // Both encoders see the same image, so n_patches is shared
    // n_patches = (224/14)^2 = 256

    // =====================================================================
    // Shared input image
    // =====================================================================
    ggml_tensor * inp_raw = build_inp_raw();

    // =====================================================================
    // DINOv2 encoder
    // =====================================================================
    ggml_tensor * dinov2_inp;
    {
        // Patch embedding (conv2d)
        dinov2_inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw,
            patch_size, patch_size, 0, 0, 1, 1);
        dinov2_inp = ggml_reshape_2d(ctx0, dinov2_inp, n_patches, n_embd);
        dinov2_inp = ggml_cont(ctx0, ggml_transpose(ctx0, dinov2_inp));
        if (model.patch_bias) {
            dinov2_inp = ggml_add(ctx0, dinov2_inp, model.patch_bias);
        }
        // dinov2_inp: [n_embd, n_patches]

        // Prepend CLS token: [n_embd, n_patches+1]
        // class_embedding may be F16 while dinov2_inp is F32 (conv2d→mul_mat always yields F32)
        ggml_tensor * cls_emb = model.class_embedding;
        if (cls_emb->type != dinov2_inp->type) {
            cls_emb = ggml_cast(ctx0, cls_emb, dinov2_inp->type);
        }
        dinov2_inp = ggml_concat(ctx0, cls_emb, dinov2_inp, 1);

        // Add position embeddings to [CLS, patches]
        // pos_embd shape: [n_embd, n_patches+1]
        const int n_pos_dinov2 = n_patches + 1;
        ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos_dinov2);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);
        ggml_tensor * pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);
        if (pos_embd->type != dinov2_inp->type) {
            pos_embd = ggml_cast(ctx0, pos_embd, dinov2_inp->type);
        }
        dinov2_inp = ggml_add(ctx0, dinov2_inp, pos_embd);

        // Insert register tokens after CLS: [CLS, reg1..reg4, patch1..patch256]
        // Split CLS and patches
        ggml_tensor * cls_tok = ggml_view_2d(ctx0, dinov2_inp,
            n_embd, 1, dinov2_inp->nb[1], 0);
        ggml_tensor * patches_tok = ggml_view_2d(ctx0, dinov2_inp,
            n_embd, n_patches, dinov2_inp->nb[1], 1 * dinov2_inp->nb[1]);

        // reg_embedding: [n_embd, n_reg]
        ggml_tensor * reg_emb = model.reg_embedding;
        if (reg_emb->type != cls_tok->type) {
            reg_emb = ggml_cast(ctx0, reg_emb, cls_tok->type);
        }
        ggml_tensor * seq = ggml_concat(ctx0, cls_tok, reg_emb, 1);
        seq = ggml_concat(ctx0, seq, patches_tok, 1);
        // seq: [n_embd, 1+n_reg+n_patches] = [1024, 261]

        dinov2_inp = seq;
    }

    // Run DINOv2 transformer (penultimate: n_layer-1 blocks)
    const int n_pos_full = 1 + n_reg + n_patches; // 261
    ggml_tensor * dinov2_out = dinov2_inp;
    {
        ggml_tensor * inpL = dinov2_out;

        for (int il = 0; il < dinov2_run_layers; il++) {
            auto & layer = model.layers[il];
            ggml_tensor * cur = inpL;

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
            cb(cur, "d_layer_inp_normed", il);

            // self-attention (fused QKV for DINOv2)
            {
                ggml_tensor * Qcur, * Kcur, * Vcur;
                if (layer.qkv_w) {
                    cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
                    if (layer.qkv_b) {
                        cur = ggml_add(ctx0, cur, layer.qkv_b);
                    }
                    Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos_full,
                        ggml_row_size(cur->type, d_head), cur->nb[1], 0);
                    Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos_full,
                        ggml_row_size(cur->type, d_head), cur->nb[1],
                        ggml_row_size(cur->type, n_embd));
                    Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos_full,
                        ggml_row_size(cur->type, d_head), cur->nb[1],
                        ggml_row_size(cur->type, 2 * n_embd));
                } else {
                    Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                    if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                    Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                    if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                    Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                    if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                    Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos_full);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos_full);
                    Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos_full);
                }

                cur = build_attn(layer.o_w, layer.o_b,
                    Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            }

            // LayerScale on attention output
            if (layer.ls_1_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            }

            // residual
            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);

            // FFN (GELU)
            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                FFN_GELU, il);

            // LayerScale on FFN output
            if (layer.ls_2_w) {
                cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            }

            // residual
            cur = ggml_add(ctx0, inpL, cur);
            inpL = cur;
        }

        // NO post-layernorm (penultimate layer extraction)
        dinov2_out = inpL;
    }

    // Extract patch tokens from DINOv2 output (skip CLS + registers)
    // dinov2_out: [n_embd, n_pos_full] = [1024, 261]
    // want: [n_embd, n_patches] = [1024, 256]
    const int n_prefix = 1 + n_reg; // 5
    ggml_tensor * dinov2_patches = ggml_view_2d(ctx0, dinov2_out,
        n_embd, n_patches, dinov2_out->nb[1], n_prefix * dinov2_out->nb[1]);
    dinov2_patches = ggml_cont(ctx0, dinov2_patches);

    // =====================================================================
    // SigLIP encoder
    // =====================================================================

    // The input image (inp_raw) is ImageNet-normalized for DINOv2:
    //   dinov2_pixel = (pixel - inet_mean) / inet_std
    // But SigLIP expects different normalization (mean=0.5, std=0.5):
    //   siglip_pixel = (pixel - 0.5) / 0.5
    // Algebraic conversion (per channel):
    //   siglip_pixel = dinov2_pixel * (inet_std / 0.5) + (inet_mean - 0.5) / 0.5
    //
    // Channel values:
    //   R: scale=0.229/0.5=0.458,  bias=(0.485-0.5)/0.5=-0.030
    //   G: scale=0.224/0.5=0.448,  bias=(0.456-0.5)/0.5=-0.088
    //   B: scale=0.225/0.5=0.450,  bias=(0.406-0.5)/0.5=-0.188
    ggml_tensor * siglip_raw;
    {
        const float renorm_scale[3] = {0.458f, 0.448f, 0.450f};
        const float renorm_bias[3]  = {-0.030f, -0.088f, -0.188f};

        // Build per-channel scale and bias tensors: [1, 1, 3]
        // (broadcast over W and H dimensions)
        ggml_tensor * scale_t = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, 3);
        ggml_set_name(scale_t, "siglip_renorm_scale");
        ggml_set_input(scale_t);

        ggml_tensor * bias_t = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, 3);
        ggml_set_name(bias_t, "siglip_renorm_bias");
        ggml_set_input(bias_t);

        siglip_raw = ggml_mul(ctx0, inp_raw, scale_t);
        siglip_raw = ggml_add(ctx0, siglip_raw, bias_t);
    }

    ggml_tensor * siglip_inp;
    {
        // Patch embedding (conv2d) using fused encoder weights
        siglip_inp = ggml_conv_2d(ctx0, model.fused_patch_embeddings, siglip_raw,
            patch_size, patch_size, 0, 0, 1, 1);
        siglip_inp = ggml_reshape_2d(ctx0, siglip_inp, n_patches, fused_n_embd);
        siglip_inp = ggml_cont(ctx0, ggml_transpose(ctx0, siglip_inp));
        // siglip_inp: [fused_n_embd, n_patches] = [1152, 256]

        // Add position embeddings
        // fused_position_embeddings: [fused_n_embd, n_patches]
        ggml_tensor * fused_pos = model.fused_position_embeddings;
        if (fused_pos->type != siglip_inp->type) {
            fused_pos = ggml_cast(ctx0, fused_pos, siglip_inp->type);
        }
        siglip_inp = ggml_add(ctx0, siglip_inp, fused_pos);
    }

    // Run SigLIP transformer (penultimate: fused_n_layer-1 blocks)
    ggml_tensor * siglip_out = siglip_inp;
    {
        ggml_tensor * inpL = siglip_out;

        for (int il = 0; il < siglip_run_layers; il++) {
            auto & layer = model.fused_layers[il];
            ggml_tensor * cur = inpL;

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, fused_eps, il);
            cb(cur, "s_layer_inp_normed", il);

            // self-attention (fused QKV for SigLIP)
            {
                ggml_tensor * Qcur, * Kcur, * Vcur;
                if (layer.qkv_w) {
                    cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
                    if (layer.qkv_b) {
                        cur = ggml_add(ctx0, cur, layer.qkv_b);
                    }
                    Qcur = ggml_view_3d(ctx0, cur, fused_d_head, fused_n_head, n_patches,
                        ggml_row_size(cur->type, fused_d_head), cur->nb[1], 0);
                    Kcur = ggml_view_3d(ctx0, cur, fused_d_head, fused_n_head, n_patches,
                        ggml_row_size(cur->type, fused_d_head), cur->nb[1],
                        ggml_row_size(cur->type, fused_n_embd));
                    Vcur = ggml_view_3d(ctx0, cur, fused_d_head, fused_n_head, n_patches,
                        ggml_row_size(cur->type, fused_d_head), cur->nb[1],
                        ggml_row_size(cur->type, 2 * fused_n_embd));
                } else {
                    Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                    if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                    Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                    if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                    Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                    if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                    Qcur = ggml_reshape_3d(ctx0, Qcur, fused_d_head, fused_n_head, n_patches);
                    Kcur = ggml_reshape_3d(ctx0, Kcur, fused_d_head, fused_n_head, n_patches);
                    Vcur = ggml_reshape_3d(ctx0, Vcur, fused_d_head, fused_n_head, n_patches);
                }

                cur = build_attn(layer.o_w, layer.o_b,
                    Qcur, Kcur, Vcur, nullptr, fused_kq_scale, il);
            }

            // SigLIP does not have LayerScale

            // residual
            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            // layernorm2
            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, fused_eps, il);

            // FFN (GELU)
            cur = build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                FFN_GELU, il);

            // residual
            cur = ggml_add(ctx0, inpL, cur);
            inpL = cur;
        }

        // NO post-layernorm (penultimate layer extraction)
        siglip_out = inpL;
    }

    // siglip_out: [fused_n_embd, n_patches] = [1152, 256]

    // =====================================================================
    // Fuse: concatenate along embedding dimension
    // =====================================================================
    // dinov2_patches: [n_embd=1024, n_patches=256]
    // siglip_out:     [fused_n_embd=1152, n_patches=256]
    // result:         [2176, 256]
    if (dinov2_patches->type != siglip_out->type) {
        siglip_out = ggml_cast(ctx0, siglip_out, dinov2_patches->type);
    }
    ggml_tensor * fused = ggml_concat(ctx0, dinov2_patches, siglip_out, 0);

    // =====================================================================
    // 3-layer MLP projector: fc1 → GELU → fc2 → GELU → fc3
    // =====================================================================
    // mm_0: [2176] → [4096]
    fused = ggml_mul_mat(ctx0, model.mm_0_w, fused);
    fused = ggml_add(ctx0, fused, model.mm_0_b);
    fused = ggml_gelu(ctx0, fused);

    // mm_2: [4096] → [4096]
    fused = ggml_mul_mat(ctx0, model.mm_2_w, fused);
    fused = ggml_add(ctx0, fused, model.mm_2_b);
    fused = ggml_gelu(ctx0, fused);

    // mm_4: [4096] → [4096]
    fused = ggml_mul_mat(ctx0, model.mm_4_w, fused);
    fused = ggml_add(ctx0, fused, model.mm_4_b);

    // Output: [4096, 256] = [n_mmproj_embd, n_patches]
    ggml_build_forward_expand(gf, fused);
    return gf;
}
