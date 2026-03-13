#include "models.h"

// OpenVLA-7B: dual vision encoder (DINO v2 + SigLIP) + 3-layer MLP projector
//
// DINO v2 ViT-Large: 24 blocks, 1024 dim, 16 heads, patch14, 224x224
//   - Has CLS token, register tokens (4), LayerScale
//   - Extract second-to-last layer features, remove CLS + reg tokens
//
// SigLIP ViT-so400m: 27 blocks, 1152 dim, 16 heads, patch14, 224x224
//   - No CLS token, no LayerScale
//   - Extract second-to-last layer features
//
// Projector: fc1(2176->8704) -> GELU -> fc2(8704->4096) -> GELU -> fc3(4096->4096)
//   Input = concat(DINO features, SigLIP features) along embedding dim

ggml_cgraph * clip_graph_openvla::build() {
    // DINO encoder params (primary)
    const int dino_n_embd    = hparams.n_embd;      // 1024
    const int dino_n_head    = hparams.n_head;       // 16
    const int dino_d_head    = dino_n_embd / dino_n_head; // 64
    const int dino_n_layer   = hparams.n_layer;      // 24
    const int dino_feature_layer = dino_n_layer - 2; // 22 (second-to-last)

    // SigLIP encoder params (fused)
    const int siglip_n_embd  = hparams.fused_n_embd;  // 1152
    const int siglip_n_head  = hparams.fused_n_head;  // 16
    const int siglip_d_head  = siglip_n_embd / siglip_n_head; // 72
    const int siglip_n_layer = hparams.fused_n_layer;  // 27
    const int siglip_feature_layer = hparams.fused_feature_layer >= 0
        ? hparams.fused_feature_layer
        : siglip_n_layer - 2; // 25

    // Common
    const int n_patches_x = img.nx / patch_size;
    const int n_patches_y = img.ny / patch_size;
    const int n_patches_total = n_patches_x * n_patches_y; // 256

    // Build 6-channel raw input: [W, H, 6]
    ggml_tensor * inp_raw = build_inp_raw(6);

    // Split into DINO channels (0-2) and SigLIP channels (3-5)
    // inp_raw shape: [W, H, 6]
    ggml_tensor * inp_dino = ggml_view_3d(ctx0, inp_raw,
        img.nx, img.ny, 3,
        inp_raw->nb[1], inp_raw->nb[2],
        0);
    ggml_set_name(inp_dino, "inp_dino");

    ggml_tensor * inp_siglip = ggml_view_3d(ctx0, inp_raw,
        img.nx, img.ny, 3,
        inp_raw->nb[1], inp_raw->nb[2],
        3 * inp_raw->nb[2]);
    ggml_set_name(inp_siglip, "inp_siglip");

    // === DINO v2 Encoder ===

    // Patch embedding via Conv2D
    ggml_tensor * dino_emb = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_dino, patch_size, patch_size, 0, 0, 1, 1);
    dino_emb = ggml_reshape_2d(ctx0, dino_emb, n_patches_total, dino_n_embd);
    dino_emb = ggml_cont(ctx0, ggml_transpose(ctx0, dino_emb));
    if (model.patch_bias) {
        dino_emb = ggml_add(ctx0, dino_emb, model.patch_bias);
    }
    cb(dino_emb, "dino_patch_emb", -1);

    // Add position embeddings (only to patch tokens)
    dino_emb = ggml_add(ctx0, dino_emb, model.position_embeddings);
    cb(dino_emb, "dino_pos_emb", -1);

    // Prepend CLS token: [n_patches, dim] -> [1 + n_patches, dim]
    // Note: model tensors are f16, computed tensors are f32 — cast for ggml_concat
    if (model.class_embedding) {
        ggml_tensor * cls = ggml_reshape_2d(ctx0, model.class_embedding, dino_n_embd, 1);
        cls = ggml_cast(ctx0, cls, GGML_TYPE_F32);
        dino_emb = ggml_concat(ctx0, cls, dino_emb, 1);
        cb(dino_emb, "dino_cls_prepend", -1);
    }

    // Insert register tokens after CLS: [1 + n_patches, dim] -> [1 + 4 + n_patches, dim]
    // Result order: [CLS, reg0, reg1, reg2, reg3, patch0, patch1, ...]
    if (model.reg_embedding) {
        // Extract CLS token (row 0)
        ggml_tensor * cls_tok = ggml_view_2d(ctx0, dino_emb, dino_n_embd, 1,
            dino_emb->nb[1], 0);
        // Extract patch tokens (rows 1..n_patches)
        ggml_tensor * patch_toks = ggml_view_2d(ctx0, dino_emb, dino_n_embd, n_patches_total,
            dino_emb->nb[1], 1 * dino_emb->nb[1]);
        // Cast reg_embedding from f16 to f32
        ggml_tensor * reg = ggml_cast(ctx0, model.reg_embedding, GGML_TYPE_F32);
        // Concatenate: CLS + reg + patches
        dino_emb = ggml_concat(ctx0, cls_tok,
                       ggml_concat(ctx0, reg, patch_toks, 1), 1);
        cb(dino_emb, "dino_reg_prepend", -1);
    }

    const int dino_n_pos = (int)dino_emb->ne[1]; // 1 + 4 + 256 = 261
    const float dino_kq_scale = 1.0f / sqrtf((float) dino_d_head);

    // Transformer blocks
    ggml_tensor * dino_cur = dino_emb;
    for (int il = 0; il < dino_feature_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * residual = dino_cur;

        // LayerNorm 1
        dino_cur = build_norm(dino_cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(dino_cur, "dino_ln1", il);

        // Self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, dino_cur);
            if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, dino_cur);
            if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, dino_cur);
            if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, dino_d_head, dino_n_head, dino_n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, dino_d_head, dino_n_head, dino_n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, dino_d_head, dino_n_head, dino_n_pos);

            dino_cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, dino_kq_scale, il);
        }
        cb(dino_cur, "dino_attn_out", il);

        // LayerScale 1
        if (layer.ls_1_w) {
            dino_cur = ggml_mul(ctx0, dino_cur, layer.ls_1_w);
        }

        // Residual
        dino_cur = ggml_add(ctx0, dino_cur, residual);
        residual = dino_cur;

        // LayerNorm 2
        dino_cur = build_norm(dino_cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(dino_cur, "dino_ln2", il);

        // FFN (GELU)
        dino_cur = build_ffn(dino_cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            FFN_GELU, il);
        cb(dino_cur, "dino_ffn", il);

        // LayerScale 2
        if (layer.ls_2_w) {
            dino_cur = ggml_mul(ctx0, dino_cur, layer.ls_2_w);
        }

        // Residual
        dino_cur = ggml_add(ctx0, dino_cur, residual);
    }

    // Extract patch tokens only (skip CLS + register tokens)
    // dino_cur shape: [dino_n_embd, dino_n_pos]
    // We want the last n_patches_total rows (patch tokens)
    int n_prefix_tokens = dino_n_pos - n_patches_total; // CLS + reg tokens
    ggml_tensor * dino_patches = ggml_view_2d(ctx0, dino_cur,
        dino_n_embd, n_patches_total,
        dino_cur->nb[1],
        n_prefix_tokens * dino_cur->nb[1]);
    dino_patches = ggml_cont(ctx0, dino_patches);
    cb(dino_patches, "dino_patches", -1);

    // === SigLIP Encoder ===

    // Patch embedding via Conv2D
    ggml_tensor * siglip_emb = ggml_conv_2d(ctx0, model.fused_patch_embeddings, inp_siglip, patch_size, patch_size, 0, 0, 1, 1);
    siglip_emb = ggml_reshape_2d(ctx0, siglip_emb, n_patches_total, siglip_n_embd);
    siglip_emb = ggml_cont(ctx0, ggml_transpose(ctx0, siglip_emb));
    if (model.fused_patch_bias) {
        siglip_emb = ggml_add(ctx0, siglip_emb, model.fused_patch_bias);
    }
    cb(siglip_emb, "siglip_patch_emb", -1);

    // Add position embeddings
    siglip_emb = ggml_add(ctx0, siglip_emb, model.fused_position_embeddings);
    cb(siglip_emb, "siglip_pos_emb", -1);

    const float siglip_kq_scale = 1.0f / sqrtf((float) siglip_d_head);

    // Transformer blocks
    ggml_tensor * siglip_cur = siglip_emb;
    for (int il = 0; il < siglip_feature_layer; il++) {
        auto & layer = model.fused_layers[il];
        ggml_tensor * residual = siglip_cur;

        // LayerNorm 1
        siglip_cur = build_norm(siglip_cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(siglip_cur, "siglip_ln1", il);

        // Self-attention
        {
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.q_w, siglip_cur);
            if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);

            ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.k_w, siglip_cur);
            if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);

            ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.v_w, siglip_cur);
            if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, siglip_d_head, siglip_n_head, n_patches_total);
            Kcur = ggml_reshape_3d(ctx0, Kcur, siglip_d_head, siglip_n_head, n_patches_total);
            Vcur = ggml_reshape_3d(ctx0, Vcur, siglip_d_head, siglip_n_head, n_patches_total);

            siglip_cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, siglip_kq_scale, il);
        }
        cb(siglip_cur, "siglip_attn_out", il);

        // No LayerScale for SigLIP

        // Residual
        siglip_cur = ggml_add(ctx0, siglip_cur, residual);
        residual = siglip_cur;

        // LayerNorm 2
        siglip_cur = build_norm(siglip_cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(siglip_cur, "siglip_ln2", il);

        // FFN (GELU)
        siglip_cur = build_ffn(siglip_cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            FFN_GELU, il);
        cb(siglip_cur, "siglip_ffn", il);

        // Residual
        siglip_cur = ggml_add(ctx0, siglip_cur, residual);
    }

    ggml_tensor * siglip_patches = siglip_cur;
    cb(siglip_patches, "siglip_patches", -1);

    // === Concatenate DINO + SigLIP features ===
    // dino_patches: [1024, 256], siglip_patches: [1152, 256]
    // concat along dim 0 (embedding dim) -> [2176, 256]
    ggml_tensor * embeddings = ggml_concat(ctx0, dino_patches, siglip_patches, 0);
    cb(embeddings, "fused_features", -1);

    // === Projector: 3-layer MLP ===
    // fc1: [2176, 8704] -> GELU
    embeddings = ggml_mul_mat(ctx0, model.mm_0_w, embeddings);
    embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
    embeddings = ggml_gelu(ctx0, embeddings);
    cb(embeddings, "proj_fc1", -1);

    // fc2: [8704, 4096] -> GELU
    embeddings = ggml_mul_mat(ctx0, model.mm_2_w, embeddings);
    embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);
    embeddings = ggml_gelu(ctx0, embeddings);
    cb(embeddings, "proj_fc2", -1);

    // fc3: [4096, 4096]
    embeddings = ggml_mul_mat(ctx0, model.mm_4_w, embeddings);
    embeddings = ggml_add(ctx0, embeddings, model.mm_4_b);
    cb(embeddings, "proj_fc3", -1);

    // Build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
