// PI0.5 action prediction example.
//
// Flow-matching VLA: SigLIP (vision) -> PaliGemma 2B (prefix) -> Action Expert
// 300M with adaRMS (suffix). Differs from pi0: no state token in the suffix
// (state is folded into the discrete prefix tokens) and the timestep is injected
// via adaptive RMSNorm. See pi0_5-common.h.
//
// Usage:
//   llama-pi0_5 -m <model_dir> --image a.jpg,b.jpg,c.jpg -p "pick up the red block"
//               [--backend cpu|auto|<name>] [--kv-type f16] [--act-type f16]

#include "pi0_5-common.h"

#include "arg.h"
#include "log.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    pi05_cli cli = pi05_strip_cli_args(argc, argv);
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0.5 action prediction via flow matching\n\n");
        printf("Usage: %s -m <model_dir> \\\n", argv[0]);
        printf("         --image img1.jpg,img2.jpg,img3.jpg \\\n");
        printf("         -p <instruction> [--backend cpu|auto|<name>] [--kv-type f16] [--act-type f16]\n\n");
        printf("  -m <dir>     Model directory containing pi05-gemma-2b.gguf,\n");
        printf("               pi05-mmproj.gguf, pi05-action-expert.gguf\n");
        printf("  --backend    'cpu' (default), 'auto'/'gpu', or a concrete name (OpenCL, ...)\n");
        printf("  --kv-type    KV cache element type: f16 (default), f32, q8_0, q4_0\n");
        printf("  --act-type   activation dtype into matmuls: f16 (default), f32 (exact CPU parity)\n");
    };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) return 1;

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    std::string model_dir   = resolve_model_dir(params.model.path);
    std::string pali_path   = model_dir + "/pi05-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi05-mmproj.gguf";
    std::string expert_path = model_dir + "/pi05-action-expert.gguf";

    // Optional fused-QKV PaliGemma variant for the pi05 loader; llama's own
    // loader (tokenizer) still needs the unfused pi05-gemma-2b.gguf.
    std::string pali_pi05_path = model_dir + "/pi05-gemma-2b-fused.gguf";
    if (FILE * f = fopen(pali_pi05_path.c_str(), "rb")) { fclose(f); } else { pali_pi05_path = pali_path; }

    if (!params.mmproj.path.empty()) mmproj_path = params.mmproj.path;

    LOG_INF("Model directory: %s\n", model_dir.c_str());
    LOG_INF("  PaliGemma 2B:  %s\n", pali_path.c_str());
    LOG_INF("  Vision mmproj: %s\n", mmproj_path.c_str());
    LOG_INF("  Action expert: %s\n", expert_path.c_str());

    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    if (cli.backend == "cpu" || cli.backend == "CPU") {
        params.n_gpu_layers   = 0;
        params.mmproj_use_gpu = false;
    }

    ggml_backend_t backend = pi05_init_backend(cli.backend);
    if (!backend) { LOG_ERR("Failed to init backend '%s'\n", cli.backend.c_str()); return 1; }

    pi05_model model;
    if (!load_pi05_model(model, pali_pi05_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0.5 model\n"); return 1;
    }
    LOG_INF("Model loaded\n");
    pi05_dump_tensor_types(model);

    auto llama_init = common_init_from_params(params);
    llama_model   * llm_model = llama_init->model();
    llama_context * llm_ctx   = llama_init->context();
    if (!llm_model || !llm_ctx) { LOG_ERR("Failed to load LLM for tokenizer\n"); return 1; }

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) { LOG_ERR("Failed to load vision model\n"); return 1; }

    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) { LOG_ERR("Failed to load image: %s\n", img_path.c_str()); return 1; }
        bitmaps.entries.push_back(std::move(bmp));
        LOG_INF("Loaded image: %s\n", img_path.c_str());
    }

    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) prompt += mtmd_default_marker();
    prompt += params.prompt;
    LOG_INF("Prompt: %s\n", prompt.c_str());

    mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    if (mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &text,
                      bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        LOG_ERR("Failed to tokenize\n"); return 1;
    }

    // Prefix = [ image tokens (SigLIP, unscaled) ] ++ [ text tokens (scaled) ].
    // (pi0.5 folds the robot state into the text tokens upstream; here it is just
    // whatever the prompt contains, identical preprocessing to pi0.)
    const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    std::vector<float> prefix_embeddings;
    int actual_prefix_len = 0;
    std::vector<float> emb_row(PALI_N_EMBD);
    const float embd_scale = sqrtf((float) PALI_N_EMBD);

    for (int ic = 0; ic < n_chunks; ic++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (mtmd_encode_chunk(ctx_vision.get(), chunk) != 0) {
            LOG_ERR("Failed to encode image chunk %d\n", ic); return 1;
        }
        float * img_embd = mtmd_get_output_embd(ctx_vision.get());
        size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        prefix_embeddings.insert(prefix_embeddings.end(), img_embd, img_embd + n_img_tokens * PALI_N_EMBD);
        actual_prefix_len += (int) n_img_tokens;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llm_model);
    std::vector<llama_token> text_tokens;
    text_tokens.push_back(llama_vocab_bos(vocab));
    {
        std::vector<llama_token> body = common_tokenize(llm_ctx, params.prompt, false, false);
        text_tokens.insert(text_tokens.end(), body.begin(), body.end());
        std::vector<llama_token> nl = common_tokenize(llm_ctx, "\n", false, false);
        text_tokens.insert(text_tokens.end(), nl.begin(), nl.end());
    }
    for (llama_token tok : text_tokens) {
        embd_lookup_f32(model.pali_embed, tok, emb_row.data());
        for (float & e : emb_row) e *= embd_scale;
        prefix_embeddings.insert(prefix_embeddings.end(), emb_row.begin(), emb_row.end());
        actual_prefix_len++;
    }

    LOG_INF("Assembled prefix: %d tokens x %d dims (%zu text tokens)\n",
            actual_prefix_len, PALI_N_EMBD, text_tokens.size());

    if (const char * dump_path = getenv("PI05_DUMP_PREFIX")) {
        FILE * pf = fopen(dump_path, "wb");
        if (pf) {
            fwrite(prefix_embeddings.data(), sizeof(float), prefix_embeddings.size(), pf);
            fclose(pf);
            LOG_INF("Dumped prefix embeddings to %s\n", dump_path);
        }
    }

    pi05_session session;
    session.kv_type  = pi05_ggml_type_from_string(cli.kv_type);
    session.act_type = pi05_resolve_act_type(cli.act_type, cli.backend);
    if (session.kv_type == GGML_TYPE_COUNT || session.act_type == GGML_TYPE_COUNT) {
        LOG_ERR("Bad --kv-type/--act-type (try f16, f32, q8_0, q4_0)\n"); return 1;
    }
    LOG_INF("KV cache: %s | activations: %s\n",
            ggml_type_name(session.kv_type), ggml_type_name(session.act_type));

    LOG_INF("Running PaliGemma prefix pass...\n");
    if (!run_paligemma_prefix(model, session, backend, prefix_embeddings.data(), actual_prefix_len)) {
        LOG_ERR("Prefix pass failed\n"); return 1;
    }
    LOG_INF("PaliGemma prefix KV cached (%d tokens)\n", actual_prefix_len);

    // Initial noise at t=1. PI05_NOISE_BIN loads an exact reference noise tensor
    // (row-major [HORIZON][DIM]) so output can be compared against openpi.
    std::vector<float> x_t(PI05_ACTION_DIM * PI05_ACTION_HORIZON);
    if (const char * noise_path = getenv("PI05_NOISE_BIN")) {
        FILE * nf = fopen(noise_path, "rb");
        if (!nf) { LOG_ERR("PI05_NOISE_BIN: cannot open %s\n", noise_path); return 1; }
        size_t got = fread(x_t.data(), sizeof(float), x_t.size(), nf);
        fclose(nf);
        if (got != x_t.size()) { LOG_ERR("PI05_NOISE_BIN: expected %zu floats, read %zu\n", x_t.size(), got); return 1; }
        LOG_INF("Loaded initial noise from %s\n", noise_path);
    } else {
        std::mt19937 rng(42);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto & v : x_t) v = normal(rng);
    }

    LOG_INF("\nRunning flow matching diffusion (%d steps)...\n", PI05_NUM_STEPS);
    const float dt = -1.0f / PI05_NUM_STEPS;

    FILE * vt_dump = nullptr;
    if (const char * vt_path = getenv("PI05_DUMP_VT")) {
        vt_dump = fopen(vt_path, "wb");
        if (!vt_dump) LOG_ERR("PI05_DUMP_VT: cannot open %s\n", vt_path);
    }

    std::vector<float> v_t(PI05_ACTION_DIM * PI05_ACTION_HORIZON);
    for (int step = 0; step < PI05_NUM_STEPS; step++) {
        float t = 1.0f + step * dt;
        LOG_INF("  Step %d/%d (t=%.2f)\n", step + 1, PI05_NUM_STEPS, t);
        if (!run_expert_step(model, session, backend, x_t.data(), t, v_t.data())) {
            LOG_ERR("Expert step %d failed\n", step); return 1;
        }
        if (vt_dump) fwrite(v_t.data(), sizeof(float), v_t.size(), vt_dump);
        for (int i = 0; i < PI05_ACTION_DIM * PI05_ACTION_HORIZON; i++) x_t[i] += dt * v_t[i];
    }
    if (vt_dump) fclose(vt_dump);

    printf("\n=== PI0.5 Action Predictions ===\n");
    printf("Instruction: %s\n", params.prompt.c_str());
    printf("Action horizon: %d steps, Action dim: %d\n\n", PI05_ACTION_HORIZON, PI05_ACTION_DIM);
    for (int t = 0; t < PI05_ACTION_HORIZON; t++) {
        printf("  step %2d: [", t);
        for (int d = 0; d < PI05_ACTION_DIM; d++) {
            printf("% .4f", x_t[t * PI05_ACTION_DIM + d]);
            if (d < PI05_ACTION_DIM - 1) printf(", ");
        }
        printf("]\n");
    }

    pi05_session_free(session);
    free_pi05_model(model);
    ggml_backend_free(backend);
    return 0;
}
