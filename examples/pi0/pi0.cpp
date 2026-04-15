// PI0 action prediction example
//
// Uses flow matching diffusion to predict robot actions from images + text.
// Architecture: SigLIP (vision) → PaliGemma 2B (prefix) → Action Expert 300M (suffix)
//
// Usage:
//   llama-pi0 -m pi0-gemma-2b.gguf --mmproj pi0-mmproj.gguf
//       --expert pi0-action-expert.gguf
//       --image img1.jpg --image img2.jpg --image img3.jpg
//       -p "pick up the red block"

#include "pi0-common.h"

// ---- Main ----

int main(int argc, char ** argv) {
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("PI0 action prediction via flow matching\n\n");
        printf("Usage: %s -m <model_dir> \\\n", argv[0]);
        printf("         --image img1.jpg,img2.jpg,img3.jpg \\\n");
        printf("         -p <instruction>\n\n");
        printf("  -m <dir>   Model directory containing:\n");
        printf("               pi0-gemma-2b.gguf\n");
        printf("               pi0-mmproj.gguf\n");
        printf("               pi0-action-expert.gguf\n");
    };

    // Parse using common params (reuse --image for multiple images)
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    // Resolve model directory from -m path
    std::string model_dir = resolve_model_dir(params.model.path);

    std::string pali_path   = model_dir + "/pi0-gemma-2b.gguf";
    std::string mmproj_path = model_dir + "/pi0-mmproj.gguf";
    std::string expert_path = model_dir + "/pi0-action-expert.gguf";

    // Allow --mmproj override
    if (!params.mmproj.path.empty()) {
        mmproj_path = params.mmproj.path;
    }

    LOG_INF("Model directory: %s\n", model_dir.c_str());
    LOG_INF("  PaliGemma 2B:    %s\n", pali_path.c_str());
    LOG_INF("  Vision mmproj:   %s\n", mmproj_path.c_str());
    LOG_INF("  Action expert:   %s\n", expert_path.c_str());

    // Override params so llama loads the right model for tokenizer
    params.model.path  = pali_path;
    params.mmproj.path = mmproj_path;

    // Initialize backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG_ERR("Failed to init CPU backend\n");
        return 1;
    }

    // Load PI0 model weights
    pi0_model model;
    if (!load_pi0_model(model, pali_path.c_str(), expert_path.c_str(), backend)) {
        LOG_ERR("Failed to load PI0 model\n");
        return 1;
    }
    LOG_INF("Model loaded successfully\n");

    // Initialize LLM (needed for tokenizer and mtmd compatibility)
    auto llama_init = common_init_from_params(params);
    llama_model   * llm_model = llama_init->model();
    llama_context * llm_ctx   = llama_init->context();

    if (!llm_model || !llm_ctx) {
        LOG_ERR("Failed to load LLM for tokenizer\n");
        return 1;
    }

    // Initialize vision context
    mtmd_context_params mparams = mtmd_context_params_default(); // is it cuda by default?
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup; // what is warmup here? 
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), llm_model, mparams));
    if (!ctx_vision.get()) {
        LOG_ERR("Failed to load vision model\n");
        return 1;
    }

    // Load images (up to 3)
    mtmd::bitmaps bitmaps;
    for (const auto & img_path : params.image) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), img_path.c_str()));
        if (!bmp.ptr) {
            LOG_ERR("Failed to load image: %s\n", img_path.c_str());
            return 1;
        }
        bitmaps.entries.push_back(std::move(bmp));
        LOG_INF("Loaded image: %s\n", img_path.c_str());
    }

    // Build prompt with image markers
    std::string prompt;
    for (size_t i = 0; i < params.image.size(); i++) {
        prompt += mtmd_default_marker();
    }
    prompt += params.prompt;
    LOG_INF("Prompt: %s\n", prompt.c_str());

    // Tokenize
    mtmd_input_text text;
    text.text = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &text,
                                bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Failed to tokenize, res = %d\n", res);
        return 1;
    }

    // Build prefix embeddings: encode images via mtmd, look up text tokens from embedding table
    // Load embedding table to CPU once (for text token lookup)
    std::vector<float> embd_table(ggml_nelements(model.pali_embed));
    tensor_to_f32(model.pali_embed, embd_table.data());

    const int n_chunks = mtmd_input_chunks_size(chunks.ptr.get());
    std::vector<float> prefix_embeddings;
    int actual_prefix_len = 0;
    const int n_embd_out = PALI_N_EMBD; // mmproj projects to this dim

    for (int ic = 0; ic < n_chunks; ic++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.ptr.get(), ic);
        mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            // Text tokens - look up embeddings from PaliGemma embedding table
            size_t n_tokens = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);

            for (size_t t = 0; t < n_tokens; t++) {
                llama_token token_id = tokens[t];
                const float * emb = &embd_table[(size_t)token_id * PALI_N_EMBD];
                prefix_embeddings.insert(prefix_embeddings.end(), emb, emb + PALI_N_EMBD);
                actual_prefix_len++;
            }
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            // Image chunk - encode through vision model to get embeddings
            if (mtmd_encode_chunk(ctx_vision.get(), chunk) != 0) {
                LOG_ERR("Failed to encode image chunk %d\n", ic);
                return 1;
            }
            float * img_embd = mtmd_get_output_embd(ctx_vision.get());
            size_t n_img_tokens = mtmd_input_chunk_get_n_tokens(chunk);

            prefix_embeddings.insert(prefix_embeddings.end(),
                img_embd, img_embd + n_img_tokens * n_embd_out);
            actual_prefix_len += (int)n_img_tokens;
        }
    }

    LOG_INF("Assembled prefix: %d tokens × %d dims\n", actual_prefix_len, PALI_N_EMBD);

    // Run PaliGemma prefix to get KV cache
    LOG_INF("Running PaliGemma prefix pass...\n");
    if (!run_paligemma_prefix_v2(model, backend, prefix_embeddings.data(), actual_prefix_len)) {
        LOG_ERR("Prefix pass failed\n");
        return 1;
    }
    LOG_INF("PaliGemma prefix KV cached (%d tokens)\n", actual_prefix_len);

    // Initialize noisy actions (pure noise at t=1)
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<float> x_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
    for (auto & v : x_t) v = normal(rng);

    // Robot state (zeros for demo - in production, read from robot)
    std::vector<float> robot_state(PI0_ACTION_DIM, 0.0f);

    // Diffusion loop: denoise from t=1 to t=0
    LOG_INF("\nRunning flow matching diffusion (%d steps)...\n", PI0_NUM_STEPS);
    const float dt = -1.0f / PI0_NUM_STEPS;

    for (int step = 0; step < PI0_NUM_STEPS; step++) {
        float t = 1.0f + step * dt;
        LOG_INF("  Step %d/%d (t=%.2f)\n", step + 1, PI0_NUM_STEPS, t);

        // Prepare suffix embeddings
        std::vector<float> suffix_emb(EXPERT_N_EMBD * (PI0_ACTION_HORIZON + 1));
        prepare_suffix(model, robot_state.data(), x_t.data(), t, suffix_emb.data());

        // Run action expert
        std::vector<float> v_t(PI0_ACTION_DIM * PI0_ACTION_HORIZON);
        if (!run_expert_step(model, backend, suffix_emb.data(), v_t.data())) {
            LOG_ERR("Expert step %d failed\n", step);
            return 1;
        }

        // Update: x_t = x_t + dt * v_t
        for (int i = 0; i < PI0_ACTION_DIM * PI0_ACTION_HORIZON; i++) {
            x_t[i] += dt * v_t[i];
        }
    }

    // Output final actions
    printf("\n=== PI0 Action Predictions ===\n");
    printf("Instruction: %s\n", params.prompt.c_str());
    printf("Action horizon: %d steps, Action dim: %d\n\n", PI0_ACTION_HORIZON, PI0_ACTION_DIM);

    for (int t = 0; t < PI0_ACTION_HORIZON; t++) {
        printf("  step %2d: [", t);
        for (int d = 0; d < PI0_ACTION_DIM; d++) {
            printf("% .4f", x_t[t * PI0_ACTION_DIM + d]);
            if (d < PI0_ACTION_DIM - 1) printf(", ");
        }
        printf("]\n");
    }

    // Cleanup
    if (model.buf_pali)   ggml_backend_buffer_free(model.buf_pali);
    if (model.buf_expert) ggml_backend_buffer_free(model.buf_expert);
    if (model.ctx_pali)   ggml_free(model.ctx_pali);
    if (model.ctx_expert) ggml_free(model.ctx_expert);
    ggml_backend_free(backend);

    return 0;
}
