// OpenVLA-7B action prediction example
//
// Takes an image and a text instruction, outputs 7 continuous robot action values.
// Actions: [x, y, z, rx, ry, rz, gripper]
//
// Usage:
//   llama-openvla -m openvla-llm.gguf --mmproj openvla-mmproj.gguf \
//       --image image.jpg -p "pick up the red block"

#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

// OpenVLA action decoding constants
static constexpr int N_ACTION_BINS    = 256;
static constexpr int N_ACTION_DIMS    = 7;
static constexpr int VOCAB_SIZE_ORIG  = 32000; // LLaMA-2 original vocab (before padding)

// Decode a single action token ID to a normalized value in [-1, 1]
static float decode_action_token(llama_token token_id) {
    // bin_centers = midpoints of linspace(-1, 1, 256) → 255 values
    // discretized = vocab_size_orig - token_id
    // index = clip(discretized - 1, 0, 254)
    int discretized = VOCAB_SIZE_ORIG - token_id;
    int index = discretized - 1;
    if (index < 0) index = 0;
    if (index > N_ACTION_BINS - 2) index = N_ACTION_BINS - 2;

    // bin_centers[i] = midpoint of [bins[i], bins[i+1]]
    // bins = linspace(-1, 1, 256), step = 2/255
    // bin_centers[i] = -1 + (2*i + 1) / 255
    float step = 2.0f / (N_ACTION_BINS - 1);
    float bin_left  = -1.0f + index * step;
    float bin_right = -1.0f + (index + 1) * step;
    return (bin_left + bin_right) / 2.0f;
}

int main(int argc, char ** argv) {
    common_params params;

    auto show_help = [](int, char ** argv) {
        printf("OpenVLA-7B action prediction\n\n");
        printf("Usage: %s -m <llm.gguf> --mmproj <mmproj.gguf> --image <img> -p <instruction>\n\n", argv[0]);
        printf("  Outputs 7 continuous robot action values:\n");
        printf("    [x, y, z, rx, ry, rz, gripper]\n\n");
    };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_help)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        LOG_ERR("Missing --mmproj argument\n");
        return 1;
    }
    if (params.image.empty()) {
        LOG_ERR("Missing --image argument\n");
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("Missing -p <instruction> argument\n");
        return 1;
    }

    // Initialize LLM
    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (!model || !lctx) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    // Initialize vision context
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = params.mmproj_use_gpu;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.warmup    = params.warmup;
    mtmd::context_ptr ctx_vision(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!ctx_vision.get()) {
        LOG_ERR("Failed to load vision model from %s\n", params.mmproj.path.c_str());
        return 1;
    }

    // Load image
    mtmd::bitmaps bitmaps;
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), params.image[0].c_str()));
    if (!bmp.ptr) {
        LOG_ERR("Failed to load image: %s\n", params.image[0].c_str());
        return 1;
    }
    bitmaps.entries.push_back(std::move(bmp));

    // Format prompt: prepend image marker
    // OpenVLA prompt format: "In: What action should the robot take to {instruction}?\nOut:"
    std::string prompt = std::string(mtmd_default_marker()) +
        "In: What action should the robot take to " + params.prompt + "?\nOut:";

    LOG_INF("Prompt: %s\n", prompt.c_str());

    // Tokenize with image
    mtmd_input_text text;
    text.text        = prompt.c_str();
    text.add_special = true;
    text.parse_special = true;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx_vision.get(), chunks.ptr.get(), &text,
                                bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Failed to tokenize prompt, res = %d\n", res);
        return 1;
    }

    // Evaluate prompt (text + image embeddings)
    llama_pos n_past = 0;
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx_vision.get(), lctx, chunks.ptr.get(),
                                n_past, 0, params.n_batch, true, &new_n_past)) {
        LOG_ERR("Failed to evaluate prompt\n");
        return 1;
    }
    n_past = new_n_past;

    // Generate action tokens
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    llama_batch batch = llama_batch_init(1, 0, 1);

    std::vector<llama_token> action_tokens;
    float normalized_actions[N_ACTION_DIMS];

    LOG_INF("\nGenerating %d action tokens...\n", N_ACTION_DIMS);

    for (int i = 0; i < N_ACTION_DIMS; i++) {
        llama_token token_id = common_sampler_sample(smpl, lctx, -1);
        action_tokens.push_back(token_id);
        common_sampler_accept(smpl, token_id, true);

        normalized_actions[i] = decode_action_token(token_id);

        LOG_INF("  action[%d]: token_id=%5d  normalized=% .4f\n",
                i, token_id, normalized_actions[i]);

        if (llama_vocab_is_eog(vocab, token_id)) {
            LOG_WRN("  Warning: got EOG token at action[%d], stopping early\n", i);
            break;
        }

        // Evaluate this token for next prediction
        if (i < N_ACTION_DIMS - 1) {
            common_batch_clear(batch);
            common_batch_add(batch, token_id, n_past++, {0}, true);
            if (llama_decode(lctx, batch)) {
                LOG_ERR("Failed to decode action token\n");
                return 1;
            }
        }
    }

    // Print results
    printf("\n=== OpenVLA Action Prediction ===\n");
    printf("Image: %s\n", params.image[0].c_str());
    printf("Instruction: %s\n", params.prompt.c_str());
    printf("\nRaw token IDs:      [");
    for (size_t i = 0; i < action_tokens.size(); i++) {
        printf("%d%s", action_tokens[i], i + 1 < action_tokens.size() ? ", " : "");
    }
    printf("]\n");

    printf("Normalized actions: [");
    for (size_t i = 0; i < action_tokens.size(); i++) {
        printf("% .4f%s", normalized_actions[i], i + 1 < action_tokens.size() ? ", " : "");
    }
    printf("]\n");

    printf("\nAction dimensions:\n");
    const char * dim_names[] = {"  x (translation)", "  y (translation)", "  z (translation)",
                                " rx (rotation)   ", " ry (rotation)   ", " rz (rotation)   ",
                                " gripper         "};
    for (size_t i = 0; i < action_tokens.size() && i < N_ACTION_DIMS; i++) {
        printf("  %s: % .4f\n", dim_names[i], normalized_actions[i]);
    }
    printf("\nNote: These are normalized actions in [-1, 1].\n");
    printf("To get real-world values, unnormalize using dataset-specific\n");
    printf("statistics (q01/q99) from the model's config.json norm_stats.\n");

    // Cleanup
    llama_batch_free(batch);
    common_sampler_free(smpl);

    return 0;
}
