#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

static void print_usage(int, char ** argv) {
    printf("\nOpenVLA example usage:\n");
    printf("\n    %s -m model.gguf --mmproj mmproj.gguf -i image.jpg -p \"prompt\"\n", argv[0]);
    printf("\nOptions:\n");
    printf("  -m TEXT_MODEL      Path to LLaMA2 text GGUF file\n");
    printf("  --mmproj MM_PROJ    Path to vision projector GGUF file\n");
    printf("  -i IMAGE          Path to input image\n");
    printf("  -p PROMPT         Text prompt\n");
    printf("  -n N_PREDICT      Number of tokens to predict (default: 256)\n");
    printf("  -ngl N_GPU_LAYERS  GPU offload layers (default: 99, -1 = all)\n");
    printf("  --no-mmproj-offload  Disable GPU for vision model\n");
    printf("  -t N_THREADS      Number of threads (default: 4)\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    // parse command line arguments

    std::string model_path;
    std::string mmproj_path;
    std::string image_path;
    std::string prompt = "What action should I take?";
    int n_predict = 256;
    int ngl = 99;
    bool mmproj_use_gpu = true;
    int n_threads = 4;

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--mmproj") == 0) {
                if (i + 1 < argc) {
                    mmproj_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-i") == 0) {
                if (i + 1 < argc) {
                    image_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-p") == 0) {
                if (i + 1 < argc) {
                    prompt = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "--no-mmproj-offload") == 0) {
                mmproj_use_gpu = false;
            } else if (strcmp(argv[i], "-t") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_threads = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        }
        if (model_path.empty()) {
            fprintf(stderr, "Error: -m TEXT_MODEL is required\n");
            print_usage(argc, argv);
            return 1;
        }
        if (mmproj_path.empty()) {
            fprintf(stderr, "Error: --mmproj MM_PROJ is required\n");
            print_usage(argc, argv);
            return 1;
        }
        if (image_path.empty()) {
            fprintf(stderr, "Error: -i IMAGE is required\n");
            print_usage(argc, argv);
            return 1;
        }
    }

    // load dynamic backends

    ggml_backend_load_all();

    // initialize the text model (LLaMA2-7B)

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "Error: unable to load text model from %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // initialize the text context

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 2048;

    llama_context * lctx = llama_init_from_model(model, ctx_params);

    if (lctx == NULL) {
        fprintf(stderr, "Error: failed to create llama_context\n");
        return 1;
    }

    // initialize the vision model (OpenVLA encoders + projector)

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = mmproj_use_gpu;
    mparams.print_timings = true;
    mparams.n_threads = n_threads;

    mtmd_context * ctx_vision = mtmd_init_from_file(
        mmproj_path.c_str(),
        model,
        mparams
    );

    if (ctx_vision == NULL) {
        fprintf(stderr, "Error: unable to load vision model from %s\n", mmproj_path.c_str());
        return 1;
    }

    // load image

    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision, image_path.c_str()));
    if (bmp.ptr == nullptr) {
        fprintf(stderr, "Error: unable to load image from %s\n", image_path.c_str());
        return 1;
    }

    mtmd::bitmaps bitmaps;
    bitmaps.entries.push_back(std::move(bmp));

    // prepare input with image marker

    const char * marker = mtmd_default_marker(); // Returns "<__media__>"
    std::string prompt_with_image = std::string(marker) + prompt;

    mtmd_input_text text;
    text.text = prompt_with_image.c_str();
    text.add_special = true;
    text.parse_special = true;

    // tokenize the input

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(
        ctx_vision,
        chunks.ptr.get(),
        &text,
        bitmaps_c_ptr.data(),
        bitmaps_c_ptr.size()
    );

    if (res != 0) {
        fprintf(stderr, "Error: tokenization failed, res = %d\n", res);
        return 1;
    }

    bitmaps.entries.clear();

    // evaluate the chunks (vision encoding + text encoding)

    llama_pos n_past = 0;
    llama_pos new_n_past;
    int32_t n_batch = 512;

    if (mtmd_helper_eval_chunks(
        ctx_vision,
        lctx,
        chunks.ptr.get(),
        n_past,
        0,  // seq_id
        n_batch,
        true,  // logits_last
        &new_n_past
    )) {
        fprintf(stderr, "Error: eval chunks failed\n");
        return 1;
    }
    n_past = new_n_past;

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // initialize batch for generation

    llama_batch batch = llama_batch_init(1, 0, 1);

    // main generation loop

    int n_decode = 0;
    llama_token new_token_id;

    printf("Response: ");

    for (int i = 0; i < n_predict; i++) {
        // sample the next token
        new_token_id = llama_sampler_sample(smpl, lctx, -1);

        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // convert token to text and print
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "Error: failed to convert token to piece\n");
            break;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
        fflush(stdout);

        // decode the token
        batch = llama_batch_get_one(&new_token_id, 1);

        n_decode++;
    }

    printf("\n\n");

    // print statistics

    fprintf(stderr, "Decoded %d tokens\n\n", n_decode);
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(lctx);
    fprintf(stderr, "\n");

    // cleanup

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(lctx);
    llama_model_free(model);
    mtmd_free(ctx_vision);

    return 0;
}
