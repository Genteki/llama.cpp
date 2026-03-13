# OpenVLA Example

Example demonstrating how to use OpenVLA (Open Vision-Language-Action) models with llama.cpp.

OpenVLA is a vision-language-action model that can understand images and generate robot actions based on visual input.

## Model Architecture

OpenVLA-7B contains:
- **LLaMA2-7B language model** - 7B parameter text decoder
- **DINOv2 ViT-L/14 vision encoder** - 1024-dim embeddings, 261 tokens (CLS + 4 register + 256 patches)
- **SigLIP ViT-So400M/14 vision encoder** - 1152-dim embeddings, 256 patches
- **3-layer MLP projector** - Concatenates encoder outputs [2176] → [4096] → [4096] → [4096]

The vision encoders process a 224x224 image and produce 256 patch tokens that are then projected to match the LLaMA2 embedding dimension.

## Prerequisites

- llama.cpp built with libmtmd support
- OpenVLA GGUF model files (text + mmproj)

## Building

```bash
cd /path/to/llama.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target llama-openvla
```

## Usage

```bash
./build/bin/llama-openvla \
    -m /path/to/openvla-7b-text.gguf \
    --mmproj /path/to/mmproj-openvla-7b-f16.gguf \
    -i /path/to/image.jpg \
    -p "What action should I take?" \
    -n 256 \
    -ngl 99
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `-m TEXT_MODEL` | Path to LLaMA2 text GGUF file |
| `--mmproj MM_PROJ` | Path to vision projector GGUF file (required) |
| `-i IMAGE` | Path to input image (JPEG/PNG supported) |
| `-p PROMPT` | Text prompt (default: "What action should I take?") |
| `-n N_PREDICT` | Number of tokens to predict (default: 256) |
| `-ngl N_GPU_LAYERS` | GPU offload layers (default: 99, -1 = all) |
| `--no-mmproj-offload` | Disable GPU for vision model |
| `-t N_THREADS` | Number of threads (default: 4) |

## Model Conversion

To use this example, you need to convert the HuggingFace OpenVLA model to GGUF format.

### Option 1: Use llama-mtmd-cli (Recommended)

Let llama.cpp handle the conversion automatically:

```bash
./build/bin/llama-mtmd-cli -hf Stanford/openvla-7b --image test.jpg -p "What is this?"
```

This will download and convert the model on the fly.

### Option 2: Manual Conversion

Use the conversion script:

```bash
cd /path/to/llama.cpp
python tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py \
    --model-dir /path/to/openvla-7b \
    --output-dir /path/to/gguf \
    --outtype f16
```

This will create:
- `openvla-7b-text.gguf` - LLaMA2 language model
- `mmproj-openvla-7b-f16.gguf` - Vision encoders + projector

## Example Output

```bash
$ ./llama-openvla \
    -m gguf/openvla-7b-text.gguf \
    --mmproj gguf/mmproj-openvla-7b-f16.gguf \
    -i test_image.jpg \
    -p "Pick up the red ball"

Response: Move the end effector towards the red ball located at coordinates (0.5, 0.3, 0.1) and close the gripper to grasp it.

Decoded 42 tokens

llama_perf_sampler_print:    samplers time =       5.23 ms /    42 runs
llama_perf_context_print:        load time =    1234.56 ms
llama_perf_context_print: prompt eval time =     123.45 ms /   256 tokens (    0.48 ms per token,  2073.12 tokens per second)
llama_perf_context_print:        eval time =      78.90 ms /    42 runs   (    1.88 ms per token,   531.91 tokens per second)
llama_perf_context_print:       total time =    1436.91 ms /   298 tokens
```

## Notes

- The `<__media__>` marker is automatically inserted for vision input
- Image preprocessing (normalization, resizing) is handled by `mtmd_helper_bitmap_init_from_file()`
- DINOv2 uses ImageNet normalization
- SigLIP uses 0.5/0.5 normalization
- OpenVLA supports multiple datasets with different normalization statistics (stored in config.json)

## References

- OpenVLA: https://openvla.github.io/
- HuggingFace: https://huggingface.co/openvla/openvla-7b
- llama.cpp: https://github.com/ggerganov/llama.cpp
