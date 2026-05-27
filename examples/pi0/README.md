# PI0 on llama.cpp

## Overview

PI0 is a robotics flow-matching model that predicts continuous robot actions from images and text instructions. It uses iterative denoising (10 steps) instead of autoregressive token generation.

### Architecture

```
Images (3 views)          Text Prompt              Robot State          Noisy Actions + Timestep
      |                        |                       |                        |
  SigLIP So400m          Gemma Tokenizer        state_proj(Linear)    action_in_proj + sincos + MLP
      |                        |                       |                        |
  Image Tokens            Text Tokens            State Token (1)       Action Tokens (50)
      |________________________|                       |________________________|
               |                                                |
         PREFIX tokens                                   SUFFIX tokens
               |                                                |
      PaLI-Gemma 2B (18 layers)                   Action Expert 300M (18 layers)
      width=2048, 8 heads                          width=1024, 8 heads
               |                                                |
         KV cached ──────────── cross-attention ──────────> suffix output
                                                                |
                                              suffix_out[:, -50:] -> action_out_proj -> v_t
```

**Key insight**: Both experts have `num_kv_heads=1, head_dim=256`, so KV cache from PaliGemma can be injected into Action Expert's attention. During inference:
1. PaliGemma processes prefix once → KV cached
2. Action Expert runs 10 times (diffusion steps), each time cross-attending to cached prefix KV

### GGUF Files

| File | Contents | Size (f16) |
|------|----------|------------|
| `pi0-mmproj.gguf` | SigLIP So400m (27 layers, 1152-dim) + Linear projector (1152→2048) | ~794 MB |
| `pi0-gemma-2b.gguf` | PaliGemma 2B language model (Gemma arch, 18 layers, 2048-dim) | ~5.8 GB |
| `pi0-action-expert.gguf` | Action Expert 300M (18 layers, 1024-dim) + action/state projections | ~600 MB |

## Build
### Step 1: Convert Models to GGUF

**Prerequisites**: Python 3.10+, numpy, gguf-py (included in llama.cpp)

```bash
# From the llama.cpp directory
python tools/mtmd/convert_pi0_to_gguf.py \
    --model /path/to/pi0-hf \
    --output-dir /path/to/output \
    --type f16
```

This reads the HuggingFace PI0 model (`model.safetensors`) and produces 3 GGUF files:
- `pi0-mmproj.gguf` — vision encoder + projector
- `pi0-gemma-2b.gguf` — PaliGemma 2B (standard Gemma format)
- `pi0-action-expert.gguf` — Action Expert + action projections

**Example** (using the default paths from this project):
```bash
python tools/mtmd/convert_pi0_to_gguf.py \
    --model /home/genteki/hf-models/pi0 \
    --output-dir /home/genteki/gguf/pi0 \
    --type f16
```

### Step 2: Build llama.cpp

```bash
cd llama.cpp
cmake -B build
cmake --build build --target llama-pi0 -j$(nproc)
cmake --build build --target llama-pi0-bench -j$(nproc)
```

The `llama-pi0` binary will be in `build/bin/`.

### Step 3: Run Inference

### Single inference
```bash
# Set the expert model path
export PI0_EXPERT_PATH=/path/to/pi0-action-expert.gguf

./build/bin/llama-pi0 \
    -m /home/genteki/gguf/pi0 \
    --image ../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg,../data/cam_high.jpg \
    -p "pick up the red block"
    --backend cpu --kv-type f32
```

**Arguments**:
- `-m` — Path to pi0 gguf models dir
- `--image` — Camera images (up to 3 views; each produces 256 tokens)
- `-p` — Text instruction for the robot

### Bench
```
./build/bin/llama-pi0-bench \
  -m /home/genteki/gguf/pi0-q8 \
  --image ../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg,../data/cam_high.jpg \
  -p "pick up the red block" \
  -n 20 --warmup 3
  --backend cpu --kv-type f16
```

OpenCL:
```
./build-cl/bin/llama-pi0-bench -m /home/genteki/gguf/pi0 \
  --image ../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg,../data/cam_high.jpg \
  -p "pick up the red block" \
  -n 5 --warmup 1 \
  --backend OpenCL --kv-type f32
```

### Understanding the Output

The output is a `50 × 32` matrix of continuous action values:

```
=== PI0 Action Predictions ===
Instruction: pick up the red block
Action horizon: 50 steps, Action dim: 32

  step  0: [ 0.0123, -0.0456, ...]
  step  1: [ 0.0234, -0.0567, ...]
  ...
  step 49: [ 0.0345, -0.0678, ...]
```

- **50 steps** = action horizon (future timesteps to predict)
- **32 dimensions** = action features per step (joint positions, gripper, etc.)
- Values are in the model's learned action space. To convert to real robot commands, apply the dataset-specific normalization statistics from the model's config.

## Flow Matching Process

The model uses 10 denoising steps to refine actions from pure noise:
```
t=1.0 (noise) → t=0.9 → t=0.8 → ... → t=0.1 → t=0.0 (predicted actions)
```

At each step: `x_t = x_t + dt × v_t`, where `v_t` is the velocity predicted by the Action Expert.

## Files Changed in llama.cpp

| File | Change |
|------|--------|
| `tools/mtmd/convert_pi0_to_gguf.py` | New: GGUF conversion script |
| `tools/mtmd/clip-impl.h` | Added `PROJECTOR_TYPE_PI0` |
| `tools/mtmd/clip.cpp` | Added PI0 to vision pipeline switches |
| `tools/mtmd/models/siglip.cpp` | Added PI0 linear projector branch |
| `examples/pi0/pi0.cpp` | New: main inference example |
| `examples/pi0/CMakeLists.txt` | New: build config |
| `examples/CMakeLists.txt` | Added `add_subdirectory(pi0)` |


## Android Build Guide

### Prerequisites
Android NDK — download from developer.android.com/ndk or install via Android Studio's SDK Manager. Note the path (e.g. $HOME/android-sdk/ndk/26.3.11579264).

CMake 3.21+ on your host machine.

Build steps
'''
cd /home/genteki/gentekis_document/research/pi0/llama.cpp
'''

### Set your NDK path
'''
export NDK=~/android-sdk/Sdk/ndk/<version>   # adjust to your actual path
'''

### Quantize
```
./build/bin/llama-quantize \
  /home/genteki/gguf/pi0/pi0-gemma-2b.gguf \
  /home/genteki/gguf/pi0-q4/pi0-gemma-2b.gguf \
  Q4_K_M
```


### Configure for Android (arm64-v8a is standard for modern phones)
'''
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=OFF \
  -DBUILD_SHARED_LIBS=OFF
'''

### Build just the pi0 target (and its dependencies)
**Build**
'''
cmake --build build-android --target llama-pi0 -j$(nproc)
cmake --build build-android --target llama-pi0-bench -j$(nproc)
'''

**Deploy to device**
'''
adb shell mkdir -p /data/local/tmp/pi0/data
adb push build-android/bin/llama-pi0       /data/local/tmp/pi0/
adb push build-android/bin/llama-pi0-bench /data/local/tmp/pi0/
adb push /home/genteki/gguf/pi0-q4        /data/local/tmp/pi0/gguf

# Test images — adjust paths if they live elsewhere
adb push data/cam_left_wrist.jpg data/cam_right_wrist.jpg data/cam_high.jpg \
         /data/local/tmp/pi0/data/

adb shell
cd /data/local/tmp/pi0
chmod +x llama-pi0 llama-pi0-bench  
'''

**On device**
'''
adb shell
cd /data/local/tmp
chmod +x llama-pi0
./llama-pi0 -m pi0.gguf [other args]
'''


Inference
```
./llama-pi0 -m gguf/pi0-q4 \
    --image data/cam_left_wrist.jpg \
    --image data/cam_right_wrist.jpg \
    --image data/cam_high.jpg \
    -p "pick up the red block" \
    --backend cpu/auto/GPUOpenCL --kv-type f16
```
Bench
```
./llama-pi0-bench -m gguf/pi0-q4 \
    --image data/cam_left_wrist.jpg,data/cam_right_wrist.jpg,data/cam_high.jpg \
    -p "pick up the red block" \
    -n 1 --warmup 1 \
    --backend GPUOpenCL --kv-type f16
```
(CPU)
```
./llama-pi0-bench -m gguf/pi0-q4 \
    --image data/cam_left_wrist.jpg,data/cam_right_wrist.jpg,data/cam_high.jpg \
    -p "pick up the red block" \
    -n 1 --warmup 1 -ngl 0 \
    --backend cpu --kv-type f16
```