# PI0 on llama.cpp

## ⚡ Full-Optimization Quickstart (Adreno GPU)

> **To get the optimized numbers you MUST enable all of these — none are on by default.**
> Missing `PI0_VIT_IMG` alone makes Vision ~2.5× slower (ViT attention falls to a naive
> `mul_mat_f32_f32` at ~14 GFLOPS instead of the padded Ab_Bi path).

**1. Env flags (all required):**

| Flag | Effect | If missing |
|------|--------|-----------|
| `PI0_VIT_IMG=1` | Pads ViT attn head_dim 72→80 → Ab_Bi/`l4_lm` (clip.cpp) | Vision attn on `mul_mat_f32_f32`, **17× slower** |
| `PI0_VIT_WEIGHT_PRETRANS=1` | Transpose static ViT f16 weights to `[K][M]` **once** (first forward, cached in-place) instead of every frame | `transpose_16_buf` re-transposes weights each forward, **Vision +156 ms** (bit-identical either way) |
| `PI0_ATTN_IMG=1` | Routes prefix/diffusion QK·AV → `Ab_Bi_8x4_f16` image kernel | Attn AV ~639ms instead of ~21ms |
| `PI0_ATTN_COLLAPSE=1` | MQA head-fold → 2D GEMM (**required** for `ATTN_IMG`) | un-collapsed batched MQA stays on slow kernel |
| `PI0_SPLITK=1` | Diffusion FFN down/o_proj split-K (occupancy) | diffusion FFN under-occupied |
| `PI0_NO_FLASH_ATTN=1` | Keep attn on the routable Ab_Bi kernels | flash bypasses → QK/AV not routed |

**2. Model dir must be the fused package** (`pi0-q4best`): fused-QKV prefix
(`pi0-gemma-2b-fused.gguf`) + fused-QKV q4 expert. Fusion is auto-detected from the
`attn_qkv.weight` tensor / the `-fused.gguf` file — no flag.

**3. Device OpenCL env** (Adreno 830, Snapdragon 8 Elite): use the **vendor** ICD.

```bash
cd /data/local/tmp/pi0/build-android-cl
env LD_PRELOAD=/system/lib64/libbinder.so \
    LD_LIBRARY_PATH=.:/vendor/lib64 \
    OCL_ICD_FILENAMES=/vendor/lib64/libOpenCL_adreno.so \
    PI0_VIT_IMG=1 PI0_VIT_WEIGHT_PRETRANS=1 PI0_ATTN_IMG=1 PI0_ATTN_COLLAPSE=1 PI0_SPLITK=1 PI0_NO_FLASH_ATTN=1 \
    ./llama-pi0-bench -m gguf/pi0-q4best \
      --image data/cam_left_wrist.jpg,data/cam_right_wrist.jpg,data/cam_high.jpg \
      -p "pick up the red block" \
      -n 6 --warmup 2 --backend GPUOpenCL --kv-type f16
```

**Reference (Adreno 830, 3 views, cooled, all flags on):** Vision ~635 ms · Prefix ~1480 ms
· Diffusion(×10) ~390 ms (Total ~2550 ms). Verify flags took: the per-kernel summary should show
`Ab_Bi_8x4_f16` for attention, `mul_mat_f32_f32` reduced to a handful of calls, and
`transpose_16_buf` down to ~9 ms (warmup-only) rather than ~163 ms.

> **Measurement hygiene (mobile GPU):** the Adreno throttles hard under sustained load —
> a no-cooldown 20-iter run drifts Vision 2005→4126 ms. Insert a cooldown (≥60 s) before
> each measurement and use a small `-n` (min/median), or numbers are not reproducible.

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
## Hexagon (NPU) Backend

PI0's **prefix** (PaliGemma-2B) and **flow-matching diffusion** (action expert) both run
on the Qualcomm **Hexagon NPU** via the in-tree `ggml-hexagon` backend (HMX matrix unit
over FastRPC). The **vision** encoder (SigLIP) stays on the GPU — HTP has no CONV, so it
falls back automatically. On an SM8750 (Snapdragon 8 Elite / Hexagon v79) the NPU runs
the prefix **~3.3× faster** than the tuned Adreno OpenCL path.

### Requirements
- Snapdragon device with a Hexagon NPU (tested: 8 Elite / **v79**; skels also build for
  v73 / v75 / v81).
- Hexagon SDK + NDK. Easiest via the official toolchain container
  `ghcr.io/snapdragon-toolchain/arm64-android:v0.3` (NDK r28b + Hexagon SDK 6.4.0.2).

### Build
```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/workspace" ghcr.io/snapdragon-toolchain/arm64-android:v0.3 bash -lc '
    cd /workspace
    cmake --preset arm64-android-snapdragon-release -B build-snapdragon   # GGML_HEXAGON=ON + GGML_OPENCL=ON
    cmake --build build-snapdragon --target llama-pi0-bench htp-v79 -j$(nproc)
  '
```
Produces `libggml-hexagon.so` (host) + `libggml-htp-v79.so` (the DSP skel).

### Deploy + run
```bash
DST=/data/local/tmp/pi0
adb push build-snapdragon/bin/*.so                                  $DST/lib/
adb push build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-v79.so  $DST/lib/
adb push build-snapdragon/bin/llama-pi0-bench                       $DST/bin/

adb shell "cd $DST && LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
  ./bin/llama-pi0-bench --backend HTP0 --kv-type f16 -n 5 --warmup 1 \
     -m gguf/pi0-q4 --image data/cam_left_wrist.jpg,data/cam_right_wrist.jpg,data/cam_high.jpg"
```
- `--backend HTP0` selects the NPU (prefix + diffusion offload; vision → OpenCL).
- `ADSP_LIBRARY_PATH` must point at the dir holding `libggml-htp-v79.so`.
- **Keep flash attention ON** (default): on the NPU the HMX flash kernel is **~4× faster**
  than standard attention for PI0's short KV cache — do **not** set `PI0_NO_FLASH_ATTN=1` here.
- Gotcha: for a `GPUOpenCL` / `cpu` run in a build that also links Hexagon, set
  `GGML_HEXAGON_NDEV=0` to skip HTP session-open (a lingering session otherwise aborts the run).

### Performance (SM8750 · Adreno 830 / Hexagon v79 · pi0-q4, kv f16)
| phase | tuned OpenCL (Adreno) | **HTP0 (NPU)** | speedup |
|---|---|---|---|
| Prefix (PaliGemma-2B, 773 tok) | ~1520 ms | **~460 ms** | **~3.3×** |
| Diffusion (10 flow-matching steps) | ~420 ms  | **~310 ms** | ~1.4× |
| Vision (SigLIP, stays on GPU) | ~620 ms | ~620 ms | — |

The NPU wins big on the prefix — the FFN/proj GEMMs hit **8–15 TFLOPS** on HMX. Diffusion
is closer to parity (small seq=51 underutilizes HMX + per-op FastRPC overhead).

### Per-op NPU profiling
Set `GGML_HEXAGON_PROFILE=1` to dump `htp_profiling.csv` (one row per DSP op:
`phase / op / names / dims / types / usec / cycles`). Aggregate by op × output-shape to
see where NPU time goes and which ops are HMX-bound vs memory/host-bound.

### Notes
- Activations are **f32** on purpose: Gemma's residual stream overflows f16's ±65504
  range (f16 activations → garbage). The matmul already downconverts the *post-norm*
  activation to f16 internally where it is safe.
- The Gemma FFN GeGLU (`gelu(gate)*up`) is fused into one HTP `glu_geglu` op (−8% prefix).
