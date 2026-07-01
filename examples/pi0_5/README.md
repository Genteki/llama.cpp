# PI0.5 on llama.cpp

PI0.5 (π₀.₅) is the flow-matching VLA that succeeds PI0. It predicts continuous
robot actions from images + a text instruction via 10 denoising (Euler) steps.
This example mirrors [`examples/pi0`](../pi0) but follows the pi0.5 architecture
exactly, and defaults to **f16 activations + f16 KV cache** with per-tensor
weight quantization (q4 / q8 / f16).

## What changes vs PI0 (the action expert only)

The SigLIP vision tower and the PaliGemma 2B prefix are byte-for-byte the pi0
path. The **action expert** differs:

| | PI0 | PI0.5 |
|---|---|---|
| Suffix tokens | `state` + 50 action = **51** | **50** action only (state is folded into the discrete prefix text tokens) |
| Time injection | `concat(action, sincos(t))` → `action_time_mlp` (SiLU) | `action_in_proj(noisy)`; time path `sincos(t) → time_mlp_in → swish → time_mlp_out → swish = cond` |
| Expert norms | plain RMSNorm (`x·weight`) | **adaRMS**: each norm has `dense: Linear(1024→3072)` → `(scale,shift,gate)`; `normed = rms(x)·(1+scale)+shift`, residual `x + y·gate` |
| `max_token_len` | 48 | 200 (longer prefix) |
| Removed weights | — | `state_proj`, `action_time_mlp_*` |
| Added weights | — | per-layer `attn_norm.dense.*`, `ffn_norm.dense.*`, `output_norm.dense.*`, `time_mlp_in/out.*` |

The suffix attention is fully visible (all 50 action tokens see each other and
the full prefix), so the expert runs with **no attention mask** (unlike pi0's
state-token mask).

```
Images (3 views)        Text (+ discrete state)            Noisy actions + t
  SigLIP So400m          Gemma tokenizer            action_in_proj | time_mlp→cond
        └──────── PREFIX (image + text) ────────┘     └──── SUFFIX (50 actions) ────┘
            PaliGemma 2B (18L, d=2048)               Action Expert 300M adaRMS (18L, d=1024)
                 run ONCE, writes KV ──cross-attn──▶ run 10× (denoise), cond = time MLP
                                                       suffix_out[:, -50:] → action_out_proj → v_t
```

## Step 1 — Convert to GGUF

Source checkpoint: **[lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)**
(single `model.safetensors`, lerobot/`paligemma_with_expert.*` naming — the same
HF format as the pi0 checkpoint, plus the adaRMS `dense` + `time_mlp` tensors).

```bash
# from the llama.cpp directory; reuses the pi0 converter with --pi05
../openpi/.venv/bin/python tools/mtmd/convert_pi0_to_gguf.py \
    --model     /home/genteki/hf-models/pi05 \
    --tokenizer /home/genteki/.cache/openpi/big_vision/paligemma_tokenizer.model \
    --output-dir /home/genteki/gguf/pi05/f16 \
    --type f16 --pi05
```

Produces `pi05-mmproj.gguf`, `pi05-gemma-2b.gguf`, `pi05-action-expert.gguf`.
The adaRMS `dense` matrices are stored **without** the Gemma `+1` offset (the
`(1+scale)` is applied at runtime, since the scale is dynamic).

Per-tensor weight quant (the W axis) uses the standard tooling, e.g.:
```bash
./build/bin/llama-quantize /home/genteki/gguf/pi05/f16/pi05-action-expert.gguf \
    /home/genteki/gguf/pi05/q4/pi05-action-expert.gguf Q4_0
```

## Step 2 — Build

```bash
cmake -B build
cmake --build build --target llama-pi0_5       -j$(nproc)
cmake --build build --target llama-pi0_5-bench -j$(nproc)
# OpenCL per-kernel profiling (for --mode profile → cl_profiling.csv):
# cmake -B build-cl -DGGML_OPENCL=ON -DGGML_OPENCL_PROFILING=ON
```

## Step 3 — Run

```bash
./build/bin/llama-pi0_5 -m /home/genteki/gguf/pi05/f16 \
    --image ../data/cam_high.jpg,../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg \
    -p "pick up the red block" \
    --backend cpu --kv-type f32
```

Output is a `50 × 32` action matrix. (Image order is base/high, left-wrist,
right-wrist — the openpi observation order.)

**Flags**
- `--backend` — `cpu` (default), `auto`/`gpu`, or a name (`OpenCL`, `Vulkan`, …)
- `--kv-type` — KV cache dtype: `f16` (default), `f32`, `q8_0`, `q4_0`
- `--act-type` — activation dtype into matmuls: `auto` (default), `f16`, `f32`

> `auto` resolves to **f16 on GPU** backends (the deploy target / default
> precision) and **f32 on CPU**, because CPU `mul_mat` cannot take f16
> activations for quantized weights. On OpenCL the Adreno kernels take f16
> activations even for q4_0 — that is the intended fast path.

## Step 4 — Benchmark & profile

One tool, four modes:

```bash
# 1) per-phase wall-clock + analytical GFLOPS
./build/bin/llama-pi0_5-bench -m /home/genteki/gguf/pi05/f16 \
    --image ../data/cam_high.jpg,../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg \
    -p "pick up the red block" \
    --mode phase -n 20 --warmup 3 --backend OpenCL --kv-type f16 --act-type f16

# 2) targeted operators, baseline vs optimized (needs only --backend):
#    prefix QK^T (batched 8× vs MQA-collapse) and expert qkv_proj/ffn (f16 vs q8_0 vs q4_0)
./build/bin/llama-pi0_5-bench --mode op -n 50 --warmup 5 --backend OpenCL --act-type f16

# 3) quant comparison — sweeps activation precision (A16 vs A32) at the loaded
#    weights; run once per -m dir (…/q4 / …/q8 / …/f16) for the W axis (W4A16, W8A16, …)
./build/bin/llama-pi0_5-bench -m /home/genteki/gguf/pi05/q4 \
    --image ../data/cam_high.jpg,../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg \
    -p "pick up the red block" --mode quant -n 10 --backend OpenCL

# 4) per-kernel profiling — one clean inference; OpenCL build with
#    -DGGML_OPENCL_PROFILING=ON writes cl_profiling.csv (per-kernel ms + GFLOPS)
./build/bin/llama-pi0_5-bench -m /home/genteki/gguf/pi05/f16 \
    --image ../data/cam_high.jpg,../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg \
    -p "pick up the red block" --mode profile --backend OpenCL
```

`--mode op` is self-contained (synthetic weights at the real pi0.5 shapes), so it
isolates a kernel without loading the full model. The optimized variants reuse the
kernels already in `ggml/src/ggml-opencl` (MQA-collapse fold, q4_0/q8_0 GEMM, the
f16 image-B path). Toggle attention variants in the full-inference modes with
`PI0_NO_FLASH_ATTN=1`, `PI0_ATTN_COLLAPSE=1`, `PI0_FIX_Q_LAYOUT=0`.

## Parity with openpi

**Verified** against openpi's `PI0Pytorch` (which loads the same lerobot/pi05_base
safetensors). Isolating the LM+expert by feeding both stacks the *same* prefix
embeddings + initial noise (so vision/state/tokenization preprocessing is factored
out):

| config | per-step v_t cos | final action cos |
|---|---|---|
| no-flash, f32 weights | **1.000000** (all 10 steps) | **1.000000** |
| no-flash, f16 weights | **1.000000** | **1.000000** |
| default flash, f16 weights + f16 KV | 1.0000 | **0.999994** |

> Implementation note: the action expert runs one **persistent** graph across all
> 10 denoise steps. The long-lived adaRMS `cond` and the suffix-token stack input
> are pinned via `ggml_set_output` so `ggml_gallocr` does not reuse their buffers —
> without that, steps 2..10 silently corrupt (step 1 stays clean), which looks like
> a flow-matching drift but is an allocator-reuse bug. See `build_expert_graph`.

To reproduce the exact (cos 1.0) check on CPU with f32:

```bash
# feed an identical initial-noise tensor (row-major [50][32] float32) and dump v_t
PI05_NOISE_BIN=noise.bin PI05_DUMP_VT=vt.bin \
  ./build/bin/llama-pi0_5 -m /home/genteki/gguf/pi05/f16 \
    --image ../data/cam_high.jpg,../data/cam_left_wrist.jpg,../data/cam_right_wrist.jpg \
    -p "..." --backend cpu --kv-type f32 --act-type f32
```
(For the strictest comparison, also convert the weights with `--type f32` into
`/home/genteki/gguf/pi05/f32` and point `-m` there.)
`PI05_DUMP_PREFIX` / `PI05_DUMP_PREFIX_OUT` dump the assembled prefix embeddings
and the PaliGemma final hidden state to isolate vision/text vs LM vs flow-matching
stages (same staging method as the pi0 `compare/` harness).
