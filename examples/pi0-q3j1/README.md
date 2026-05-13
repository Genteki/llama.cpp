# PI0 Q3J1_TQ

Real Q3J1_TQ KV-cache in HBM, dispatched through `ggml_flash_attn_ext`'s `type_traits` path — the same mechanism Q4_0 / Q8_0 use as KV-cache types in standard llama.cpp.

## Implementation

Mirrors `examples/pi0/{pi0.cpp, pi0-bench.cpp, pi0-common.h}` with three pieces swapped:

1. **Prefix-pass output**: F32 K/V is converted to `GGML_TYPE_Q3J1_TQ` via a `ggml_cast` node at the end of the prefix graph; only Q3J1_TQ bytes (3.83 MiB total) are downloaded to host.
2. **Expert-step prefix tensors**: allocated as `GGML_TYPE_Q3J1_TQ` inputs and uploaded as Q3J1_TQ bytes — so prefix K/V actually lives in the compute buffer (HBM) at 7.1× smaller size.
3. **Concat replacement**: `ggml_concat` is broken for quantized types (its `concat_any` path iterates per-element with `type_size`, which is the block size for quant), so `build_gemma_layer` allocates a full `[kv_dim, total_kv]` Q3J1_TQ buffer and writes prefix + current K/V into it via `ggml_cpy`-into-views (same pattern standard llama.cpp uses for KV cache writes).

`ggml_flash_attn_ext` then consumes the Q3J1_TQ K/V directly: dispatches via `type_traits_cpu->vec_dot` for K (Q3J1_TQ has `vec_dot_type=F32`) and `type_traits->to_float` for V.

`examples/pi0/` is unchanged.

## Build guide

```bash
cd /home/genteki/gentekis_document/research/pi0/llama.cpp

# Re-run cmake to pick up the pi0-q3j1 subdirectory, then build
cmake -S . -B build
cmake --build build -j$(nproc) --target \
  llama-pi0 llama-pi0-bench \
  llama-pi0-q3j1 llama-pi0-q3j1-bench

# Baseline (fp16 K/V)
./build/bin/llama-pi0-bench \
  -m /home/genteki/gguf/pi0 \
  --image /home/genteki/gentekis_document/research/pi0/data/cam_left_wrist.jpg,/home/genteki/gentekis_document/research/pi0/data/cam_right_wrist.jpg,/home/genteki/gentekis_document/research/pi0/data/cam_high.jpg \
  -p "pick up the red block" -n 5 --warmup 2

# Q3J1_TQ K/V in HBM
# NOTE: each inference is ~86 s on CPU (see "Performance" below); use -n 3 --warmup 1
./build/bin/llama-pi0-q3j1-bench \
  -m /home/genteki/gguf/pi0 \
  --image /home/genteki/gentekis_document/research/pi0/data/cam_left_wrist.jpg,/home/genteki/gentekis_document/research/pi0/data/cam_right_wrist.jpg,/home/genteki/gentekis_document/research/pi0/data/cam_high.jpg \
  -p "pick up the red block" -n 3 --warmup 1
```

## Numbers

CPU-only build, AVX2 host. Baseline = 5 iter, 2 warmup. Q3J1_TQ = 3 iter, 1 warmup (each inference ~14-86 s).

The Q3J1_TQ "codebook-only" column drops the QJL Sᵀ·sign reconstruction from `to_float` (set `Q3J1_TQ_SKIP_QJL=1` env var). It loses the 1-bit residual correction during dequant — same K/V bytes in HBM, but attention only sees the 3-bit codebook value. Used here to isolate the QJL cost from everything else.

| Phase             | FP16 baseline | Q3J1_TQ + QJL | Q3J1_TQ codebook-only |
|---|---:|---:|---:|
| Vision Encode     |       947 ms  |          914 ms |          904 ms |
| Prefix Pass       |      4243 ms  |         4219 ms |         4181 ms |
| Suffix Prep ×10   |       606 ms  |          608 ms |          604 ms |
| **Expert Step ×10** |   **947 ms** |   **80 012 ms** |    **8 634 ms** |
| Diffusion Total   |      1553 ms  |        80 620 ms |         9 238 ms |
| Inference Total   |      6743 ms  |        85 753 ms |        14 323 ms |
| Slowdown vs fp16  |          1.0× |       **12.7×** |         **2.1×** |
| Expert step slowdown vs fp16 |  1.0× |       **84.5×** |          **9.1×** |

| Resource             | FP16  | Q3J1_TQ | Ratio |
|---|---:|---:|---:|
| Prefix K+V in HBM    | 28.5 MiB | **4.0 MiB** | **7.1× smaller** |
| Bytes uploaded per expert step | 28.5 MiB | 4.0 MiB | 7.1× |

## Ablation: where the 85× expert-step slowdown comes from

Comparing Q3J1_TQ+QJL (80 012 ms) vs codebook-only (8 634 ms) vs FP16 (947 ms) gives a clean decomposition:

| Source | Multiplier | What it is |
|---|---:|---|
| **QJL Sᵀ·sign in `to_float`** | **9.3×** | 80 012 / 8 634 — the 32×32 random projection per K block, every dot product |
| **Everything else** | **9.1×** | 8 634 / 947 — flash-attn fast path bypass + scalar (no-SIMD) `vec_dot_q3j1_tq_f32` |
| Compound (multiplicative) | 84.6× | 9.3 × 9.1 ✓ |

Roughly **half-and-half** — both factors contribute equally to the slowdown. The original hypothesis ("flash-attn fallback + no SIMD") accounts for ~9×, and the algorithmic cost of doing real QJL math during dequant accounts for the other ~9×. They multiply, not add.

## Performance analysis (informed by the ablation above)

The slowdown is honest, not a bug. Two compounding factors:

1. **Flash-attn fast path bypassed.** `ggml_compute_forward_flash_attn_ext_f16` has hardcoded gates: `use_split_kv_path` and `use_tiled` both require K/V to be F32 or F16. Q3J1_TQ falls through to the per-row reference path (`_f16_one_chunk`) — no SIMD tiling, vec_dot called once per (Q-row, K-row) pair. Combined with the scalar reference `vec_dot_q3j1_tq_f32` (no SIMD), this contributes the **9.1× factor** observed in the codebook-only ablation.

2. **The reference `dequantize_row_q3j1_tq` does full QJL Sᵀ·sign reconstruction per block.** 1024 mul-adds for the 32×32 projection + 32 codebook lookups. This contributes an additional **9.3× factor** on top of (1).

Q4_0/Q8_0 avoid both of these: hand-tuned SIMD vec_dots **and** dirt-cheap `to_float` (4-bit codebook lookup, no projection). Q3J1_TQ + QJL does the full random projection every block.

### Toggling the ablation

```bash
# Full Q3J1_TQ + QJL (default)
./build/bin/llama-pi0-q3j1-bench -m /home/genteki/gguf/pi0 ...

# Codebook-only (drops the 1-bit residual correction in attention)
Q3J1_TQ_SKIP_QJL=1 ./build/bin/llama-pi0-q3j1-bench -m /home/genteki/gguf/pi0 ...
```

The toggle lives in `dequantize_row_q3j1_tq` (`ggml/src/ggml-quants.c`) — checked once per call via cached env-var read. K/V bytes in HBM are unchanged either way; only the dequant math changes.

## Optimization paths (ranked by payoff)

| Optimization | Expected speedup | Effort |
|---|---|---|
| **SIMD `vec_dot_q3j1_tq`** (AVX2/NEON 32-wide accum, fuse codebook + Sᵀ·sign per block) | 8-16× | medium, no algorithm change |
| **Drop QJL Sᵀ·sign from `to_float`** (return only codebook value — gives up the 1-bit residual correction) | ~30× (matches Q4_0-class cost) | trivial; quality regression to measure |
| **True QJL-aware flash-attn** — `q·k ≈ √(π/2)·‖k‖·⟨Sq, sign(Sk)⟩/m` directly, no K reconstruction | ~50× + fastest | hard; new kernel, custom flash-attn path |
| **Q3J1_TQ-aware fast path in `ggml_flash_attn_ext`** (extend the F32/F16 allowlist to dispatch a Q3J1 tile kernel) | additional 2-4× | hard; ggml-internal |

## Files in this directory

- [pi0-q3j1-common.h](pi0-q3j1-common.h) — model loading, prefix-pass with F32→Q3J1_TQ cast, expert-step with Q3J1_TQ inputs, `build_gemma_layer` with cpy-into-views concat
- [pi0-q3j1.cpp](pi0-q3j1.cpp) — single-shot inference (mirror of pi0/pi0.cpp)
- [pi0-q3j1-bench.cpp](pi0-q3j1-bench.cpp) — phase-timing benchmark (mirror of pi0/pi0-bench.cpp)
