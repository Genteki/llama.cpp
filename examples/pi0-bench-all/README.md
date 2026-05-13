# PI0 unified bench

Phase 1 deliverable for the Q3J1_TQ Snapdragon work: one bench binary that runs all 5 KV-cache modes and produces a comparison table. Stays on CPU; numbers feed the GPU port decisions.

## Modes

| `--kv-mode`        | KV type | Codebook | 1-bit QJL | SIMD | What it measures |
|---|---|:---:|:---:|:---:|---|
| `fp16`             | F32     | n/a | n/a | tiled FA | Uncompressed baseline (label retained for parity with pi0-q3j1; storage is actually F32) |
| `q4_0`             | Q4_0    | 4-bit linear | n/a | hand-tuned | Standard llama.cpp KV compression at 4.5 bpw (same byte budget as Q3J1_TQ) |
| `q3j1_trivial`     | Q3J1_TQ | ✓ | ✗ skipped | scalar | Codebook only (env `Q3J1_TQ_SKIP_QJL=1`); fastest Q3J1 path, lowest fidelity |
| `q3j1_qjl_scalar`  | Q3J1_TQ | ✓ | ✓ Sᵀ·sign | scalar | Full QJL with scalar dequant (env `Q3J1_TQ_DISABLE_SIMD=1`) — apples-to-apples baseline for SIMD attribution |
| `q3j1_simd`        | Q3J1_TQ | ✓ | ✓ Sᵀ·sign | AVX2 | Same workload as `q3j1_qjl_scalar`, with the AVX2 fused vec_dot |
| `q3j1_fa`          | Q3J1_TQ | ✓ | ✓ via Sq identity | scalar | True-QJL FA: precompute `Sq` per Q row, dot with stored sign bits (env `Q3J1_TQ_USE_FA=1`); V is codebook-only |
| `q3j1_predeq`      | Q3J1_TQ | ✓ | ✓ Sᵀ·sign | scalar (cached) | Pre-dequant K and V to fp32 once at the top of each thread's `_one_chunk`, then plain fp32 inner dot. Removes the ~408× per-Q-row dequant redundancy (env `Q3J1_TQ_PRE_DEQUANT_KV=1`) |

**Pairs that isolate one factor at a time:**

| Comparison | What it isolates |
|---|---|
| `q3j1_qjl_scalar` vs `q3j1_simd` | Pure SIMD speedup on identical workload |
| `q3j1_qjl_scalar` vs `q3j1_fa`   | Algorithmic speedup from the QJL identity (skips K reconstruction) |
| `q3j1_qjl_scalar` vs `q3j1_trivial` | Cost of reconstructing the 1-bit residual at all |
| `q3j1_simd` vs `q3j1_predeq` | Cost of redundant per-Q-row K/V dequant (only relevant in `_one_chunk` ref path) |
| `fp16` vs `q4_0` | Cost of falling off the FA tiled fast path (Q4_0 has industrial-grade SIMD vec_dot) |

## What's wired

| Piece | Where |
|---|---|
| `pi0-bench-all-common.h` | Generalized prefix/expert pipeline; takes `kv_type` runtime arg |
| `pi0-bench-all.cpp` | `--kv-mode` dispatch + per-phase timing + `BENCH_RESULT` summary line |
| AVX2 vec_dot | [ggml-quants.c](../../ggml/src/ggml-quants.c) (`ggml_vec_dot_q3j1_tq_f32_avx2_impl`); selected at compile time in [ggml-cpu/quants.c](../../ggml/src/ggml-cpu/quants.c) |
| QJL-FA path | [ops.cpp](../../ggml/src/ggml-cpu/ops.cpp) `_one_chunk` — branches on `q3j1_tq_should_use_fa()`, calls `ggml_q3j1_tq_precompute_Sq` + `ggml_vec_dot_q3j1_tq_qjl_fa` + `dequantize_row_q3j1_tq_codebook` |
| `pi0-q4/` | Standalone Q4_0 example (parallel to `pi0/`, `pi0-q3j1/`) |

## Build

```bash
cd /home/genteki/gentekis_document/research/pi0/llama.cpp
cmake -S . -B build
cmake --build build -j$(nproc) --target llama-pi0-bench-all llama-pi0-q4 llama-pi0-q4-bench
```

## Run

Single mode:
```bash
./build/bin/llama-pi0-bench-all \
  -m /home/genteki/gguf/pi0 \
  --image /home/genteki/gentekis_document/research/pi0/data/cam_left_wrist.jpg,/home/genteki/gentekis_document/research/pi0/data/cam_right_wrist.jpg,/home/genteki/gentekis_document/research/pi0/data/cam_high.jpg \
  -p "pick up the red block" \
  --kv-mode q3j1_fa -n 3 --warmup 1
```

All modes + summary table:
```bash
./examples/pi0-bench-all/run-bench-all.sh -n 3 --warmup 1
```

The wrapper writes per-mode logs to `examples/pi0-bench-all/bench-logs/` and prints a table comparing inference time and expert-step time vs the fp16 baseline.

## Measured shape (CPU, AVX-512 host with `-march=native`)

First full run, 3 iter / 1 warmup. Numbers shift with iteration count but the ordering is stable.

| Mode             | Inference ms | Expert step ×10 ms | Notes |
|---|---:|---:|---|
| fp16             |  6 627 |   976 | Tiled FA fast path (~50 ns/dot) |
| q4_0             |  6 933 | 1 175 | `_one_chunk` reference + hand-tuned vpdpbusd vec_dot |
| q3j1_trivial     | 13 636 | 7 787 | 256 codebook lookups per K row, scalar |
| q3j1_qjl_scalar  | (re-run pending) | — | Apples-to-apples baseline for SIMD attribution |
| q3j1_simd        | 62 191 | 56 398 | Same QJL workload as `q3j1_qjl_scalar`, AVX2-fused |
| q3j1_fa          | 24 815 | 19 003 | QJL identity: Sq once per Q row, sign-bit dot per K row |

Read it as: `q3j1_simd / q3j1_qjl_scalar` is the actual SIMD speedup on the QJL workload. `q3j1_fa / q3j1_qjl_scalar` is the algorithmic-redesign speedup. The two together tell the story; q3j1_trivial is just there as the lower bound (no QJL math).

If `q3j1_fa` lands near `fp16` once Sq+sign-dot is also SIMD-ized, the algorithmic redesign worked and the GPU port has a clear target.
