#!/usr/bin/env python3
"""Quantize a pi0.5 action-expert GGUF to q4_0 / q8_0.

`llama-quantize` cannot process the action expert because it uses the custom
`pi0-action-expert` architecture (unknown to llama.cpp's model loader). This
standalone tool re-quantizes only the 2-D matmul weights (attn q/k/v/o, ffn
gate/up/down, the adaRMS `*.dense` matrices, time_mlp, action_in/out_proj) whose
in-features are a multiple of the 32-element block size; biases and anything else
are copied through unchanged. All GGUF metadata is preserved.

The PaliGemma 2B prefix uses the standard `gemma` arch, so quantize that one with
the normal tool:  llama-quantize pi05-gemma-2b.gguf out.gguf q4_0

Usage:
    python quantize_expert.py <in-f16-expert.gguf> <out.gguf> {q4_0|q8_0}
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf
from gguf.quants import quantize

QTYPES = {"q4_0": gguf.GGMLQuantizationType.Q4_0, "q8_0": gguf.GGMLQuantizationType.Q8_0}


def main():
    ap = argparse.ArgumentParser(description="Quantize a pi0.5 action-expert GGUF")
    ap.add_argument("src", help="input f16/f32 pi05-action-expert.gguf")
    ap.add_argument("dst", help="output quantized gguf")
    ap.add_argument("qtype", choices=list(QTYPES), help="target quant type")
    ap.add_argument("--fuse-qkv", action="store_true",
                    help="concat attn_q/k/v.weight -> attn_qkv.weight (out order q,k,v) so the "
                         "graph runs one wide GEMM instead of q(Ab_Bi)+k/v(slow GEMV). The "
                         "pi0_5 loader auto-uses attn_qkv when present (pi0_5-common.cpp).")
    args = ap.parse_args()
    qt = QTYPES[args.qtype]

    r = gguf.GGUFReader(args.src)
    w = gguf.GGUFWriter(args.dst, arch="pi0-action-expert")

    # Copy all scalar/array KV metadata (the writer re-adds the structural keys).
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for field in r.fields.values():
        if field.name in skip:
            continue
        try:
            w.add_key_value(field.name, field.contents(), field.types[0])
        except Exception:
            pass

    def quantizable(t):
        # 2-D weight whose ne0 (in-features) is a multiple of the block size (32).
        return t.name.endswith(".weight") and len(t.shape) == 2 and int(t.shape[0]) % 32 == 0

    # --fuse-qkv: fold each layer's attn_q/k/v.weight into one attn_qkv.weight.
    # np.array(t.data) is [out, in]; concat along out (axis 0) in q,k,v order to
    # match the graph's view-split [q_dim | kv_dim | kv_dim].
    by_name = {t.name: t for t in r.tensors}
    skip_qkv = set()
    if args.fuse_qkv:
        for name in by_name:
            if name.endswith(".attn_q.weight"):
                base = name[: -len(".attn_q.weight")]
                skip_qkv.update(f"{base}.attn_{x}.weight" for x in ("q", "k", "v"))

    def emit(name, arr):
        w.add_tensor(name, quantize(arr.astype(np.float32), qt), raw_dtype=qt)

    nq = nc = nf = 0
    for t in r.tensors:
        data = np.array(t.data)
        if t.name in skip_qkv:
            if t.name.endswith(".attn_q.weight"):  # emit the fused tensor once, at q's slot
                base = t.name[: -len(".attn_q.weight")]
                qkv = np.concatenate([np.array(by_name[f"{base}.attn_{x}.weight"].data)
                                      for x in ("q", "k", "v")], axis=0)
                emit(f"{base}.attn_qkv.weight", qkv)
                nf += 1
            continue
        if quantizable(t):
            # quantize() returns uint8 [rows, row_bytes]; the writer derives the
            # logical shape from the byte shape + raw_dtype (no raw_shape needed).
            emit(t.name, data)
            nq += 1
        else:
            w.add_tensor(t.name, data)
            nc += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.qtype}: quantized {nq} tensors, fused-qkv {nf}, copied {nc} -> {args.dst}")


if __name__ == "__main__":
    main()
