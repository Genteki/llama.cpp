#!/usr/bin/env python3
"""
Convert PI0 HF model to 3 GGUF files for llama.cpp inference.

Outputs:
  1. pi0-mmproj.gguf       — SigLIP So400m vision encoder + linear projector
  2. pi0-gemma-2b.gguf     — PaliGemma 2B language model (standard Gemma format)
  3. pi0-action-expert.gguf — Action Expert 300M + action/state projections

Usage:
    python convert_pi0_to_gguf.py --model /path/to/pi0-hf --output-dir /path/to/output
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import sentencepiece

sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


def load_tensor(model_dir: Path, tensor_name: str) -> np.ndarray:
    """Load a single tensor from single-shard safetensors file."""
    shard_path = model_dir / "model.safetensors"
    with open(shard_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

        info = header[tensor_name]
        shape = info["shape"]
        offsets = info["data_offsets"]

        data_start = 8 + header_size + offsets[0]
        data_end = 8 + header_size + offsets[1]
        f.seek(data_start)
        raw = f.read(data_end - data_start)

        if info["dtype"] == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            arr32 = np.zeros(arr.shape, dtype=np.uint32)
            arr32[:] = arr.astype(np.uint32) << 16
            return arr32.view(np.float32)
        elif info["dtype"] == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        else:
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


# Tensors that are used in ggml_mul_mat and can be stored as f16
MUL_MAT_NAMES = {
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_out.weight",
    "ffn_up.weight", "ffn_down.weight", "mm.0.weight",
    # Gemma LLM names
    "attn_output.weight", "ffn_gate.weight",
    # Action expert custom
    "action_in_proj.weight", "action_out_proj.weight", "state_proj.weight",
    "action_time_mlp_in.weight", "action_time_mlp_out.weight",
}


def add_tensor(writer, name: str, data: np.ndarray, use_f16: bool):
    can_f16 = any(name.endswith(s) for s in MUL_MAT_NAMES)
    if use_f16 and can_f16:
        data = data.astype(np.float16)
    else:
        data = data.astype(np.float32)
    writer.add_tensor(name, data)
    print(f"  {name}: {list(data.shape)} ({data.dtype})")


def add_tokenizer(writer, tokenizer_path: Path):
    """Embed SentencePiece tokenizer into GGUF."""
    print(f"\n--- Tokenizer from {tokenizer_path} ---")
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(tokenizer_path))
    vocab_size = sp.GetPieceSize()

    tokens = []
    scores = []
    toktypes = []

    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        toktype = 1  # normal

        if sp.IsUnknown(i):
            toktype = 2  # unknown
        elif sp.IsControl(i):
            toktype = 3  # control
        elif sp.IsUnused(i):
            toktype = 5  # unused
        elif sp.IsByte(i):
            toktype = 6  # byte

        tokens.append(piece.encode("utf-8"))
        scores.append(score)
        toktypes.append(toktype)

    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)
    writer.add_uint32("tokenizer.ggml.bos_token_id", sp.bos_id())
    writer.add_uint32("tokenizer.ggml.eos_token_id", sp.eos_id())
    writer.add_uint32("tokenizer.ggml.padding_token_id", sp.pad_id())
    writer.add_uint32("tokenizer.ggml.unknown_token_id", sp.unk_id())
    print(f"  Vocab size: {vocab_size}, BOS: {sp.bos_id()}, EOS: {sp.eos_id()}")


def convert_mmproj(model_dir: Path, output_path: Path, use_f16: bool):
    """Convert SigLIP So400m vision encoder + linear projector."""
    print("\n=== Converting mmproj (SigLIP + projector) ===")

    writer = gguf.GGUFWriter(str(output_path), arch="clip")
    writer.add_type(gguf.GGUFType.MMPROJ)
    writer.add_string("general.name", "pi0")

    # SigLIP config
    n_embd = 1152
    n_head = 16
    n_layer = 27
    n_ff = 4304
    image_size = 224
    patch_size = 14

    writer.add_clip_projector_type("pi0")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_vision_image_size(image_size)
    writer.add_vision_patch_size(patch_size)
    writer.add_vision_embedding_length(n_embd)
    writer.add_vision_feed_forward_length(n_ff)
    writer.add_vision_block_count(n_layer)
    writer.add_vision_head_count(n_head)
    writer.add_vision_attention_layernorm_eps(1e-6)
    writer.add_vision_image_mean([0.5, 0.5, 0.5])
    writer.add_vision_image_std([0.5, 0.5, 0.5])
    writer.add_vision_use_gelu(True)
    writer.add_uint32("clip.vision.projection_dim", 0)

    vt_prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"

    # Patch embedding
    data = load_tensor(model_dir, f"{vt_prefix}.embeddings.patch_embedding.weight")
    add_tensor(writer, "v.patch_embd.weight", data, use_f16)
    data = load_tensor(model_dir, f"{vt_prefix}.embeddings.patch_embedding.bias")
    add_tensor(writer, "v.patch_embd.bias", data, use_f16)

    # Position embedding
    data = load_tensor(model_dir, f"{vt_prefix}.embeddings.position_embedding.weight")
    add_tensor(writer, "v.position_embd.weight", data, use_f16)

    # Post layernorm
    data = load_tensor(model_dir, f"{vt_prefix}.post_layernorm.weight")
    add_tensor(writer, "v.post_ln.weight", data, use_f16)
    data = load_tensor(model_dir, f"{vt_prefix}.post_layernorm.bias")
    add_tensor(writer, "v.post_ln.bias", data, use_f16)

    # Transformer blocks
    for il in range(n_layer):
        prefix = f"{vt_prefix}.encoder.layers.{il}"
        blk = f"v.blk.{il}"

        # LayerNorm 1
        data = load_tensor(model_dir, f"{prefix}.layer_norm1.weight")
        add_tensor(writer, f"{blk}.ln1.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.layer_norm1.bias")
        add_tensor(writer, f"{blk}.ln1.bias", data, use_f16)

        # Attention Q, K, V (separate in PI0's SigLIP)
        data = load_tensor(model_dir, f"{prefix}.self_attn.q_proj.weight")
        add_tensor(writer, f"{blk}.attn_q.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.self_attn.q_proj.bias")
        add_tensor(writer, f"{blk}.attn_q.bias", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.k_proj.weight")
        add_tensor(writer, f"{blk}.attn_k.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.self_attn.k_proj.bias")
        add_tensor(writer, f"{blk}.attn_k.bias", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.v_proj.weight")
        add_tensor(writer, f"{blk}.attn_v.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.self_attn.v_proj.bias")
        add_tensor(writer, f"{blk}.attn_v.bias", data, use_f16)

        # Attention output
        data = load_tensor(model_dir, f"{prefix}.self_attn.out_proj.weight")
        add_tensor(writer, f"{blk}.attn_out.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.self_attn.out_proj.bias")
        add_tensor(writer, f"{blk}.attn_out.bias", data, use_f16)

        # LayerNorm 2
        data = load_tensor(model_dir, f"{prefix}.layer_norm2.weight")
        add_tensor(writer, f"{blk}.ln2.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.layer_norm2.bias")
        add_tensor(writer, f"{blk}.ln2.bias", data, use_f16)

        # MLP
        data = load_tensor(model_dir, f"{prefix}.mlp.fc1.weight")
        add_tensor(writer, f"{blk}.ffn_up.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.mlp.fc1.bias")
        add_tensor(writer, f"{blk}.ffn_up.bias", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.mlp.fc2.weight")
        add_tensor(writer, f"{blk}.ffn_down.weight", data, use_f16)
        data = load_tensor(model_dir, f"{prefix}.mlp.fc2.bias")
        add_tensor(writer, f"{blk}.ffn_down.bias", data, use_f16)

    # Projector: simple Linear(1152 -> 2048)
    print("\n--- Projector ---")
    mm_prefix = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    data = load_tensor(model_dir, f"{mm_prefix}.weight")
    add_tensor(writer, "mm.0.weight", data, use_f16)
    data = load_tensor(model_dir, f"{mm_prefix}.bias")
    add_tensor(writer, "mm.0.bias", data, use_f16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Written: {output_path} ({os.path.getsize(output_path) / 1024**2:.1f} MB)")


def convert_gemma_2b(model_dir: Path, output_path: Path, use_f16: bool, tokenizer_path: Path = None):
    """Convert PaliGemma 2B language model to standard Gemma GGUF."""
    print("\n=== Converting PaliGemma 2B (Gemma format) ===")

    writer = gguf.GGUFWriter(str(output_path), arch="gemma")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_string("general.name", "pi0-paligemma-2b")

    # Gemma 2B config
    n_embd = 2048
    n_head = 8
    n_kv_head = 1
    head_dim = 256
    n_layer = 18
    n_ff = 16384
    vocab_size = 257152

    writer.add_context_length(2048)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(n_ff)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_vocab_size(vocab_size)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if use_f16 else gguf.LlamaFileType.ALL_F32)

    # Tokenizer - embed SentencePiece model
    add_tokenizer(writer, tokenizer_path)

    lm_prefix = "paligemma_with_expert.paligemma.model.language_model.layers"

    # Embedding: use lm_head as tied embedding (Gemma convention)
    data = load_tensor(model_dir, "paligemma_with_expert.paligemma.lm_head.weight")
    add_tensor(writer, "token_embd.weight", data, use_f16)

    # Transformer blocks
    for il in range(n_layer):
        prefix = f"{lm_prefix}.{il}"
        blk = f"blk.{il}"

        # RMS norms (Gemma convention: +1 offset)
        data = load_tensor(model_dir, f"{prefix}.input_layernorm.weight")
        add_tensor(writer, f"{blk}.attn_norm.weight", data + 1.0, False)

        data = load_tensor(model_dir, f"{prefix}.post_attention_layernorm.weight")
        add_tensor(writer, f"{blk}.ffn_norm.weight", data + 1.0, False)

        # Attention
        data = load_tensor(model_dir, f"{prefix}.self_attn.q_proj.weight")
        add_tensor(writer, f"{blk}.attn_q.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.k_proj.weight")
        add_tensor(writer, f"{blk}.attn_k.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.v_proj.weight")
        add_tensor(writer, f"{blk}.attn_v.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.o_proj.weight")
        add_tensor(writer, f"{blk}.attn_output.weight", data, use_f16)

        # FFN
        data = load_tensor(model_dir, f"{prefix}.mlp.gate_proj.weight")
        add_tensor(writer, f"{blk}.ffn_gate.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.mlp.up_proj.weight")
        add_tensor(writer, f"{blk}.ffn_up.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.mlp.down_proj.weight")
        add_tensor(writer, f"{blk}.ffn_down.weight", data, use_f16)

    # Final norm
    data = load_tensor(model_dir, "paligemma_with_expert.paligemma.model.language_model.norm.weight")
    add_tensor(writer, "output_norm.weight", data + 1.0, False)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Written: {output_path} ({os.path.getsize(output_path) / 1024**2:.1f} MB)")


def convert_action_expert(model_dir: Path, output_path: Path, use_f16: bool):
    """Convert Action Expert 300M + action projections to custom GGUF."""
    print("\n=== Converting Action Expert 300M ===")

    writer = gguf.GGUFWriter(str(output_path), arch="pi0-action-expert")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_string("general.name", "pi0-action-expert")

    # Gemma 300M config
    n_embd = 1024
    n_head = 8
    n_kv_head = 1
    head_dim = 256
    n_layer = 18
    n_ff = 4096
    action_dim = 32
    action_horizon = 50

    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(n_ff)
    writer.add_block_count(n_layer)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_kv_head)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_uint32("pi0.action_dim", action_dim)
    writer.add_uint32("pi0.action_horizon", action_horizon)

    expert_prefix = "paligemma_with_expert.gemma_expert.model.layers"

    # Transformer blocks
    for il in range(n_layer):
        prefix = f"{expert_prefix}.{il}"
        blk = f"blk.{il}"

        # RMS norms (+1 for Gemma convention)
        data = load_tensor(model_dir, f"{prefix}.input_layernorm.weight")
        add_tensor(writer, f"{blk}.attn_norm.weight", data + 1.0, False)

        data = load_tensor(model_dir, f"{prefix}.post_attention_layernorm.weight")
        add_tensor(writer, f"{blk}.ffn_norm.weight", data + 1.0, False)

        # Attention
        data = load_tensor(model_dir, f"{prefix}.self_attn.q_proj.weight")
        add_tensor(writer, f"{blk}.attn_q.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.k_proj.weight")
        add_tensor(writer, f"{blk}.attn_k.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.v_proj.weight")
        add_tensor(writer, f"{blk}.attn_v.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.self_attn.o_proj.weight")
        add_tensor(writer, f"{blk}.attn_output.weight", data, use_f16)

        # FFN
        data = load_tensor(model_dir, f"{prefix}.mlp.gate_proj.weight")
        add_tensor(writer, f"{blk}.ffn_gate.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.mlp.up_proj.weight")
        add_tensor(writer, f"{blk}.ffn_up.weight", data, use_f16)

        data = load_tensor(model_dir, f"{prefix}.mlp.down_proj.weight")
        add_tensor(writer, f"{blk}.ffn_down.weight", data, use_f16)

    # Final norm
    data = load_tensor(model_dir, "paligemma_with_expert.gemma_expert.model.norm.weight")
    add_tensor(writer, "output_norm.weight", data + 1.0, False)

    # Action projections
    print("\n--- Action Projections ---")
    for name in ["action_in_proj", "action_out_proj", "state_proj",
                  "action_time_mlp_in", "action_time_mlp_out"]:
        data = load_tensor(model_dir, f"{name}.weight")
        add_tensor(writer, f"{name}.weight", data, use_f16)
        data = load_tensor(model_dir, f"{name}.bias")
        add_tensor(writer, f"{name}.bias", data, use_f16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Written: {output_path} ({os.path.getsize(output_path) / 1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert PI0 HF model to GGUF")
    parser.add_argument("--model", type=str, required=True, help="Path to PI0 HF model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for GGUF files")
    parser.add_argument("--type", type=str, default="f16", choices=["f16", "f32"], help="Output data type")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to paligemma_tokenizer.model (SentencePiece)")
    args = parser.parse_args()

    model_dir = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer)

    use_f16 = args.type == "f16"

    convert_mmproj(model_dir, output_dir / "pi0-mmproj.gguf", use_f16)
    convert_gemma_2b(model_dir, output_dir / "pi0-gemma-2b.gguf", use_f16, tokenizer_path)
    convert_action_expert(model_dir, output_dir / "pi0-action-expert.gguf", use_f16)

    print("\n=== Done! ===")
    for name in ["pi0-mmproj.gguf", "pi0-gemma-2b.gguf", "pi0-action-expert.gguf"]:
        path = output_dir / name
        print(f"  {path}: {os.path.getsize(path) / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
