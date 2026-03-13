#!/usr/bin/env python3
"""
Convert OpenVLA-7B vision encoder + projector to GGUF mmproj format.

OpenVLA uses a dual vision backbone:
  - DINO v2 ViT-Large (featurizer): 24 blocks, 1024 dim, 16 heads
  - SigLIP ViT-so400m (fused_featurizer): 27 blocks, 1152 dim, 16 heads
  - 3-layer MLP projector: fc1(2176->8704) -> GELU -> fc2(8704->4096) -> GELU -> fc3(4096->4096)

Usage:
    python convert_openvla_to_gguf.py --model /path/to/openvla-7b --output openvla-mmproj.gguf
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Add gguf-py to path
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


def load_safetensors_metadata(model_dir: Path) -> dict:
    """Load tensor metadata (shapes, dtypes) from safetensors index."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    # Single shard
    return {}


def load_tensor_from_safetensors(model_dir: Path, tensor_name: str, weight_map: dict) -> np.ndarray:
    """Load a single tensor from safetensors files."""
    if weight_map:
        shard_file = weight_map[tensor_name]
    else:
        # Single shard
        shard_file = "model.safetensors"

    shard_path = model_dir / shard_file
    with open(shard_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

        info = header[tensor_name]
        dtype_str = info["dtype"]
        shape = info["shape"]
        offsets = info["data_offsets"]

        # Map dtype
        dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": "bfloat16",
        }
        np_dtype = dtype_map.get(dtype_str, dtype_str)

        data_start = 8 + header_size + offsets[0]
        data_end = 8 + header_size + offsets[1]
        f.seek(data_start)
        raw = f.read(data_end - data_start)

        if np_dtype == "bfloat16":
            # Convert bfloat16 to float32
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            # bfloat16 -> float32: shift left by 16 bits
            arr32 = np.zeros(arr.shape, dtype=np.uint32)
            arr32[:] = arr.astype(np.uint32) << 16
            return arr32.view(np.float32)
        else:
            return np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()


def split_qkv(qkv_weight: np.ndarray, n_head: int) -> tuple:
    """Split fused QKV weight [3*dim, dim] into separate Q, K, V."""
    dim = qkv_weight.shape[1]
    head_dim = dim // n_head
    # QKV is simply concatenated: [Q; K; V]
    q = qkv_weight[:dim, :]
    k = qkv_weight[dim:2*dim, :]
    v = qkv_weight[2*dim:, :]
    return q, k, v


def split_qkv_bias(qkv_bias: np.ndarray, dim: int) -> tuple:
    """Split fused QKV bias [3*dim] into separate Q, K, V."""
    q = qkv_bias[:dim]
    k = qkv_bias[dim:2*dim]
    v = qkv_bias[2*dim:]
    return q, k, v


def main():
    parser = argparse.ArgumentParser(description="Convert OpenVLA vision encoder to GGUF")
    parser.add_argument("--model", type=str, required=True, help="Path to OpenVLA HF model directory")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF file path")
    parser.add_argument("--type", type=str, default="f16", choices=["f16", "f32"], help="Output data type")
    args = parser.parse_args()

    model_dir = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_dtype = gguf.GGMLQuantizationType.F16 if args.type == "f16" else gguf.GGMLQuantizationType.F32

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    with open(model_dir / "preprocessor_config.json") as f:
        preproc = json.load(f)

    print(f"Model: {config.get('model_type', 'unknown')}")
    print(f"Vision backbone: {config.get('vision_backbone_id', 'unknown')}")

    # Load weight map
    weight_map = load_safetensors_metadata(model_dir)

    # DINO v2 config
    dino_n_embd = 1024
    dino_n_head = 16
    dino_n_layer = 24
    dino_n_ff = 4096
    dino_feature_layer = 22  # second-to-last layer (0-indexed)

    # SigLIP config
    siglip_n_embd = 1152
    siglip_n_head = 16
    siglip_n_layer = 27
    siglip_n_ff = 4304
    siglip_feature_layer = 25  # second-to-last layer (0-indexed)

    # Common config
    image_size = 224
    patch_size = 14

    # Normalization - preprocessor_config uses "means" and "stds" arrays
    # Index 0 = DINO (ImageNet), Index 1 = SigLIP (0.5/0.5)
    dino_mean = preproc["means"][0]   # [0.485, 0.456, 0.406]
    dino_std = preproc["stds"][0]     # [0.229, 0.224, 0.225]
    siglip_mean = preproc["means"][1] # [0.5, 0.5, 0.5]
    siglip_std = preproc["stds"][1]   # [0.5, 0.5, 0.5]

    print(f"DINO: {dino_n_layer} layers, {dino_n_embd} dim, {dino_n_head} heads")
    print(f"SigLIP: {siglip_n_layer} layers, {siglip_n_embd} dim, {siglip_n_head} heads")
    print(f"Image: {image_size}x{image_size}, patch: {patch_size}")

    # Create GGUF writer
    writer = gguf.GGUFWriter(str(output_path), arch="clip")
    writer.add_type(gguf.GGUFType.MMPROJ)
    writer.add_string("general.name", "openvla")

    # Vision metadata (primary = DINO)
    writer.add_clip_projector_type("openvla")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_vision_image_size(image_size)
    writer.add_vision_patch_size(patch_size)
    writer.add_vision_embedding_length(dino_n_embd)
    writer.add_vision_feed_forward_length(dino_n_ff)
    writer.add_vision_block_count(dino_n_layer)
    writer.add_vision_head_count(dino_n_head)
    writer.add_vision_attention_layernorm_eps(1e-6)
    writer.add_vision_image_mean(dino_mean)
    writer.add_vision_image_std(dino_std)
    writer.add_vision_use_gelu(True)
    writer.add_uint32("clip.vision.projection_dim", 0)
    writer.add_array("clip.vision.feature_layer", [dino_feature_layer])

    # Fused encoder metadata (SigLIP) - custom keys
    writer.add_uint32("clip.vision.fused_embedding_length", siglip_n_embd)
    writer.add_uint32("clip.vision.fused_feed_forward_length", siglip_n_ff)
    writer.add_uint32("clip.vision.fused_block_count", siglip_n_layer)
    writer.add_uint32("clip.vision.fused_head_count", siglip_n_head)
    writer.add_array("clip.vision.fused_image_mean", siglip_mean)
    writer.add_array("clip.vision.fused_image_std", siglip_std)
    writer.add_uint32("clip.vision.fused_feature_layer", siglip_feature_layer)

    # Tensors used only in ggml_mul_mat (supports mixed f32/f16)
    MUL_MAT_TENSORS = {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                       "attn_out.weight", "ffn_up.weight", "ffn_down.weight",
                       "mm.0.weight", "mm.2.weight", "mm.4.weight"}

    def add_tensor(name: str, data: np.ndarray):
        # Only mul_mat weight tensors can be f16; everything else must be f32
        # because ggml CPU element-wise ops (add, mul) don't support mixed types
        suffix = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
        can_f16 = any(name.endswith(s) for s in MUL_MAT_TENSORS)
        if args.type == "f16" and can_f16:
            data = data.astype(np.float16)
        else:
            data = data.astype(np.float32)
        writer.add_tensor(name, data)
        print(f"  {name}: {list(data.shape)} ({data.dtype})")

    # === DINO v2 Encoder (prefix: v.) ===
    print("\n--- DINO v2 Encoder ---")

    # Patch embedding
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.featurizer.patch_embed.proj.weight", weight_map)
    add_tensor("v.patch_embd.weight", data)

    data = load_tensor_from_safetensors(model_dir, "vision_backbone.featurizer.patch_embed.proj.bias", weight_map)
    add_tensor("v.patch_embd.bias", data)

    # CLS token: [1, 1, 1024] -> [1024]
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.featurizer.cls_token", weight_map)
    add_tensor("v.class_embd", data.squeeze())

    # Register tokens: [1, 4, 1024] -> [4, 1024]
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.featurizer.reg_token", weight_map)
    add_tensor("v.reg_embd", data.squeeze(0))

    # Position embeddings: [1, 256, 1024] -> [256, 1024]
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.featurizer.pos_embed", weight_map)
    add_tensor("v.position_embd.weight", data.squeeze(0))

    # Post-layernorm
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.featurizer.norm.weight", weight_map)
    add_tensor("v.post_ln.weight", data)
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.featurizer.norm.bias", weight_map)
    add_tensor("v.post_ln.bias", data)

    # Transformer blocks (only up to feature_layer)
    for il in range(dino_n_layer):
        prefix = f"vision_backbone.featurizer.blocks.{il}"
        blk = f"v.blk.{il}"

        # Layer norm 1
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm1.weight", weight_map)
        add_tensor(f"{blk}.ln1.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm1.bias", weight_map)
        add_tensor(f"{blk}.ln1.bias", data)

        # Attention: split fused QKV
        qkv_w = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.qkv.weight", weight_map)
        q_w, k_w, v_w = split_qkv(qkv_w, dino_n_head)
        add_tensor(f"{blk}.attn_q.weight", q_w)
        add_tensor(f"{blk}.attn_k.weight", k_w)
        add_tensor(f"{blk}.attn_v.weight", v_w)

        qkv_b = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.qkv.bias", weight_map)
        q_b, k_b, v_b = split_qkv_bias(qkv_b, dino_n_embd)
        add_tensor(f"{blk}.attn_q.bias", q_b)
        add_tensor(f"{blk}.attn_k.bias", k_b)
        add_tensor(f"{blk}.attn_v.bias", v_b)

        # Attention output projection
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.proj.weight", weight_map)
        add_tensor(f"{blk}.attn_out.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.proj.bias", weight_map)
        add_tensor(f"{blk}.attn_out.bias", data)

        # Layer scale
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.ls1.scale_factor", weight_map)
        add_tensor(f"{blk}.ls1.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.ls2.scale_factor", weight_map)
        add_tensor(f"{blk}.ls2.weight", data)

        # Layer norm 2
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm2.weight", weight_map)
        add_tensor(f"{blk}.ln2.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm2.bias", weight_map)
        add_tensor(f"{blk}.ln2.bias", data)

        # MLP: fc1 = ff_up, fc2 = ff_down
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc1.weight", weight_map)
        add_tensor(f"{blk}.ffn_up.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc1.bias", weight_map)
        add_tensor(f"{blk}.ffn_up.bias", data)

        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc2.weight", weight_map)
        add_tensor(f"{blk}.ffn_down.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc2.bias", weight_map)
        add_tensor(f"{blk}.ffn_down.bias", data)

    # === SigLIP Encoder (prefix: v2.) ===
    print("\n--- SigLIP Encoder ---")

    # Patch embedding
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.fused_featurizer.patch_embed.proj.weight", weight_map)
    add_tensor("v2.patch_embd.weight", data)

    data = load_tensor_from_safetensors(model_dir, "vision_backbone.fused_featurizer.patch_embed.proj.bias", weight_map)
    add_tensor("v2.patch_embd.bias", data)

    # Position embeddings: [1, 256, 1152] -> [256, 1152]
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.fused_featurizer.pos_embed", weight_map)
    add_tensor("v2.position_embd.weight", data.squeeze(0))

    # Post-layernorm
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.fused_featurizer.norm.weight", weight_map)
    add_tensor("v2.post_ln.weight", data)
    data = load_tensor_from_safetensors(model_dir, "vision_backbone.fused_featurizer.norm.bias", weight_map)
    add_tensor("v2.post_ln.bias", data)

    # Transformer blocks
    for il in range(siglip_n_layer):
        prefix = f"vision_backbone.fused_featurizer.blocks.{il}"
        blk = f"v2.blk.{il}"

        # Layer norm 1
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm1.weight", weight_map)
        add_tensor(f"{blk}.ln1.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm1.bias", weight_map)
        add_tensor(f"{blk}.ln1.bias", data)

        # Attention: split fused QKV
        qkv_w = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.qkv.weight", weight_map)
        q_w, k_w, v_w = split_qkv(qkv_w, siglip_n_head)
        add_tensor(f"{blk}.attn_q.weight", q_w)
        add_tensor(f"{blk}.attn_k.weight", k_w)
        add_tensor(f"{blk}.attn_v.weight", v_w)

        qkv_b = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.qkv.bias", weight_map)
        q_b, k_b, v_b = split_qkv_bias(qkv_b, siglip_n_embd)
        add_tensor(f"{blk}.attn_q.bias", q_b)
        add_tensor(f"{blk}.attn_k.bias", k_b)
        add_tensor(f"{blk}.attn_v.bias", v_b)

        # Attention output projection
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.proj.weight", weight_map)
        add_tensor(f"{blk}.attn_out.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.attn.proj.bias", weight_map)
        add_tensor(f"{blk}.attn_out.bias", data)

        # Layer norm 2
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm2.weight", weight_map)
        add_tensor(f"{blk}.ln2.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.norm2.bias", weight_map)
        add_tensor(f"{blk}.ln2.bias", data)

        # MLP
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc1.weight", weight_map)
        add_tensor(f"{blk}.ffn_up.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc1.bias", weight_map)
        add_tensor(f"{blk}.ffn_up.bias", data)

        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc2.weight", weight_map)
        add_tensor(f"{blk}.ffn_down.weight", data)
        data = load_tensor_from_safetensors(model_dir, f"{prefix}.mlp.fc2.bias", weight_map)
        add_tensor(f"{blk}.ffn_down.bias", data)

    # === Projector (prefix: mm.) ===
    print("\n--- Projector ---")

    # fc1 -> mm.0
    data = load_tensor_from_safetensors(model_dir, "projector.fc1.weight", weight_map)
    add_tensor("mm.0.weight", data)
    data = load_tensor_from_safetensors(model_dir, "projector.fc1.bias", weight_map)
    add_tensor("mm.0.bias", data)

    # fc2 -> mm.2
    data = load_tensor_from_safetensors(model_dir, "projector.fc2.weight", weight_map)
    add_tensor("mm.2.weight", data)
    data = load_tensor_from_safetensors(model_dir, "projector.fc2.bias", weight_map)
    add_tensor("mm.2.bias", data)

    # fc3 -> mm.4
    data = load_tensor_from_safetensors(model_dir, "projector.fc3.weight", weight_map)
    add_tensor("mm.4.weight", data)
    data = load_tensor_from_safetensors(model_dir, "projector.fc3.bias", weight_map)
    add_tensor("mm.4.bias", data)

    # Write
    print(f"\nWriting to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! File size: {file_size:.1f} MB")


if __name__ == "__main__":
    main()
