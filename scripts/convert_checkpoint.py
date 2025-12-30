#!/usr/bin/env python3
"""Convert Flax Linen checkpoint to ONNX format.

This script converts a trained IntentionNetwork checkpoint to ONNX format
for browser-based inference using ONNX Runtime Web.

Approach: Build ONNX graph manually using onnx Python API
Since tf2onnx has compatibility issues with NumPy 2.x and JAX, we directly
construct the ONNX graph from the checkpoint weights.

Usage:
    python convert_checkpoint.py \
        --checkpoint /path/to/checkpoint \
        --output model.onnx
"""

import argparse
import sys
from pathlib import Path

# Add track-mjx to path for imports
TRACK_MJX_PATH = Path(__file__).parent.parent.parent / "track-mjx"
sys.path.insert(0, str(TRACK_MJX_PATH))

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from track_mjx.agent import checkpointing


def build_decoder_only_onnx_model(normalizer_params, decoder_params, cfg):
    """Build decoder-only ONNX model for latent space random walk mode.

    This model takes a latent vector and proprioceptive observation as inputs,
    normalizes the proprioceptive observation, and runs through the decoder.

    Args:
        normalizer_params: RunningStatisticsState for observation normalization
        decoder_params: Dict containing decoder Dense and LayerNorm params
        cfg: Configuration dict from checkpoint

    Returns:
        ONNX ModelProto
    """
    reference_obs_size = cfg.network_config.reference_obs_size
    proprio_obs_size = cfg.network_config.proprioceptive_obs_size
    latent_size = cfg.network_config.intention_size

    # Convert params to numpy - only need proprio portion of normalization
    norm_mean_full = np.array(normalizer_params.mean).astype(np.float32)
    norm_std_full = np.array(normalizer_params.std).astype(np.float32)
    proprio_norm_mean = norm_mean_full[reference_obs_size:]
    proprio_norm_std = norm_std_full[reference_obs_size:] + 1e-8
    decoder_params_np = jax.tree.map(lambda x: np.array(x).astype(np.float32), decoder_params)

    # Build list of initializers
    initializers = [
        numpy_helper.from_array(proprio_norm_mean, "proprio_norm_mean"),
        numpy_helper.from_array(proprio_norm_std, "proprio_norm_std"),
    ]

    # Add decoder params as initializers
    decoder_layer_names = ['hidden_0', 'hidden_1', 'hidden_2', 'hidden_3']
    for i, layer_key in enumerate(decoder_layer_names):
        if layer_key not in decoder_params_np:
            break
        layer = decoder_params_np[layer_key]
        initializers.append(numpy_helper.from_array(layer['kernel'], f"dec_{layer_key}_w"))
        initializers.append(numpy_helper.from_array(layer['bias'], f"dec_{layer_key}_b"))
        ln_key = f'LayerNorm_{i}'
        if ln_key in decoder_params_np:
            ln = decoder_params_np[ln_key]
            initializers.append(numpy_helper.from_array(ln['scale'], f"dec_ln{i}_scale"))
            initializers.append(numpy_helper.from_array(ln['bias'], f"dec_ln{i}_bias"))

    # Build computation graph
    nodes = []
    node_idx = 0

    def make_node(op_type, inputs, outputs, name=None, **attrs):
        nonlocal node_idx
        if name is None:
            name = f"node_{node_idx}"
        node_idx += 1
        return helper.make_node(op_type, inputs, outputs, name=name, **attrs)

    # Inputs: latent (batch, 16), proprio_obs (batch, 277)
    # 1. Normalize proprio_obs: (proprio_obs - mean) / std
    nodes.append(make_node("Sub", ["proprio_obs", "proprio_norm_mean"], ["proprio_centered"]))
    nodes.append(make_node("Div", ["proprio_centered", "proprio_norm_std"], ["proprio_normalized"]))

    # 2. Concat [latent, proprio_normalized]
    nodes.append(make_node("Concat", ["latent", "proprio_normalized"], ["decoder_input"], axis=1))

    # 3. Decoder forward pass (same as full model)
    x = "decoder_input"
    for i, layer_key in enumerate(decoder_layer_names):
        if f"dec_{layer_key}_w" not in [init.name for init in initializers]:
            break

        # Dense
        nodes.append(make_node("MatMul", [x, f"dec_{layer_key}_w"], [f"dec_{layer_key}_mm"]))
        nodes.append(make_node("Add", [f"dec_{layer_key}_mm", f"dec_{layer_key}_b"], [f"dec_{layer_key}_out"]))

        # Check if LayerNorm exists for this layer
        ln_scale_name = f"dec_ln{i}_scale"
        has_ln = ln_scale_name in [init.name for init in initializers]

        if has_ln:
            # SiLU
            nodes.append(make_node("Sigmoid", [f"dec_{layer_key}_out"], [f"dec_{layer_key}_sig"]))
            nodes.append(make_node("Mul", [f"dec_{layer_key}_out", f"dec_{layer_key}_sig"], [f"dec_{layer_key}_silu"]))

            # LayerNorm
            nodes.append(make_node("ReduceMean", [f"dec_{layer_key}_silu"], [f"dec_ln{i}_mean"], axes=[-1], keepdims=1))
            nodes.append(make_node("Sub", [f"dec_{layer_key}_silu", f"dec_ln{i}_mean"], [f"dec_ln{i}_centered"]))
            nodes.append(make_node("Mul", [f"dec_ln{i}_centered", f"dec_ln{i}_centered"], [f"dec_ln{i}_sq"]))
            nodes.append(make_node("ReduceMean", [f"dec_ln{i}_sq"], [f"dec_ln{i}_var"], axes=[-1], keepdims=1))

            eps_name = f"dec_ln{i}_eps"
            initializers.append(numpy_helper.from_array(np.array([1e-5], dtype=np.float32), eps_name))
            nodes.append(make_node("Add", [f"dec_ln{i}_var", eps_name], [f"dec_ln{i}_var_eps"]))
            nodes.append(make_node("Sqrt", [f"dec_ln{i}_var_eps"], [f"dec_ln{i}_std"]))
            nodes.append(make_node("Div", [f"dec_ln{i}_centered", f"dec_ln{i}_std"], [f"dec_ln{i}_norm"]))
            nodes.append(make_node("Mul", [f"dec_ln{i}_norm", f"dec_ln{i}_scale"], [f"dec_ln{i}_scaled"]))
            nodes.append(make_node("Add", [f"dec_ln{i}_scaled", f"dec_ln{i}_bias"], [f"dec_{layer_key}_ln"]))
            x = f"dec_{layer_key}_ln"
        else:
            x = f"dec_{layer_key}_out"

    # Final output
    nodes.append(make_node("Identity", [x], ["action_logits"]))

    # Define inputs/outputs
    inputs = [
        helper.make_tensor_value_info("latent", TensorProto.FLOAT, [None, latent_size]),
        helper.make_tensor_value_info("proprio_obs", TensorProto.FLOAT, [None, proprio_obs_size]),
    ]
    outputs = [helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, [None, 76])]

    # Create graph
    graph = helper.make_graph(
        nodes,
        "decoder_only",
        inputs,
        outputs,
        initializers,
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    return model


def build_onnx_model(normalizer_params, encoder_params, decoder_params, cfg):
    """Build ONNX model graph manually from checkpoint weights.

    Args:
        normalizer_params: RunningStatisticsState for observation normalization
        encoder_params: Dict containing encoder Dense and LayerNorm params
        decoder_params: Dict containing decoder Dense and LayerNorm params
        cfg: Configuration dict from checkpoint

    Returns:
        ONNX ModelProto
    """
    reference_obs_size = cfg.network_config.reference_obs_size
    obs_size = cfg.network_config.observation_size

    # Convert params to numpy
    norm_mean = np.array(normalizer_params.mean).astype(np.float32)
    norm_std = np.array(normalizer_params.std).astype(np.float32)
    encoder_params_np = jax.tree.map(lambda x: np.array(x).astype(np.float32), encoder_params)
    decoder_params_np = jax.tree.map(lambda x: np.array(x).astype(np.float32), decoder_params)

    # Build list of initializers (constants in the graph)
    initializers = [
        numpy_helper.from_array(norm_mean, "norm_mean"),
        numpy_helper.from_array(norm_std + 1e-8, "norm_std"),  # Add epsilon here
        numpy_helper.from_array(np.array([reference_obs_size], dtype=np.int64), "ref_obs_size"),
    ]

    # Add encoder params as initializers
    for i, layer_key in enumerate(['hidden_0', 'hidden_1', 'hidden_2']):
        layer = encoder_params_np[layer_key]
        initializers.append(numpy_helper.from_array(layer['kernel'], f"enc_{layer_key}_w"))
        initializers.append(numpy_helper.from_array(layer['bias'], f"enc_{layer_key}_b"))
        ln = encoder_params_np[f'LayerNorm_{i}']
        initializers.append(numpy_helper.from_array(ln['scale'], f"enc_ln{i}_scale"))
        initializers.append(numpy_helper.from_array(ln['bias'], f"enc_ln{i}_bias"))

    initializers.append(numpy_helper.from_array(encoder_params_np['fc2_mean']['kernel'], "enc_mean_w"))
    initializers.append(numpy_helper.from_array(encoder_params_np['fc2_mean']['bias'], "enc_mean_b"))

    # Add decoder params as initializers
    decoder_layer_names = ['hidden_0', 'hidden_1', 'hidden_2', 'hidden_3']
    for i, layer_key in enumerate(decoder_layer_names):
        if layer_key not in decoder_params_np:
            break
        layer = decoder_params_np[layer_key]
        initializers.append(numpy_helper.from_array(layer['kernel'], f"dec_{layer_key}_w"))
        initializers.append(numpy_helper.from_array(layer['bias'], f"dec_{layer_key}_b"))
        ln_key = f'LayerNorm_{i}'
        if ln_key in decoder_params_np:
            ln = decoder_params_np[ln_key]
            initializers.append(numpy_helper.from_array(ln['scale'], f"dec_ln{i}_scale"))
            initializers.append(numpy_helper.from_array(ln['bias'], f"dec_ln{i}_bias"))

    # Build computation graph
    nodes = []
    node_idx = 0

    def make_node(op_type, inputs, outputs, name=None, **attrs):
        nonlocal node_idx
        if name is None:
            name = f"node_{node_idx}"
        node_idx += 1
        return helper.make_node(op_type, inputs, outputs, name=name, **attrs)

    # Input: obs (batch, 904)
    # 1. Normalize: (obs - mean) / std
    nodes.append(make_node("Sub", ["obs", "norm_mean"], ["obs_centered"]))
    nodes.append(make_node("Div", ["obs_centered", "norm_std"], ["obs_normalized"]))

    # 2. Split into reference and proprioceptive
    # Use Slice to get ref_obs = obs[:, :640] and proprio_obs = obs[:, 640:]
    nodes.append(helper.make_node(
        "Slice",
        inputs=["obs_normalized", "starts_ref", "ends_ref", "axes_ref"],
        outputs=["ref_obs"],
        name="slice_ref"
    ))
    nodes.append(helper.make_node(
        "Slice",
        inputs=["obs_normalized", "starts_prop", "ends_prop", "axes_prop"],
        outputs=["proprio_obs"],
        name="slice_prop"
    ))

    # Add slice constants
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "starts_ref"))
    initializers.append(numpy_helper.from_array(np.array([reference_obs_size], dtype=np.int64), "ends_ref"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_ref"))
    initializers.append(numpy_helper.from_array(np.array([reference_obs_size], dtype=np.int64), "starts_prop"))
    initializers.append(numpy_helper.from_array(np.array([obs_size], dtype=np.int64), "ends_prop"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_prop"))

    # 3. Encoder forward pass
    x = "ref_obs"
    for i, layer_key in enumerate(['hidden_0', 'hidden_1', 'hidden_2']):
        # Dense: x = x @ W + b
        nodes.append(make_node("MatMul", [x, f"enc_{layer_key}_w"], [f"enc_{layer_key}_mm"]))
        nodes.append(make_node("Add", [f"enc_{layer_key}_mm", f"enc_{layer_key}_b"], [f"enc_{layer_key}_out"]))

        # SiLU: x * sigmoid(x)
        nodes.append(make_node("Sigmoid", [f"enc_{layer_key}_out"], [f"enc_{layer_key}_sig"]))
        nodes.append(make_node("Mul", [f"enc_{layer_key}_out", f"enc_{layer_key}_sig"], [f"enc_{layer_key}_silu"]))

        # LayerNorm
        # mean = ReduceMean(x, axis=-1)
        # var = ReduceMean((x - mean)^2, axis=-1)
        # normalized = (x - mean) / sqrt(var + eps)
        # out = normalized * scale + bias
        nodes.append(make_node("ReduceMean", [f"enc_{layer_key}_silu"], [f"enc_ln{i}_mean"], axes=[-1], keepdims=1))
        nodes.append(make_node("Sub", [f"enc_{layer_key}_silu", f"enc_ln{i}_mean"], [f"enc_ln{i}_centered"]))
        nodes.append(make_node("Mul", [f"enc_ln{i}_centered", f"enc_ln{i}_centered"], [f"enc_ln{i}_sq"]))
        nodes.append(make_node("ReduceMean", [f"enc_ln{i}_sq"], [f"enc_ln{i}_var"], axes=[-1], keepdims=1))

        # Add epsilon constant for this layer
        eps_name = f"enc_ln{i}_eps"
        initializers.append(numpy_helper.from_array(np.array([1e-5], dtype=np.float32), eps_name))
        nodes.append(make_node("Add", [f"enc_ln{i}_var", eps_name], [f"enc_ln{i}_var_eps"]))
        nodes.append(make_node("Sqrt", [f"enc_ln{i}_var_eps"], [f"enc_ln{i}_std"]))
        nodes.append(make_node("Div", [f"enc_ln{i}_centered", f"enc_ln{i}_std"], [f"enc_ln{i}_norm"]))
        nodes.append(make_node("Mul", [f"enc_ln{i}_norm", f"enc_ln{i}_scale"], [f"enc_ln{i}_scaled"]))
        nodes.append(make_node("Add", [f"enc_ln{i}_scaled", f"enc_ln{i}_bias"], [f"enc_{layer_key}_ln"]))

        x = f"enc_{layer_key}_ln"

    # Get latent mean
    nodes.append(make_node("MatMul", [x, "enc_mean_w"], ["enc_mean_mm"]))
    nodes.append(make_node("Add", ["enc_mean_mm", "enc_mean_b"], ["latent_mean"]))

    # 4. Decoder forward pass
    # Concat [latent_mean, proprio_obs]
    nodes.append(make_node("Concat", ["latent_mean", "proprio_obs"], ["decoder_input"], axis=1))

    x = "decoder_input"
    for i, layer_key in enumerate(decoder_layer_names):
        if f"dec_{layer_key}_w" not in [init.name for init in initializers]:
            break

        # Dense
        nodes.append(make_node("MatMul", [x, f"dec_{layer_key}_w"], [f"dec_{layer_key}_mm"]))
        nodes.append(make_node("Add", [f"dec_{layer_key}_mm", f"dec_{layer_key}_b"], [f"dec_{layer_key}_out"]))

        # Check if LayerNorm exists for this layer (not for the last layer)
        ln_scale_name = f"dec_ln{i}_scale"
        has_ln = ln_scale_name in [init.name for init in initializers]

        if has_ln:
            # SiLU
            nodes.append(make_node("Sigmoid", [f"dec_{layer_key}_out"], [f"dec_{layer_key}_sig"]))
            nodes.append(make_node("Mul", [f"dec_{layer_key}_out", f"dec_{layer_key}_sig"], [f"dec_{layer_key}_silu"]))

            # LayerNorm
            nodes.append(make_node("ReduceMean", [f"dec_{layer_key}_silu"], [f"dec_ln{i}_mean"], axes=[-1], keepdims=1))
            nodes.append(make_node("Sub", [f"dec_{layer_key}_silu", f"dec_ln{i}_mean"], [f"dec_ln{i}_centered"]))
            nodes.append(make_node("Mul", [f"dec_ln{i}_centered", f"dec_ln{i}_centered"], [f"dec_ln{i}_sq"]))
            nodes.append(make_node("ReduceMean", [f"dec_ln{i}_sq"], [f"dec_ln{i}_var"], axes=[-1], keepdims=1))

            eps_name = f"dec_ln{i}_eps"
            initializers.append(numpy_helper.from_array(np.array([1e-5], dtype=np.float32), eps_name))
            nodes.append(make_node("Add", [f"dec_ln{i}_var", eps_name], [f"dec_ln{i}_var_eps"]))
            nodes.append(make_node("Sqrt", [f"dec_ln{i}_var_eps"], [f"dec_ln{i}_std"]))
            nodes.append(make_node("Div", [f"dec_ln{i}_centered", f"dec_ln{i}_std"], [f"dec_ln{i}_norm"]))
            nodes.append(make_node("Mul", [f"dec_ln{i}_norm", f"dec_ln{i}_scale"], [f"dec_ln{i}_scaled"]))
            nodes.append(make_node("Add", [f"dec_ln{i}_scaled", f"dec_ln{i}_bias"], [f"dec_{layer_key}_ln"]))
            x = f"dec_{layer_key}_ln"
        else:
            x = f"dec_{layer_key}_out"

    # Final output
    nodes.append(make_node("Identity", [x], ["action_logits"]))

    # Define input/output
    inputs = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, obs_size])]
    outputs = [
        helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, [None, 76]),
        helper.make_tensor_value_info("latent_mean", TensorProto.FLOAT, [None, cfg.network_config.intention_size]),
    ]

    # Create graph
    graph = helper.make_graph(
        nodes,
        "intention_network",
        inputs,
        outputs,
        initializers,
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    return model


def extract_encoder_decoder_params(policy_params):
    """Extract encoder and decoder params from the nested policy params structure."""
    params = policy_params['params']
    encoder_params = params['encoder']
    decoder_params = params['decoder']
    return encoder_params, decoder_params


def convert_to_onnx(checkpoint_path: str, output_path: str, step: int = None):
    """Convert checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to checkpoint directory
        output_path: Path for output ONNX file
        step: Optional checkpoint step (default: latest)
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    ckpt = checkpointing.load_checkpoint_for_eval(checkpoint_path, step=step)
    cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]

    # Unpack policy params: (normalizer_state, network_params)
    normalizer_params, network_params = policy_params

    print(f"Observation size: {cfg.network_config.observation_size}")
    print(f"Reference obs size: {cfg.network_config.reference_obs_size}")
    print(f"Action size: {cfg.network_config.action_size}")
    print(f"Intention size: {cfg.network_config.intention_size}")

    # Extract encoder and decoder params
    encoder_params, decoder_params = extract_encoder_decoder_params(network_params)

    # Build full ONNX model (encoder + decoder)
    print("Building full ONNX model...")
    model = build_onnx_model(normalizer_params, encoder_params, decoder_params, cfg)

    # Check model
    print("Checking full ONNX model...")
    onnx.checker.check_model(model)

    # Save model
    print(f"Saving full ONNX model to: {output_path}")
    onnx.save(model, output_path)

    # Build decoder-only ONNX model (for latent space random walk mode)
    print("Building decoder-only ONNX model...")
    decoder_model = build_decoder_only_onnx_model(normalizer_params, decoder_params, cfg)

    # Check decoder model
    print("Checking decoder-only ONNX model...")
    onnx.checker.check_model(decoder_model)

    # Save decoder model
    decoder_output_path = str(Path(output_path).with_name("decoder_only.onnx"))
    print(f"Saving decoder-only ONNX model to: {decoder_output_path}")
    onnx.save(decoder_model, decoder_output_path)

    # Verify with ONNX Runtime
    print("Verifying with ONNX Runtime...")
    import onnxruntime as ort

    session = ort.InferenceSession(output_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Test input
    obs_size = cfg.network_config.observation_size
    test_input = np.zeros((1, obs_size), dtype=np.float32)

    ort_output = session.run([output_name], {input_name: test_input})[0]
    print(f"ONNX Runtime output shape: {ort_output.shape}")

    # Compare with original Flax module
    print("Comparing with original Flax module...")
    ppo_network = checkpointing.make_ppo_network_from_cfg(cfg)
    dummy_key = jax.random.PRNGKey(0)
    flax_output, _, _ = ppo_network.policy_network.apply(
        normalizer_params, network_params, jnp.array(test_input), dummy_key,
        deterministic=True, get_activation=False
    )

    max_diff = np.max(np.abs(np.array(flax_output) - ort_output))
    print(f"Max difference between Flax and ONNX: {max_diff}")

    if max_diff > 1e-3:
        print("WARNING: Large difference detected!")
    else:
        print("Verification passed!")

    # Verify decoder-only model
    print("\nVerifying decoder-only ONNX model...")
    decoder_session = ort.InferenceSession(decoder_output_path)
    latent_size = cfg.network_config.intention_size
    proprio_size = cfg.network_config.proprioceptive_obs_size

    test_latent = np.zeros((1, latent_size), dtype=np.float32)
    test_proprio = np.zeros((1, proprio_size), dtype=np.float32)

    decoder_output = decoder_session.run(
        [decoder_session.get_outputs()[0].name],
        {"latent": test_latent, "proprio_obs": test_proprio}
    )[0]
    print(f"Decoder-only output shape: {decoder_output.shape}")
    print(f"Decoder-only model verified successfully!")

    # Save normalization parameters
    norm_params_path = Path(output_path).with_suffix(".norm.npz")
    np.savez(
        norm_params_path,
        mean=np.array(normalizer_params.mean),
        std=np.array(normalizer_params.std),
        reference_obs_size=cfg.network_config.reference_obs_size,
        proprioceptive_obs_size=cfg.network_config.proprioceptive_obs_size,
    )
    print(f"Saved normalization parameters to: {norm_params_path}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert Flax checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to load (default: latest)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_to_onnx(args.checkpoint, str(output_path), args.step)


if __name__ == "__main__":
    main()
