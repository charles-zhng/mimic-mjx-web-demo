#!/usr/bin/env python3
"""Convert Flax Linen checkpoint to ONNX format.

This script converts a trained IntentionNetwork checkpoint to ONNX format
for browser-based inference using ONNX Runtime Web.

Approach: Build ONNX graph manually using onnx Python API
Since tf2onnx has compatibility issues with NumPy 2.x and JAX, we directly
construct the ONNX graph from the checkpoint weights.

Usage:
    # Convert MIMIC tracking policy (encoder + decoder)
    python convert_checkpoint.py \
        --checkpoint /path/to/checkpoint \
        --output model.onnx

    # Convert high-level joystick policy
    python convert_checkpoint.py \
        --highlevel-checkpoint /path/to/highlevel_checkpoint \
        --output highlevel_policy.onnx
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Add track-mjx to path for imports
TRACK_MJX_PATH = Path(__file__).parent.parent.parent / "track-mjx"
sys.path.insert(0, str(TRACK_MJX_PATH))

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from orbax import checkpoint as ocp

from track_mjx.agent import checkpointing


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails."""
    pass


def validate_checkpoint(ckpt: dict) -> None:
    """Validate checkpoint structure before conversion.

    Args:
        ckpt: Loaded checkpoint dictionary

    Raises:
        CheckpointValidationError: If required fields are missing or malformed
    """
    # Check top-level keys
    required_keys = ['cfg', 'policy']
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise CheckpointValidationError(
            f"Missing required checkpoint keys: {missing}. "
            f"Found keys: {list(ckpt.keys())}"
        )

    # Validate policy structure: (normalizer_params, network_params)
    policy = ckpt['policy']
    if not isinstance(policy, (list, tuple)) or len(policy) != 2:
        raise CheckpointValidationError(
            f"Expected policy to be (normalizer_params, network_params) tuple, "
            f"got {type(policy).__name__}"
        )

    normalizer_params, network_params = policy

    # Validate normalizer params (DictRunningStatisticsState with imitation_target and proprioception)
    if not hasattr(normalizer_params, 'imitation_target') or not hasattr(normalizer_params, 'proprioception'):
        raise CheckpointValidationError(
            "Invalid normalizer_params: expected DictRunningStatisticsState with 'imitation_target' and 'proprioception'. "
            f"Found attributes: {[a for a in dir(normalizer_params) if not a.startswith('_')]}"
        )

    # Validate network params structure
    if not isinstance(network_params, dict) or 'params' not in network_params:
        raise CheckpointValidationError(
            f"Expected network_params to be dict with 'params' key, "
            f"got {type(network_params).__name__} with keys: {list(network_params.keys()) if isinstance(network_params, dict) else 'N/A'}"
        )

    params = network_params['params']
    required_subkeys = ['encoder', 'decoder']
    missing_subkeys = [k for k in required_subkeys if k not in params]
    if missing_subkeys:
        raise CheckpointValidationError(
            f"Missing required network params: {missing_subkeys}. "
            f"Found: {list(params.keys())}"
        )

    # Validate encoder has fc2_mean for latent projection
    encoder = params['encoder']
    if 'fc2_mean' not in encoder:
        raise CheckpointValidationError(
            f"Encoder missing 'fc2_mean' layer for latent projection. "
            f"Found layers: {list(encoder.keys())}"
        )

    # Validate config has network_config
    cfg = ckpt['cfg']
    if not hasattr(cfg, 'network_config'):
        raise CheckpointValidationError(
            "Config missing 'network_config' attribute"
        )

    nc = cfg.network_config
    required_nc_attrs = ['action_size', 'intention_size', 'encoder_layer_sizes', 'decoder_layer_sizes']
    missing_nc = [a for a in required_nc_attrs if not hasattr(nc, a)]
    if missing_nc:
        raise CheckpointValidationError(
            f"network_config missing required attributes: {missing_nc}"
        )

    print("Checkpoint validation passed")


@dataclass
class NetworkDims:
    """Centralized network dimensions extracted from checkpoint config."""
    obs_size: int
    reference_obs_size: int
    proprio_obs_size: int
    action_size: int
    latent_size: int
    encoder_layer_sizes: list[int]
    decoder_layer_sizes: list[int]

    @classmethod
    def from_config(cls, cfg) -> 'NetworkDims':
        obs_size, ref_size, proprio_size = get_obs_sizes(cfg)
        nc = cfg.network_config
        return cls(
            obs_size=obs_size,
            reference_obs_size=ref_size,
            proprio_obs_size=proprio_size,
            action_size=nc.action_size,
            latent_size=nc.intention_size,
            encoder_layer_sizes=list(nc.encoder_layer_sizes),
            decoder_layer_sizes=list(nc.decoder_layer_sizes),
        )

    @property
    def output_size(self) -> int:
        """Action output size: means + log_stds."""
        return self.action_size * 2

    @property
    def num_encoder_layers(self) -> int:
        return len(self.encoder_layer_sizes)

    @property
    def num_decoder_layers(self) -> int:
        return len(self.decoder_layer_sizes)

    def encoder_layer_names(self) -> list[str]:
        return [f'hidden_{i}' for i in range(self.num_encoder_layers)]

    def decoder_layer_names(self) -> list[str]:
        # Decoder has len(decoder_layer_sizes) + 1 layers:
        # hidden_0..hidden_N are the hidden layers, plus one final output layer
        return [f'hidden_{i}' for i in range(self.num_decoder_layers + 1)]


def get_obs_sizes(cfg):
    """Extract observation sizes from config."""
    nc = cfg.network_config
    reference_obs_size = nc.obs_sizes.imitation_target
    proprio_obs_size = nc.obs_sizes.proprioception
    obs_size = reference_obs_size + proprio_obs_size
    return obs_size, reference_obs_size, proprio_obs_size


def get_normalizer_arrays(normalizer_params):
    """Extract mean and std arrays from DictRunningStatisticsState normalizer.

    Returns:
        norm_mean: Full observation mean (float32 numpy array)
        norm_std: Full observation std (float32 numpy array)
    """
    ref_mean = np.array(normalizer_params.imitation_target.mean).astype(np.float32)
    ref_std = np.array(normalizer_params.imitation_target.std).astype(np.float32)
    proprio_mean = np.array(normalizer_params.proprioception.mean).astype(np.float32)
    proprio_std = np.array(normalizer_params.proprioception.std).astype(np.float32)
    norm_mean = np.concatenate([ref_mean, proprio_mean])
    norm_std = np.concatenate([ref_std, proprio_std])
    return norm_mean, norm_std


def add_dense_layer_params(
    initializers: list,
    params_np: dict,
    prefix: str,
    layer_key: str,
    layer_idx: int,
    has_layer_norm: bool,
) -> None:
    """Add Dense layer weights and optional LayerNorm params to initializers."""
    layer = params_np[layer_key]
    initializers.append(numpy_helper.from_array(layer['kernel'], f"{prefix}_{layer_key}_w"))
    initializers.append(numpy_helper.from_array(layer['bias'], f"{prefix}_{layer_key}_b"))
    if has_layer_norm:
        ln_key = f'LayerNorm_{layer_idx}'
        if ln_key in params_np:
            ln = params_np[ln_key]
            initializers.append(numpy_helper.from_array(ln['scale'], f"{prefix}_ln{layer_idx}_scale"))
            initializers.append(numpy_helper.from_array(ln['bias'], f"{prefix}_ln{layer_idx}_bias"))


def add_mlp_block(
    nodes: list,
    initializers: list,
    prefix: str,
    layer_key: str,
    layer_idx: int,
    input_name: str,
    has_layer_norm: bool,
    make_node: callable,
) -> str:
    """Build Dense -> SiLU -> LayerNorm block, returns output tensor name.

    Args:
        nodes: List to append ONNX nodes to
        initializers: List of ONNX initializers (for epsilon constant)
        prefix: Layer prefix ("enc" or "dec")
        layer_key: Layer name (e.g., "hidden_0")
        layer_idx: Layer index for LayerNorm naming
        input_name: Input tensor name
        has_layer_norm: Whether to include LayerNorm after SiLU
        make_node: Node factory function

    Returns:
        Output tensor name
    """
    # Dense: x = x @ W + b
    nodes.append(make_node("MatMul", [input_name, f"{prefix}_{layer_key}_w"], [f"{prefix}_{layer_key}_mm"]))
    nodes.append(make_node("Add", [f"{prefix}_{layer_key}_mm", f"{prefix}_{layer_key}_b"], [f"{prefix}_{layer_key}_out"]))

    if not has_layer_norm:
        return f"{prefix}_{layer_key}_out"

    # SiLU: x * sigmoid(x)
    nodes.append(make_node("Sigmoid", [f"{prefix}_{layer_key}_out"], [f"{prefix}_{layer_key}_sig"]))
    nodes.append(make_node("Mul", [f"{prefix}_{layer_key}_out", f"{prefix}_{layer_key}_sig"], [f"{prefix}_{layer_key}_silu"]))

    # LayerNorm: (x - mean) / sqrt(var + eps) * scale + bias
    ln_prefix = f"{prefix}_ln{layer_idx}"
    nodes.append(make_node("ReduceMean", [f"{prefix}_{layer_key}_silu"], [f"{ln_prefix}_mean"], axes=[-1], keepdims=1))
    nodes.append(make_node("Sub", [f"{prefix}_{layer_key}_silu", f"{ln_prefix}_mean"], [f"{ln_prefix}_centered"]))
    nodes.append(make_node("Mul", [f"{ln_prefix}_centered", f"{ln_prefix}_centered"], [f"{ln_prefix}_sq"]))
    nodes.append(make_node("ReduceMean", [f"{ln_prefix}_sq"], [f"{ln_prefix}_var"], axes=[-1], keepdims=1))

    eps_name = f"{ln_prefix}_eps"
    initializers.append(numpy_helper.from_array(np.array([1e-5], dtype=np.float32), eps_name))
    nodes.append(make_node("Add", [f"{ln_prefix}_var", eps_name], [f"{ln_prefix}_var_eps"]))
    nodes.append(make_node("Sqrt", [f"{ln_prefix}_var_eps"], [f"{ln_prefix}_std"]))
    nodes.append(make_node("Div", [f"{ln_prefix}_centered", f"{ln_prefix}_std"], [f"{ln_prefix}_norm"]))
    nodes.append(make_node("Mul", [f"{ln_prefix}_norm", f"{ln_prefix}_scale"], [f"{ln_prefix}_scaled"]))
    nodes.append(make_node("Add", [f"{ln_prefix}_scaled", f"{ln_prefix}_bias"], [f"{prefix}_{layer_key}_ln"]))

    return f"{prefix}_{layer_key}_ln"


def build_decoder_only_onnx_model(normalizer_params, decoder_params, dims: NetworkDims):
    """Build decoder-only ONNX model for latent space random walk mode.

    This model takes a latent vector and proprioceptive observation as inputs,
    normalizes the proprioceptive observation, and runs through the decoder.

    Args:
        normalizer_params: RunningStatisticsState for observation normalization
        decoder_params: Dict containing decoder Dense and LayerNorm params
        dims: NetworkDims with all dimension info

    Returns:
        ONNX ModelProto
    """
    # Convert params to numpy - only need proprio portion of normalization
    norm_mean_full, norm_std_full = get_normalizer_arrays(normalizer_params)
    proprio_norm_mean = norm_mean_full[dims.reference_obs_size:]
    proprio_norm_std = norm_std_full[dims.reference_obs_size:] + 1e-8
    decoder_params_np = jax.tree.map(lambda x: np.array(x).astype(np.float32), decoder_params)

    # Build list of initializers
    initializers = [
        numpy_helper.from_array(proprio_norm_mean, "proprio_norm_mean"),
        numpy_helper.from_array(proprio_norm_std, "proprio_norm_std"),
    ]

    # Add decoder params as initializers (dynamic layer count)
    decoder_layer_names = dims.decoder_layer_names()
    for i, layer_key in enumerate(decoder_layer_names):
        is_last_layer = (i == len(decoder_layer_names) - 1)
        has_ln = not is_last_layer and f'LayerNorm_{i}' in decoder_params_np
        add_dense_layer_params(initializers, decoder_params_np, "dec", layer_key, i, has_ln)

    # Build computation graph
    nodes = []
    node_idx = 0

    def make_node(op_type, inputs, outputs, name=None, **attrs):
        nonlocal node_idx
        if name is None:
            name = f"node_{node_idx}"
        node_idx += 1
        return helper.make_node(op_type, inputs, outputs, name=name, **attrs)

    # 1. Normalize proprio_obs: (proprio_obs - mean) / std
    nodes.append(make_node("Sub", ["proprio_obs", "proprio_norm_mean"], ["proprio_centered"]))
    nodes.append(make_node("Div", ["proprio_centered", "proprio_norm_std"], ["proprio_normalized"]))

    # 2. Concat [latent, proprio_normalized]
    nodes.append(make_node("Concat", ["latent", "proprio_normalized"], ["decoder_input"], axis=1))

    # 3. Decoder forward pass (dynamic layer count)
    x = "decoder_input"
    for i, layer_key in enumerate(decoder_layer_names):
        is_last_layer = (i == len(decoder_layer_names) - 1)
        has_ln = not is_last_layer and f'LayerNorm_{i}' in decoder_params_np
        x = add_mlp_block(nodes, initializers, "dec", layer_key, i, x, has_ln, make_node)

    # Final output
    nodes.append(make_node("Identity", [x], ["action_logits"]))

    # Define inputs/outputs (using dims for sizes)
    inputs = [
        helper.make_tensor_value_info("latent", TensorProto.FLOAT, [None, dims.latent_size]),
        helper.make_tensor_value_info("proprio_obs", TensorProto.FLOAT, [None, dims.proprio_obs_size]),
    ]
    outputs = [helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, [None, dims.output_size])]

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


def build_onnx_model(normalizer_params, encoder_params, decoder_params, dims: NetworkDims):
    """Build ONNX model graph manually from checkpoint weights.

    Args:
        normalizer_params: RunningStatisticsState for observation normalization
        encoder_params: Dict containing encoder Dense and LayerNorm params
        decoder_params: Dict containing decoder Dense and LayerNorm params
        dims: NetworkDims with all dimension info

    Returns:
        ONNX ModelProto
    """
    # Convert params to numpy
    norm_mean, norm_std = get_normalizer_arrays(normalizer_params)
    encoder_params_np = jax.tree.map(lambda x: np.array(x).astype(np.float32), encoder_params)
    decoder_params_np = jax.tree.map(lambda x: np.array(x).astype(np.float32), decoder_params)

    # Build list of initializers (constants in the graph)
    initializers = [
        numpy_helper.from_array(norm_mean, "norm_mean"),
        numpy_helper.from_array(norm_std + 1e-8, "norm_std"),  # Add epsilon here
        numpy_helper.from_array(np.array([dims.reference_obs_size], dtype=np.int64), "ref_obs_size"),
    ]

    # Add encoder params as initializers (dynamic layer count)
    encoder_layer_names = dims.encoder_layer_names()
    for i, layer_key in enumerate(encoder_layer_names):
        # All encoder layers have LayerNorm
        add_dense_layer_params(initializers, encoder_params_np, "enc", layer_key, i, has_layer_norm=True)

    # Add latent mean projection layer
    initializers.append(numpy_helper.from_array(encoder_params_np['fc2_mean']['kernel'], "enc_mean_w"))
    initializers.append(numpy_helper.from_array(encoder_params_np['fc2_mean']['bias'], "enc_mean_b"))

    # Add decoder params as initializers (dynamic layer count)
    decoder_layer_names = dims.decoder_layer_names()
    for i, layer_key in enumerate(decoder_layer_names):
        is_last_layer = (i == len(decoder_layer_names) - 1)
        has_ln = not is_last_layer and f'LayerNorm_{i}' in decoder_params_np
        add_dense_layer_params(initializers, decoder_params_np, "dec", layer_key, i, has_ln)

    # Build computation graph
    nodes = []
    node_idx = 0

    def make_node(op_type, inputs, outputs, name=None, **attrs):
        nonlocal node_idx
        if name is None:
            name = f"node_{node_idx}"
        node_idx += 1
        return helper.make_node(op_type, inputs, outputs, name=name, **attrs)

    # 1. Normalize: (obs - mean) / std
    nodes.append(make_node("Sub", ["obs", "norm_mean"], ["obs_centered"]))
    nodes.append(make_node("Div", ["obs_centered", "norm_std"], ["obs_normalized"]))

    # 2. Split into reference and proprioceptive
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

    # Add slice constants (using dims for sizes)
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "starts_ref"))
    initializers.append(numpy_helper.from_array(np.array([dims.reference_obs_size], dtype=np.int64), "ends_ref"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_ref"))
    initializers.append(numpy_helper.from_array(np.array([dims.reference_obs_size], dtype=np.int64), "starts_prop"))
    initializers.append(numpy_helper.from_array(np.array([dims.obs_size], dtype=np.int64), "ends_prop"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_prop"))

    # 3. Encoder forward pass (dynamic layer count, all have LayerNorm)
    x = "ref_obs"
    for i, layer_key in enumerate(encoder_layer_names):
        x = add_mlp_block(nodes, initializers, "enc", layer_key, i, x, has_layer_norm=True, make_node=make_node)

    # Get latent mean
    nodes.append(make_node("MatMul", [x, "enc_mean_w"], ["enc_mean_mm"]))
    nodes.append(make_node("Add", ["enc_mean_mm", "enc_mean_b"], ["latent_mean"]))

    # 4. Decoder forward pass (dynamic layer count)
    nodes.append(make_node("Concat", ["latent_mean", "proprio_obs"], ["decoder_input"], axis=1))

    x = "decoder_input"
    for i, layer_key in enumerate(decoder_layer_names):
        is_last_layer = (i == len(decoder_layer_names) - 1)
        has_ln = not is_last_layer and f'LayerNorm_{i}' in decoder_params_np
        x = add_mlp_block(nodes, initializers, "dec", layer_key, i, x, has_ln, make_node)

    # Final output
    nodes.append(make_node("Identity", [x], ["action_logits"]))

    # Define input/output (using dims for all sizes)
    inputs = [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [None, dims.obs_size])]
    outputs = [
        helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, [None, dims.output_size]),
        helper.make_tensor_value_info("latent_mean", TensorProto.FLOAT, [None, dims.latent_size]),
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

    # Validate checkpoint structure before processing
    print("Validating checkpoint structure...")
    validate_checkpoint(ckpt)

    cfg = ckpt["cfg"]
    policy_params = ckpt["policy"]

    # Unpack policy params: (normalizer_state, network_params)
    normalizer_params, network_params = policy_params

    # Extract all dimensions from config
    dims = NetworkDims.from_config(cfg)
    print(f"Observation size: {dims.obs_size}")
    print(f"Reference obs size: {dims.reference_obs_size}")
    print(f"Proprioceptive obs size: {dims.proprio_obs_size}")
    print(f"Action size: {dims.action_size}")
    print(f"Latent size: {dims.latent_size}")
    print(f"Encoder layers: {dims.encoder_layer_sizes}")
    print(f"Decoder layers: {dims.decoder_layer_sizes}")

    # Extract encoder and decoder params
    encoder_params, decoder_params = extract_encoder_decoder_params(network_params)

    # Build full ONNX model (encoder + decoder)
    print("Building full ONNX model...")
    model = build_onnx_model(normalizer_params, encoder_params, decoder_params, dims)

    # Check model
    print("Checking full ONNX model...")
    onnx.checker.check_model(model)

    # Save model
    print(f"Saving full ONNX model to: {output_path}")
    onnx.save(model, output_path)

    # Build decoder-only ONNX model (for latent space random walk mode)
    print("Building decoder-only ONNX model...")
    decoder_model = build_decoder_only_onnx_model(normalizer_params, decoder_params, dims)

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

    # Test input (flat for ONNX)
    test_input = np.zeros((1, dims.obs_size), dtype=np.float32)

    ort_output = session.run([output_name], {input_name: test_input})[0]
    print(f"ONNX Runtime output shape: {ort_output.shape}")

    # Compare with original Flax module
    print("Comparing with original Flax module...")
    ppo_network = checkpointing.make_ppo_network_from_cfg(cfg)
    dummy_key = jax.random.PRNGKey(0)

    test_input_dict = {
        'imitation_target': jnp.zeros((1, dims.reference_obs_size)),
        'proprioception': jnp.zeros((1, dims.proprio_obs_size)),
    }
    flax_output, _, _ = ppo_network.policy_network.apply(
        normalizer_params, network_params, test_input_dict, dummy_key,
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

    test_latent = np.zeros((1, dims.latent_size), dtype=np.float32)
    test_proprio = np.zeros((1, dims.proprio_obs_size), dtype=np.float32)

    decoder_output = decoder_session.run(
        [decoder_session.get_outputs()[0].name],
        {"latent": test_latent, "proprio_obs": test_proprio}
    )[0]
    print(f"Decoder-only output shape: {decoder_output.shape}")
    print(f"Decoder-only model verified successfully!")

    # Save normalization parameters
    norm_params_path = Path(output_path).with_suffix(".norm.npz")
    norm_mean, norm_std = get_normalizer_arrays(normalizer_params)
    np.savez(
        norm_params_path,
        mean=norm_mean,
        std=norm_std,
        reference_obs_size=dims.reference_obs_size,
        proprioceptive_obs_size=dims.proprio_obs_size,
    )
    print(f"Saved normalization parameters to: {norm_params_path}")

    # Save network metadata as JSON for frontend
    import json
    metadata = {
        "observation_size": dims.obs_size,
        "reference_obs_size": dims.reference_obs_size,
        "proprioceptive_obs_size": dims.proprio_obs_size,
        "action_size": dims.action_size,
        "latent_size": dims.latent_size,
        "encoder_layer_sizes": dims.encoder_layer_sizes,
        "decoder_layer_sizes": dims.decoder_layer_sizes,
    }
    metadata_path = Path(output_path).parent / "network_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved network metadata to: {metadata_path}")

    print("Done!")


# =============================================================================
# High-Level Policy Conversion (for joystick mode)
# =============================================================================

@dataclass
class HighLevelNetworkDims:
    """Network dimensions for high-level policy."""
    task_obs_size: int
    latent_size: int
    hidden_layer_sizes: list[int]

    @property
    def num_hidden_layers(self) -> int:
        return len(self.hidden_layer_sizes)


def build_highlevel_onnx_model(
    normalizer_state: dict,
    policy_params: dict,
    dims: HighLevelNetworkDims,
):
    """Build ONNX model for high-level joystick policy.

    The high-level policy is a simple MLP that maps task observations to latent
    space. It has embedded normalization (mean/std from running statistics).

    Architecture:
        task_obs -> normalize -> hidden layers (SiLU) -> latent

    Args:
        normalizer_state: Dict with 'mean', 'std' arrays (from orbax restore)
        policy_params: Dict with 'params' containing MLP layer weights
        dims: HighLevelNetworkDims with all dimension info

    Returns:
        ONNX ModelProto
    """
    # Extract normalization statistics (dict access since we restore as numpy)
    norm_mean = np.array(normalizer_state['mean']).astype(np.float32)
    norm_std = np.array(normalizer_state['std']).astype(np.float32) + 1e-8

    # Convert policy params to numpy
    params = policy_params['params']
    params_np = jax.tree.map(lambda x: np.array(x).astype(np.float32), params)

    # Build initializers
    initializers = [
        numpy_helper.from_array(norm_mean, "norm_mean"),
        numpy_helper.from_array(norm_std, "norm_std"),
    ]

    # Add hidden layer weights
    # Brax PPO uses naming: hidden_0, hidden_1, ... for hidden layers
    # Final layer for latent output
    for i in range(dims.num_hidden_layers):
        layer_key = f'hidden_{i}'
        layer = params_np[layer_key]
        initializers.append(numpy_helper.from_array(layer['kernel'], f"hidden_{i}_w"))
        initializers.append(numpy_helper.from_array(layer['bias'], f"hidden_{i}_b"))

    # Output layer (maps to latent)
    # In brax PPO, this is typically named based on num_hidden_layers
    output_layer_key = f'hidden_{dims.num_hidden_layers}'
    output_layer = params_np[output_layer_key]
    initializers.append(numpy_helper.from_array(output_layer['kernel'], "output_w"))
    initializers.append(numpy_helper.from_array(output_layer['bias'], "output_b"))

    # Build computation graph
    nodes = []
    node_idx = 0

    def make_node(op_type, inputs, outputs, name=None, **attrs):
        nonlocal node_idx
        if name is None:
            name = f"node_{node_idx}"
        node_idx += 1
        return helper.make_node(op_type, inputs, outputs, name=name, **attrs)

    # 1. Normalize: (task_obs - mean) / std
    nodes.append(make_node("Sub", ["task_obs", "norm_mean"], ["obs_centered"]))
    nodes.append(make_node("Div", ["obs_centered", "norm_std"], ["obs_normalized"]))

    # 2. Hidden layers with SiLU activation
    x = "obs_normalized"
    for i in range(dims.num_hidden_layers):
        # Dense: x = x @ W + b
        nodes.append(make_node("MatMul", [x, f"hidden_{i}_w"], [f"hidden_{i}_mm"]))
        nodes.append(make_node("Add", [f"hidden_{i}_mm", f"hidden_{i}_b"], [f"hidden_{i}_pre"]))
        # SiLU: x * sigmoid(x)
        nodes.append(make_node("Sigmoid", [f"hidden_{i}_pre"], [f"hidden_{i}_sig"]))
        nodes.append(make_node("Mul", [f"hidden_{i}_pre", f"hidden_{i}_sig"], [f"hidden_{i}_out"]))
        x = f"hidden_{i}_out"

    # 3. Output layer (outputs mean + log_std for Gaussian)
    nodes.append(make_node("MatMul", [x, "output_w"], ["output_mm"]))
    nodes.append(make_node("Add", ["output_mm", "output_b"], ["action_logits"]))

    # 4. Extract mean (first half) and apply tanh to get latent
    # For NormalTanhDistribution, mode = tanh(mean)
    actual_latent_size = dims.latent_size // 2  # Output is mean + log_std
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "slice_start"))
    initializers.append(numpy_helper.from_array(np.array([actual_latent_size], dtype=np.int64), "slice_end"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "slice_axis"))

    nodes.append(helper.make_node(
        "Slice",
        inputs=["action_logits", "slice_start", "slice_end", "slice_axis"],
        outputs=["action_mean"],
        name="slice_mean"
    ))
    nodes.append(make_node("Tanh", ["action_mean"], ["latent"]))

    # Define inputs/outputs
    inputs = [
        helper.make_tensor_value_info("task_obs", TensorProto.FLOAT, [None, dims.task_obs_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("latent", TensorProto.FLOAT, [None, actual_latent_size]),
    ]

    # Create graph
    graph = helper.make_graph(
        nodes,
        "highlevel_policy",
        inputs,
        outputs,
        initializers,
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    return model


def find_latest_checkpoint_step(checkpoint_path: str) -> str:
    """Find the latest checkpoint step in an orbax checkpoint directory."""
    ckpt_dir = Path(checkpoint_path)
    steps = []
    for item in ckpt_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            steps.append(int(item.name))
    if not steps:
        raise ValueError(f"No checkpoint steps found in {checkpoint_path}")
    return str(max(steps))


def load_highlevel_checkpoint(checkpoint_path: str):
    """Load high-level brax PPO checkpoint with orbax.

    Handles the cuda:0 sharding issue by restoring as numpy arrays.

    Args:
        checkpoint_path: Path to orbax checkpoint step directory

    Returns:
        Tuple of (normalizer_state, policy_params, value_params) as numpy arrays
    """
    from orbax.checkpoint._src.metadata import value as value_metadata
    from orbax.checkpoint._src.serialization import type_handlers
    from orbax.checkpoint import args as ocp_args

    handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
    checkpointer = ocp.Checkpointer(handler)

    # Get metadata to know the structure
    meta = checkpointer.metadata(checkpoint_path)

    # Build restore args to restore as numpy arrays (avoids sharding issues)
    def build_restore_args(meta_tree):
        if isinstance(meta_tree, value_metadata.ArrayMetadata):
            return type_handlers.RestoreArgs(
                restore_type=np.ndarray,
                dtype=meta_tree.dtype,
            )
        elif isinstance(meta_tree, dict):
            return {k: build_restore_args(v) for k, v in meta_tree.items()}
        elif isinstance(meta_tree, (list, tuple)):
            return type(meta_tree)(build_restore_args(v) for v in meta_tree)
        else:
            return meta_tree

    restore_args = build_restore_args(meta.item_metadata.tree)
    params = checkpointer.restore(checkpoint_path, args=ocp_args.PyTreeRestore(item=restore_args))

    return params[0], params[1], params[2]


def convert_highlevel_to_onnx(checkpoint_path: str, output_path: str, latent_size: int | None = None):
    """Convert high-level brax PPO checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to orbax checkpoint directory
        output_path: Path for output ONNX file
        latent_size: Size of latent output (auto-detected if None)
    """
    import json

    print(f"Loading high-level checkpoint from: {checkpoint_path}")

    # Find latest step
    step = find_latest_checkpoint_step(checkpoint_path)
    ckpt_step_path = str(Path(checkpoint_path) / step)
    print(f"Using checkpoint step: {step}")

    # Load orbax checkpoint as numpy arrays
    normalizer_state, policy_params, value_params = load_highlevel_checkpoint(ckpt_step_path)
    del value_params  # Not needed for inference

    print(f"Normalizer mean shape: {normalizer_state['mean'].shape}")
    print(f"Normalizer std shape: {normalizer_state['std'].shape}")

    # Infer network dimensions from params
    policy_keys = list(policy_params['params'].keys())
    print(f"Policy param keys: {policy_keys}")

    # Count hidden layers (hidden_0, hidden_1, ..., hidden_N where N is output)
    hidden_layer_count = sum(1 for k in policy_keys if k.startswith('hidden_'))
    num_hidden_layers = hidden_layer_count - 1  # Last one is output

    # Get hidden layer sizes from weights
    hidden_layer_sizes = []
    for i in range(num_hidden_layers):
        kernel = policy_params['params'][f'hidden_{i}']['kernel']
        hidden_layer_sizes.append(kernel.shape[1])

    task_obs_size = int(normalizer_state['mean'].shape[0])

    # Auto-detect latent size from final layer output
    final_kernel = policy_params['params'][f'hidden_{num_hidden_layers}']['kernel']
    detected_latent_size = final_kernel.shape[1]
    if latent_size is not None and latent_size != detected_latent_size:
        print(f"Warning: specified latent_size={latent_size} differs from detected={detected_latent_size}")
        print(f"Using detected latent size: {detected_latent_size}")
    latent_size = detected_latent_size

    dims = HighLevelNetworkDims(
        task_obs_size=task_obs_size,
        latent_size=latent_size,
        hidden_layer_sizes=hidden_layer_sizes,
    )
    print(f"Task obs size: {dims.task_obs_size}")
    print(f"Network output size: {dims.latent_size} (mean + log_std)")
    print(f"Actual latent size: {dims.latent_size // 2} (after extracting mean and tanh)")
    print(f"Hidden layer sizes: {dims.hidden_layer_sizes}")

    # Build ONNX model
    print("Building high-level ONNX model...")
    model = build_highlevel_onnx_model(normalizer_state, policy_params, dims)

    # Check model
    print("Checking ONNX model...")
    onnx.checker.check_model(model)

    # Save model
    print(f"Saving ONNX model to: {output_path}")
    onnx.save(model, output_path)

    # Verify with ONNX Runtime
    print("Verifying with ONNX Runtime...")
    import onnxruntime as ort

    session = ort.InferenceSession(output_path)
    print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"  Outputs: {[o.name for o in session.get_outputs()]}")

    test_input = np.zeros((1, dims.task_obs_size), dtype=np.float32)
    ort_output = session.run(["latent"], {"task_obs": test_input})[0]
    print(f"  Output shape: {ort_output.shape}")
    print(f"  ONNX Runtime verification passed!")

    # Update network_metadata.json if it exists
    # Note: latent_size is the actual output size (half of network output due to mean+logstd)
    actual_latent_size = dims.latent_size // 2
    metadata_path = Path(output_path).parent / "network_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["highlevel_task_obs_size"] = dims.task_obs_size
        metadata["highlevel_hidden_layer_sizes"] = dims.hidden_layer_sizes
        metadata["highlevel_latent_size"] = actual_latent_size
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated network metadata at: {metadata_path}")
    else:
        print(f"Note: {metadata_path} not found, skipping metadata update")

    # Log checkpoint provenance (same format as update_checkpoint.sh)
    from datetime import datetime
    log_path = Path(output_path).parent / "checkpoint.log"
    checkpoint_name = Path(checkpoint_path).name
    timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
    log_entry = f"{timestamp} | highlevel: {checkpoint_name} | {checkpoint_path}\n"
    with open(log_path, "a") as f:
        f.write(log_entry)
    print(f"Logged checkpoint provenance to: {log_path}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Convert Flax checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to MIMIC tracking checkpoint directory",
    )
    parser.add_argument(
        "--highlevel-checkpoint",
        type=str,
        default=None,
        help="Path to high-level (joystick) brax PPO checkpoint directory",
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
    parser.add_argument(
        "--latent-size",
        type=int,
        default=16,
        help="Latent size for high-level policy (default: 16)",
    )

    args = parser.parse_args()

    # Validate args
    if args.checkpoint is None and args.highlevel_checkpoint is None:
        parser.error("Must specify either --checkpoint or --highlevel-checkpoint")
    if args.checkpoint is not None and args.highlevel_checkpoint is not None:
        parser.error("Cannot specify both --checkpoint and --highlevel-checkpoint")

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        convert_to_onnx(args.checkpoint, str(output_path), args.step)
    else:
        convert_highlevel_to_onnx(args.highlevel_checkpoint, str(output_path), args.latent_size)


if __name__ == "__main__":
    main()
