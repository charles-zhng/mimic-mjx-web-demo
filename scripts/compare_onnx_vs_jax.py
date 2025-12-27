#!/usr/bin/env python3
"""Compare ONNX inference with original JAX model inference."""

import sys
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jp
import mujoco

# Add paths
TRACK_MJX_PATH = Path(__file__).parent.parent.parent / "track-mjx"
VNL_PATH = Path(__file__).parent.parent.parent / "vnl-playground"
sys.path.insert(0, str(TRACK_MJX_PATH))
sys.path.insert(0, str(VNL_PATH))

from track_mjx.agent import checkpointing
from brax import math as brax_math
from brax.training.agents.ppo import networks as ppo_networks


def build_observation(mj_model, mj_data, clip_qpos, clip_xpos, current_frame, prev_action):
    """Build observation matching CORRECTED TypeScript implementation.

    Layout:
    - Reference (640): all roots(15), all quats(20), all joints(335), all bodies(270)
      - Bodies are (18 bodies, 5 frames) not (5 frames, 18 bodies)
    - Proprioception (277): joints, vels, forces, height, zaxis, appendages,
                            kinematic(9), touch(4), prev_action(38)
    """
    BODY_INDICES_REF = [2, 9, 10, 11, 12, 14, 15, 16, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66]
    BODY_INDICES_MODEL = [i + 1 for i in BODY_INDICES_REF]
    END_EFFECTOR_INDICES_MODEL = [65, 60, 17, 13, 56]
    TORSO_INDEX_MODEL = 3
    WALKER_INDEX_MODEL = 2
    REFERENCE_LENGTH = 5
    NUM_JOINTS = 67

    obs = []
    root_pos = mj_data.qpos[:3]
    root_quat = mj_data.qpos[3:7]
    cur_body_xpos = [mj_data.xpos[idx].copy() for idx in BODY_INDICES_MODEL]

    num_frames = len(clip_qpos)

    # === Reference Observation (640 dims) ===
    # All roots first (5 × 3 = 15)
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        ref_root_pos = ref_qpos[:3]
        rel_pos = ref_root_pos - root_pos
        ego_pos = brax_math.rotate(jp.array(rel_pos), jp.array(root_quat))
        obs.extend(np.array(ego_pos).tolist())

    # All quats (5 × 4 = 20)
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        ref_quat = ref_qpos[3:7]
        rel_quat = brax_math.relative_quat(jp.array(ref_quat), jp.array(root_quat))
        obs.extend(np.array(rel_quat).tolist())

    # All joints (5 × 67 = 335)
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        for j in range(NUM_JOINTS):
            obs.append(ref_qpos[7 + j] - mj_data.qpos[7 + j])

    # All bodies (18 bodies × 5 frames × 3 = 270) - bodies outer loop!
    for b, ref_body_idx in enumerate(BODY_INDICES_REF):
        for f in range(REFERENCE_LENGTH):
            frame_idx = min(current_frame + 1 + f, num_frames - 1)
            ref_xpos = np.array(clip_xpos[frame_idx])
            ref_body_pos = np.array(ref_xpos[ref_body_idx])
            cur_body_pos = cur_body_xpos[b]
            rel_body_pos = ref_body_pos - cur_body_pos
            ego_body_pos = brax_math.rotate(jp.array(rel_body_pos), jp.array(root_quat))
            obs.extend(np.array(ego_body_pos).tolist())

    # === Proprioception (277 dims) ===
    # Joint angles (67)
    for i in range(7, mj_model.nq):
        obs.append(mj_data.qpos[i])
    # Joint velocities (67)
    for i in range(6, mj_model.nv):
        obs.append(mj_data.qvel[i])
    # qfrc_actuator (73)
    for i in range(mj_model.nv):
        obs.append(mj_data.qfrc_actuator[i])
    # Body height (1)
    obs.append(mj_data.xpos[TORSO_INDEX_MODEL, 2])
    # World z-axis (3)
    walker_xmat = mj_data.xmat[WALKER_INDEX_MODEL].reshape(3, 3)
    obs.extend(walker_xmat.flatten()[6:9].tolist())
    # Appendages (15)
    torso_pos = mj_data.xpos[TORSO_INDEX_MODEL]
    torso_mat = mj_data.xmat[TORSO_INDEX_MODEL].reshape(3, 3)
    for body_idx in END_EFFECTOR_INDICES_MODEL:
        body_pos = mj_data.xpos[body_idx]
        rel = body_pos - torso_pos
        ego = rel @ torso_mat
        obs.extend(ego.tolist())
    # Kinematic sensors (9) + Touch sensors (4) - BEFORE prev_action
    obs.extend(mj_data.sensordata[:13].tolist())
    # Prev action (38) - LAST
    obs.extend(prev_action.tolist())

    return np.array(obs, dtype=np.float32)


def main():
    print("=" * 60)
    print("ONNX vs JAX INFERENCE COMPARISON")
    print("=" * 60)

    # Load checkpoint
    checkpoint_path = str(TRACK_MJX_PATH / "model_checkpoints" / "251224_200417_996678")
    ckpt = checkpointing.load_checkpoint_for_eval(checkpoint_path)
    cfg = ckpt["cfg"]
    normalizer_state, policy_params = ckpt["policy"]

    print(f"\n1. CHECKPOINT INFO")
    print(f"   Path: {checkpoint_path}")
    print(f"   Obs normalizer mean shape: {normalizer_state.mean.shape}")
    print(f"   Obs normalizer std shape: {normalizer_state.std.shape}")

    # Create inference function
    inference_fn = checkpointing.load_inference_fn(cfg, ckpt["policy"], deterministic=True)

    # Load web demo model and clips
    mj_model = mujoco.MjModel.from_xml_path(
        '/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/models/rodent.xml'
    )
    mj_data = mujoco.MjData(mj_model)

    with open('/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/motions/clips.json', 'r') as f:
        web_clips = json.load(f)
    clip = web_clips['clips'][0]

    # Set initial state
    ref_qpos = np.array(clip['qpos'][0])
    mj_data.qpos[:len(ref_qpos)] = ref_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # Build observation
    prev_action = np.zeros(38, dtype=np.float32)
    obs = build_observation(mj_model, mj_data, clip['qpos'], clip['xpos'], 0, prev_action)

    print(f"\n2. OBSERVATION")
    print(f"   Shape: {obs.shape}")
    print(f"   Range: [{obs.min():.4f}, {obs.max():.4f}]")

    # JAX inference
    print(f"\n3. JAX INFERENCE")

    # The inference_fn takes raw observation and normalizes internally
    jax_obs = jp.array(obs).reshape(1, -1)
    jax_action, _ = inference_fn(jax_obs, jax.random.PRNGKey(0))
    jax_action = np.array(jax_action[0])

    # Also show normalized observation for debugging
    obs_normalized = (obs - np.array(normalizer_state.mean)) / np.array(normalizer_state.std)
    print(f"   Normalized obs range: [{obs_normalized.min():.4f}, {obs_normalized.max():.4f}]")

    print(f"   JAX action shape: {jax_action.shape}")
    print(f"   JAX action range: [{jax_action.min():.4f}, {jax_action.max():.4f}]")
    print(f"   JAX first 5 actions: {jax_action[:5]}")

    # ONNX inference
    print(f"\n4. ONNX INFERENCE")
    import onnxruntime as ort

    session = ort.InferenceSession(
        '/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/nn/intention_network.onnx'
    )

    # ONNX model has normalization built-in, so pass raw observation
    onnx_input = obs.reshape(1, -1)
    outputs = session.run(None, {'obs': onnx_input})
    logits = outputs[0][0]
    onnx_action = np.tanh(logits[:38])

    print(f"   ONNX logits shape: {logits.shape}")
    print(f"   ONNX action range: [{onnx_action.min():.4f}, {onnx_action.max():.4f}]")
    print(f"   ONNX first 5 actions: {onnx_action[:5]}")

    # Compare
    print(f"\n5. COMPARISON")
    diff = np.abs(jax_action - onnx_action)
    print(f"   Max action diff: {diff.max():.6f}")
    print(f"   Mean action diff: {diff.mean():.6f}")

    if diff.max() > 0.01:
        worst_idx = np.argmax(diff)
        print(f"   WARNING: Significant difference at index {worst_idx}")
        print(f"     JAX: {jax_action[worst_idx]:.6f}")
        print(f"     ONNX: {onnx_action[worst_idx]:.6f}")

        # Check normalization
        print(f"\n6. NORMALIZATION CHECK")
        onnx_model = __import__('onnx').load('/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/nn/intention_network.onnx')
        for init in onnx_model.graph.initializer:
            if init.name == 'norm_mean':
                onnx_mean = __import__('onnx').numpy_helper.to_array(init)
            if init.name == 'norm_std':
                onnx_std = __import__('onnx').numpy_helper.to_array(init)

        mean_diff = np.abs(np.array(normalizer_state.mean) - onnx_mean)
        std_diff = np.abs(np.array(normalizer_state.std) - onnx_std)
        print(f"   Mean diff max: {mean_diff.max():.10f}")
        print(f"   Std diff max: {std_diff.max():.10f}")
    else:
        print(f"   Actions match closely!")

    # Test multiple frames
    print(f"\n7. MULTI-FRAME COMPARISON")
    max_diffs = []
    for frame in range(10):
        obs = build_observation(mj_model, mj_data, clip['qpos'], clip['xpos'], frame, prev_action)

        # JAX
        jax_obs = jp.array(obs).reshape(1, -1)
        jax_action, _ = inference_fn(jax_obs, jax.random.PRNGKey(0))
        jax_action = np.array(jax_action[0])

        # ONNX
        outputs = session.run(None, {'obs': obs.reshape(1, -1)})
        onnx_action = np.tanh(outputs[0][0][:38])

        diff = np.abs(jax_action - onnx_action).max()
        max_diffs.append(diff)

    print(f"   Max diffs over 10 frames: {[f'{d:.6f}' for d in max_diffs]}")
    print(f"   Overall max diff: {max(max_diffs):.6f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
