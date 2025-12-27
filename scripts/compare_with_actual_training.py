#!/usr/bin/env python3
"""Compare web demo observations with ACTUAL training environment observations."""

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

from vnl_playground.tasks.rodent import imitation, consts
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips
from brax import math as brax_math
from track_mjx.agent import checkpointing


def build_web_demo_observation(mj_model, mj_data, clip_qpos, clip_xpos, current_frame, prev_action):
    """Build observation matching CORRECTED TypeScript web demo implementation."""
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

    # Root targets (5 × 3 = 15)
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        ref_root_pos = ref_qpos[:3]
        rel_pos = ref_root_pos - root_pos
        ego_pos = brax_math.rotate(jp.array(rel_pos), jp.array(root_quat))
        obs.extend(np.array(ego_pos).tolist())

    # Quat targets (5 × 4 = 20)
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        ref_quat = ref_qpos[3:7]
        rel_quat = brax_math.relative_quat(jp.array(ref_quat), jp.array(root_quat))
        obs.extend(np.array(rel_quat).tolist())

    # Joint targets (5 × 67 = 335)
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        for j in range(NUM_JOINTS):
            obs.append(ref_qpos[7 + j] - mj_data.qpos[7 + j])

    # Body targets (18 bodies × 5 frames × 3 = 270)
    for b, ref_body_idx in enumerate(BODY_INDICES_REF):
        for f in range(REFERENCE_LENGTH):
            frame_idx = min(current_frame + 1 + f, num_frames - 1)
            ref_xpos = np.array(clip_xpos[frame_idx])
            ref_body_pos = np.array(ref_xpos[ref_body_idx])
            cur_body_pos = cur_body_xpos[b]
            rel_body_pos = ref_body_pos - cur_body_pos
            ego_body_pos = brax_math.rotate(jp.array(rel_body_pos), jp.array(root_quat))
            obs.extend(np.array(ego_body_pos).tolist())

    # Proprioception
    for i in range(7, mj_model.nq):
        obs.append(mj_data.qpos[i])
    for i in range(6, mj_model.nv):
        obs.append(mj_data.qvel[i])
    for i in range(mj_model.nv):
        obs.append(mj_data.qfrc_actuator[i])
    obs.append(mj_data.xpos[TORSO_INDEX_MODEL, 2])
    walker_xmat = mj_data.xmat[WALKER_INDEX_MODEL].reshape(3, 3)
    obs.extend(walker_xmat.flatten()[6:9].tolist())
    torso_pos = mj_data.xpos[TORSO_INDEX_MODEL]
    torso_mat = mj_data.xmat[TORSO_INDEX_MODEL].reshape(3, 3)
    for body_idx in END_EFFECTOR_INDICES_MODEL:
        body_pos = mj_data.xpos[body_idx]
        rel = body_pos - torso_pos
        ego = rel @ torso_mat
        obs.extend(ego.tolist())
    # Sensors before prev_action
    obs.extend(mj_data.sensordata[:13].tolist())
    obs.extend(prev_action.tolist())

    return np.array(obs, dtype=np.float32)


def main():
    print("=" * 70)
    print("COMPARISON: Web Demo vs ACTUAL Training Environment")
    print("=" * 70)

    # Load training environment with reference clips
    print("\n1. LOADING TRAINING ENVIRONMENT")
    reference_path = TRACK_MJX_PATH / "data" / "rodent" / "rodent_reference_clips.h5"
    clips = ReferenceClips(str(reference_path), n_frames_per_clip=250, keep_clips_idx=np.array([0]))

    config = imitation.default_config()
    config.ctrl_dt = 0.01
    env = imitation.Imitation(config, clips=clips)

    print(f"   Model: nq={env.mj_model.nq}, nv={env.mj_model.nv}, nu={env.mj_model.nu}")

    # Reset environment to clip 0, frame 0
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng, clip_idx=0, start_frame=0)

    # Get flattened observation from training env
    training_obs, _ = jax.flatten_util.ravel_pytree(state.obs)
    training_obs = np.array(training_obs)

    print(f"   Training obs shape: {training_obs.shape}")
    print(f"   Training obs range: [{training_obs.min():.4f}, {training_obs.max():.4f}]")

    # Load web demo clips
    print("\n2. LOADING WEB DEMO CLIPS")
    with open('/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/motions/clips.json', 'r') as f:
        web_clips = json.load(f)
    web_clip = web_clips['clips'][0]
    print(f"   Web clip name: {web_clip['name']}")
    print(f"   Web clip frames: {web_clip['num_frames']}")

    # Use training env's MuJoCo model for consistency
    mj_data = mujoco.MjData(env.mj_model)

    # Set initial state from web clip
    ref_qpos = np.array(web_clip['qpos'][0])
    mj_data.qpos[:len(ref_qpos)] = ref_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(env.mj_model, mj_data)

    # Build web demo observation
    prev_action = np.zeros(38, dtype=np.float32)
    web_obs = build_web_demo_observation(
        env.mj_model, mj_data, web_clip['qpos'], web_clip['xpos'], 0, prev_action
    )

    print(f"   Web demo obs shape: {web_obs.shape}")
    print(f"   Web demo obs range: [{web_obs.min():.4f}, {web_obs.max():.4f}]")

    # Compare observations
    print("\n3. OBSERVATION COMPARISON")

    # First, check if clips match
    training_clip_qpos = clips.qpos[0, 0]  # First clip, first frame
    web_clip_qpos = np.array(web_clip['qpos'][0])

    qpos_diff = np.abs(training_clip_qpos[:len(web_clip_qpos)] - web_clip_qpos).max()
    print(f"   Clip qpos diff (frame 0): {qpos_diff:.6f}")

    if qpos_diff > 0.001:
        print(f"   WARNING: Clips don't match! Using same clip for fair comparison...")
        # Reset with training clip data instead
        mj_data.qpos[:] = training_clip_qpos
        mj_data.qvel[:] = 0
        mujoco.mj_forward(env.mj_model, mj_data)

        # Convert training clips to web format
        train_clip_qpos = [np.array(clips.qpos[0, f]).tolist() for f in range(clips.qpos.shape[1])]
        train_clip_xpos = [np.array(clips.xpos[0, f]).tolist() for f in range(clips.xpos.shape[1])]

        web_obs = build_web_demo_observation(
            env.mj_model, mj_data, train_clip_qpos, train_clip_xpos, 0, prev_action
        )

    # Detailed section comparison
    sections = [
        ("root_targets", 0, 15),
        ("quat_targets", 15, 35),
        ("joint_targets", 35, 370),
        ("body_targets", 370, 640),
        ("joint_angles", 640, 707),
        ("joint_vels", 707, 774),
        ("actuator_ctrl", 774, 847),
        ("body_height", 847, 848),
        ("world_zaxis", 848, 851),
        ("appendages", 851, 866),
        ("kinematic_sensors", 866, 875),
        ("touch_sensors", 875, 879),
        ("prev_action", 879, 917),
    ]

    print(f"\n   Section-by-section comparison:")
    total_diff = 0
    for name, start, end in sections:
        train_section = training_obs[start:end]
        web_section = web_obs[start:end]
        diff = np.abs(train_section - web_section)
        max_diff = diff.max()
        mean_diff = diff.mean()
        total_diff += diff.sum()
        status = "✓" if max_diff < 0.001 else "✗"
        print(f"   {status} {name:20s} [{start:3d}:{end:3d}]: max={max_diff:.6f}, mean={mean_diff:.6f}")
        if max_diff >= 0.001:
            worst_idx = np.argmax(diff)
            print(f"      Worst at {start + worst_idx}: train={train_section[worst_idx]:.6f}, web={web_section[worst_idx]:.6f}")

    overall_max = np.abs(training_obs - web_obs).max()
    print(f"\n   Overall max diff: {overall_max:.6f}")
    print(f"   Overall mean diff: {np.abs(training_obs - web_obs).mean():.6f}")

    # ONNX comparison
    print("\n4. ONNX INFERENCE COMPARISON")
    import onnxruntime as ort

    session = ort.InferenceSession(
        '/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/nn/intention_network.onnx'
    )

    # ONNX with training obs
    train_outputs = session.run(None, {'obs': training_obs.reshape(1, -1).astype(np.float32)})
    train_action = np.tanh(train_outputs[0][0][:38])

    # ONNX with web obs
    web_outputs = session.run(None, {'obs': web_obs.reshape(1, -1)})
    web_action = np.tanh(web_outputs[0][0][:38])

    action_diff = np.abs(train_action - web_action)
    print(f"   Train action range: [{train_action.min():.4f}, {train_action.max():.4f}]")
    print(f"   Web action range: [{web_action.min():.4f}, {web_action.max():.4f}]")
    print(f"   Action diff max: {action_diff.max():.6f}")
    print(f"   Action diff mean: {action_diff.mean():.6f}")

    # JAX inference comparison
    print("\n5. JAX INFERENCE COMPARISON")
    checkpoint_path = str(TRACK_MJX_PATH / "model_checkpoints" / "251224_200417_996678")
    ckpt = checkpointing.load_checkpoint_for_eval(checkpoint_path)
    inference_fn = checkpointing.load_inference_fn(ckpt["cfg"], ckpt["policy"], deterministic=True)

    jax_obs = jp.array(training_obs).reshape(1, -1)
    jax_action, _ = inference_fn(jax_obs, jax.random.PRNGKey(0))
    jax_action = np.array(jax_action[0])

    jax_onnx_diff = np.abs(jax_action - train_action)
    print(f"   JAX action range: [{jax_action.min():.4f}, {jax_action.max():.4f}]")
    print(f"   JAX vs ONNX diff max: {jax_onnx_diff.max():.6f}")

    # Summary
    print("\n" + "=" * 70)
    if overall_max < 0.001:
        print("SUCCESS: Observations match between web demo and training environment!")
    else:
        print("WARNING: Observations differ. Check sections marked with ✗ above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
