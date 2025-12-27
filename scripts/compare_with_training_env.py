#!/usr/bin/env python3
"""Compare web demo observation building with actual training environment."""

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

from vnl_playground.tasks.rodent import imitation
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips
from omegaconf import OmegaConf
from brax import math as brax_math


def load_training_env():
    """Load the actual training environment."""
    # Load config
    config_path = TRACK_MJX_PATH / "configs" / "imitation.yaml"
    config = OmegaConf.load(config_path)

    # Create environment
    env = imitation.Imitation(config.env)
    return env, config


def build_web_demo_observation(mj_model, mj_data, clip_qpos, clip_xpos, current_frame, prev_action):
    """Build observation matching the CORRECTED TypeScript web demo implementation.

    Key layout differences from naive per-frame iteration:
    1. Reference observation: all roots, all quats, all joints, then all bodies
       - Bodies iterate as (18 bodies, 5 frames) not (5 frames, 18 bodies)
    2. Proprioception: sensors come BEFORE prev_action
    """

    # Constants matching observation.ts
    BODY_INDICES_REF = [2, 9, 10, 11, 12, 14, 15, 16, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66]
    BODY_INDICES_MODEL = [i + 1 for i in BODY_INDICES_REF]
    END_EFFECTOR_INDICES_MODEL = [65, 60, 17, 13, 56]
    TORSO_INDEX_MODEL = 3
    WALKER_INDEX_MODEL = 2
    REFERENCE_LENGTH = 5
    NUM_JOINTS = 67

    obs = []

    # Root from qpos (matching TypeScript)
    root_pos = mj_data.qpos[:3]
    root_quat = mj_data.qpos[3:7]

    # Current body positions
    cur_body_xpos = [mj_data.xpos[idx].copy() for idx in BODY_INDICES_MODEL]

    # === Reference Observation (640 dims) ===
    # Training layout: all roots (15), all quats (20), all joints (335), all bodies (270)
    num_frames = len(clip_qpos)

    # Root targets: 5 frames × 3 = 15
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        ref_root_pos = ref_qpos[:3]
        rel_pos = ref_root_pos - root_pos
        ego_pos = brax_math.rotate(jp.array(rel_pos), jp.array(root_quat))
        obs.extend(np.array(ego_pos).tolist())

    # Quat targets: 5 frames × 4 = 20
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        ref_quat = ref_qpos[3:7]
        rel_quat = brax_math.relative_quat(jp.array(ref_quat), jp.array(root_quat))
        obs.extend(np.array(rel_quat).tolist())

    # Joint targets: 5 frames × 67 = 335
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        for j in range(NUM_JOINTS):
            ref_joint = ref_qpos[7 + j]
            cur_joint = mj_data.qpos[7 + j]
            obs.append(ref_joint - cur_joint)

    # Body targets: 18 bodies × 5 frames × 3 = 270 (bodies outer loop!)
    for b, ref_body_idx in enumerate(BODY_INDICES_REF):
        for f in range(REFERENCE_LENGTH):
            frame_idx = min(current_frame + 1 + f, num_frames - 1)
            ref_xpos = np.array(clip_xpos[frame_idx])
            ref_body_pos = np.array(ref_xpos[ref_body_idx])
            cur_body_pos = cur_body_xpos[b]
            rel_body_pos = ref_body_pos - cur_body_pos
            ego_body_pos = brax_math.rotate(jp.array(rel_body_pos), jp.array(root_quat))
            obs.extend(np.array(ego_body_pos).tolist())

    # === Proprioceptive Observation (277 dims) ===

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

    # Sensors BEFORE prev_action (training order)
    # Kinematic (9) + touch (4)
    obs.extend(mj_data.sensordata[:13].tolist())

    # Prev action (38) - LAST
    obs.extend(prev_action.tolist())

    return np.array(obs, dtype=np.float32)


def main():
    print("=" * 60)
    print("COMPREHENSIVE COMPARISON: Web Demo vs Training Environment")
    print("=" * 60)

    # Load web demo model and clips
    mj_model = mujoco.MjModel.from_xml_path(
        '/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/models/rodent.xml'
    )
    mj_data = mujoco.MjData(mj_model)

    with open('/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/motions/clips.json', 'r') as f:
        web_clips = json.load(f)
    clip = web_clips['clips'][0]

    print(f"\n1. MODEL COMPARISON")
    print(f"   Web demo: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}, nbody={mj_model.nbody}")

    # Try to load training environment
    try:
        env, config = load_training_env()
        print(f"   Training: nq={env.mj_model.nq}, nv={env.mj_model.nv}, nu={env.mj_model.nu}, nbody={env.mj_model.nbody}")

        # Check if dimensions match
        if mj_model.nq != env.mj_model.nq:
            print(f"   WARNING: nq mismatch! Web={mj_model.nq}, Training={env.mj_model.nq}")
        if mj_model.nv != env.mj_model.nv:
            print(f"   WARNING: nv mismatch! Web={mj_model.nv}, Training={env.mj_model.nv}")
    except Exception as e:
        print(f"   Could not load training env: {e}")
        env = None

    print(f"\n2. ACTUATOR COMPARISON")
    print(f"   Web demo first 5 actuators:")
    for i in range(min(5, mj_model.nu)):
        name = mj_model.actuator(i).name
        gain = mj_model.actuator_gainprm[i][0]
        bias = mj_model.actuator_biastype[i]
        print(f"     {name}: gain={gain:.4f}, biastype={bias}")

    if env:
        print(f"   Training first 5 actuators:")
        for i in range(min(5, env.mj_model.nu)):
            name = env.mj_model.actuator(i).name
            gain = env.mj_model.actuator_gainprm[i][0]
            bias = env.mj_model.actuator_biastype[i]
            print(f"     {name}: gain={gain:.4f}, biastype={bias}")

    print(f"\n3. INITIAL STATE TEST")
    # Set initial state from clip
    ref_qpos = np.array(clip['qpos'][0])
    mj_data.qpos[:len(ref_qpos)] = ref_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    print(f"   Initial root pos: {mj_data.qpos[:3]}")
    print(f"   Initial root quat: {mj_data.qpos[3:7]}")
    print(f"   Initial torso height: {mj_data.xpos[3, 2]:.4f}")

    print(f"\n4. OBSERVATION BUILDING TEST")
    prev_action = np.zeros(38, dtype=np.float32)
    web_obs = build_web_demo_observation(
        mj_model, mj_data, clip['qpos'], clip['xpos'],
        current_frame=0, prev_action=prev_action
    )
    print(f"   Web demo observation shape: {web_obs.shape}")
    print(f"   Web demo obs range: [{web_obs.min():.4f}, {web_obs.max():.4f}]")

    # Check for NaN/Inf
    if np.any(np.isnan(web_obs)):
        nan_idx = np.where(np.isnan(web_obs))[0]
        print(f"   WARNING: NaN values at indices: {nan_idx}")
    if np.any(np.isinf(web_obs)):
        inf_idx = np.where(np.isinf(web_obs))[0]
        print(f"   WARNING: Inf values at indices: {inf_idx}")

    print(f"\n5. OBSERVATION SECTIONS")
    sections = [
        ("reference", 0, 640),
        ("joint_angles", 640, 707),
        ("joint_vels", 707, 774),
        ("qfrc_actuator", 774, 847),
        ("body_height", 847, 848),
        ("world_zaxis", 848, 851),
        ("appendages", 851, 866),
        ("kinematic_sensors", 866, 875),
        ("touch_sensors", 875, 879),
        ("prev_action", 879, 917),
    ]

    for name, start, end in sections:
        section = web_obs[start:end]
        print(f"   {name} ({start}-{end}): range=[{section.min():.4f}, {section.max():.4f}], mean={section.mean():.4f}")

    print(f"\n6. ONNX INFERENCE TEST")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            '/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/nn/intention_network.onnx'
        )

        input_tensor = web_obs.reshape(1, -1)
        outputs = session.run(None, {'obs': input_tensor})
        logits = outputs[0][0]

        print(f"   ONNX output shape: {logits.shape}")
        print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

        # Extract action (tanh of first 38)
        action = np.tanh(logits[:38])
        print(f"   Action (tanh) range: [{action.min():.4f}, {action.max():.4f}]")
        print(f"   First 5 actions: {action[:5]}")

        # Check for NaN/Inf
        if np.any(np.isnan(logits)):
            print(f"   WARNING: NaN in logits!")
        if np.any(np.isinf(logits)):
            print(f"   WARNING: Inf in logits!")

    except Exception as e:
        print(f"   ONNX test failed: {e}")

    print(f"\n7. SIMULATION STEP TEST")
    # Apply action and step
    mj_data.ctrl[:] = action

    # Step 5 times (matching training n_steps)
    for _ in range(5):
        mujoco.mj_step(mj_model, mj_data)

    print(f"   After 1 control step:")
    print(f"   Root pos: {mj_data.qpos[:3]}")
    print(f"   Root quat: {mj_data.qpos[3:7]}")
    print(f"   Torso height: {mj_data.xpos[3, 2]:.4f}")

    # Check if robot is stable
    if mj_data.xpos[3, 2] < 0.01:
        print(f"   WARNING: Robot may have collapsed (torso height < 0.01)")

    print(f"\n8. MULTI-STEP STABILITY TEST")
    # Reset and run multiple steps
    mj_data.qpos[:len(ref_qpos)] = ref_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    prev_action = np.zeros(38, dtype=np.float32)

    heights = []
    for step in range(20):
        # Build observation
        obs = build_web_demo_observation(
            mj_model, mj_data, clip['qpos'], clip['xpos'],
            current_frame=step, prev_action=prev_action
        )

        # Run inference
        outputs = session.run(None, {'obs': obs.reshape(1, -1)})
        action = np.tanh(outputs[0][0][:38])

        # Apply and step
        mj_data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(mj_model, mj_data)

        prev_action = action.astype(np.float32)
        heights.append(mj_data.xpos[3, 2])

    print(f"   Torso heights over 20 steps: {[f'{h:.3f}' for h in heights]}")

    if heights[-1] < 0.01:
        print(f"   FAILURE: Robot collapsed!")
    elif max(heights) - min(heights) > 0.1:
        print(f"   WARNING: Large height variation (unstable)")
    else:
        print(f"   Robot appears stable")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
