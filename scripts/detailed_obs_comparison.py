#!/usr/bin/env python3
"""Detailed comparison of observation structure between training env and TypeScript."""

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
from ml_collections import config_dict
from brax import math as brax_math


def main():
    print("=" * 60)
    print("DETAILED OBSERVATION STRUCTURE COMPARISON")
    print("=" * 60)

    # Load the imitation environment
    print("\n1. LOADING IMITATION ENVIRONMENT")
    config = imitation.default_config()
    config.ctrl_dt = 0.01  # Match web demo
    env = imitation.Imitation(config)
    print(f"   nq={env.mj_model.nq}, nv={env.mj_model.nv}, nu={env.mj_model.nu}")

    # Load web demo clips
    with open('/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/motions/clips.json', 'r') as f:
        web_clips = json.load(f)
    clip = web_clips['clips'][0]

    # Reset environment
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng, clip_idx=0, start_frame=0)

    print("\n2. OBSERVATION STRUCTURE FROM TRAINING ENV")
    obs = state.obs

    # Imitation target structure
    print("\n   imitation_target:")
    for key, val in obs['imitation_target'].items():
        print(f"     {key}: shape={np.array(val).shape}")

    # Proprioception structure
    print("\n   proprioception:")
    prop = env._get_proprioception(state.data, state.info, flatten=False)
    for key, val in prop.items():
        if isinstance(val, dict):
            print(f"     {key}:")
            for subkey, subval in val.items():
                print(f"       {subkey}: shape={np.array(subval).shape}")
        else:
            print(f"     {key}: shape={np.array(val).shape}")

    # Flatten and show order
    print("\n3. FLATTENED OBSERVATION ORDER (TRAINING)")
    flat_obs, _ = jax.flatten_util.ravel_pytree(obs)
    print(f"   Total size: {len(flat_obs)}")

    # Show the exact flattening order
    imitation_target = obs['imitation_target']
    prop_flat = env._get_proprioception(state.data, state.info, flatten=True)

    idx = 0
    print("\n   imitation_target flattening:")
    for key, val in imitation_target.items():
        arr = np.array(val)
        size = arr.size
        print(f"     {key}: indices [{idx}, {idx+size}) = {size} elements, shape {arr.shape}")
        idx += size

    ref_obs_size = idx
    print(f"   Reference observation total: {ref_obs_size}")

    print("\n   proprioception flattening:")
    prop_unflat = env._get_proprioception(state.data, state.info, flatten=False)
    for key, val in prop_unflat.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                arr = np.array(subval)
                size = arr.size
                print(f"     {key}.{subkey}: indices [{idx}, {idx+size}) = {size} elements")
                idx += size
        else:
            arr = np.array(val)
            size = arr.size
            print(f"     {key}: indices [{idx}, {idx+size}) = {size} elements")
            idx += size

    print(f"   Proprioception total: {idx - ref_obs_size}")
    print(f"   Grand total: {idx}")

    # Show TypeScript expected order
    print("\n4. TYPESCRIPT EXPECTED ORDER (from observation.ts)")
    print("   Reference (640 dims):")
    print("     For each of 5 frames:")
    print("       root_targets: 3")
    print("       quat_targets: 4")
    print("       joint_targets: 67")
    print("       body_targets: 18 × 3 = 54")
    print("     Total per frame: 128")
    print("     Total: 5 × 128 = 640")
    print("")
    print("   Proprioceptive (277 dims):")
    print("     joint_angles: 67 (indices 640-707)")
    print("     joint_vels: 67 (indices 707-774)")
    print("     qfrc_actuator: 73 (indices 774-847)")
    print("     body_height: 1 (index 847)")
    print("     world_zaxis: 3 (indices 848-851)")
    print("     appendages: 15 (indices 851-866)")
    print("     prev_action: 38 (indices 866-904)")  # TypeScript order
    print("     kinematic_sensors: 9 (indices 904-913)")
    print("     touch_sensors: 4 (indices 913-917)")
    print("     Total: 277")

    # Key difference analysis
    print("\n5. KEY DIFFERENCES FOUND")

    # Check body_targets ordering
    body_targets = np.array(imitation_target['body'])
    print(f"\n   body_targets shape: {body_targets.shape}")
    if body_targets.shape == (18, 5, 3):
        print("   Training flattens as: body0_frame0_xyz, body0_frame1_xyz, ..., body17_frame4_xyz")
        print("   TypeScript builds as: frame0_body0_xyz, frame0_body1_xyz, ..., frame4_body17_xyz")
        print("   MISMATCH: Need to transpose before flattening!")

    # Check proprioception ordering
    print("\n   Proprioception order:")
    print("   Training: appendages(15), kinematic(9), touch(4), prev_action(38)")
    print("   TypeScript: appendages(15), prev_action(38), kinematic(9), touch(4)")
    print("   MISMATCH: prev_action and sensors are in wrong order!")

    # Verify the body names match
    print("\n6. BODY NAME VERIFICATION")
    print(f"   consts.BODIES: {consts.BODIES}")
    print(f"   consts.END_EFFECTORS: {consts.END_EFFECTORS}")

    # Create correct observation in Python matching training order
    print("\n7. BUILDING CORRECT OBSERVATION")

    # Get raw data
    mj_data = mujoco.MjData(env.mj_model)
    ref_qpos = np.array(clip['qpos'][0])
    mj_data.qpos[:len(ref_qpos)] = ref_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(env.mj_model, mj_data)

    # Build observation matching training order
    obs_correct = []

    # Reference observation
    root_pos = mj_data.qpos[:3]
    root_quat = mj_data.qpos[3:7]

    # Collect reference frames (5 frames × (3+4+67) then 18 bodies × 5 frames × 3)
    ref_length = 5
    num_frames = len(clip['qpos'])

    # First: root(5,3), quat(5,4), joint(5,67)
    for f in range(ref_length):
        frame_idx = min(1 + f, num_frames - 1)
        ref_qpos_f = np.array(clip['qpos'][frame_idx])

        # Root target
        ref_root_pos = ref_qpos_f[:3]
        rel_pos = ref_root_pos - root_pos
        ego_pos = brax_math.rotate(jp.array(rel_pos), jp.array(root_quat))
        obs_correct.extend(np.array(ego_pos).tolist())

    for f in range(ref_length):
        frame_idx = min(1 + f, num_frames - 1)
        ref_qpos_f = np.array(clip['qpos'][frame_idx])

        # Quat target
        ref_quat = ref_qpos_f[3:7]
        rel_quat = brax_math.relative_quat(jp.array(ref_quat), jp.array(root_quat))
        obs_correct.extend(np.array(rel_quat).tolist())

    for f in range(ref_length):
        frame_idx = min(1 + f, num_frames - 1)
        ref_qpos_f = np.array(clip['qpos'][frame_idx])

        # Joint targets
        for j in range(67):
            obs_correct.append(ref_qpos_f[7 + j] - mj_data.qpos[7 + j])

    # Then: body(18, 5, 3) - iterate bodies first, then frames!
    # Get current body positions
    BODY_NAMES = consts.BODIES  # 18 bodies
    body_indices = []
    for name in BODY_NAMES:
        body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{name}-rodent")
        body_indices.append(body_id)

    cur_body_xpos = [mj_data.xpos[idx].copy() for idx in body_indices]

    # Get reference clip body indices
    ref_xpos = np.array(clip['xpos'][0])
    # The clip stores xpos with original model indices (without floor)
    # BODY_INDICES_REF in TypeScript
    BODY_INDICES_REF = [2, 9, 10, 11, 12, 14, 15, 16, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66]

    # Body targets: (18 bodies, 5 frames, 3 coords)
    for b, ref_body_idx in enumerate(BODY_INDICES_REF):  # 18 bodies
        for f in range(ref_length):  # 5 frames
            frame_idx = min(1 + f, num_frames - 1)
            ref_xpos_f = np.array(clip['xpos'][frame_idx])
            ref_body_pos = ref_xpos_f[ref_body_idx]
            cur_body_pos = cur_body_xpos[b]
            rel_body_pos = ref_body_pos - cur_body_pos
            ego_body_pos = brax_math.rotate(jp.array(rel_body_pos), jp.array(root_quat))
            obs_correct.extend(np.array(ego_body_pos).tolist())

    print(f"   Reference obs size: {len(obs_correct)} (expected 640)")

    # Proprioception
    # joint_angles (67)
    for i in range(7, env.mj_model.nq):
        obs_correct.append(mj_data.qpos[i])
    # joint_ang_vels (67)
    for i in range(6, env.mj_model.nv):
        obs_correct.append(mj_data.qvel[i])
    # actuator_ctrl / qfrc_actuator (73)
    for i in range(env.mj_model.nv):
        obs_correct.append(mj_data.qfrc_actuator[i])
    # body_height (1)
    torso_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso-rodent")
    obs_correct.append(mj_data.xpos[torso_id, 2])
    # world_zaxis (3)
    walker_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "walker-rodent")
    walker_xmat = mj_data.xmat[walker_id].reshape(3, 3)
    obs_correct.extend(walker_xmat.flatten()[6:9].tolist())

    # appendages_pos (15)
    torso_pos = mj_data.xpos[torso_id]
    torso_mat = mj_data.xmat[torso_id].reshape(3, 3)
    for name in consts.END_EFFECTORS:
        body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{name}-rodent")
        body_pos = mj_data.xpos[body_id]
        rel = body_pos - torso_pos
        ego = rel @ torso_mat
        obs_correct.extend(ego.tolist())

    # kinematic_sensors (9) - CORRECT ORDER
    obs_correct.extend(mj_data.sensordata[:9].tolist())

    # touch_sensors (4)
    obs_correct.extend(mj_data.sensordata[9:13].tolist())

    # prev_action (38) - LAST in training order
    prev_action = np.zeros(38)
    obs_correct.extend(prev_action.tolist())

    obs_correct = np.array(obs_correct, dtype=np.float32)
    print(f"   Total obs size: {len(obs_correct)} (expected 917)")

    # Compare with training observation
    print("\n8. COMPARISON WITH TRAINING OBSERVATION")
    training_flat = np.array(flat_obs)

    if len(obs_correct) == len(training_flat):
        diff = np.abs(obs_correct - training_flat)
        print(f"   Max diff: {diff.max():.6f}")
        print(f"   Mean diff: {diff.mean():.6f}")

        if diff.max() > 0.001:
            worst_idx = np.argmax(diff)
            print(f"   Worst diff at index {worst_idx}:")
            print(f"     Built: {obs_correct[worst_idx]:.6f}")
            print(f"     Training: {training_flat[worst_idx]:.6f}")

            # Find section
            if worst_idx < 640:
                print(f"     Section: reference observation")
            elif worst_idx < 707:
                print(f"     Section: joint_angles")
            elif worst_idx < 774:
                print(f"     Section: joint_vels")
            elif worst_idx < 847:
                print(f"     Section: qfrc_actuator")
            elif worst_idx < 848:
                print(f"     Section: body_height")
            elif worst_idx < 851:
                print(f"     Section: world_zaxis")
            elif worst_idx < 866:
                print(f"     Section: appendages")
            elif worst_idx < 875:
                print(f"     Section: kinematic_sensors")
            elif worst_idx < 879:
                print(f"     Section: touch_sensors")
            else:
                print(f"     Section: prev_action")
        else:
            print("   Observations match!")
    else:
        print(f"   Size mismatch: built={len(obs_correct)}, training={len(training_flat)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
