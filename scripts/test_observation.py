#!/usr/bin/env python3
"""Test observation building by comparing Python imitation env with TypeScript implementation."""

import sys
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jp
from brax import math as brax_math
import mujoco

# Add track-mjx to path
TRACK_MJX_PATH = Path(__file__).parent.parent.parent / "track-mjx"
sys.path.insert(0, str(TRACK_MJX_PATH))

from vnl_playground.tasks.rodent import imitation, wrappers, consts
from vnl_playground.tasks.rodent.reference_clips import ReferenceClips
from omegaconf import OmegaConf
from ml_collections import ConfigDict
from track_mjx.agent import checkpointing


def build_observation_typescript_style(
    mj_model, mj_data, clip_qpos, clip_xpos, current_frame, prev_action
):
    """Build observation matching the TypeScript implementation."""

    # Constants from observation.ts
    BODY_INDICES_REF = [2, 9, 10, 11, 12, 14, 15, 16, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66]
    BODY_INDICES_MODEL = [i + 1 for i in BODY_INDICES_REF]
    END_EFFECTOR_INDICES_MODEL = [65, 60, 17, 13, 56]
    TORSO_INDEX_MODEL = 3
    WALKER_INDEX_MODEL = 2
    REFERENCE_LENGTH = 5
    NUM_JOINTS = 67
    ACTION_SIZE = 38

    obs = []

    # Current state from qpos (like TypeScript does)
    root_pos = mj_data.qpos[:3]
    root_quat = mj_data.qpos[3:7]

    # Current body positions for body targets (using MODEL indices)
    cur_body_xpos = []
    for body_idx in BODY_INDICES_MODEL:
        cur_body_xpos.append(mj_data.xpos[body_idx].copy())

    # === Reference Observation (640 dims) ===
    num_frames = len(clip_qpos)
    for f in range(REFERENCE_LENGTH):
        frame_idx = min(current_frame + 1 + f, num_frames - 1)
        ref_qpos = np.array(clip_qpos[frame_idx])
        ref_xpos = np.array(clip_xpos[frame_idx])

        # Root position target (relative, rotated to egocentric)
        ref_root_pos = ref_qpos[:3]
        rel_pos = ref_root_pos - root_pos
        ego_pos = brax_math.rotate(jp.array(rel_pos), jp.array(root_quat))
        obs.extend(np.array(ego_pos).tolist())

        # Quaternion target (relative) - FIXED: q2 * q1^-1
        ref_quat = ref_qpos[3:7]
        rel_quat = brax_math.relative_quat(jp.array(ref_quat), jp.array(root_quat))
        obs.extend(np.array(rel_quat).tolist())

        # Joint angle differences (reference - current)
        for j in range(NUM_JOINTS):
            ref_joint = ref_qpos[7 + j]
            cur_joint = mj_data.qpos[7 + j]
            obs.append(ref_joint - cur_joint)

        # Body position targets (relative, rotated to egocentric)
        for b, ref_body_idx in enumerate(BODY_INDICES_REF):
            ref_body_pos = np.array(ref_xpos[ref_body_idx])
            cur_body_pos = cur_body_xpos[b]
            rel_body_pos = ref_body_pos - cur_body_pos
            ego_body_pos = brax_math.rotate(jp.array(rel_body_pos), jp.array(root_quat))
            obs.extend(np.array(ego_body_pos).tolist())

    # === Proprioceptive Observation (277 dims) ===

    # Joint angles (excluding root 7 dofs): 67 values
    for i in range(7, mj_model.nq):
        obs.append(mj_data.qpos[i])

    # Joint velocities (excluding root 6 dofs): 67 values
    for i in range(6, mj_model.nv):
        obs.append(mj_data.qvel[i])

    # Actuator forces (qfrc_actuator): 73 values
    for i in range(mj_model.nv):
        obs.append(mj_data.qfrc_actuator[i])

    # Body height (torso z position): 1 value
    torso_z = mj_data.xpos[TORSO_INDEX_MODEL, 2]
    obs.append(torso_z)

    # World z-axis in body frame: 3 values
    walker_xmat = mj_data.xmat[WALKER_INDEX_MODEL].reshape(3, 3)
    obs.extend(walker_xmat.flatten()[6:9].tolist())

    # Appendages positions (egocentric relative to torso): 15 values
    torso_pos = mj_data.xpos[TORSO_INDEX_MODEL]
    torso_mat = mj_data.xmat[TORSO_INDEX_MODEL].reshape(3, 3)
    for body_idx in END_EFFECTOR_INDICES_MODEL:
        body_pos = mj_data.xpos[body_idx]
        rel = body_pos - torso_pos
        # jp.dot(rel, torso_mat) matches TypeScript: rel @ torso_mat
        ego = rel @ torso_mat
        obs.extend(ego.tolist())

    # Previous action: 38 values
    obs.extend(prev_action.tolist())

    # Kinematic sensors: 9 values
    obs.extend(mj_data.sensordata[:9].tolist())

    # Touch sensors: 4 values
    obs.extend(mj_data.sensordata[9:13].tolist())

    return np.array(obs, dtype=np.float32)


def get_imitation_env_observation(env, state):
    """Get flattened observation from imitation environment."""
    # The FlattenObsWrapper flattens the observation
    return np.array(state.obs).flatten()


def main():
    # Load the web demo model directly for comparison
    mj_model = mujoco.MjModel.from_xml_path(
        '/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/models/rodent.xml'
    )
    mj_data = mujoco.MjData(mj_model)

    # Load web demo clips
    with open('/Users/charles/MIMIC-MJX/mimic-mjx-web-demo/public/motions/clips.json', 'r') as f:
        web_clips = json.load(f)
    clip = web_clips['clips'][0]

    # Set initial state from clip frame 0
    ref_qpos = np.array(clip['qpos'][0])
    mj_data.qpos[:len(ref_qpos)] = ref_qpos
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    print(f"Model nq={mj_model.nq}, nv={mj_model.nv}")
    print(f"Initial qpos[:7]: {mj_data.qpos[:7]}")

    # Build TypeScript-style observation
    prev_action = np.zeros(38, dtype=np.float32)
    ts_obs = build_observation_typescript_style(
        mj_model, mj_data, clip['qpos'], clip['xpos'],
        current_frame=0, prev_action=prev_action
    )
    print(f"TypeScript-style observation shape: {ts_obs.shape}")

    # Now build the observation using the TRAINING code's logic
    # This matches what vnl_playground/tasks/rodent/imitation.py does

    def build_training_style_obs(mj_data, clip_qpos, clip_xpos, current_frame, prev_action):
        """Build observation matching the training environment."""
        BODIES = ["torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
                  "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
                  "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
                  "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R"]
        END_EFFECTORS = ["lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"]

        # Body name to index mapping for web demo model
        body_name_to_idx = {mj_model.body(i).name: i for i in range(mj_model.nbody)}

        # Get body indices
        BODY_INDICES_MODEL = [body_name_to_idx[name] for name in BODIES]
        END_EFFECTOR_INDICES_MODEL = [body_name_to_idx[name] for name in END_EFFECTORS]
        TORSO_IDX = body_name_to_idx["torso"]
        WALKER_IDX = body_name_to_idx["walker"]

        # Reference body indices (in the clip xpos, which has 67 bodies)
        # These come from the h5 file structure
        ref_body_names = ['world', 'walker', 'torso', 'vertebra_1', 'vertebra_2', 'vertebra_3',
                          'vertebra_4', 'vertebra_5', 'vertebra_6', 'pelvis', 'upper_leg_L',
                          'lower_leg_L', 'foot_L', 'toe_L', 'upper_leg_R', 'lower_leg_R',
                          'foot_R', 'toe_R', 'vertebra_C1', 'vertebra_C2', 'vertebra_C3',
                          'vertebra_C4', 'vertebra_C5', 'vertebra_C6', 'vertebra_C7', 'vertebra_C8',
                          'vertebra_C9', 'vertebra_C10', 'vertebra_C11', 'vertebra_C12', 'vertebra_C13',
                          'vertebra_C14', 'vertebra_C15', 'vertebra_C16', 'vertebra_C17', 'vertebra_C18',
                          'vertebra_C19', 'vertebra_C20', 'vertebra_C21', 'vertebra_C22', 'vertebra_C23',
                          'vertebra_C24', 'vertebra_C25', 'vertebra_C26', 'vertebra_C27', 'vertebra_C28',
                          'vertebra_C29', 'vertebra_C30', 'vertebra_cervical_5', 'vertebra_cervical_4',
                          'vertebra_cervical_3', 'vertebra_cervical_2', 'vertebra_cervical_1',
                          'vertebra_axis', 'vertebra_atlant', 'skull', 'jaw', 'scapula_L',
                          'upper_arm_L', 'lower_arm_L', 'hand_L', 'finger_L', 'scapula_R',
                          'upper_arm_R', 'lower_arm_R', 'hand_R', 'finger_R']
        BODY_INDICES_REF = [ref_body_names.index(name) for name in BODIES]

        REFERENCE_LENGTH = 5
        NUM_JOINTS = 67

        obs = []

        # Training code uses walker body's xpos and xquat
        root_pos = mj_data.xpos[WALKER_IDX]
        root_quat = mj_data.xquat[WALKER_IDX]

        # Current body positions
        cur_body_xpos = [mj_data.xpos[idx].copy() for idx in BODY_INDICES_MODEL]

        # === Reference Observation ===
        num_frames = len(clip_qpos)
        for f in range(REFERENCE_LENGTH):
            frame_idx = min(current_frame + 1 + f, num_frames - 1)
            ref_qpos = np.array(clip_qpos[frame_idx])
            ref_xpos = np.array(clip_xpos[frame_idx])

            # Root position target
            ref_root_pos = ref_qpos[:3]
            rel_pos = ref_root_pos - root_pos
            ego_pos = brax_math.rotate(jp.array(rel_pos), jp.array(root_quat))
            obs.extend(np.array(ego_pos).tolist())

            # Quaternion target
            ref_quat = ref_qpos[3:7]
            rel_quat = brax_math.relative_quat(jp.array(ref_quat), jp.array(root_quat))
            obs.extend(np.array(rel_quat).tolist())

            # Joint angle differences
            for j in range(NUM_JOINTS):
                ref_joint = ref_qpos[7 + j]
                cur_joint = mj_data.qpos[7 + j]
                obs.append(ref_joint - cur_joint)

            # Body position targets
            for b, ref_body_idx in enumerate(BODY_INDICES_REF):
                ref_body_pos = np.array(ref_xpos[ref_body_idx])
                cur_body_pos = cur_body_xpos[b]
                rel_body_pos = ref_body_pos - cur_body_pos
                ego_body_pos = brax_math.rotate(jp.array(rel_body_pos), jp.array(root_quat))
                obs.extend(np.array(ego_body_pos).tolist())

        # === Proprioceptive ===
        # Joint angles
        for i in range(7, mj_model.nq):
            obs.append(mj_data.qpos[i])

        # Joint velocities
        for i in range(6, mj_model.nv):
            obs.append(mj_data.qvel[i])

        # qfrc_actuator
        for i in range(mj_model.nv):
            obs.append(mj_data.qfrc_actuator[i])

        # Body height
        obs.append(mj_data.xpos[TORSO_IDX, 2])

        # World z-axis (from walker body)
        walker_xmat = mj_data.xmat[WALKER_IDX].reshape(3, 3)
        obs.extend(walker_xmat.flatten()[6:9].tolist())

        # Appendages
        torso_pos = mj_data.xpos[TORSO_IDX]
        torso_mat = mj_data.xmat[TORSO_IDX].reshape(3, 3)
        for body_idx in END_EFFECTOR_INDICES_MODEL:
            body_pos = mj_data.xpos[body_idx]
            rel = body_pos - torso_pos
            ego = rel @ torso_mat
            obs.extend(ego.tolist())

        # Prev action
        obs.extend(prev_action.tolist())

        return np.array(obs, dtype=np.float32)

    training_obs = build_training_style_obs(
        mj_data, clip['qpos'], clip['xpos'], current_frame=0, prev_action=prev_action
    )
    print(f"Training-style observation shape: {training_obs.shape}")

    # Compare TypeScript vs Training style (should match now)
    print("\n=== Comparing TypeScript vs Training style ===")
    diff = np.abs(ts_obs[:len(training_obs)] - training_obs)
    print(f"Max diff: {diff.max():.10f}")
    if diff.max() > 1e-6:
        worst_idx = np.argmax(diff)
        print(f"Worst at index {worst_idx}: ts={ts_obs[worst_idx]:.6f}, training={training_obs[worst_idx]:.6f}")

    # Compare sections
    print("\n=== Comparing observation sections ===")

    # Reference observation (first 640)
    if len(training_obs) >= 640 and len(ts_obs) >= 640:
        ref_diff = np.abs(training_obs[:640] - ts_obs[:640])
        print(f"Reference obs (0-640) max diff: {ref_diff.max():.6f}")
        if ref_diff.max() > 0.01:
            worst_idx = np.argmax(ref_diff)
            print(f"  Worst at index {worst_idx}: training={training_obs[worst_idx]:.6f}, ts={ts_obs[worst_idx]:.6f}")
            # Find which section
            frame = worst_idx // 128
            within_frame = worst_idx % 128
            if within_frame < 3:
                section = "root_pos"
            elif within_frame < 7:
                section = "quat"
            elif within_frame < 74:
                section = "joints"
            else:
                section = "body"
            print(f"  In frame {frame}, section: {section}")

    # Proprioceptive sections
    sections = [
        ("joint_angles", 640, 707),
        ("joint_vels", 707, 774),
        ("qfrc_actuator", 774, 847),
        ("body_height", 847, 848),
        ("world_zaxis", 848, 851),
        ("appendages", 851, 866),
        ("prev_action", 866, 904),
    ]

    for name, start, end in sections:
        if len(training_obs) >= end and len(ts_obs) >= end:
            section_diff = np.abs(training_obs[start:end] - ts_obs[start:end])
            max_diff = section_diff.max()
            if max_diff > 0.001:
                worst_idx = np.argmax(section_diff) + start
                print(f"{name} ({start}-{end}) max diff: {max_diff:.6f}")
                print(f"  At index {worst_idx}: training={training_obs[worst_idx]:.6f}, ts={ts_obs[worst_idx]:.6f}")
            else:
                print(f"{name} ({start}-{end}) max diff: {max_diff:.6f} ✓")

    # Note: Training env doesn't have sensors, so we can't compare those
    print("\nNote: Training-style obs produces 904-dim obs (no sensors)")
    print("TypeScript implementation adds 13 sensor dims for 917 total")


if __name__ == "__main__":
    main()
