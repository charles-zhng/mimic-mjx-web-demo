import type { MjModel, MjData, MainModule, MotionClip } from '../types'

/**
 * Observation dimensions from the trained model.
 * Total: 917 = 640 (reference) + 277 (proprioceptive)
 *
 * Reference observation (640 dims):
 *   - root_targets: 5 frames × 3 = 15
 *   - quat_targets: 5 frames × 4 = 20
 *   - joint_targets: 5 frames × 67 = 335
 *   - body_targets: 5 frames × 18 bodies × 3 = 270
 *
 * Proprioceptive observation (277 dims):
 *   - joint_angles: 67 (nq - 7)
 *   - joint_ang_vels: 67 (nv - 6)
 *   - qfrc_actuator: 73 (nv)
 *   - body_height: 1
 *   - world_zaxis: 3
 *   - appendages_pos: 15 (5 end effectors × 3)
 *   - prev_action: 38
 *   - kinematic_sensors: 9 (accelerometer 3 + velocimeter 3 + gyro 3)
 *   - touch_sensors: 4 (palm_L, palm_R, sole_L, sole_R)
 */
export const OBS_CONFIG = {
  totalSize: 917,
  referenceObsSize: 640,
  proprioceptiveObsSize: 277,
  referenceLength: 5, // Number of future reference frames
  actionSize: 38,
  numJoints: 67,  // nq - 7
  numBodies: 18,  // Bodies tracked in reference
}

// Body indices in REFERENCE xpos array (from reference clips h5 file)
// BODIES = ["torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
//           "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
//           "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
//           "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R"]
const BODY_INDICES_REF = [2, 9, 10, 11, 12, 14, 15, 16, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66]

// Body indices in MuJoCo MODEL xpos array (offset by +1 due to added floor body)
// Model body order: world(0), floor(1), walker(2), torso(3), ...
const BODY_INDICES_MODEL = BODY_INDICES_REF.map(i => i + 1)

// End effector indices in reference xpos array
// END_EFFECTORS = ["lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"]
const END_EFFECTOR_INDICES_REF = [64, 59, 16, 12, 55]
const END_EFFECTOR_INDICES_MODEL = END_EFFECTOR_INDICES_REF.map(i => i + 1)

// Torso index in model (reference index + 1 due to floor body)
const TORSO_INDEX_MODEL = 3

// Walker (root body) index in model
const WALKER_INDEX_MODEL = 2

// Note: Observation normalization is handled inside the ONNX model.
// The model includes mean/std as initializers and normalizes internally.

/**
 * Helper to convert MuJoCo array property to typed array.
 */
function toFloat32Array(arr: unknown): Float32Array {
  if (arr instanceof Float32Array) return arr
  if (arr instanceof Float64Array) return new Float32Array(arr)
  if (Array.isArray(arr)) return new Float32Array(arr)
  return new Float32Array(0)
}

/**
 * Helper: Get Float32Array from MuJoCo data property.
 */
function getFloatArray(_mujoco: MainModule, data: MjData, propName: string, _length: number): Float32Array {
  const arr = (data as unknown as Record<string, unknown>)[propName]
  return toFloat32Array(arr)
}

/**
 * Quaternion multiplication.
 */
function quatMul(q1: number[], q2: number[]): number[] {
  const [w1, x1, y1, z1] = q1
  const [w2, x2, y2, z2] = q2
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ]
}

/**
 * Quaternion conjugate (inverse for unit quaternions).
 */
function quatConj(q: number[]): number[] {
  return [q[0], -q[1], -q[2], -q[3]]
}

/**
 * Rotate vector by quaternion.
 */
function rotateByQuat(v: number[], q: number[]): number[] {
  const qv: number[] = [0, v[0], v[1], v[2]]
  const qc = quatConj(q)
  const result = quatMul(quatMul(q, qv), qc)
  return [result[1], result[2], result[3]]
}

/**
 * Compute relative quaternion: q2 * quat_inv(q1)
 * This matches brax.math.relative_quat(q1, q2)
 */
function relativeQuat(q1: number[], q2: number[]): number[] {
  return quatMul(q2, quatConj(q1))
}

/**
 * Build the observation vector for the policy network.
 *
 * The observation consists of:
 * 1. Reference observation (640 dims): Future trajectory targets in egocentric frame
 *    - root_targets: 5 × 3 = 15 (all frames together)
 *    - quat_targets: 5 × 4 = 20 (all frames together)
 *    - joint_targets: 5 × 67 = 335 (all frames together)
 *    - body_targets: 18 × 5 × 3 = 270 (bodies outer, frames inner)
 * 2. Proprioceptive observation (277 dims): Current robot state + sensors
 *    - Order: joints, vels, forces, height, zaxis, appendages, kinematic, touch, prev_action
 */
export function buildObservation(
  mujoco: MainModule,
  model: MjModel,
  data: MjData,
  clip: MotionClip,
  currentFrame: number,
  prevAction: Float32Array
): Float32Array {
  const obs = new Float32Array(OBS_CONFIG.totalSize)

  // Get current state from MuJoCo data
  const qpos = getFloatArray(mujoco, data, 'qpos', model.nq)
  const qvel = getFloatArray(mujoco, data, 'qvel', model.nv)
  const qfrcActuator = getFloatArray(mujoco, data, 'qfrc_actuator', model.nv)
  const xpos = getFloatArray(mujoco, data, 'xpos', model.nbody * 3)
  const xmat = getFloatArray(mujoco, data, 'xmat', model.nbody * 9)

  // Root position and quaternion (first 7 elements of qpos)
  const rootPos = [qpos[0], qpos[1], qpos[2]]
  const rootQuat = [qpos[3], qpos[4], qpos[5], qpos[6]]

  // Get current body positions for body targets (using MODEL indices)
  const curBodyXpos: number[][] = []
  for (const bodyIdx of BODY_INDICES_MODEL) {
    curBodyXpos.push([
      xpos[bodyIdx * 3],
      xpos[bodyIdx * 3 + 1],
      xpos[bodyIdx * 3 + 2],
    ])
  }

  // === Reference Observation (640 dims) ===
  // Training env flattens as: all roots, all quats, all joints, all bodies
  // Body array is (18, 5, 3) - iterate bodies first, then frames

  let refIdx = 0

  // Root targets: 5 frames × 3 coords = 15
  for (let f = 0; f < OBS_CONFIG.referenceLength; f++) {
    const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
    const refQpos = clip.qpos[frameIdx]

    const refRootPos = [refQpos[0], refQpos[1], refQpos[2]]
    const relPos = [
      refRootPos[0] - rootPos[0],
      refRootPos[1] - rootPos[1],
      refRootPos[2] - rootPos[2],
    ]
    const egoPos = rotateByQuat(relPos, rootQuat)
    obs[refIdx++] = egoPos[0]
    obs[refIdx++] = egoPos[1]
    obs[refIdx++] = egoPos[2]
  }

  // Quat targets: 5 frames × 4 coords = 20
  for (let f = 0; f < OBS_CONFIG.referenceLength; f++) {
    const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
    const refQpos = clip.qpos[frameIdx]

    const refQuat = [refQpos[3], refQpos[4], refQpos[5], refQpos[6]]
    const relQuat = relativeQuat(refQuat, rootQuat)
    obs[refIdx++] = relQuat[0]
    obs[refIdx++] = relQuat[1]
    obs[refIdx++] = relQuat[2]
    obs[refIdx++] = relQuat[3]
  }

  // Joint targets: 5 frames × 67 joints = 335
  for (let f = 0; f < OBS_CONFIG.referenceLength; f++) {
    const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
    const refQpos = clip.qpos[frameIdx]

    for (let j = 0; j < OBS_CONFIG.numJoints; j++) {
      const refJoint = refQpos[7 + j]
      const curJoint = qpos[7 + j]
      obs[refIdx++] = refJoint - curJoint
    }
  }

  // Body targets: 18 bodies × 5 frames × 3 coords = 270
  // IMPORTANT: Iterate bodies first, then frames (matches training env flattening)
  for (let b = 0; b < BODY_INDICES_REF.length; b++) {
    const refBodyIdx = BODY_INDICES_REF[b]
    for (let f = 0; f < OBS_CONFIG.referenceLength; f++) {
      const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
      const refXpos = clip.xpos[frameIdx]

      const refBodyPos = [refXpos[refBodyIdx][0], refXpos[refBodyIdx][1], refXpos[refBodyIdx][2]]
      const curBodyPos = curBodyXpos[b]
      const relBodyPos = [
        refBodyPos[0] - curBodyPos[0],
        refBodyPos[1] - curBodyPos[1],
        refBodyPos[2] - curBodyPos[2],
      ]
      const egoBodyPos = rotateByQuat(relBodyPos, rootQuat)
      obs[refIdx++] = egoBodyPos[0]
      obs[refIdx++] = egoBodyPos[1]
      obs[refIdx++] = egoBodyPos[2]
    }
  }

  // === Proprioceptive Observation (277 dims) ===
  let propIdx = OBS_CONFIG.referenceObsSize

  // Joint angles (excluding root 7 dofs): 67 values
  for (let i = 7; i < model.nq; i++) {
    obs[propIdx++] = qpos[i]
  }

  // Joint velocities (excluding root 6 dofs): 67 values
  for (let i = 6; i < model.nv; i++) {
    obs[propIdx++] = qvel[i]
  }

  // Actuator forces (qfrc_actuator): 73 values (full nv)
  // Python: data.qfrc_actuator (NOT data.ctrl!)
  for (let i = 0; i < model.nv; i++) {
    obs[propIdx++] = qfrcActuator[i]
  }

  // Body height (torso z position): 1 value
  const torsoZ = xpos[TORSO_INDEX_MODEL * 3 + 2]
  obs[propIdx++] = torsoZ

  // World z-axis in body frame (from root body rotation matrix): 3 values
  // Python: self.root_body(data).xmat.flatten()[6:]
  obs[propIdx++] = xmat[WALKER_INDEX_MODEL * 9 + 6]
  obs[propIdx++] = xmat[WALKER_INDEX_MODEL * 9 + 7]
  obs[propIdx++] = xmat[WALKER_INDEX_MODEL * 9 + 8]

  // Appendages positions (egocentric relative to torso): 5 × 3 = 15 values
  // Python: uses torso.xmat for rotation, not root_quat
  const torsoPos = [
    xpos[TORSO_INDEX_MODEL * 3],
    xpos[TORSO_INDEX_MODEL * 3 + 1],
    xpos[TORSO_INDEX_MODEL * 3 + 2],
  ]
  // Get torso rotation matrix for egocentric transform
  const torsoMat = [
    xmat[TORSO_INDEX_MODEL * 9 + 0], xmat[TORSO_INDEX_MODEL * 9 + 1], xmat[TORSO_INDEX_MODEL * 9 + 2],
    xmat[TORSO_INDEX_MODEL * 9 + 3], xmat[TORSO_INDEX_MODEL * 9 + 4], xmat[TORSO_INDEX_MODEL * 9 + 5],
    xmat[TORSO_INDEX_MODEL * 9 + 6], xmat[TORSO_INDEX_MODEL * 9 + 7], xmat[TORSO_INDEX_MODEL * 9 + 8],
  ]

  for (const bodyIdx of END_EFFECTOR_INDICES_MODEL) {
    const bodyPos = [
      xpos[bodyIdx * 3],
      xpos[bodyIdx * 3 + 1],
      xpos[bodyIdx * 3 + 2],
    ]
    const rel = [
      bodyPos[0] - torsoPos[0],
      bodyPos[1] - torsoPos[1],
      bodyPos[2] - torsoPos[2],
    ]
    // Python: jp.dot(global_xpos - torso.xpos, torso.xmat)
    // This is matrix-vector multiply: rel @ torso_mat (row-major)
    const ego = [
      rel[0] * torsoMat[0] + rel[1] * torsoMat[3] + rel[2] * torsoMat[6],
      rel[0] * torsoMat[1] + rel[1] * torsoMat[4] + rel[2] * torsoMat[7],
      rel[0] * torsoMat[2] + rel[1] * torsoMat[5] + rel[2] * torsoMat[8],
    ]
    obs[propIdx++] = ego[0]
    obs[propIdx++] = ego[1]
    obs[propIdx++] = ego[2]
  }

  // Kinematic sensors: accelerometer(3) + velocimeter(3) + gyro(3) = 9 values
  // Sensor layout: accelerometer[0-2], velocimeter[3-5], gyro[6-8]
  // NOTE: Training order is sensors BEFORE prev_action
  const sensordata = getFloatArray(mujoco, data, 'sensordata', 16)
  for (let i = 0; i < 9; i++) {
    obs[propIdx++] = sensordata[i] || 0
  }

  // Touch sensors: palm_L, palm_R, sole_L, sole_R = 4 values
  // Sensor layout: palm_L[9], palm_R[10], sole_L[11], sole_R[12]
  for (let i = 9; i < 13; i++) {
    obs[propIdx++] = sensordata[i] || 0
  }

  // Previous action: 38 values (LAST in training order)
  for (let i = 0; i < OBS_CONFIG.actionSize; i++) {
    obs[propIdx++] = prevAction[i]
  }

  // Debug logging (only on first frame)
  if (currentFrame === 0 && (globalThis as unknown as Record<string, boolean>).__OBS_DEBUG__) {
    console.log('=== TypeScript Observation Debug ===')
    console.log('rootPos:', rootPos)
    console.log('rootQuat:', rootQuat)
    console.log('root_targets (obs[0:15]):', Array.from(obs.slice(0, 15)))
    console.log('quat_targets (obs[15:35]):', Array.from(obs.slice(15, 35)))
    console.log('torsoHeight (obs[847]):', obs[847])
    console.log('worldZAxis (obs[848:851]):', [obs[848], obs[849], obs[850]])
    console.log('appendages (obs[851:866]):', Array.from(obs.slice(851, 866)))
    console.log('kinematic_sensors (obs[866:875]):', Array.from(obs.slice(866, 875)))
    console.log('touch_sensors (obs[875:879]):', Array.from(obs.slice(875, 879)))
    console.log('prev_action (obs[879:917]):', Array.from(obs.slice(879, 917)))
    console.log('obsLength:', obs.length)
    console.log('obsSum:', Array.from(obs).reduce((a, b) => a + b, 0))
  }

  return obs
}
