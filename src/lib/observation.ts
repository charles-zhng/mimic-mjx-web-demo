import type { MjModel, MjData, MainModule, MotionClip } from '../types'
import type { AnimalConfig } from '../types/animal-config'
import { getBodyIndicesModel, getEndEffectorIndicesModel } from '../types/animal-config'

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
 * 1. Reference observation: Future trajectory targets in egocentric frame
 *    - root_targets: referenceLength × 3 (all frames together)
 *    - quat_targets: referenceLength × 4 (all frames together)
 *    - joint_targets: referenceLength × numJoints (all frames together)
 *    - body_targets: numBodies × referenceLength × 3 (bodies outer, frames inner)
 * 2. Proprioceptive observation: Current robot state + sensors
 *    - Order: joints, vels, forces, height, zaxis, appendages, kinematic, touch, prev_action
 *
 * @param config Animal configuration with all dimensions and indices
 */
export function buildObservation(
  mujoco: MainModule,
  model: MjModel,
  data: MjData,
  clip: MotionClip,
  currentFrame: number,
  prevAction: Float32Array,
  config: AnimalConfig
): Float32Array {
  const obs = new Float32Array(config.obs.totalSize)

  // Derive model indices from config
  const bodyIndicesRef = config.body.bodyIndicesRef
  const bodyIndicesModel = getBodyIndicesModel(config)
  const endEffectorIndicesModel = getEndEffectorIndicesModel(config)

  // Get current state from MuJoCo data
  const qpos = getFloatArray(mujoco, data, 'qpos', model.nq)
  const qvel = getFloatArray(mujoco, data, 'qvel', model.nv)
  const qfrcActuator = getFloatArray(mujoco, data, 'qfrc_actuator', model.nv)
  const xpos = getFloatArray(mujoco, data, 'xpos', model.nbody * 3)
  const xmat = getFloatArray(mujoco, data, 'xmat', model.nbody * 9)

  // Root position and quaternion (first 7 elements of qpos for floating base)
  const rootPos = [qpos[0], qpos[1], qpos[2]]
  const rootQuat = [qpos[3], qpos[4], qpos[5], qpos[6]]

  // Get current body positions for body targets (using MODEL indices)
  const curBodyXpos: number[][] = []
  for (const bodyIdx of bodyIndicesModel) {
    curBodyXpos.push([
      xpos[bodyIdx * 3],
      xpos[bodyIdx * 3 + 1],
      xpos[bodyIdx * 3 + 2],
    ])
  }

  // === Reference Observation ===
  // Training env flattens as: all roots, all quats, all joints, all bodies
  // Body array is (numBodies, referenceLength, 3) - iterate bodies first, then frames

  let refIdx = 0

  // Root targets: referenceLength frames × 3 coords
  for (let f = 0; f < config.obs.referenceLength; f++) {
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

  // Quat targets: referenceLength frames × 4 coords
  for (let f = 0; f < config.obs.referenceLength; f++) {
    const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
    const refQpos = clip.qpos[frameIdx]

    const refQuat = [refQpos[3], refQpos[4], refQpos[5], refQpos[6]]
    const relQuat = relativeQuat(refQuat, rootQuat)
    obs[refIdx++] = relQuat[0]
    obs[refIdx++] = relQuat[1]
    obs[refIdx++] = relQuat[2]
    obs[refIdx++] = relQuat[3]
  }

  // Joint targets: referenceLength frames × numJoints joints
  for (let f = 0; f < config.obs.referenceLength; f++) {
    const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
    const refQpos = clip.qpos[frameIdx]

    for (let j = 0; j < config.body.numJoints; j++) {
      const refJoint = refQpos[config.body.rootDOFs + j]
      const curJoint = qpos[config.body.rootDOFs + j]
      obs[refIdx++] = refJoint - curJoint
    }
  }

  // Body targets: numBodies bodies × referenceLength frames × 3 coords
  // IMPORTANT: Iterate bodies first, then frames (matches training env flattening)
  for (let b = 0; b < bodyIndicesRef.length; b++) {
    const refBodyIdx = bodyIndicesRef[b]
    for (let f = 0; f < config.obs.referenceLength; f++) {
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

  // === Proprioceptive Observation ===
  let propIdx = config.obs.referenceObsSize

  // Joint angles (excluding root DOFs): numJoints values
  for (let i = config.body.rootDOFs; i < model.nq; i++) {
    obs[propIdx++] = qpos[i]
  }

  // Joint velocities (excluding root vel DOFs): numJoints values
  for (let i = config.body.rootVelDOFs; i < model.nv; i++) {
    obs[propIdx++] = qvel[i]
  }

  // Actuator forces (qfrc_actuator): nv values
  for (let i = 0; i < model.nv; i++) {
    obs[propIdx++] = qfrcActuator[i]
  }

  // Body height (torso z position): 1 value
  const torsoZ = xpos[config.body.torsoIndex * 3 + 2]
  obs[propIdx++] = torsoZ

  // World z-axis in body frame (from root body rotation matrix): 3 values
  obs[propIdx++] = xmat[config.body.walkerIndex * 9 + 6]
  obs[propIdx++] = xmat[config.body.walkerIndex * 9 + 7]
  obs[propIdx++] = xmat[config.body.walkerIndex * 9 + 8]

  // Appendages positions (egocentric relative to torso): endEffectors × 3 values
  const torsoPos = [
    xpos[config.body.torsoIndex * 3],
    xpos[config.body.torsoIndex * 3 + 1],
    xpos[config.body.torsoIndex * 3 + 2],
  ]
  // Get torso rotation matrix for egocentric transform
  const torsoMat = [
    xmat[config.body.torsoIndex * 9 + 0], xmat[config.body.torsoIndex * 9 + 1], xmat[config.body.torsoIndex * 9 + 2],
    xmat[config.body.torsoIndex * 9 + 3], xmat[config.body.torsoIndex * 9 + 4], xmat[config.body.torsoIndex * 9 + 5],
    xmat[config.body.torsoIndex * 9 + 6], xmat[config.body.torsoIndex * 9 + 7], xmat[config.body.torsoIndex * 9 + 8],
  ]

  for (const bodyIdx of endEffectorIndicesModel) {
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
    // Matrix-vector multiply: rel @ torso_mat (row-major)
    const ego = [
      rel[0] * torsoMat[0] + rel[1] * torsoMat[3] + rel[2] * torsoMat[6],
      rel[0] * torsoMat[1] + rel[1] * torsoMat[4] + rel[2] * torsoMat[7],
      rel[0] * torsoMat[2] + rel[1] * torsoMat[5] + rel[2] * torsoMat[8],
    ]
    obs[propIdx++] = ego[0]
    obs[propIdx++] = ego[1]
    obs[propIdx++] = ego[2]
  }

  // Kinematic sensors: accelerometer + velocimeter + gyro
  const sensordata = getFloatArray(mujoco, data, 'sensordata', config.sensors.totalCount)

  // Accelerometer
  for (let i = config.sensors.kinematic.accelerometer[0]; i < config.sensors.kinematic.accelerometer[1]; i++) {
    obs[propIdx++] = sensordata[i] || 0
  }
  // Velocimeter
  for (let i = config.sensors.kinematic.velocimeter[0]; i < config.sensors.kinematic.velocimeter[1]; i++) {
    obs[propIdx++] = sensordata[i] || 0
  }
  // Gyro
  for (let i = config.sensors.kinematic.gyro[0]; i < config.sensors.kinematic.gyro[1]; i++) {
    obs[propIdx++] = sensordata[i] || 0
  }

  // Touch sensors
  for (const touch of config.sensors.touch) {
    obs[propIdx++] = sensordata[touch.index] || 0
  }

  // Previous action: actionSize values (LAST in training order)
  for (let i = 0; i < config.action.size; i++) {
    obs[propIdx++] = prevAction[i]
  }

  // Debug logging (only on first frame)
  if (currentFrame === 0 && (globalThis as unknown as Record<string, boolean>).__OBS_DEBUG__) {
    console.log(`=== ${config.name} Observation Debug ===`)
    console.log('rootPos:', rootPos)
    console.log('rootQuat:', rootQuat)
    console.log('root_targets (obs[0:15]):', Array.from(obs.slice(0, 15)))
    console.log('obsLength:', obs.length)
    console.log('obsSum:', Array.from(obs).reduce((a, b) => a + b, 0))
  }

  return obs
}
