import type { MjModel, MjData, MainModule, MotionClip } from '../types'
import type { AnimalConfig } from '../types/animal-config'
import { getBodyIndicesModel, getEndEffectorIndicesModel } from '../types/animal-config'

// =============================================================================
// Pre-allocated buffers to avoid per-frame allocations
// =============================================================================

// Main observation buffer (lazily initialized based on config)
let _obsBuffer: Float32Array | null = null

// Proprioceptive-only observation buffer (for latent walk mode)
let _proprioObsBuffer: Float32Array | null = null

// Working arrays for quaternion math (fixed size)
const _quatTemp1 = [0, 0, 0, 0]
const _quatTemp2 = [0, 0, 0, 0]
const _quatTemp3 = [0, 0, 0, 0]
const _vec3Temp1 = [0, 0, 0]

// Working arrays for position/quat extraction
const _rootPos = [0, 0, 0]
const _rootQuat = [0, 0, 0, 0]
const _refRootPos = [0, 0, 0]
const _relPos = [0, 0, 0]
const _refQuat = [0, 0, 0, 0]
const _torsoPos = [0, 0, 0]
const _torsoMat = [0, 0, 0, 0, 0, 0, 0, 0, 0]
const _bodyPos = [0, 0, 0]
const _rel = [0, 0, 0]
const _refBodyPos = [0, 0, 0]
const _relBodyPos = [0, 0, 0]

// Pre-allocated body position arrays (max 32 bodies)
const MAX_BODIES = 32
const _curBodyXpos: number[][] = []
for (let i = 0; i < MAX_BODIES; i++) {
  _curBodyXpos.push([0, 0, 0])
}

// =============================================================================
// Mutable quaternion math (writes to output parameter, no allocations)
// =============================================================================

/**
 * Quaternion multiplication into output array.
 */
function quatMulInto(out: number[], q1: number[], q2: number[]): void {
  const w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3]
  const w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3]
  out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
  out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
}

/**
 * Quaternion conjugate into output array.
 */
function quatConjInto(out: number[], q: number[]): void {
  out[0] = q[0]
  out[1] = -q[1]
  out[2] = -q[2]
  out[3] = -q[3]
}

/**
 * Rotate vector by quaternion, writing result to output array.
 * Uses _quatTemp1, _quatTemp2, _quatTemp3 as working space.
 */
function rotateByQuatInto(out: number[], v: number[], q: number[]): void {
  // qv = [0, v.x, v.y, v.z]
  _quatTemp1[0] = 0
  _quatTemp1[1] = v[0]
  _quatTemp1[2] = v[1]
  _quatTemp1[3] = v[2]
  // qc = conjugate(q)
  quatConjInto(_quatTemp2, q)
  // result = q * qv * qc
  quatMulInto(_quatTemp3, q, _quatTemp1)
  quatMulInto(_quatTemp1, _quatTemp3, _quatTemp2)
  out[0] = _quatTemp1[1]
  out[1] = _quatTemp1[2]
  out[2] = _quatTemp1[3]
}

/**
 * Compute relative quaternion into output: q2 * quat_inv(q1)
 * Uses _quatTemp2 as working space.
 */
function relativeQuatInto(out: number[], q1: number[], q2: number[]): void {
  quatConjInto(_quatTemp2, q1)
  quatMulInto(out, q2, _quatTemp2)
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
 * 2. Proprioceptive observation: Current robot state
 *    - Order: joints, vels, forces, height, zaxis, appendages, prev_action
 *
 * @param config Animal configuration with all dimensions and indices
 */
export function buildObservation(
  _mujoco: MainModule,
  model: MjModel,
  data: MjData,
  clip: MotionClip,
  currentFrame: number,
  prevAction: Float32Array,
  config: AnimalConfig
): Float32Array {
  // Reuse observation buffer (lazily allocate if size changed)
  if (!_obsBuffer || _obsBuffer.length !== config.obs.totalSize) {
    _obsBuffer = new Float32Array(config.obs.totalSize)
  }
  const obs = _obsBuffer
  // Note: No need to zero - we overwrite all values below

  // Derive model indices from config
  const bodyIndicesRef = config.body.bodyIndicesRef
  const bodyIndicesModel = getBodyIndicesModel(config)
  const endEffectorIndicesModel = getEndEffectorIndicesModel(config)

  // Get current state from MuJoCo data (direct access, avoid helper overhead)
  const qpos = (data as unknown as Record<string, Float64Array>).qpos
  const qvel = (data as unknown as Record<string, Float64Array>).qvel
  const qfrcActuator = (data as unknown as Record<string, Float64Array>).qfrc_actuator
  const xpos = (data as unknown as Record<string, Float64Array>).xpos
  const xmat = (data as unknown as Record<string, Float64Array>).xmat

  // Root position and quaternion (reuse pre-allocated arrays)
  _rootPos[0] = qpos[0]; _rootPos[1] = qpos[1]; _rootPos[2] = qpos[2]
  _rootQuat[0] = qpos[3]; _rootQuat[1] = qpos[4]; _rootQuat[2] = qpos[5]; _rootQuat[3] = qpos[6]

  // Get current body positions for body targets (reuse pre-allocated arrays)
  for (let i = 0; i < bodyIndicesModel.length; i++) {
    const bodyIdx = bodyIndicesModel[i]
    _curBodyXpos[i][0] = xpos[bodyIdx * 3]
    _curBodyXpos[i][1] = xpos[bodyIdx * 3 + 1]
    _curBodyXpos[i][2] = xpos[bodyIdx * 3 + 2]
  }

  // === Reference Observation ===
  // Training env flattens as: all roots, all quats, all joints, all bodies
  // Body array is (numBodies, referenceLength, 3) - iterate bodies first, then frames

  let refIdx = 0

  // Root targets: referenceLength frames × 3 coords
  for (let f = 0; f < config.obs.referenceLength; f++) {
    const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
    const refQpos = clip.qpos[frameIdx]

    // Use pre-allocated arrays
    _refRootPos[0] = refQpos[0]; _refRootPos[1] = refQpos[1]; _refRootPos[2] = refQpos[2]
    _relPos[0] = _refRootPos[0] - _rootPos[0]
    _relPos[1] = _refRootPos[1] - _rootPos[1]
    _relPos[2] = _refRootPos[2] - _rootPos[2]
    rotateByQuatInto(_vec3Temp1, _relPos, _rootQuat)
    obs[refIdx++] = _vec3Temp1[0]
    obs[refIdx++] = _vec3Temp1[1]
    obs[refIdx++] = _vec3Temp1[2]
  }

  // Quat targets: referenceLength frames × 4 coords
  for (let f = 0; f < config.obs.referenceLength; f++) {
    const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
    const refQpos = clip.qpos[frameIdx]

    // Use pre-allocated arrays
    _refQuat[0] = refQpos[3]; _refQuat[1] = refQpos[4]; _refQuat[2] = refQpos[5]; _refQuat[3] = refQpos[6]
    relativeQuatInto(_quatTemp1, _refQuat, _rootQuat)
    obs[refIdx++] = _quatTemp1[0]
    obs[refIdx++] = _quatTemp1[1]
    obs[refIdx++] = _quatTemp1[2]
    obs[refIdx++] = _quatTemp1[3]
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
    const curBodyPos = _curBodyXpos[b]
    for (let f = 0; f < config.obs.referenceLength; f++) {
      const frameIdx = Math.min(currentFrame + 1 + f, clip.num_frames - 1)
      const refXpos = clip.xpos[frameIdx]
      const refBody = refXpos[refBodyIdx]

      // Use pre-allocated arrays
      _refBodyPos[0] = refBody[0]; _refBodyPos[1] = refBody[1]; _refBodyPos[2] = refBody[2]
      _relBodyPos[0] = _refBodyPos[0] - curBodyPos[0]
      _relBodyPos[1] = _refBodyPos[1] - curBodyPos[1]
      _relBodyPos[2] = _refBodyPos[2] - curBodyPos[2]
      rotateByQuatInto(_vec3Temp1, _relBodyPos, _rootQuat)
      obs[refIdx++] = _vec3Temp1[0]
      obs[refIdx++] = _vec3Temp1[1]
      obs[refIdx++] = _vec3Temp1[2]
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
  const torsoIdx = config.body.torsoIndex
  _torsoPos[0] = xpos[torsoIdx * 3]
  _torsoPos[1] = xpos[torsoIdx * 3 + 1]
  _torsoPos[2] = xpos[torsoIdx * 3 + 2]
  // Get torso rotation matrix for egocentric transform
  const torsoMatBase = torsoIdx * 9
  _torsoMat[0] = xmat[torsoMatBase]; _torsoMat[1] = xmat[torsoMatBase + 1]; _torsoMat[2] = xmat[torsoMatBase + 2]
  _torsoMat[3] = xmat[torsoMatBase + 3]; _torsoMat[4] = xmat[torsoMatBase + 4]; _torsoMat[5] = xmat[torsoMatBase + 5]
  _torsoMat[6] = xmat[torsoMatBase + 6]; _torsoMat[7] = xmat[torsoMatBase + 7]; _torsoMat[8] = xmat[torsoMatBase + 8]

  for (const bodyIdx of endEffectorIndicesModel) {
    _bodyPos[0] = xpos[bodyIdx * 3]
    _bodyPos[1] = xpos[bodyIdx * 3 + 1]
    _bodyPos[2] = xpos[bodyIdx * 3 + 2]
    _rel[0] = _bodyPos[0] - _torsoPos[0]
    _rel[1] = _bodyPos[1] - _torsoPos[1]
    _rel[2] = _bodyPos[2] - _torsoPos[2]
    // Matrix-vector multiply: rel @ torso_mat (row-major)
    obs[propIdx++] = _rel[0] * _torsoMat[0] + _rel[1] * _torsoMat[3] + _rel[2] * _torsoMat[6]
    obs[propIdx++] = _rel[0] * _torsoMat[1] + _rel[1] * _torsoMat[4] + _rel[2] * _torsoMat[7]
    obs[propIdx++] = _rel[0] * _torsoMat[2] + _rel[1] * _torsoMat[5] + _rel[2] * _torsoMat[8]
  }

  // Previous action: actionSize values (LAST in training order)
  for (let i = 0; i < config.action.size; i++) {
    obs[propIdx++] = prevAction[i]
  }

  // Debug logging (only on first frame)
  if (currentFrame === 0 && (globalThis as unknown as Record<string, boolean>).__OBS_DEBUG__) {
    console.log(`=== ${config.name} Observation Debug ===`)
    console.log('rootPos:', _rootPos.slice())
    console.log('rootQuat:', _rootQuat.slice())
    console.log('root_targets (obs[0:15]):', Array.from(obs.slice(0, 15)))
    console.log('obsLength:', obs.length)
    console.log('obsSum:', Array.from(obs).reduce((a, b) => a + b, 0))
  }

  return obs
}

/**
 * Build proprioceptive-only observation for latent walk mode.
 * This extracts only the current robot state (no reference trajectory targets).
 *
 * Order: joints, velocities, forces, height, zaxis, appendages, prev_action
 *
 * @param config Animal configuration with all dimensions and indices
 */
export function buildProprioceptiveObservation(
  _mujoco: MainModule,
  model: MjModel,
  data: MjData,
  prevAction: Float32Array,
  config: AnimalConfig
): Float32Array {
  // Reuse buffer (lazily allocate if size changed)
  if (!_proprioObsBuffer || _proprioObsBuffer.length !== config.obs.proprioceptiveObsSize) {
    _proprioObsBuffer = new Float32Array(config.obs.proprioceptiveObsSize)
  }
  const obs = _proprioObsBuffer

  // Derive model indices from config
  const endEffectorIndicesModel = getEndEffectorIndicesModel(config)

  // Get current state from MuJoCo data
  const qpos = (data as unknown as Record<string, Float64Array>).qpos
  const qvel = (data as unknown as Record<string, Float64Array>).qvel
  const qfrcActuator = (data as unknown as Record<string, Float64Array>).qfrc_actuator
  const xpos = (data as unknown as Record<string, Float64Array>).xpos
  const xmat = (data as unknown as Record<string, Float64Array>).xmat

  let propIdx = 0

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
  const torsoIdx = config.body.torsoIndex
  _torsoPos[0] = xpos[torsoIdx * 3]
  _torsoPos[1] = xpos[torsoIdx * 3 + 1]
  _torsoPos[2] = xpos[torsoIdx * 3 + 2]
  // Get torso rotation matrix for egocentric transform
  const torsoMatBase = torsoIdx * 9
  _torsoMat[0] = xmat[torsoMatBase]; _torsoMat[1] = xmat[torsoMatBase + 1]; _torsoMat[2] = xmat[torsoMatBase + 2]
  _torsoMat[3] = xmat[torsoMatBase + 3]; _torsoMat[4] = xmat[torsoMatBase + 4]; _torsoMat[5] = xmat[torsoMatBase + 5]
  _torsoMat[6] = xmat[torsoMatBase + 6]; _torsoMat[7] = xmat[torsoMatBase + 7]; _torsoMat[8] = xmat[torsoMatBase + 8]

  for (const bodyIdx of endEffectorIndicesModel) {
    _bodyPos[0] = xpos[bodyIdx * 3]
    _bodyPos[1] = xpos[bodyIdx * 3 + 1]
    _bodyPos[2] = xpos[bodyIdx * 3 + 2]
    _rel[0] = _bodyPos[0] - _torsoPos[0]
    _rel[1] = _bodyPos[1] - _torsoPos[1]
    _rel[2] = _bodyPos[2] - _torsoPos[2]
    // Matrix-vector multiply: rel @ torso_mat (row-major)
    obs[propIdx++] = _rel[0] * _torsoMat[0] + _rel[1] * _torsoMat[3] + _rel[2] * _torsoMat[6]
    obs[propIdx++] = _rel[0] * _torsoMat[1] + _rel[1] * _torsoMat[4] + _rel[2] * _torsoMat[7]
    obs[propIdx++] = _rel[0] * _torsoMat[2] + _rel[1] * _torsoMat[5] + _rel[2] * _torsoMat[8]
  }

  // Previous action: actionSize values
  for (let i = 0; i < config.action.size; i++) {
    obs[propIdx++] = prevAction[i]
  }

  return obs
}
