import type { AnimalConfig } from '../../types/animal-config'

/**
 * Configuration for the rodent walker model.
 *
 * Body indices reference (from reference clips h5 file):
 * BODIES = ["torso", "pelvis", "upper_leg_L", "lower_leg_L", "foot_L",
 *           "upper_leg_R", "lower_leg_R", "foot_R", "skull", "jaw",
 *           "scapula_L", "upper_arm_L", "lower_arm_L", "finger_L",
 *           "scapula_R", "upper_arm_R", "lower_arm_R", "finger_R"]
 *
 * End effectors: ["lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"]
 *
 * Model body order: world(0), floor(1), walker(2), torso(3), ...
 * Reference indices need +1 offset to get model indices due to floor body.
 */
export const rodentConfig: AnimalConfig = {
  id: 'rodent',
  name: 'Rodent Walker',

  assets: {
    modelPath: '/models/rodent_scaled.xml',
    skinPath: '/models/rodent_walker_skin.skn',
    onnxPath: '/nn/intention_network.onnx?v=12',
    decoderOnnxPath: '/nn/decoder_only.onnx?v=11',
    clipsPath: '/motions/clips.json',
  },

  obs: {
    // Total: 917 = 640 (reference) + 277 (proprioceptive)
    totalSize: 917,
    referenceObsSize: 640,
    proprioceptiveObsSize: 277,
    referenceLength: 5, // 5 future frames
  },

  body: {
    numJoints: 67,     // nq - 7 (root DOFs)
    rootDOFs: 7,       // 3 position + 4 quaternion
    rootVelDOFs: 6,    // 3 linear + 3 angular velocity
    numBodies: 18,     // tracked body count
    torsoIndex: 3,     // torso body index in model
    walkerIndex: 2,    // root/walker body index in model
    // Body indices in reference clip xpos array
    bodyIndicesRef: [2, 9, 10, 11, 12, 14, 15, 16, 55, 56, 57, 58, 59, 61, 62, 63, 64, 66],
    bodyOffset: 1,     // +1 due to floor body in model
    // End effector indices in reference clip xpos array
    endEffectorIndicesRef: [64, 59, 16, 12, 55],
  },

  action: {
    size: 38, // 38 actuators
  },

  sensors: {
    totalCount: 13,
    kinematic: {
      accelerometer: [0, 3],   // indices 0-2
      velocimeter: [3, 6],     // indices 3-5
      gyro: [6, 9],            // indices 6-8
    },
    touch: [
      { name: 'palm_L', index: 9 },
      { name: 'palm_R', index: 10 },
      { name: 'sole_L', index: 11 },
      { name: 'sole_R', index: 12 },
    ],
  },

  timing: {
    ctrlDt: 0.01,    // 100Hz control
    mocapHz: 50,     // 50Hz motion capture
  },

  rendering: {
    cameraPosition: [0, -1.2, 0.5],  // Relative offset from center
    cameraTarget: [0, 0, 0],           // Center is computed dynamically
    orbitMinDistance: 0.1,
    orbitMaxDistance: 5,
    lightPosition: [2, -2, 5],
    shadowBounds: 2,
    ghostOffset: 0.3,
  },

  latentSpace: {
    size: 16,
    ouThetaV: 0.3,   // Velocity mean-reversion rate
    ouSigmaV: 0.25,  // Velocity volatility (lower = smaller movements)
    ouThetaX: 0.1,   // Position mean-reversion rate (higher = tighter bounds)
    ouMu: 0.0,       // Position target
  },

  joystick: {
    taskObsSize: 57,  // prev_action(38) + kinematic(9) + touch(4) + origin(3) + command(3)
    latentSize: 16,   // Must match decoder input latent size
    commandRanges: {
      vx: [-0.5, 0.5],    // Forward/backward m/s
      vy: [-0.3, 0.3],    // Lateral m/s
      vyaw: [-1.0, 1.0],  // Yaw rad/s
    },
    highlevelOnnxPath: '/nn/highlevel_policy.onnx',
  },
}
