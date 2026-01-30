/**
 * Configuration for an animal model and its neural controller.
 * This centralizes all animal-specific parameters that were previously hardcoded.
 */
export interface AnimalConfig {
  /** Unique identifier, e.g., "rodent", "humanoid" */
  id: string
  /** Display name, e.g., "Rodent Walker" */
  name: string

  // Asset paths (relative to public folder)
  assets: {
    /** MuJoCo XML model path, e.g., "/models/rodent_scaled.xml" */
    modelPath: string
    /** Skin file path, or null if no skin */
    skinPath: string | null
    /** ONNX neural network path, e.g., "/nn/intention_network.onnx" */
    onnxPath: string
    /** Decoder-only ONNX path for latent walk mode (optional) */
    decoderOnnxPath?: string
    /** Motion clips JSON path, e.g., "/motions/clips.json" */
    clipsPath: string
  }

  // Observation structure
  obs: {
    /** Total observation size (reference + proprioceptive) */
    totalSize: number
    /** Reference observation size (future trajectory targets) */
    referenceObsSize: number
    /** Proprioceptive observation size (current state + sensors) */
    proprioceptiveObsSize: number
    /** Number of future reference frames to include */
    referenceLength: number
  }

  // Body configuration
  body: {
    /** Number of joint DOFs (nq - rootDOFs) */
    numJoints: number
    /** Root DOFs to exclude (7 for floating base: 3 pos + 4 quat) */
    rootDOFs: number
    /** Root velocity DOFs to exclude (6 for floating base: 3 lin + 3 ang) */
    rootVelDOFs: number
    /** Number of tracked bodies for reference observation */
    numBodies: number
    /** Torso body index in MuJoCo model */
    torsoIndex: number
    /** Root/walker body index in MuJoCo model */
    walkerIndex: number
    /** Body indices in reference clip xpos array */
    bodyIndicesRef: number[]
    /** Offset to convert reference indices to model indices (e.g., +1 for floor body) */
    bodyOffset: number
    /** End effector indices in reference clip xpos array */
    endEffectorIndicesRef: number[]
  }

  // Action configuration
  action: {
    /** Action dimension (number of actuators controlled) */
    size: number
  }

  // Sensor configuration
  sensors: {
    /** Total number of sensor values */
    totalCount: number
    /** Kinematic sensor indices [start, end) for accelerometer, velocimeter, gyro */
    kinematic: {
      accelerometer: [number, number]
      velocimeter: [number, number]
      gyro: [number, number]
    }
    /** Touch sensor configuration */
    touch: { name: string; index: number }[]
  }

  // Timing configuration
  timing: {
    /** Control timestep in seconds (e.g., 0.01 for 100Hz) */
    ctrlDt: number
    /** Motion capture frame rate in Hz (e.g., 50) */
    mocapHz: number
  }

  // Rendering configuration (scale-dependent values)
  rendering: {
    /** Initial camera position [x, y, z] */
    cameraPosition: [number, number, number]
    /** Camera look-at target [x, y, z] */
    cameraTarget: [number, number, number]
    /** Minimum orbit distance */
    orbitMinDistance: number
    /** Maximum orbit distance */
    orbitMaxDistance: number
    /** Directional light position [x, y, z] */
    lightPosition: [number, number, number]
    /** Shadow camera bounds (symmetric: -bounds to +bounds) */
    shadowBounds: number
    /** Ghost reference X offset in meters */
    ghostOffset: number
  }

  // Latent space configuration for random walk mode (optional)
  // Uses double-OU: velocity follows OU, position integrates velocity with mean-reversion
  latentSpace?: {
    /** Latent vector dimension (e.g., 16) */
    size: number
    /** Velocity mean-reversion rate - higher = faster velocity decay */
    ouThetaV: number
    /** Velocity volatility - higher = more acceleration */
    ouSigmaV: number
    /** Position mean-reversion rate - gentle pull toward mean */
    ouThetaX: number
    /** Position target (typically 0) */
    ouMu: number
  }

  // Joystick mode configuration (optional)
  joystick?: {
    /** Task observation size for high-level policy (e.g., 57) */
    taskObsSize: number
    /** Latent size output by high-level policy (must match decoder input) */
    latentSize: number
    /** Command velocity ranges */
    commandRanges: {
      vx: [number, number]   // Forward/backward m/s
      vy: [number, number]   // Lateral m/s
      vyaw: [number, number] // Yaw rad/s
    }
    /** High-level policy ONNX path */
    highlevelOnnxPath: string
  }
}

/** Derive model body indices from reference indices */
export function getBodyIndicesModel(config: AnimalConfig): number[] {
  return config.body.bodyIndicesRef.map(i => i + config.body.bodyOffset)
}

/** Derive model end effector indices from reference indices */
export function getEndEffectorIndicesModel(config: AnimalConfig): number[] {
  return config.body.endEffectorIndicesRef.map(i => i + config.body.bodyOffset)
}
