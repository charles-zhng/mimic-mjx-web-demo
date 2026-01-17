/**
 * Data emitted each simulation frame for visualization.
 */
export interface VisualizationData {
  timestamp: number
  frame: number
  // Joint data for selected joint
  jointAngle: { current: number; reference: number }
  jointVelocity: { current: number; reference: number }
  // Touch sensors (4 feet)
  touchSensors: number[]
  // Latent space vector
  latent: Float32Array
}

/**
 * Rolling buffer of visualization data.
 */
export interface VisualizationHistory {
  data: VisualizationData[]
  maxLength: number
}

/**
 * Configuration for a single joint to visualize.
 */
export interface JointConfig {
  index: number
  name: string
}
