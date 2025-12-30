import { useEffect, useRef, useCallback, useState } from 'react'
import type { InferenceSession } from 'onnxruntime-web'
import type { MainModule, MjModel, MjData, MotionClip, SimulationState } from '../types'
import type { AnimalConfig } from '../types/animal-config'
import type { VisualizationData } from '../types/visualization'
import { runInference, runDecoderInference, extractActionInto } from './useONNX'
import { buildObservation, buildProprioceptiveObservation } from '../lib/observation'

export type InferenceMode = 'tracking' | 'latentWalk' | 'latentNoise'

interface UseSimulationProps {
  mujoco: MainModule | null
  model: MjModel | null
  data: MjData | null
  ghostData: MjData | null
  session: InferenceSession | null
  decoderSession: InferenceSession | null
  clips: MotionClip[] | null
  selectedClip: number
  isPlaying: boolean
  speed: number
  isReady: boolean
  config: AnimalConfig
  inferenceMode: InferenceMode
  noiseMagnitude: number
  selectedJointIndex: number
  onVisualizationData?: (data: VisualizationData) => void
}

// Ornstein-Uhlenbeck process state
interface OUState {
  x: Float32Array
  initialized: boolean
}

/**
 * Generate a standard normal random number using Box-Muller transform.
 */
function gaussianRandom(): number {
  const u1 = Math.random()
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/**
 * Step the Ornstein-Uhlenbeck process in-place.
 * dx = theta * (mu - x) * dt + sigma * sqrt(dt) * dW
 */
function stepOUProcess(
  state: Float32Array,
  dt: number,
  theta: number,
  mu: number,
  sigma: number
): void {
  const sqrtDt = Math.sqrt(dt)
  for (let i = 0; i < state.length; i++) {
    const dW = gaussianRandom()
    state[i] += theta * (mu - state[i]) * dt + sigma * sqrtDt * dW
  }
}

// Maximum control steps per browser frame to prevent lockups
const MAX_STEPS_PER_FRAME = 10

export function useSimulation({
  mujoco,
  model,
  data,
  ghostData,
  session,
  decoderSession,
  clips,
  selectedClip,
  isPlaying,
  speed,
  isReady,
  config,
  inferenceMode,
  noiseMagnitude,
  selectedJointIndex,
  onVisualizationData,
}: UseSimulationProps): SimulationState {
  const [currentFrame, setCurrentFrame] = useState(0)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)
  const accumulatedTimeRef = useRef<number>(0)
  const prevActionRef = useRef<Float32Array>(new Float32Array(config.action.size))

  // Pre-allocated buffer for latent noise mode
  const latentNoiseRef = useRef<Float32Array>(new Float32Array(config.latentSpace?.size ?? 16))

  // OU process state for latent walk mode
  const ouStateRef = useRef<OUState>({
    x: new Float32Array(config.latentSpace?.size ?? 16),
    initialized: false,
  })

  // Reset simulation to initial state
  const reset = useCallback(() => {
    if (!mujoco || !model || !data || !clips) return

    const clip = clips[selectedClip]
    if (!clip) return

    // Reset MuJoCo data
    mujoco.mj_resetData(model, data)

    // Get qpos array
    const qpos = data.qpos as unknown as Float64Array
    const refQpos = clip.qpos[0]

    // Copy reference qpos to MuJoCo (only up to model.nq)
    for (let i = 0; i < Math.min(refQpos.length, model.nq); i++) {
      qpos[i] = refQpos[i]
    }

    // Explicitly zero velocities
    const qvel = data.qvel as unknown as Float64Array
    for (let i = 0; i < model.nv; i++) {
      qvel[i] = 0
    }

    // Forward kinematics to update derived quantities
    mujoco.mj_forward(model, data)

    // Reset ghost data to match initial reference pose (with offset)
    if (ghostData) {
      mujoco.mj_resetData(model, ghostData)
      const ghostQpos = ghostData.qpos as unknown as Float64Array
      for (let i = 0; i < Math.min(refQpos.length, model.nq); i++) {
        ghostQpos[i] = refQpos[i]
      }
      ghostQpos[0] += config.rendering.ghostOffset  // Apply X offset
      const ghostQvel = ghostData.qvel as unknown as Float64Array
      for (let i = 0; i < model.nv; i++) {
        ghostQvel[i] = 0
      }
      mujoco.mj_forward(model, ghostData)
    }

    // Reset simulation state
    data.time = 0
    setCurrentFrame(0)
    accumulatedTimeRef.current = 0
    prevActionRef.current.fill(0)  // Zero the buffer instead of allocating new

    // Reset OU state for latent walk mode
    ouStateRef.current.x.fill(0)
    ouStateRef.current.initialized = false
  }, [mujoco, model, data, ghostData, clips, selectedClip, config.action.size])

  // Reset when clip or inference mode changes
  useEffect(() => {
    if (isReady && clips) {
      reset()
    }
  }, [isReady, selectedClip, clips, reset, inferenceMode])

  // Main simulation loop
  useEffect(() => {
    // For tracking mode, we need clips and session
    // For latent walk/noise modes, we need decoderSession and latentSpace config
    const canRunTracking = isReady && isPlaying && mujoco && model && data && session && clips && inferenceMode === 'tracking'
    const canRunLatentWalk = isReady && isPlaying && mujoco && model && data && decoderSession && config.latentSpace && inferenceMode === 'latentWalk'
    const canRunLatentNoise = isReady && isPlaying && mujoco && model && data && decoderSession && config.latentSpace && inferenceMode === 'latentNoise'

    if (!canRunTracking && !canRunLatentWalk && !canRunLatentNoise) {
      return
    }

    const clip = clips?.[selectedClip]

    // Cache physics stepping parameters (computed once per effect)
    const modelOpt = (model as unknown as { opt: { timestep: number } }).opt
    const numSubsteps = modelOpt?.timestep ? Math.round(config.timing.ctrlDt / modelOpt.timestep) : 5
    const ctrl = data.ctrl as unknown as Float64Array

    // Helper: apply action to ctrl and step physics
    const applyActionAndStep = (logits: Float32Array) => {
      extractActionInto(prevActionRef.current, logits)
      for (let i = 0; i < model.nu; i++) {
        ctrl[i] = prevActionRef.current[i]
      }
      for (let i = 0; i < numSubsteps; i++) {
        mujoco.mj_step(model, data)
      }
    }

    const simulate = async (timestamp: number) => {
      // Calculate delta time
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp
      }
      const deltaTime = (timestamp - lastTimeRef.current) / 1000 * speed
      lastTimeRef.current = timestamp
      accumulatedTimeRef.current += deltaTime

      const targetSimTime = accumulatedTimeRef.current

      try {
        let latent: Float32Array | null = null

        if (inferenceMode === 'tracking' && clip && session) {
          // === TRACKING MODE ===
          const targetFrame = Math.floor(targetSimTime * config.timing.mocapHz)

          // Check if we've reached the end of the clip
          if (targetFrame >= clip.num_frames - config.obs.referenceLength) {
            accumulatedTimeRef.current = 0
            setCurrentFrame(0)
            reset()
            animationRef.current = requestAnimationFrame(simulate)
            return
          }

          // Run control steps until physics catches up with target time
          let stepsThisFrame = 0
          while ((data.time as number) < targetSimTime && stepsThisFrame < MAX_STEPS_PER_FRAME) {
            stepsThisFrame++

            const physicsFrame = Math.floor((data.time as number) * config.timing.mocapHz)
            const obs = buildObservation(mujoco, model, data, clip, physicsFrame, prevActionRef.current, config)
            const result = await runInference(session, obs)
            latent = result.latent
            applyActionAndStep(result.logits)
          }

          setCurrentFrame(targetFrame)

          // Emit visualization data (once per browser frame)
          if (onVisualizationData && latent) {
            const qpos = data.qpos as unknown as Float64Array
            const qvel = data.qvel as unknown as Float64Array
            const sensordata = data.sensordata as unknown as Float64Array
            const refQpos = clip.qpos[targetFrame]
            const refQvel = clip.qvel?.[targetFrame]
            const jointQposIndex = 7 + selectedJointIndex
            const jointQvelIndex = 6 + selectedJointIndex

            onVisualizationData({
              timestamp: data.time as number,
              frame: targetFrame,
              jointAngle: {
                current: qpos[jointQposIndex] ?? 0,
                reference: refQpos?.[jointQposIndex] ?? 0,
              },
              jointVelocity: {
                current: qvel[jointQvelIndex] ?? 0,
                reference: refQvel?.[jointQvelIndex] ?? 0,
              },
              touchSensors: [
                sensordata[9] ?? 0,
                sensordata[10] ?? 0,
                sensordata[11] ?? 0,
                sensordata[12] ?? 0,
              ],
              latent: new Float32Array(latent),
            })
          }

          // Update ghost reference pose
          if (ghostData) {
            const refQpos = clip.qpos[targetFrame]
            const ghostQpos = ghostData.qpos as unknown as Float64Array
            for (let i = 0; i < Math.min(refQpos.length, model.nq); i++) {
              ghostQpos[i] = refQpos[i]
            }
            ghostQpos[0] += config.rendering.ghostOffset

            const ghostQvel = ghostData.qvel as unknown as Float64Array
            for (let i = 0; i < model.nv; i++) {
              ghostQvel[i] = 0
            }
            mujoco.mj_forward(model, ghostData)
          }
        } else if (inferenceMode === 'latentWalk' && decoderSession && config.latentSpace) {
          // === LATENT WALK MODE ===
          if (!ouStateRef.current.initialized) {
            ouStateRef.current.x.fill(0)
            ouStateRef.current.initialized = true
          }

          const { ouTheta, ouMu, ouSigma } = config.latentSpace
          let stepsThisFrame = 0

          while ((data.time as number) < targetSimTime && stepsThisFrame < MAX_STEPS_PER_FRAME) {
            stepsThisFrame++
            stepOUProcess(ouStateRef.current.x, config.timing.ctrlDt, ouTheta, ouMu, ouSigma * noiseMagnitude)
            const proprioObs = buildProprioceptiveObservation(mujoco, model, data, prevActionRef.current, config)
            const logits = await runDecoderInference(decoderSession, ouStateRef.current.x, proprioObs)
            applyActionAndStep(logits)
          }

          setCurrentFrame(Math.floor(targetSimTime * config.timing.mocapHz))
        } else if (inferenceMode === 'latentNoise' && decoderSession && config.latentSpace) {
          // === LATENT NOISE MODE ===
          const latentNoise = latentNoiseRef.current
          const noiseScale = config.latentSpace.ouSigma * noiseMagnitude
          let stepsThisFrame = 0

          while ((data.time as number) < targetSimTime && stepsThisFrame < MAX_STEPS_PER_FRAME) {
            stepsThisFrame++

            // Generate noise into pre-allocated buffer
            for (let i = 0; i < latentNoise.length; i++) {
              latentNoise[i] = gaussianRandom() * noiseScale
            }

            const proprioObs = buildProprioceptiveObservation(mujoco, model, data, prevActionRef.current, config)
            const logits = await runDecoderInference(decoderSession, latentNoise, proprioObs)
            applyActionAndStep(logits)
          }

          setCurrentFrame(Math.floor(targetSimTime * config.timing.mocapHz))
        }

        animationRef.current = requestAnimationFrame(simulate)
      } catch (e) {
        console.error('Simulation step error:', e)
        animationRef.current = requestAnimationFrame(simulate)
      }
    }

    // Start simulation loop
    lastTimeRef.current = 0
    animationRef.current = requestAnimationFrame(simulate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
    }
  }, [isReady, isPlaying, mujoco, model, data, ghostData, session, decoderSession, clips, selectedClip, speed, reset, config, inferenceMode, noiseMagnitude, selectedJointIndex, onVisualizationData])

  return {
    currentFrame,
    reset,
  }
}
