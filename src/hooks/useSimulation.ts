import { useEffect, useRef, useCallback, useState } from 'react'
import type { InferenceSession } from 'onnxruntime-web'
import type { MainModule, MjModel, MjData, MotionClip, SimulationState } from '../types'
import type { AnimalConfig } from '../types/animal-config'
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
}: UseSimulationProps): SimulationState {
  const [currentFrame, setCurrentFrame] = useState(0)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)
  const accumulatedTimeRef = useRef<number>(0)
  const prevActionRef = useRef<Float32Array>(new Float32Array(config.action.size))

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

    // Debug: log qpos dimensions
    console.log('Model nq:', model.nq, 'Reference qpos length:', refQpos.length)

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

    console.log('Simulation reset to clip:', clip.name)
  }, [mujoco, model, data, ghostData, clips, selectedClip, config.action.size])

  // Reset when clip changes
  useEffect(() => {
    if (isReady && clips) {
      reset()
    }
  }, [isReady, selectedClip, clips, reset])

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

    const simulate = async (timestamp: number) => {
      // Calculate delta time
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp
      }
      const deltaTime = (timestamp - lastTimeRef.current) / 1000 * speed
      lastTimeRef.current = timestamp
      accumulatedTimeRef.current += deltaTime

      // Calculate current frame (for tracking mode display)
      const frameTime = accumulatedTimeRef.current
      const newFrame = Math.floor(frameTime * config.timing.mocapHz)

      try {
        let logits: Float32Array

        if (inferenceMode === 'tracking' && clip && session) {
          // === TRACKING MODE ===
          // Check if we've reached the end of the clip
          if (newFrame >= clip.num_frames - config.obs.referenceLength) {
            // Loop back to start
            accumulatedTimeRef.current = 0
            setCurrentFrame(0)
            reset()
            animationRef.current = requestAnimationFrame(simulate)
            return
          }

          setCurrentFrame(newFrame)

          // Build full observation
          const obs = buildObservation(
            mujoco,
            model,
            data,
            clip,
            newFrame,
            prevActionRef.current,
            config
          )

          // Run full inference
          logits = await runInference(session, obs)

          // Update ghost reference pose (kinematics only, no physics)
          if (ghostData) {
            const refQpos = clip.qpos[newFrame]
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
          // No frame tracking in latent walk mode - runs indefinitely
          setCurrentFrame(Math.floor(accumulatedTimeRef.current * config.timing.mocapHz))

          // Initialize OU state if needed
          if (!ouStateRef.current.initialized) {
            ouStateRef.current.x.fill(0)
            ouStateRef.current.initialized = true
          }

          // Step OU process (sigma scaled by noiseMagnitude)
          const { ouTheta, ouMu, ouSigma } = config.latentSpace
          stepOUProcess(ouStateRef.current.x, config.timing.ctrlDt, ouTheta, ouMu, ouSigma * noiseMagnitude)

          // Build proprioceptive-only observation
          const proprioObs = buildProprioceptiveObservation(
            mujoco,
            model,
            data,
            prevActionRef.current,
            config
          )

          // Run decoder inference
          logits = await runDecoderInference(decoderSession, ouStateRef.current.x, proprioObs)
        } else if (inferenceMode === 'latentNoise' && decoderSession && config.latentSpace) {
          // === LATENT NOISE MODE (independent) ===
          // Sample independent Gaussian noise at each step
          setCurrentFrame(Math.floor(accumulatedTimeRef.current * config.timing.mocapHz))

          // Generate independent Gaussian noise for latent space
          const latentNoise = new Float32Array(config.latentSpace.size)
          for (let i = 0; i < latentNoise.length; i++) {
            latentNoise[i] = gaussianRandom() * config.latentSpace.ouSigma * noiseMagnitude
          }

          // Build proprioceptive-only observation
          const proprioObs = buildProprioceptiveObservation(
            mujoco,
            model,
            data,
            prevActionRef.current,
            config
          )

          // Run decoder inference
          logits = await runDecoderInference(decoderSession, latentNoise, proprioObs)
        } else {
          // Should not reach here
          animationRef.current = requestAnimationFrame(simulate)
          return
        }

        // Extract action (tanh of mean) into pre-allocated buffer
        extractActionInto(prevActionRef.current, logits)

        // Apply action to MuJoCo control
        const ctrl = data.ctrl as unknown as Float64Array
        const action = prevActionRef.current
        for (let i = 0; i < model.nu; i++) {
          ctrl[i] = action[i]
        }

        // Step the simulation
        const modelOpt = (model as unknown as { opt: { timestep: number } }).opt
        const numSubsteps = modelOpt?.timestep ? Math.round(config.timing.ctrlDt / modelOpt.timestep) : 5
        for (let i = 0; i < numSubsteps; i++) {
          mujoco.mj_step(model, data)
        }

        // Debug logging
        if ((globalThis as unknown as Record<string, boolean>).__SIM_DEBUG__) {
          const xpos = data.xpos as unknown as Float64Array
          const torsoZ = xpos[config.body.torsoIndex * 3 + 2]

          const heights = ((globalThis as unknown as Record<string, number[]>).__HEIGHTS__ ||= [])
          const frames = ((globalThis as unknown as Record<string, number[]>).__FRAMES__ ||= [])
          heights.push(torsoZ)
          frames.push(newFrame)

          if (heights.length <= 50 && heights.length % 5 === 0) {
            console.log(`Iteration ${heights.length}: frame=${newFrame}, torso=${torsoZ.toFixed(4)}, accTime=${accumulatedTimeRef.current.toFixed(3)}`)
          }
          if (heights.length === 50) {
            console.log('Heights:', heights.map(h => h.toFixed(3)).join(', '))
            console.log('Frames:', frames.join(', '))
          }
        }
      } catch (e) {
        console.error('Simulation step error:', e)
      }

      // Continue loop
      animationRef.current = requestAnimationFrame(simulate)
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
  }, [isReady, isPlaying, mujoco, model, data, ghostData, session, decoderSession, clips, selectedClip, speed, reset, config, inferenceMode, noiseMagnitude])

  return {
    currentFrame,
    reset,
  }
}
