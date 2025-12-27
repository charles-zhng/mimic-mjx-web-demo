import { useEffect, useRef, useCallback, useState } from 'react'
import type { InferenceSession } from 'onnxruntime-web'
import type { MainModule, MjModel, MjData, MotionClip, SimulationState } from '../types'
import { runInference, extractAction } from './useONNX'
import { buildObservation, OBS_CONFIG } from '../lib/observation'

interface UseSimulationProps {
  mujoco: MainModule | null
  model: MjModel | null
  data: MjData | null
  session: InferenceSession | null
  clips: MotionClip[] | null
  selectedClip: number
  isPlaying: boolean
  speed: number
  isReady: boolean
}

// Control timestep in seconds (100 Hz) - matches training checkpoint
const CTRL_DT = 0.01
// Mocap frame rate (50 Hz)
const MOCAP_HZ = 50

export function useSimulation({
  mujoco,
  model,
  data,
  session,
  clips,
  selectedClip,
  isPlaying,
  speed,
  isReady,
}: UseSimulationProps): SimulationState {
  const [currentFrame, setCurrentFrame] = useState(0)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)
  const accumulatedTimeRef = useRef<number>(0)
  const prevActionRef = useRef<Float32Array>(new Float32Array(OBS_CONFIG.actionSize))

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

    // Reset simulation state
    data.time = 0
    setCurrentFrame(0)
    accumulatedTimeRef.current = 0
    prevActionRef.current = new Float32Array(OBS_CONFIG.actionSize)

    console.log('Simulation reset to clip:', clip.name)
  }, [mujoco, model, data, clips, selectedClip])

  // Reset when clip changes
  useEffect(() => {
    if (isReady && clips) {
      reset()
    }
  }, [isReady, selectedClip, clips, reset])

  // Main simulation loop
  useEffect(() => {
    if (!isReady || !isPlaying || !mujoco || !model || !data || !session || !clips) {
      return
    }

    const clip = clips[selectedClip]
    if (!clip) return

    const simulate = async (timestamp: number) => {
      // Calculate delta time
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = timestamp
      }
      const deltaTime = (timestamp - lastTimeRef.current) / 1000 * speed
      lastTimeRef.current = timestamp
      accumulatedTimeRef.current += deltaTime

      // Calculate current reference frame
      const frameTime = accumulatedTimeRef.current
      const newFrame = Math.floor(frameTime * MOCAP_HZ)

      // Check if we've reached the end of the clip
      if (newFrame >= clip.num_frames - OBS_CONFIG.referenceLength) {
        // Loop back to start
        accumulatedTimeRef.current = 0
        setCurrentFrame(0)
        reset()
        animationRef.current = requestAnimationFrame(simulate)
        return
      }

      setCurrentFrame(newFrame)

      try {
        // Build observation (ONNX model handles normalization internally)
        const obs = buildObservation(
          mujoco,
          model,
          data,
          clip,
          newFrame,
          prevActionRef.current
        )

        // Run inference (ONNX model normalizes input internally)
        const logits = await runInference(session, obs)

        // Extract action (tanh of mean)
        const action = extractAction(logits)

        // Apply action to MuJoCo control
        // MuJoCo WASM exposes ctrl as a Float64Array view
        const ctrl = data.ctrl as unknown as Float64Array
        for (let i = 0; i < model.nu; i++) {
          ctrl[i] = action[i]
        }

        // Store action for next observation
        prevActionRef.current = action

        // Step the simulation
        // Run multiple physics substeps per control step
        const modelOpt = (model as unknown as { opt: { timestep: number } }).opt
        const numSubsteps = modelOpt?.timestep ? Math.round(CTRL_DT / modelOpt.timestep) : 5
        for (let i = 0; i < numSubsteps; i++) {
          mujoco.mj_step(model, data)
        }

        // Debug logging
        if ((globalThis as unknown as Record<string, boolean>).__SIM_DEBUG__) {
          const xpos = data.xpos as unknown as Float64Array
          const torsoZ = xpos[3 * 3 + 2]

          // Track data in global arrays
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
  }, [isReady, isPlaying, mujoco, model, data, session, clips, selectedClip, speed, reset])

  return {
    currentFrame,
    reset,
  }
}
