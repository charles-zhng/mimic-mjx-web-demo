import type { MainModule as MjMainModule, MjModel, MjData } from 'mujoco-js'

// Emscripten heap interface
interface EmscriptenHeap {
  HEAP8: Int8Array
  HEAP16: Int16Array
  HEAP32: Int32Array
  HEAPU8: Uint8Array
  HEAPU16: Uint16Array
  HEAPU32: Uint32Array
  HEAPF32: Float32Array
  HEAPF64: Float64Array
}

// Extended MainModule with Emscripten heap access
export type MainModule = MjMainModule & EmscriptenHeap

export type { MjModel, MjData }

export interface MotionClip {
  name: string
  num_frames: number
  qpos: number[][] // [num_frames, nq]
  qvel: number[][] // [num_frames, nv]
  xpos: number[][][] // [num_frames, nbody, 3]
}

export interface MotionClipsMetadata {
  num_clips: number
  clip_length: number
  qpos_dim: number
  qvel_dim: number
  mocap_hz: number
}

export interface MotionClipsData {
  metadata: MotionClipsMetadata
  clips: MotionClip[]
}

export interface SimulationState {
  currentFrame: number
  reset: () => void
}
