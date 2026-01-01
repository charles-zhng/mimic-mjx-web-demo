import { useState, useEffect, useRef } from 'react'
import { ONNXModel } from '@jax-js/onnx'
import { numpy as np, init, jit, defaultDevice, type Array as JaxArray } from '@jax-js/jax'
import type { AnimalConfig } from '../types/animal-config'

// JIT-compiled run function type
type JitRunFn = (inputs: Record<string, JaxArray>) => Record<string, JaxArray>

export interface JaxSession {
  run: JitRunFn
  dispose: () => void
}

interface UseONNXResult {
  session: JaxSession | null
  decoderSession: JaxSession | null
  isReady: boolean
  error: string | null
}

/**
 * Run one warmup inference to trigger JIT compilation.
 */
function warmup(run: JitRunFn, inputShapes: Record<string, number[]>): void {
  const inputs: Record<string, JaxArray> = {}
  for (const [name, shape] of Object.entries(inputShapes)) {
    const size = shape.reduce((a, b) => a * b, 1)
    inputs[name] = np.array(new Float32Array(size)).reshape(shape)
  }
  const output = run(inputs)
  for (const key of Object.keys(output)) {
    output[key].dispose()
  }
}

export function useONNX(config: AnimalConfig): UseONNXResult {
  const [session, setSession] = useState<JaxSession | null>(null)
  const [decoderSession, setDecoderSession] = useState<JaxSession | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const initRef = useRef(false)
  const configIdRef = useRef(config.id)
  const modelsRef = useRef<{ full: ONNXModel | null; decoder: ONNXModel | null }>({
    full: null,
    decoder: null,
  })

  useEffect(() => {
    // Reset if config changed
    if (configIdRef.current !== config.id) {
      configIdRef.current = config.id
      initRef.current = false
      setIsReady(false)
      setError(null)
      // Dispose old models
      modelsRef.current.full?.dispose()
      modelsRef.current.decoder?.dispose()
      modelsRef.current = { full: null, decoder: null }
    }

    if (initRef.current) return
    initRef.current = true

    async function initModels() {
      try {
        // 1. Initialize jax-js WebGPU backend
        const devices = await init('webgpu')
        if (!devices.includes('webgpu')) {
          throw new Error('WebGPU not available - required for jax-js inference')
        }
        defaultDevice('webgpu')
        console.log(`jax-js WebGPU backend initialized (device: ${defaultDevice()})`)

        // 2. Load model bytes in parallel
        const [fullModelBytes, decoderModelBytes] = await Promise.all([
          fetch(config.assets.onnxPath).then((r) => r.arrayBuffer()),
          config.assets.decoderOnnxPath
            ? fetch(config.assets.decoderOnnxPath).then((r) => r.arrayBuffer())
            : Promise.resolve(null),
        ])

        // 3. Create ONNXModel instances
        const fullModel = new ONNXModel(new Uint8Array(fullModelBytes))
        const decoderModel = decoderModelBytes
          ? new ONNXModel(new Uint8Array(decoderModelBytes))
          : null

        modelsRef.current = { full: fullModel, decoder: decoderModel }

        // 4. JIT compile the run functions
        const fullRun = jit(fullModel.run)
        const decoderRun = decoderModel ? jit(decoderModel.run) : null

        // 5. Warmup inference (triggers JIT compilation)
        console.log('Running warmup inference for JIT compilation...')
        const warmupStart = performance.now()
        warmup(fullRun, { obs: [1, config.obs.totalSize] })
        console.log(`  Full model warmup: ${(performance.now() - warmupStart).toFixed(0)}ms`)

        if (decoderRun && config.latentSpace) {
          const decoderWarmupStart = performance.now()
          warmup(decoderRun, {
            latent: [1, config.latentSpace.size],
            proprio_obs: [1, config.obs.proprioceptiveObsSize],
          })
          console.log(`  Decoder warmup: ${(performance.now() - decoderWarmupStart).toFixed(0)}ms`)
        }

        // 6. Create session wrappers
        const sessionWrapper: JaxSession = {
          run: fullRun,
          dispose: () => fullModel.dispose(),
        }

        const decoderWrapper: JaxSession | null = decoderRun
          ? {
              run: decoderRun,
              dispose: () => decoderModel!.dispose(),
            }
          : null

        setSession(sessionWrapper)
        setDecoderSession(decoderWrapper)
        setIsReady(true)

        console.log(`jax-js WebGPU ONNX initialized for ${config.name}`)
        console.log('  Full model ready (JIT compiled + warmed up)')
        if (decoderWrapper) {
          console.log('  Decoder model ready (JIT compiled + warmed up)')
        }
      } catch (e) {
        console.error('Failed to initialize jax-js ONNX:', e)
        setError(e instanceof Error ? e.message : 'Failed to initialize jax-js ONNX')
      }
    }

    initModels()
  }, [config])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      modelsRef.current.full?.dispose()
      modelsRef.current.decoder?.dispose()
    }
  }, [])

  return { session, decoderSession, isReady, error }
}

/**
 * Result from policy network inference.
 */
export interface InferenceResult {
  logits: Float32Array
  latent: Float32Array
}

/**
 * Run inference on the policy network.
 * Input: observation vector (size from config.obs.totalSize)
 * Output: logits (action means + action log_stds) and latent vector
 */
// Performance monitoring
let inferenceCount = 0
let totalInferenceTime = 0

export function runInference(
  session: JaxSession,
  observation: Float32Array
): InferenceResult {
  const t0 = performance.now()

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const input = np.array(observation as any).reshape([1, observation.length])
  const output = session.run({ obs: input })

  // Get references to keep arrays alive, extract data, then dispose
  const actionLogits = output['action_logits'].ref
  const latentMean = output['latent_mean'].ref

  const logits = new Float32Array(actionLogits.dataSync())
  const latent = new Float32Array(latentMean.dataSync())

  // Now dispose (decrements refcount we added with .ref)
  actionLogits.dispose()
  latentMean.dispose()

  const elapsed = performance.now() - t0
  inferenceCount++
  totalInferenceTime += elapsed

  // Log every 100 inferences
  if (inferenceCount % 100 === 0) {
    console.log(`[jax-js] Inference #${inferenceCount}: avg ${(totalInferenceTime / inferenceCount).toFixed(2)}ms`)
  }

  return { logits, latent }
}

/**
 * Extract action from network output into pre-allocated buffer.
 * Takes the first actionSize dims (mean) and applies tanh.
 * Writes result to `out` buffer to avoid per-frame allocations.
 */
export function extractActionInto(out: Float32Array, logits: Float32Array): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.tanh(logits[i])
  }
}

/**
 * Run inference on the decoder-only network (for latent walk mode).
 * Input: latent vector (e.g., 16-dim) + proprioceptive observation (e.g., 277-dim)
 * Output: logits (action means + action log_stds)
 */
export function runDecoderInference(
  session: JaxSession,
  latent: Float32Array,
  proprioObs: Float32Array
): Float32Array {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const latentInput = np.array(latent as any).reshape([1, latent.length])
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const proprioInput = np.array(proprioObs as any).reshape([1, proprioObs.length])
  const output = session.run({ latent: latentInput, proprio_obs: proprioInput })

  // Get reference, extract data, then dispose
  const keys = Object.keys(output)
  const outputRef = output[keys[0]].ref
  const result = new Float32Array(outputRef.dataSync())
  outputRef.dispose()

  return result
}
