import { useState, useEffect, useRef } from 'react'
import * as ort from 'onnxruntime-web'
import type { AnimalConfig } from '../types/animal-config'

interface UseONNXResult {
  session: ort.InferenceSession | null
  decoderSession: ort.InferenceSession | null
  isReady: boolean
  error: string | null
}

export function useONNX(config: AnimalConfig): UseONNXResult {
  const [session, setSession] = useState<ort.InferenceSession | null>(null)
  const [decoderSession, setDecoderSession] = useState<ort.InferenceSession | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const initRef = useRef(false)
  const configIdRef = useRef(config.id)

  useEffect(() => {
    // Reset if config changed
    if (configIdRef.current !== config.id) {
      configIdRef.current = config.id
      initRef.current = false
      setIsReady(false)
      setError(null)
    }

    if (initRef.current) return
    initRef.current = true

    async function init() {
      try {
        // Configure ONNX Runtime for WASM backend
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/'

        const sessionOptions = {
          executionProviders: ['wasm'] as const,
          graphOptimizationLevel: 'all' as const,
        }

        // Load full model and decoder model in parallel
        const [onnxSession, decoderOnnxSession] = await Promise.all([
          ort.InferenceSession.create(config.assets.onnxPath, sessionOptions),
          config.assets.decoderOnnxPath
            ? ort.InferenceSession.create(config.assets.decoderOnnxPath, sessionOptions)
            : Promise.resolve(null),
        ])

        setSession(onnxSession)
        setDecoderSession(decoderOnnxSession)
        setIsReady(true)

        console.log(`ONNX Runtime initialized for ${config.name}`)
        console.log('  Full model inputs:', onnxSession.inputNames)
        console.log('  Full model outputs:', onnxSession.outputNames)
        if (decoderOnnxSession) {
          console.log('  Decoder model inputs:', decoderOnnxSession.inputNames)
          console.log('  Decoder model outputs:', decoderOnnxSession.outputNames)
        }
      } catch (e) {
        console.error('Failed to initialize ONNX Runtime:', e)
        setError(e instanceof Error ? e.message : 'Failed to initialize ONNX Runtime')
      }
    }

    init()
  }, [config])

  return { session, decoderSession, isReady, error }
}

/**
 * Run inference on the policy network.
 * Input: observation vector (size from config.obs.totalSize)
 * Output: logits (action means + action log_stds)
 */
export async function runInference(
  session: ort.InferenceSession,
  observation: Float32Array
): Promise<Float32Array> {
  const inputTensor = new ort.Tensor('float32', observation, [1, observation.length])
  const feeds = { obs: inputTensor }
  const results = await session.run(feeds)
  const output = results[session.outputNames[0]]
  return output.data as Float32Array
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
export async function runDecoderInference(
  session: ort.InferenceSession,
  latent: Float32Array,
  proprioObs: Float32Array
): Promise<Float32Array> {
  const latentTensor = new ort.Tensor('float32', latent, [1, latent.length])
  const proprioTensor = new ort.Tensor('float32', proprioObs, [1, proprioObs.length])
  const feeds = { latent: latentTensor, proprio_obs: proprioTensor }
  const results = await session.run(feeds)
  const output = results[session.outputNames[0]]
  return output.data as Float32Array
}
