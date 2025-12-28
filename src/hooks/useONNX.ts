import { useState, useEffect, useRef } from 'react'
import * as ort from 'onnxruntime-web'
import type { AnimalConfig } from '../types/animal-config'

interface UseONNXResult {
  session: ort.InferenceSession | null
  isReady: boolean
  error: string | null
}

export function useONNX(config: AnimalConfig): UseONNXResult {
  const [session, setSession] = useState<ort.InferenceSession | null>(null)
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

        // Create inference session
        const onnxSession = await ort.InferenceSession.create(
          config.assets.onnxPath,
          {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
          }
        )

        setSession(onnxSession)
        setIsReady(true)

        console.log(`ONNX Runtime initialized for ${config.name}`)
        console.log('  Input names:', onnxSession.inputNames)
        console.log('  Output names:', onnxSession.outputNames)
      } catch (e) {
        console.error('Failed to initialize ONNX Runtime:', e)
        setError(e instanceof Error ? e.message : 'Failed to initialize ONNX Runtime')
      }
    }

    init()
  }, [config])

  return { session, isReady, error }
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
 * Extract action from network output.
 * Takes the first actionSize dims (mean) and applies tanh.
 */
export function extractAction(logits: Float32Array, actionSize: number): Float32Array {
  const action = new Float32Array(actionSize)
  for (let i = 0; i < actionSize; i++) {
    action[i] = Math.tanh(logits[i])
  }
  return action
}
