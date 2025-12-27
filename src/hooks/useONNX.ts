import { useState, useEffect, useRef } from 'react'
import * as ort from 'onnxruntime-web'

interface UseONNXResult {
  session: ort.InferenceSession | null
  isReady: boolean
  error: string | null
}

export function useONNX(): UseONNXResult {
  const [session, setSession] = useState<ort.InferenceSession | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const initRef = useRef(false)

  useEffect(() => {
    if (initRef.current) return
    initRef.current = true

    async function init() {
      try {
        // Configure ONNX Runtime for WASM backend
        // Use CDN for production, local for development
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/'

        // Create inference session
        const onnxSession = await ort.InferenceSession.create(
          '/nn/intention_network.onnx?v=3',
          {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
          }
        )

        setSession(onnxSession)
        setIsReady(true)

        console.log('ONNX Runtime initialized successfully')
        console.log('  Input names:', onnxSession.inputNames)
        console.log('  Output names:', onnxSession.outputNames)
      } catch (e) {
        console.error('Failed to initialize ONNX Runtime:', e)
        setError(e instanceof Error ? e.message : 'Failed to initialize ONNX Runtime')
      }
    }

    init()
  }, [])

  return { session, isReady, error }
}

/**
 * Run inference on the policy network.
 * Input: 917-dim observation vector
 * Output: 76-dim logits (38 action means + 38 action log_stds)
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
 * Takes the first 38 dims (mean) and applies tanh.
 */
export function extractAction(logits: Float32Array): Float32Array {
  const actionSize = 38
  const action = new Float32Array(actionSize)
  for (let i = 0; i < actionSize; i++) {
    action[i] = Math.tanh(logits[i])
  }
  return action
}
