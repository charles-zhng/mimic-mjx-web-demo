/**
 * Benchmark comparing onnxruntime-web vs jax-js/onnx (WebGPU) for neural network inference.
 *
 * Usage: Import and call runBenchmark() or access via the benchmark.html page.
 */

import * as ort from 'onnxruntime-web'
import { ONNXModel } from '@jax-js/onnx'
import { numpy as np, init, defaultDevice, jit } from '@jax-js/jax'

export interface BenchmarkResult {
  library: string
  warmupIterations: number
  benchmarkIterations: number
  warmupTimeMs: number
  totalTimeMs: number
  avgTimeMs: number
  minTimeMs: number
  maxTimeMs: number
  stdDevMs: number
  throughputInferencesPerSec: number
}

export interface BenchmarkConfig {
  modelPath: string
  inputShape: number[]
  warmupIterations: number
  benchmarkIterations: number
  onProgress?: (message: string) => void
}

const defaultConfig: BenchmarkConfig = {
  modelPath: '/nn/intention_network.onnx',
  inputShape: [1, 917],
  warmupIterations: 50,
  benchmarkIterations: 500,
}

function calculateStats(times: number[]): {
  avg: number
  min: number
  max: number
  stdDev: number
} {
  const avg = times.reduce((a, b) => a + b, 0) / times.length
  const min = Math.min(...times)
  const max = Math.max(...times)
  const variance = times.reduce((sum, t) => sum + (t - avg) ** 2, 0) / times.length
  const stdDev = Math.sqrt(variance)
  return { avg, min, max, stdDev }
}

/**
 * Benchmark onnxruntime-web inference
 */
async function benchmarkOnnxRuntime(config: BenchmarkConfig): Promise<BenchmarkResult> {
  const log = config.onProgress ?? console.log

  log('[onnxruntime-web] Loading model...')

  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/'

  const session = await ort.InferenceSession.create(config.modelPath, {
    executionProviders: ['wasm'] as const,
    graphOptimizationLevel: 'all' as const,
  })
  log(`[onnxruntime-web] Model loaded. Inputs: ${session.inputNames}`)

  const inputData = new Float32Array(config.inputShape.reduce((a, b) => a * b, 1))
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.random() * 2 - 1
  }

  // Warmup
  log(`[onnxruntime-web] Running ${config.warmupIterations} warmup iterations...`)
  const warmupStart = performance.now()
  for (let i = 0; i < config.warmupIterations; i++) {
    const inputTensor = new ort.Tensor('float32', inputData, config.inputShape)
    await session.run({ obs: inputTensor })
  }
  const warmupTime = performance.now() - warmupStart

  // Benchmark
  log(`[onnxruntime-web] Running ${config.benchmarkIterations} benchmark iterations...`)
  const times: number[] = []
  const totalStart = performance.now()

  for (let i = 0; i < config.benchmarkIterations; i++) {
    const start = performance.now()
    const inputTensor = new ort.Tensor('float32', inputData, config.inputShape)
    await session.run({ obs: inputTensor })
    times.push(performance.now() - start)
  }

  const totalTime = performance.now() - totalStart
  const stats = calculateStats(times)

  return {
    library: 'onnxruntime-web',
    warmupIterations: config.warmupIterations,
    benchmarkIterations: config.benchmarkIterations,
    warmupTimeMs: warmupTime,
    totalTimeMs: totalTime,
    avgTimeMs: stats.avg,
    minTimeMs: stats.min,
    maxTimeMs: stats.max,
    stdDevMs: stats.stdDev,
    throughputInferencesPerSec: 1000 / stats.avg,
  }
}

/**
 * Benchmark jax-js/onnx inference with WebGPU backend
 */
async function benchmarkJaxJsWebGpu(config: BenchmarkConfig): Promise<BenchmarkResult> {
  const log = config.onProgress ?? console.log

  log('[jax-js webgpu] Initializing backend...')

  const devices = await init('webgpu')
  if (!devices.includes('webgpu')) {
    throw new Error('WebGPU not available')
  }
  defaultDevice('webgpu')
  log(`[jax-js webgpu] Backend initialized: ${defaultDevice()}`)

  log('[jax-js webgpu] Loading model...')
  const modelBytes = await fetch(config.modelPath).then((r) => r.arrayBuffer())
  const model = new ONNXModel(new Uint8Array(modelBytes))
  const run = jit(model.run)
  log('[jax-js webgpu] Model loaded and JIT-compiled.')

  const inputSize = config.inputShape.reduce((a, b) => a * b, 1)
  const inputData = new Float32Array(inputSize)
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.random() * 2 - 1
  }

  // Warmup
  log(`[jax-js webgpu] Running ${config.warmupIterations} warmup iterations...`)
  const warmupStart = performance.now()
  for (let i = 0; i < config.warmupIterations; i++) {
    const input = np.array(inputData).reshape(config.inputShape)
    const output = run({ obs: input })
    for (const key of Object.keys(output)) {
      const arr = output[key].ref
      new Float32Array(arr.dataSync())
      arr.dispose()
    }
  }
  const warmupTime = performance.now() - warmupStart

  // Benchmark
  log(`[jax-js webgpu] Running ${config.benchmarkIterations} benchmark iterations...`)
  const times: number[] = []
  const totalStart = performance.now()

  for (let i = 0; i < config.benchmarkIterations; i++) {
    const start = performance.now()
    const input = np.array(inputData).reshape(config.inputShape)
    const output = run({ obs: input })
    for (const key of Object.keys(output)) {
      const arr = output[key].ref
      new Float32Array(arr.dataSync())
      arr.dispose()
    }
    times.push(performance.now() - start)
  }

  const totalTime = performance.now() - totalStart
  const stats = calculateStats(times)

  model.dispose()

  return {
    library: 'jax-js-webgpu',
    warmupIterations: config.warmupIterations,
    benchmarkIterations: config.benchmarkIterations,
    warmupTimeMs: warmupTime,
    totalTimeMs: totalTime,
    avgTimeMs: stats.avg,
    minTimeMs: stats.min,
    maxTimeMs: stats.max,
    stdDevMs: stats.stdDev,
    throughputInferencesPerSec: 1000 / stats.avg,
  }
}

export interface FullBenchmarkResult {
  onnxRuntime: BenchmarkResult
  jaxJsWebGpu: BenchmarkResult
}

/**
 * Run benchmark comparing onnxruntime-web vs jax-js WebGPU
 */
export async function runBenchmark(
  config: Partial<BenchmarkConfig> = {}
): Promise<FullBenchmarkResult> {
  const fullConfig = { ...defaultConfig, ...config }
  const log = fullConfig.onProgress ?? console.log

  log('=== ONNX Inference Benchmark ===')
  log(`Model: ${fullConfig.modelPath}`)
  log(`Input shape: [${fullConfig.inputShape.join(', ')}]`)
  log(`Warmup: ${fullConfig.warmupIterations}, Benchmark: ${fullConfig.benchmarkIterations}`)
  log('')

  const onnxRuntime = await benchmarkOnnxRuntime(fullConfig)
  log('')

  const jaxJsWebGpu = await benchmarkJaxJsWebGpu(fullConfig)
  log('')

  // Results
  log('=== Results ===')
  log('')
  log(`onnxruntime-web:`)
  log(`  Avg: ${onnxRuntime.avgTimeMs.toFixed(3)} ms`)
  log(`  Throughput: ${onnxRuntime.throughputInferencesPerSec.toFixed(1)} inferences/sec`)
  log('')
  log(`jax-js (WebGPU):`)
  log(`  Avg: ${jaxJsWebGpu.avgTimeMs.toFixed(3)} ms`)
  log(`  Throughput: ${jaxJsWebGpu.throughputInferencesPerSec.toFixed(1)} inferences/sec`)
  log('')

  const speedup = onnxRuntime.avgTimeMs / jaxJsWebGpu.avgTimeMs
  if (speedup > 1) {
    log(`jax-js WebGPU is ${speedup.toFixed(2)}x faster`)
  } else {
    log(`onnxruntime-web is ${(1 / speedup).toFixed(2)}x faster`)
  }

  return { onnxRuntime, jaxJsWebGpu }
}

export { benchmarkOnnxRuntime, benchmarkJaxJsWebGpu }
