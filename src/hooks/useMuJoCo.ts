import { useState, useEffect, useRef } from 'react'
import type { MainModule, MjModel, MjData } from '../types'
import type { AnimalConfig } from '../types/animal-config'

interface UseMuJoCoResult {
  mujoco: MainModule | null
  model: MjModel | null
  data: MjData | null
  isReady: boolean
  error: string | null
}

export function useMuJoCo(config: AnimalConfig): UseMuJoCoResult {
  const [mujoco, setMujoco] = useState<MainModule | null>(null)
  const [model, setModel] = useState<MjModel | null>(null)
  const [data, setData] = useState<MjData | null>(null)
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
        // Dynamic import of MuJoCo WASM
        const mujocoModule = await import('mujoco-js')
        const mj = await mujocoModule.default() as MainModule
        setMujoco(mj)

        // Load model XML
        const modelResponse = await fetch(config.assets.modelPath)
        const modelXml = await modelResponse.text()
        const modelFilename = config.assets.modelPath.split('/').pop() || 'model.xml'
        mj.FS.writeFile(`/${modelFilename}`, modelXml)

        // Load skin file if specified
        if (config.assets.skinPath) {
          const skinResponse = await fetch(config.assets.skinPath)
          const skinData = await skinResponse.arrayBuffer()
          const skinFilename = config.assets.skinPath.split('/').pop() || 'skin.skn'
          mj.FS.writeFile(`/${skinFilename}`, new Uint8Array(skinData))
        }

        // Load the model
        const mjModel = mj.MjModel.loadFromXML(`/${modelFilename}`)
        setModel(mjModel)

        // Create simulation data
        const mjData = new mj.MjData(mjModel)
        setData(mjData)

        // Reset to initial state
        mj.mj_resetData(mjModel, mjData)
        mj.mj_forward(mjModel, mjData)

        setIsReady(true)
        console.log(`MuJoCo initialized for ${config.name}`)
        console.log(`  nq=${mjModel.nq}, nv=${mjModel.nv}, nu=${mjModel.nu}`)
        console.log(`  nbody=${mjModel.nbody}, ngeom=${mjModel.ngeom}`)
      } catch (e) {
        console.error('Failed to initialize MuJoCo:', e)
        setError(e instanceof Error ? e.message : 'Failed to initialize MuJoCo')
      }
    }

    init()
  }, [config])

  return { mujoco, model, data, isReady, error }
}
