import { useState, useEffect, useRef } from 'react'
import type { MainModule, MjModel, MjData } from '../types'

interface UseMuJoCoResult {
  mujoco: MainModule | null
  model: MjModel | null
  data: MjData | null
  isReady: boolean
  error: string | null
}

export function useMuJoCo(): UseMuJoCoResult {
  const [mujoco, setMujoco] = useState<MainModule | null>(null)
  const [model, setModel] = useState<MjModel | null>(null)
  const [data, setData] = useState<MjData | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const initRef = useRef(false)

  useEffect(() => {
    if (initRef.current) return
    initRef.current = true

    async function init() {
      try {
        // Dynamic import of MuJoCo WASM
        const mujocoModule = await import('mujoco-js')
        const mj = await mujocoModule.default() as MainModule
        setMujoco(mj)

        // Load scaled rodent model (includes arena, scaled to 0.9 to match training)
        const rodentResponse = await fetch('/models/rodent_scaled.xml')
        const rodentXml = await rodentResponse.text()
        mj.FS.writeFile('/rodent_scaled.xml', rodentXml)

        // Load skin file if referenced by the model
        const skinResponse = await fetch('/models/rodent_walker_skin.skn')
        const skinData = await skinResponse.arrayBuffer()
        mj.FS.writeFile('/rodent_walker_skin.skn', new Uint8Array(skinData))

        // Load the model - rodent_scaled.xml includes arena
        const mjModel = mj.MjModel.loadFromXML('/rodent_scaled.xml')
        setModel(mjModel)

        // Create simulation data
        const mjData = new mj.MjData(mjModel)
        setData(mjData)

        // Reset to initial state
        mj.mj_resetData(mjModel, mjData)
        mj.mj_forward(mjModel, mjData)

        setIsReady(true)
        console.log('MuJoCo initialized successfully')
        console.log(`  nq=${mjModel.nq}, nv=${mjModel.nv}, nu=${mjModel.nu}`)
        console.log(`  nbody=${mjModel.nbody}, ngeom=${mjModel.ngeom}`)
      } catch (e) {
        console.error('Failed to initialize MuJoCo:', e)
        setError(e instanceof Error ? e.message : 'Failed to initialize MuJoCo')
      }
    }

    init()
  }, [])

  return { mujoco, model, data, isReady, error }
}
