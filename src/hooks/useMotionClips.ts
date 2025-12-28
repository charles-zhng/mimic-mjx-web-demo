import { useState, useEffect, useRef } from 'react'
import type { MotionClip, MotionClipsData, MotionClipsMetadata } from '../types'
import type { AnimalConfig } from '../types/animal-config'

interface UseMotionClipsResult {
  clips: MotionClip[] | null
  metadata: MotionClipsMetadata | null
  isReady: boolean
  error: string | null
}

export function useMotionClips(config: AnimalConfig): UseMotionClipsResult {
  const [clips, setClips] = useState<MotionClip[] | null>(null)
  const [metadata, setMetadata] = useState<MotionClipsMetadata | null>(null)
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

    async function load() {
      try {
        // Load motion clips JSON
        const response = await fetch(config.assets.clipsPath)
        if (!response.ok) {
          throw new Error(`Failed to fetch motion clips: ${response.status}`)
        }

        const data: MotionClipsData = await response.json()

        setClips(data.clips)
        setMetadata(data.metadata)
        setIsReady(true)

        console.log(`Motion clips loaded for ${config.name}`)
        console.log(`  ${data.clips.length} clips`)
        console.log(`  qpos_dim=${data.metadata.qpos_dim}, qvel_dim=${data.metadata.qvel_dim}`)
        console.log('  Clip names:', data.clips.map(c => c.name).join(', '))
      } catch (e) {
        console.error('Failed to load motion clips:', e)
        setError(e instanceof Error ? e.message : 'Failed to load motion clips')
      }
    }

    load()
  }, [config])

  return { clips, metadata, isReady, error }
}
