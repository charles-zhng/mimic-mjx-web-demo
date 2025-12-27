import { useState, useEffect, useRef } from 'react'
import type { MotionClip, MotionClipsData, MotionClipsMetadata } from '../types'

interface UseMotionClipsResult {
  clips: MotionClip[] | null
  metadata: MotionClipsMetadata | null
  isReady: boolean
  error: string | null
}

export function useMotionClips(): UseMotionClipsResult {
  const [clips, setClips] = useState<MotionClip[] | null>(null)
  const [metadata, setMetadata] = useState<MotionClipsMetadata | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const initRef = useRef(false)

  useEffect(() => {
    if (initRef.current) return
    initRef.current = true

    async function load() {
      try {
        // Load motion clips JSON
        const response = await fetch('/motions/clips.json')
        if (!response.ok) {
          throw new Error(`Failed to fetch motion clips: ${response.status}`)
        }

        const data: MotionClipsData = await response.json()

        setClips(data.clips)
        setMetadata(data.metadata)
        setIsReady(true)

        console.log('Motion clips loaded successfully')
        console.log(`  ${data.clips.length} clips`)
        console.log(`  qpos_dim=${data.metadata.qpos_dim}, qvel_dim=${data.metadata.qvel_dim}`)
        console.log('  Clip names:', data.clips.map(c => c.name).join(', '))
      } catch (e) {
        console.error('Failed to load motion clips:', e)
        setError(e instanceof Error ? e.message : 'Failed to load motion clips')
      }
    }

    load()
  }, [])

  return { clips, metadata, isReady, error }
}
