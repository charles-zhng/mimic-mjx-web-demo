import { useState, useMemo } from 'react'
import Viewer from './components/Viewer'
import Controls from './components/Controls'
import { useMuJoCo } from './hooks/useMuJoCo'
import { useONNX } from './hooks/useONNX'
import { useMotionClips } from './hooks/useMotionClips'
import { useSimulation } from './hooks/useSimulation'
import { getAnimalConfig, defaultAnimalId } from './config/animals'

function App() {
  const [animalId, setAnimalId] = useState(defaultAnimalId)
  const [selectedClip, setSelectedClip] = useState(3) // clip_0280
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1.0)

  // Get config for the selected animal (memoized to prevent unnecessary re-renders)
  const config = useMemo(() => getAnimalConfig(animalId), [animalId])

  const { mujoco, model, data, isReady: mujocoReady, error: mujocoError } = useMuJoCo(config)
  const { session, isReady: onnxReady, error: onnxError } = useONNX(config)
  const { clips, isReady: clipsReady, error: clipsError } = useMotionClips(config)

  const isReady = mujocoReady && onnxReady && clipsReady
  const error = mujocoError || onnxError || clipsError

  const simulation = useSimulation({
    mujoco,
    model,
    data,
    session,
    clips,
    selectedClip,
    isPlaying,
    speed,
    isReady,
    config,
  })

  const handleReset = () => {
    simulation.reset()
  }

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying)
  }

  // Handle animal change (reset everything)
  const handleAnimalChange = (newAnimalId: string) => {
    setIsPlaying(false)
    setSelectedClip(0)
    setAnimalId(newAnimalId)
  }

  // Expose for debugging/future use
  void handleAnimalChange

  if (error) {
    return (
      <div className="app">
        <div className="loading-overlay">
          <h2>Error</h2>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <div className="viewer-container">
        {!isReady && (
          <div className="loading-overlay">
            <h2>Loading {config.name}...</h2>
            <p>
              {!mujocoReady && 'Loading MuJoCo WASM...'}
              {mujocoReady && !onnxReady && 'Loading neural network...'}
              {mujocoReady && onnxReady && !clipsReady && 'Loading motion clips...'}
            </p>
          </div>
        )}

        <Viewer
          mujoco={mujoco}
          model={model}
          data={data}
          isReady={isReady}
        />

        <Controls
          clips={clips}
          selectedClip={selectedClip}
          onClipChange={setSelectedClip}
          isPlaying={isPlaying}
          onTogglePlay={handleTogglePlay}
          onReset={handleReset}
          speed={speed}
          onSpeedChange={setSpeed}
        />

        <div className="status-bar">
          {config.name} | Frame: {simulation.currentFrame} / {clips?.[selectedClip]?.num_frames ?? 0}
          {' | '}
          Speed: {speed.toFixed(1)}x
        </div>
      </div>
    </div>
  )
}

export default App
