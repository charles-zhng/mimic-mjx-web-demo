import { useState } from 'react'
import Viewer from './components/Viewer'
import Controls from './components/Controls'
import { useMuJoCo } from './hooks/useMuJoCo'
import { useONNX } from './hooks/useONNX'
import { useMotionClips } from './hooks/useMotionClips'
import { useSimulation } from './hooks/useSimulation'

function App() {
  const [selectedClip, setSelectedClip] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1.0)

  const { mujoco, model, data, isReady: mujocoReady, error: mujocoError } = useMuJoCo()
  const { session, isReady: onnxReady, error: onnxError } = useONNX()
  const { clips, isReady: clipsReady, error: clipsError } = useMotionClips()

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
  })

  const handleReset = () => {
    simulation.reset()
  }

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying)
  }

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
            <h2>Loading...</h2>
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
          Frame: {simulation.currentFrame} / {clips?.[selectedClip]?.num_frames ?? 0}
          {' | '}
          Speed: {speed.toFixed(1)}x
        </div>
      </div>
    </div>
  )
}

export default App
