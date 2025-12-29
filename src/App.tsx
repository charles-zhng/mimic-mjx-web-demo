import { useState, useMemo } from 'react'
import Viewer from './components/Viewer'
import Controls from './components/Controls'
import { useMuJoCo } from './hooks/useMuJoCo'
import { useONNX } from './hooks/useONNX'
import { useMotionClips } from './hooks/useMotionClips'
import { useSimulation, type InferenceMode } from './hooks/useSimulation'
import { getAnimalConfig, defaultAnimalId } from './config/animals'

export type LatentWalkInitialPose = 'default' | 'rearing'

// Clip indices for initial poses in latent walk mode
const LATENT_WALK_INITIAL_POSE_CLIPS: Record<LatentWalkInitialPose, number> = {
  default: 2,  // Walk_4
  rearing: 3,  // FastWalk_76
}

function App() {
  const [animalId, setAnimalId] = useState(defaultAnimalId)
  const [selectedClip, setSelectedClip] = useState(3) // clip_0280
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1.0)
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>('tracking')
  const [latentWalkInitialPose, setLatentWalkInitialPose] = useState<LatentWalkInitialPose>('default')
  const [noiseMagnitude, setNoiseMagnitude] = useState(1.0)

  // Get config for the selected animal (memoized to prevent unnecessary re-renders)
  const config = useMemo(() => getAnimalConfig(animalId), [animalId])

  const { mujoco, model, data, ghostData, isReady: mujocoReady, error: mujocoError } = useMuJoCo(config)
  const { session, decoderSession, isReady: onnxReady, error: onnxError } = useONNX(config)
  const { clips, isReady: clipsReady, error: clipsError } = useMotionClips(config)

  const isReady = mujocoReady && onnxReady && clipsReady
  const error = mujocoError || onnxError || clipsError

  // Determine which clip to use for initial pose
  const initialPoseClipIndex = (inferenceMode === 'latentWalk' || inferenceMode === 'latentNoise')
    ? LATENT_WALK_INITIAL_POSE_CLIPS[latentWalkInitialPose]
    : selectedClip

  const simulation = useSimulation({
    mujoco,
    model,
    data,
    ghostData,
    session,
    decoderSession,
    clips,
    selectedClip: initialPoseClipIndex,
    isPlaying,
    speed,
    isReady,
    config,
    inferenceMode,
    noiseMagnitude,
  })

  const handleReset = () => {
    simulation.reset()
  }

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying)
  }

  // Handle inference mode change (reset selections to defaults)
  const handleModeChange = (mode: InferenceMode) => {
    setInferenceMode(mode)
    setSelectedClip(3) // FastWalk_76
    setLatentWalkInitialPose('default')
    setNoiseMagnitude(1.0)
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
          ghostData={ghostData}
          isReady={isReady}
          config={config}
          inferenceMode={inferenceMode}
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
          inferenceMode={inferenceMode}
          onModeChange={handleModeChange}
          latentWalkInitialPose={latentWalkInitialPose}
          onLatentWalkInitialPoseChange={setLatentWalkInitialPose}
          noiseMagnitude={noiseMagnitude}
          onNoiseMagnitudeChange={setNoiseMagnitude}
        />

        <div className="status-bar">
          {config.name} | {inferenceMode === 'tracking' ? 'Tracking' : 'Latent Walk'}
          {inferenceMode === 'tracking' && ` | Frame: ${simulation.currentFrame} / ${clips?.[selectedClip]?.num_frames ?? 0}`}
          {' | '}
          Speed: {speed.toFixed(1)}x
        </div>

        <div className="project-link">
          <div>
            Part of the{' '}
            <a href="https://mimic-mjx.talmolab.org/" target="_blank" rel="noopener noreferrer">
              MIMIC-MJX
            </a>{' '}
            project
          </div>
          <div style={{ textAlign: 'center' }}>
            Created by{' '}
            <a href="https://charles-zhng.github.io/" target="_blank" rel="noopener noreferrer">
              Charles Zhang
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
