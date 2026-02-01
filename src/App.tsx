import { useState, useMemo, useCallback } from 'react'
import Viewer from './components/Viewer'
import Controls from './components/Controls'
import AnalysisPanel from './components/AnalysisPanel'
import { useMuJoCo } from './hooks/useMuJoCo'
import { useONNX } from './hooks/useONNX'
import { useMotionClips } from './hooks/useMotionClips'
import { useSimulation, type InferenceMode } from './hooks/useSimulation'
import { useKeyboardInput } from './hooks/useKeyboardInput'
import { getAnimalConfig, defaultAnimalId } from './config/animals'
import type { VisualizationData, VisualizationHistory } from './types/visualization'

export type LatentWalkInitialPose = 'default' | 'rearing'

// Clip and frame indices for initial poses in latent walk mode
const LATENT_WALK_INITIAL_POSES: Record<LatentWalkInitialPose, { clip: number; frame: number }> = {
  default: { clip: 5, frame: 0 },    // Walk_4, first frame
  rearing: { clip: 9, frame: 100 },  // Rear_149, peak rearing frame
}

function App() {
  const [animalId, setAnimalId] = useState(defaultAnimalId)
  const [selectedClip, setSelectedClip] = useState(0) // FastWalk_76
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1.0)
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>('tracking')
  const [latentWalkInitialPose, setLatentWalkInitialPose] = useState<LatentWalkInitialPose>('default')
  const [noiseMagnitude, setNoiseMagnitude] = useState(1.0)

  // Analysis panel state
  const [showAnalysis, setShowAnalysis] = useState(false)
  const [selectedJointIndex, setSelectedJointIndex] = useState(8) // hip_L_extend
  const [vizHistory, setVizHistory] = useState<VisualizationHistory>({
    data: [],
    maxLength: 200, // ~2 seconds at 100Hz
  })

  // Get config for the selected animal (memoized to prevent unnecessary re-renders)
  const config = useMemo(() => getAnimalConfig(animalId), [animalId])

  const { mujoco, model, data, ghostData, isReady: mujocoReady, error: mujocoError } = useMuJoCo(config)
  const { session, decoderSession, highlevelSession, metadata, isReady: onnxReady, error: onnxError } = useONNX(config)
  const { clips, isReady: clipsReady, error: clipsError } = useMotionClips(config)

  // Keyboard input for joystick mode
  const joystickCommand = useKeyboardInput(
    inferenceMode === 'joystick',
    config.joystick?.commandRanges
  )

  const isReady = mujocoReady && onnxReady && clipsReady
  const error = mujocoError || onnxError || clipsError

  // Determine which clip and frame to use for initial pose
  const initialPose = (inferenceMode === 'latentWalk' || inferenceMode === 'latentNoise' || inferenceMode === 'joystick')
    ? LATENT_WALK_INITIAL_POSES[latentWalkInitialPose]
    : { clip: selectedClip, frame: 0 }

  // Visualization data callback
  const handleVisualizationData = useCallback((vizData: VisualizationData) => {
    setVizHistory((prev) => ({
      ...prev,
      data: [...prev.data.slice(-(prev.maxLength - 1)), vizData],
    }))
  }, [])

  // Clear visualization history when clip changes or simulation resets
  const clearVizHistory = useCallback(() => {
    setVizHistory((prev) => ({ ...prev, data: [] }))
  }, [])

  const simulation = useSimulation({
    mujoco,
    model,
    data,
    ghostData,
    session,
    decoderSession,
    highlevelSession,
    metadata,
    clips,
    selectedClip: initialPose.clip,
    initialFrame: initialPose.frame,
    isPlaying,
    speed,
    isReady,
    config,
    inferenceMode,
    noiseMagnitude,
    selectedJointIndex,
    joystickCommand: inferenceMode === 'joystick' ? joystickCommand : null,
    onVisualizationData: inferenceMode === 'tracking' && showAnalysis ? handleVisualizationData : undefined,
  })

  const handleReset = () => {
    simulation.reset()
    clearVizHistory()
  }

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying)
  }

  // Handle inference mode change (reset selections to defaults)
  const handleModeChange = (mode: InferenceMode) => {
    setInferenceMode(mode)
    setSelectedClip(0) // FastWalk_76
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

        {inferenceMode === 'tracking' && (
          <AnalysisPanel
            history={vizHistory}
            selectedJointIndex={selectedJointIndex}
            onJointChange={setSelectedJointIndex}
            isVisible={showAnalysis}
            onToggle={() => setShowAnalysis(!showAnalysis)}
            latentSize={metadata?.latent_size}
          />
        )}

        <div className="status-bar">
          {config.name} | {
            inferenceMode === 'tracking' ? 'Tracking' :
            inferenceMode === 'joystick' ? 'Joystick' :
            'Latent Walk'
          }
          {inferenceMode === 'tracking' && ` | Frame: ${simulation.currentFrame} / ${clips?.[selectedClip]?.num_frames ?? 0}`}
          {inferenceMode === 'joystick' && joystickCommand && (
            ` | vx: ${joystickCommand.vx.toFixed(2)} vyaw: ${joystickCommand.vyaw.toFixed(2)}`
          )}
          {' | '}
          Speed: {speed.toFixed(1)}x
        </div>

        <div className="viewer-help">
          <div className="viewer-help-title">
            Camera <span className="viewer-help-title-note"></span>
          </div>
          <div className="viewer-help-items">
            <div className="viewer-help-item">
              <svg className="input-icon mouse left" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6.5" y="2.5" width="11" height="19" rx="5.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
                <line x1="12" y1="2.5" x2="12" y2="9" stroke="currentColor" strokeWidth="1.5" />
                <path d="M6.5 2.5h5.5v6.5H6.5z" fill="currentColor" opacity="0.6" />
              </svg>
              <span>Rotate</span>
            </div>
            <div className="viewer-help-item">
              <svg className="input-icon mouse right" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6.5" y="2.5" width="11" height="19" rx="5.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
                <line x1="12" y1="2.5" x2="12" y2="9" stroke="currentColor" strokeWidth="1.5" />
                <path d="M12 2.5h5.5v6.5H12z" fill="currentColor" opacity="0.6" />
              </svg>
              <span>Pan</span>
            </div>
            <div className="viewer-help-item">
              <svg className="input-icon mouse wheel" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6.5" y="2.5" width="11" height="19" rx="5.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
                <line x1="12" y1="2.5" x2="12" y2="9" stroke="currentColor" strokeWidth="1.5" />
                <rect x="10.5" y="6.5" width="3" height="5" rx="1.2" fill="currentColor" opacity="0.7" />
              </svg>
              <span>Zoom</span>
            </div>
            <div className="viewer-help-item">
              <span className="keycap">Shift</span>
              <svg className="input-icon mouse left" viewBox="0 0 24 24" aria-hidden="true">
                <rect x="6.5" y="2.5" width="11" height="19" rx="5.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
                <line x1="12" y1="2.5" x2="12" y2="9" stroke="currentColor" strokeWidth="1.5" />
                <path d="M6.5 2.5h5.5v6.5H6.5z" fill="currentColor" opacity="0.6" />
              </svg>
              <span>Pan</span>
            </div>
          </div>
        </div>

        <div className="project-link">
          <div>
            Part of the{' '}
            <a href="https://mimic-mjx.talmolab.org/" target="_blank" rel="noopener noreferrer">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ verticalAlign: 'middle', marginRight: '3px' }}>
                <circle cx="12" cy="12" r="10" />
                <line x1="2" y1="12" x2="22" y2="12" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
              MIMIC-MJX
            </a>{' '}
            project
          </div>
          <div style={{ textAlign: 'center' }}>
            Created by{' '}
            <a href="https://charles-zhng.github.io/" target="_blank" rel="noopener noreferrer">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ verticalAlign: 'middle', marginRight: '3px' }}>
                <circle cx="12" cy="8" r="4" />
                <path d="M20 21a8 8 0 1 0-16 0" />
              </svg>
              Charles Zhang
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
