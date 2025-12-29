import type { MotionClip } from '../types'
import type { InferenceMode } from '../hooks/useSimulation'
import type { LatentWalkInitialPose } from '../App'

interface ControlsProps {
  clips: MotionClip[] | null
  selectedClip: number
  onClipChange: (index: number) => void
  isPlaying: boolean
  onTogglePlay: () => void
  onReset: () => void
  speed: number
  onSpeedChange: (speed: number) => void
  inferenceMode: InferenceMode
  onModeChange: (mode: InferenceMode) => void
  latentWalkInitialPose: LatentWalkInitialPose
  onLatentWalkInitialPoseChange: (pose: LatentWalkInitialPose) => void
}

export default function Controls({
  clips,
  selectedClip,
  onClipChange,
  isPlaying,
  onTogglePlay,
  onReset,
  speed,
  onSpeedChange,
  inferenceMode,
  onModeChange,
  latentWalkInitialPose,
  onLatentWalkInitialPoseChange,
}: ControlsProps) {
  return (
    <div className="controls-panel">
      <h2>MIMIC-MJX Demo</h2>

      <div className="control-group">
        <label>Inference Mode</label>
        <select
          value={inferenceMode}
          onChange={(e) => onModeChange(e.target.value as InferenceMode)}
        >
          <option value="tracking">Motion Tracking</option>
          <option value="latentWalk">Latent Random Walk</option>
        </select>
      </div>

      {inferenceMode === 'tracking' ? (
        <div className="control-group">
          <label>Motion Clip</label>
          <select
            value={selectedClip}
            onChange={(e) => onClipChange(Number(e.target.value))}
            disabled={!clips}
          >
            {clips?.map((clip, index) => (
              <option key={clip.name} value={index}>
                {clip.name}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div className="control-group">
          <label>Initial Pose</label>
          <select
            value={latentWalkInitialPose}
            onChange={(e) => onLatentWalkInitialPoseChange(e.target.value as LatentWalkInitialPose)}
          >
            <option value="default">Default</option>
            <option value="rearing">Rearing</option>
          </select>
        </div>
      )}

      <div className="control-group">
        <label>Speed: {speed.toFixed(1)}x</label>
        <input
          type="range"
          min="0.1"
          max="2.0"
          step="0.1"
          value={speed}
          onChange={(e) => onSpeedChange(Number(e.target.value))}
        />
        <div className="speed-display">0.1x — 2.0x</div>
      </div>

      <div className="button-group">
        <button className="primary" onClick={onTogglePlay}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button className="secondary" onClick={onReset}>
          Reset
        </button>
      </div>
    </div>
  )
}
