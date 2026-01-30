import type { MotionClip } from '../types'
import type { InferenceMode } from '../hooks/useSimulation'
import type { LatentWalkInitialPose } from '../App'
import Select from './Select'

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
  noiseMagnitude: number
  onNoiseMagnitudeChange: (magnitude: number) => void
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
  noiseMagnitude,
  onNoiseMagnitudeChange,
}: ControlsProps) {
  return (
    <div className="controls-panel">
      <h2>MIMIC-MJX Demo</h2>

      <div className="control-group">
        <label>Inference Mode</label>
        <Select
          value={inferenceMode}
          options={[
            { value: 'tracking', label: 'Motion Tracking' },
            { value: 'joystick', label: 'Joystick Control (W QE)' },
            { value: 'latentNoise', label: 'Latent Random Noise (independent)' },
            { value: 'latentWalk', label: 'Latent Random Walk (OU process)' },
          ]}
          onChange={(value) => onModeChange(value as InferenceMode)}
        />
      </div>

      {inferenceMode === 'tracking' ? (
        <div className="control-group">
          <label>Motion Clip</label>
          <Select
            value={String(selectedClip)}
            options={clips?.map((clip, index) => ({
              value: String(index),
              label: clip.name,
            })) ?? []}
            onChange={(value) => onClipChange(Number(value))}
            disabled={!clips}
          />
        </div>
      ) : inferenceMode === 'joystick' ? (
        <>
          <div className="control-group">
            <label>Initial Pose</label>
            <Select
              value={latentWalkInitialPose}
              options={[
                { value: 'default', label: 'Default' },
                { value: 'rearing', label: 'Rearing' },
              ]}
              onChange={(value) => onLatentWalkInitialPoseChange(value as LatentWalkInitialPose)}
            />
          </div>
          <div className="keyboard-help">
            <div className="keyboard-help-title">Keyboard Controls</div>
            <div className="keyboard-help-row">
              <span className="keycap">W</span> Forward
            </div>
            <div className="keyboard-help-row">
              <span className="keycap">Q</span> Turn Right
              <span className="keycap">E</span> Turn Left
            </div>
          </div>
        </>
      ) : (
        <div className="control-group">
          <label>Initial Pose</label>
          <Select
            value={latentWalkInitialPose}
            options={[
              { value: 'default', label: 'Default' },
              { value: 'rearing', label: 'Rearing' },
            ]}
            onChange={(value) => onLatentWalkInitialPoseChange(value as LatentWalkInitialPose)}
          />
        </div>
      )}

      {(inferenceMode === 'latentWalk' || inferenceMode === 'latentNoise') && (
        <div className="control-group">
          <label>Noise: {noiseMagnitude.toFixed(1)}x</label>
          <input
            type="range"
            min="0"
            max="2.0"
            step="0.1"
            value={noiseMagnitude}
            onChange={(e) => onNoiseMagnitudeChange(Number(e.target.value))}
            style={{ '--fill-percent': `${(noiseMagnitude / 2.0) * 100}%` } as React.CSSProperties}
          />
          <div className="speed-display">0x — 2.0x</div>
        </div>
      )}

      <div className="control-group">
        <label>Speed: {speed.toFixed(1)}x</label>
        <input
          type="range"
          min="0"
          max="2.0"
          step="0.1"
          value={speed}
          onChange={(e) => onSpeedChange(Math.max(0.1, Number(e.target.value)))}
          style={{ '--fill-percent': `${(speed / 2.0) * 100}%` } as React.CSSProperties}
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
