import type { MotionClip } from '../types'

interface ControlsProps {
  clips: MotionClip[] | null
  selectedClip: number
  onClipChange: (index: number) => void
  isPlaying: boolean
  onTogglePlay: () => void
  onReset: () => void
  speed: number
  onSpeedChange: (speed: number) => void
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
}: ControlsProps) {
  return (
    <div className="controls-panel">
      <h2>MIMIC-MJX Demo</h2>

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
