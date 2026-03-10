import { useState, useCallback } from 'react'

interface TouchButtonsProps {
  pressKey: (key: string) => void
  releaseKey: (key: string) => void
}

interface ButtonConfig {
  key: string
  label: string
}

const BUTTONS: ButtonConfig[] = [
  { key: 'w', label: 'W' },
  { key: 'q', label: 'Q' },
  { key: 'e', label: 'E' },
]

export default function TouchButtons({ pressKey, releaseKey }: TouchButtonsProps) {
  const [pressed, setPressed] = useState<Set<string>>(new Set())

  const handlePress = useCallback((key: string) => {
    setPressed(prev => { const next = new Set(prev); next.add(key); return next })
    pressKey(key)
  }, [pressKey])

  const handleRelease = useCallback((key: string) => {
    setPressed(prev => { const next = new Set(prev); next.delete(key); return next })
    releaseKey(key)
  }, [releaseKey])

  return (
    <div className="touch-buttons">
      {BUTTONS.map(({ key, label }) => (
        <div
          key={key}
          className={`touch-button touch-button-${key}${pressed.has(key) ? ' pressed' : ''}`}
          onTouchStart={(e) => { e.preventDefault(); handlePress(key) }}
          onTouchEnd={(e) => { e.preventDefault(); handleRelease(key) }}
          onTouchCancel={(e) => { e.preventDefault(); handleRelease(key) }}
          onMouseDown={() => handlePress(key)}
          onMouseUp={() => handleRelease(key)}
          onMouseLeave={() => { if (pressed.has(key)) handleRelease(key) }}
        >
          {label}
        </div>
      ))}
    </div>
  )
}
