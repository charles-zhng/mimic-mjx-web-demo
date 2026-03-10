import { useState, useCallback } from 'react'

interface TouchButtonsProps {
  pressKey: (key: string) => void
  releaseKey: (key: string) => void
}

interface ButtonConfig {
  key: string
  label: React.ReactNode
}

const Arrow = ({ d }: { d: string }) => (
  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
)

const LEFT_BUTTONS: ButtonConfig[] = [
  { key: 'w', label: <Arrow d="M12 19V5M5 12l7-7 7 7" /> },
]

const RIGHT_BUTTONS: ButtonConfig[] = [
  { key: 'q', label: <Arrow d="M19 12H5M12 5l-7 7 7 7" /> },
  { key: 'e', label: <Arrow d="M5 12h14M12 5l7 7-7 7" /> },
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

  const renderButton = ({ key, label }: ButtonConfig) => (
    <div
      key={key}
      className={`touch-button${pressed.has(key) ? ' pressed' : ''}`}
      onTouchStart={(e) => { e.preventDefault(); handlePress(key) }}
      onTouchEnd={(e) => { e.preventDefault(); handleRelease(key) }}
      onTouchCancel={(e) => { e.preventDefault(); handleRelease(key) }}
      onMouseDown={() => handlePress(key)}
      onMouseUp={() => handleRelease(key)}
      onMouseLeave={() => { if (pressed.has(key)) handleRelease(key) }}
    >
      {label}
    </div>
  )

  return (
    <div className="touch-buttons">
      <div className="touch-buttons-left">
        {LEFT_BUTTONS.map(renderButton)}
      </div>
      <div className="touch-buttons-right">
        {RIGHT_BUTTONS.map(renderButton)}
      </div>
    </div>
  )
}
