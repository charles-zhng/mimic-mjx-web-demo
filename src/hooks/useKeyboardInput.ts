import { useState, useEffect, useCallback, useRef } from 'react'

/**
 * Joystick command vector for velocity control.
 */
export interface JoystickCommand {
  /** Forward/backward velocity in m/s */
  vx: number
  /** Lateral velocity (strafe) in m/s */
  vy: number
  /** Yaw rotation rate in rad/s */
  vyaw: number
}

interface CommandRanges {
  vx: [number, number]
  vy: [number, number]
  vyaw: [number, number]
}

const DEFAULT_RANGES: CommandRanges = {
  vx: [-0.5, 0.5],
  vy: [-0.3, 0.3],
  vyaw: [-1.0, 1.0],
}

/**
 * Hook that captures WASD QE keyboard input and converts to joystick command.
 *
 * Key mappings:
 * - W: Forward (+vx)
 * - Q: Turn right (+vyaw)
 * - E: Turn left (-vyaw)
 *
 * @param enabled Whether to capture keyboard input
 * @param ranges Command value ranges for each axis
 * @returns Current joystick command based on pressed keys
 */
export function useKeyboardInput(
  enabled: boolean = true,
  ranges: CommandRanges = DEFAULT_RANGES
): JoystickCommand {
  // Track which keys are currently pressed
  const keysRef = useRef<Set<string>>(new Set())
  const [command, setCommand] = useState<JoystickCommand>({ vx: 0, vy: 0, vyaw: 0 })

  // Compute command from pressed keys
  const updateCommand = useCallback(() => {
    const keys = keysRef.current

    let vx = 0
    let vy = 0
    let vyaw = 0

    // Forward (W only)
    if (keys.has('w') || keys.has('W')) vx += ranges.vx[1]

    // Turn right/left (Q/E)
    if (keys.has('q') || keys.has('Q')) vyaw += ranges.vyaw[1]
    if (keys.has('e') || keys.has('E')) vyaw += ranges.vyaw[0]

    setCommand({ vx, vy, vyaw })
  }, [ranges])

  useEffect(() => {
    if (!enabled) {
      // Clear command when disabled
      setCommand({ vx: 0, vy: 0, vyaw: 0 })
      keysRef.current.clear()
      return
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      const key = e.key
      if (!keysRef.current.has(key)) {
        keysRef.current.add(key)
        updateCommand()
      }
    }

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key
      if (keysRef.current.has(key)) {
        keysRef.current.delete(key)
        updateCommand()
      }
    }

    // Clear keys when window loses focus
    const handleBlur = () => {
      keysRef.current.clear()
      setCommand({ vx: 0, vy: 0, vyaw: 0 })
    }

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', handleBlur)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleBlur)
    }
  }, [enabled, updateCommand])

  return command
}
