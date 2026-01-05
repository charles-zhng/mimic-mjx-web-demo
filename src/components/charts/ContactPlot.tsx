import { useRef, useEffect } from 'react'

interface ContactPlotProps {
  /** Array of sensor data over time: [timeStep][sensorIndex] */
  data: boolean[][]
  labels: string[]
  width?: number
  height?: number
  colors?: string[]
}

const DEFAULT_COLORS = ['#4aff9f', '#ff4a9f', '#9f4aff', '#ffff4a']

export default function ContactPlot({
  data,
  labels,
  width = 200,
  height = 60,
  colors = DEFAULT_COLORS,
}: ContactPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    ctx.scale(dpr, dpr)

    // Clear
    ctx.fillStyle = '#141414'
    ctx.fillRect(0, 0, width, height)

    const numSensors = labels.length
    const marginLeft = 50
    const marginRight = 5
    const marginTop = 4
    const marginBottom = 4
    const chartWidth = width - marginLeft - marginRight
    const chartHeight = height - marginTop - marginBottom
    const rowHeight = chartHeight / numSensors
    const rowPadding = 2

    // Draw row backgrounds and labels
    ctx.font = '9px Inter, sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'

    for (let i = 0; i < numSensors; i++) {
      const y = marginTop + i * rowHeight

      // Row background
      ctx.fillStyle = '#1c1c1c'
      ctx.fillRect(marginLeft, y + rowPadding, chartWidth, rowHeight - rowPadding * 2)

      // Label
      ctx.fillStyle = colors[i] ?? '#888'
      ctx.fillText(labels[i], marginLeft - 4, y + rowHeight / 2)
    }

    // Draw contact rectangles
    if (data.length > 0) {
      const timeStep = chartWidth / Math.max(data.length - 1, 1)

      for (let t = 0; t < data.length; t++) {
        const x = marginLeft + t * timeStep
        const rectWidth = Math.max(timeStep, 2)

        for (let s = 0; s < numSensors; s++) {
          if (data[t]?.[s]) {
            const y = marginTop + s * rowHeight + rowPadding
            ctx.fillStyle = colors[s] ?? '#4a9eff'
            ctx.fillRect(x, y, rectWidth, rowHeight - rowPadding * 2)
          }
        }
      }
    }

    // Draw grid lines between rows
    ctx.strokeStyle = '#2c2c2c'
    ctx.lineWidth = 0.5
    for (let i = 1; i < numSensors; i++) {
      const y = marginTop + i * rowHeight
      ctx.beginPath()
      ctx.moveTo(marginLeft, y)
      ctx.lineTo(width - marginRight, y)
      ctx.stroke()
    }
  }, [data, labels, width, height, colors])

  return <canvas ref={canvasRef} style={{ width, height }} />
}
