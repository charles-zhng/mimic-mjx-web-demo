import { useRef, useEffect, useMemo } from 'react'

interface LineData {
  label: string
  values: number[]
  color: string
}

interface LineChartProps {
  lines: LineData[]
  width?: number
  height?: number
  yLabel?: string
  showLegend?: boolean
}

export default function LineChart({
  lines,
  width = 200,
  height = 100,
  yLabel,
  showLegend = true,
}: LineChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Calculate y-axis bounds
  const { yMin, yMax } = useMemo(() => {
    let min = Infinity
    let max = -Infinity
    for (const line of lines) {
      for (const v of line.values) {
        if (v < min) min = v
        if (v > max) max = v
      }
    }
    // Add padding
    const range = max - min || 1
    return {
      yMin: min - range * 0.1,
      yMax: max + range * 0.1,
    }
  }, [lines])

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

    // Margins
    const marginLeft = 35
    const marginRight = 10
    const marginTop = 10
    const marginBottom = 15
    const chartWidth = width - marginLeft - marginRight
    const chartHeight = height - marginTop - marginBottom

    // Draw grid lines
    ctx.strokeStyle = '#2c2c2c'
    ctx.lineWidth = 0.5
    const numGridLines = 4
    for (let i = 0; i <= numGridLines; i++) {
      const y = marginTop + (chartHeight * i) / numGridLines
      ctx.beginPath()
      ctx.moveTo(marginLeft, y)
      ctx.lineTo(width - marginRight, y)
      ctx.stroke()
    }

    // Draw y-axis labels
    ctx.fillStyle = '#666'
    ctx.font = '9px Inter, sans-serif'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let i = 0; i <= numGridLines; i++) {
      const y = marginTop + (chartHeight * i) / numGridLines
      const value = yMax - ((yMax - yMin) * i) / numGridLines
      ctx.fillText(value.toFixed(1), marginLeft - 3, y)
    }

    // Draw y-label
    if (yLabel) {
      ctx.save()
      ctx.translate(8, height / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillStyle = '#888'
      ctx.font = '9px Inter, sans-serif'
      ctx.fillText(yLabel, 0, 0)
      ctx.restore()
    }

    // Draw lines
    for (const line of lines) {
      if (line.values.length < 2) continue

      ctx.strokeStyle = line.color
      ctx.lineWidth = 1.5
      ctx.beginPath()

      const xStep = chartWidth / (line.values.length - 1)
      for (let i = 0; i < line.values.length; i++) {
        const x = marginLeft + i * xStep
        const normalizedY = (line.values[i] - yMin) / (yMax - yMin)
        const y = marginTop + chartHeight * (1 - normalizedY)

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      }
      ctx.stroke()
    }

    // Draw legend
    if (showLegend && lines.length > 0) {
      const legendX = marginLeft + 5
      const legendY = marginTop + 5
      ctx.font = '8px Inter, sans-serif'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'

      lines.forEach((line, i) => {
        const y = legendY + i * 12
        ctx.fillStyle = line.color
        ctx.fillRect(legendX, y + 2, 8, 8)
        ctx.fillStyle = '#aaa'
        ctx.fillText(line.label, legendX + 12, y)
      })
    }
  }, [lines, width, height, yMin, yMax, yLabel, showLegend])

  return (
    <canvas
      ref={canvasRef}
      style={{ width, height }}
    />
  )
}
