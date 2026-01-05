import { useRef, useEffect, useMemo } from 'react'

interface Point {
  x: number
  y: number
}

interface ScatterPlotProps {
  points: Point[]
  width?: number
  height?: number
  xLabel?: string
  yLabel?: string
  xDim: number
  yDim: number
  onXDimChange: (dim: number) => void
  onYDimChange: (dim: number) => void
  numDims: number
}

export default function ScatterPlot({
  points,
  width = 200,
  height = 150,
  xLabel,
  yLabel,
  xDim,
  yDim,
  onXDimChange,
  onYDimChange,
  numDims,
}: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Calculate bounds with padding
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    if (points.length === 0) {
      return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 }
    }

    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity

    for (const p of points) {
      if (p.x < minX) minX = p.x
      if (p.x > maxX) maxX = p.x
      if (p.y < minY) minY = p.y
      if (p.y > maxY) maxY = p.y
    }

    // Add padding and ensure minimum range
    const xRange = Math.max(maxX - minX, 0.1)
    const yRange = Math.max(maxY - minY, 0.1)

    return {
      xMin: minX - xRange * 0.1,
      xMax: maxX + xRange * 0.1,
      yMin: minY - yRange * 0.1,
      yMax: maxY + yRange * 0.1,
    }
  }, [points])

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
    const marginLeft = 30
    const marginRight = 10
    const marginTop = 10
    const marginBottom = 25
    const chartWidth = width - marginLeft - marginRight
    const chartHeight = height - marginTop - marginBottom

    // Draw grid
    ctx.strokeStyle = '#2c2c2c'
    ctx.lineWidth = 0.5

    // Vertical grid lines
    for (let i = 0; i <= 4; i++) {
      const x = marginLeft + (chartWidth * i) / 4
      ctx.beginPath()
      ctx.moveTo(x, marginTop)
      ctx.lineTo(x, marginTop + chartHeight)
      ctx.stroke()
    }

    // Horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = marginTop + (chartHeight * i) / 4
      ctx.beginPath()
      ctx.moveTo(marginLeft, y)
      ctx.lineTo(marginLeft + chartWidth, y)
      ctx.stroke()
    }

    // Draw axes labels
    ctx.fillStyle = '#666'
    ctx.font = '8px Inter, sans-serif'

    // X-axis labels
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (let i = 0; i <= 4; i++) {
      const x = marginLeft + (chartWidth * i) / 4
      const value = xMin + ((xMax - xMin) * i) / 4
      ctx.fillText(value.toFixed(1), x, marginTop + chartHeight + 2)
    }

    // Y-axis labels
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    for (let i = 0; i <= 4; i++) {
      const y = marginTop + (chartHeight * i) / 4
      const value = yMax - ((yMax - yMin) * i) / 4
      ctx.fillText(value.toFixed(1), marginLeft - 3, y)
    }

    // Draw axis labels
    if (xLabel) {
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = '#888'
      ctx.fillText(xLabel, marginLeft + chartWidth / 2, height - 8)
    }

    if (yLabel) {
      ctx.save()
      ctx.translate(8, marginTop + chartHeight / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.fillStyle = '#888'
      ctx.fillText(yLabel, 0, 0)
      ctx.restore()
    }

    // Draw points with trail effect
    if (points.length > 0) {
      // Draw trail (older points more transparent)
      for (let i = 0; i < points.length - 1; i++) {
        const alpha = 0.1 + (0.5 * i) / points.length
        const normalizedX = (points[i].x - xMin) / (xMax - xMin)
        const normalizedY = (points[i].y - yMin) / (yMax - yMin)
        const x = marginLeft + normalizedX * chartWidth
        const y = marginTop + (1 - normalizedY) * chartHeight

        ctx.beginPath()
        ctx.arc(x, y, 2, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`
        ctx.fill()
      }

      // Draw current point (last point, brightest)
      const current = points[points.length - 1]
      const normalizedX = (current.x - xMin) / (xMax - xMin)
      const normalizedY = (current.y - yMin) / (yMax - yMin)
      const x = marginLeft + normalizedX * chartWidth
      const y = marginTop + (1 - normalizedY) * chartHeight

      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fillStyle = '#fff'
      ctx.fill()
      ctx.strokeStyle = '#707070'
      ctx.lineWidth = 1
      ctx.stroke()
    }
  }, [points, width, height, xMin, xMax, yMin, yMax, xLabel, yLabel])

  // Generate dimension options
  const dimOptions = Array.from({ length: numDims }, (_, i) => i)

  return (
    <div className="scatter-plot">
      <div className="scatter-controls">
        <label>
          X:
          <select value={xDim} onChange={(e) => onXDimChange(Number(e.target.value))}>
            {dimOptions.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label>
          Y:
          <select value={yDim} onChange={(e) => onYDimChange(Number(e.target.value))}>
            {dimOptions.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
      </div>
      <canvas ref={canvasRef} style={{ width, height }} />
    </div>
  )
}
