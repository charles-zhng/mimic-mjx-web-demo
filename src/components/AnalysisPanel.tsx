import { useState, useMemo } from 'react'
import type { VisualizationHistory, JointConfig } from '../types/visualization'
import LineChart from './charts/LineChart'
import ContactPlot from './charts/ContactPlot'
import ScatterPlot from './charts/ScatterPlot'
import Select from './Select'

// Force threshold for binary contact detection
const CONTACT_THRESHOLD = 0.1

// Common joints for visualization
const JOINT_OPTIONS: JointConfig[] = [
  { index: 0, name: 'vertebra_1_extend' },
  { index: 1, name: 'vertebra_1_bend' },
  { index: 2, name: 'vertebra_1_twist' },
  { index: 3, name: 'vertebra_2_extend' },
  { index: 6, name: 'hip_L_supinate' },
  { index: 7, name: 'hip_L_abduct' },
  { index: 8, name: 'hip_L_extend' },
  { index: 9, name: 'knee_L' },
  { index: 10, name: 'ankle_L' },
  { index: 14, name: 'hip_R_supinate' },
  { index: 15, name: 'hip_R_abduct' },
  { index: 16, name: 'hip_R_extend' },
  { index: 17, name: 'knee_R' },
  { index: 18, name: 'ankle_R' },
  { index: 34, name: 'shoulder_L' },
  { index: 37, name: 'elbow_L' },
  { index: 42, name: 'shoulder_R' },
  { index: 45, name: 'elbow_R' },
]

interface AnalysisPanelProps {
  history: VisualizationHistory
  selectedJointIndex: number
  onJointChange: (index: number) => void
  isVisible: boolean
  onToggle: () => void
}

export default function AnalysisPanel({
  history,
  selectedJointIndex,
  onJointChange,
  isVisible,
  onToggle,
}: AnalysisPanelProps) {
  const [latentXDim, setLatentXDim] = useState(0)
  const [latentYDim, setLatentYDim] = useState(1)

  // Extract data for charts
  const jointAngleData = useMemo(() => {
    const current = history.data.map((d) => d.jointAngle.current)
    const reference = history.data.map((d) => d.jointAngle.reference)
    return [
      { label: 'Current', values: current, color: '#4a9eff' },
      { label: 'Reference', values: reference, color: '#ff9f4a' },
    ]
  }, [history.data])

  // Convert touch sensor forces to binary contact states
  const contactData = useMemo(() => {
    return history.data.map((d) =>
      d.touchSensors.map((force) => force > CONTACT_THRESHOLD)
    )
  }, [history.data])

  const contactLabels = ['Palm L', 'Palm R', 'Sole L', 'Sole R']
  const contactColors = ['#4aff9f', '#ff4a9f', '#9f4aff', '#ffff4a']

  const latentPoints = useMemo(() => {
    return history.data.map((d) => ({
      x: d.latent[latentXDim] ?? 0,
      y: d.latent[latentYDim] ?? 0,
    }))
  }, [history.data, latentXDim, latentYDim])

  const selectedJoint = JOINT_OPTIONS.find((j) => j.index === selectedJointIndex)

  return (
    <>
      {/* Toggle button - always visible */}
      <button className="analysis-toggle" onClick={onToggle}>
        {isVisible ? '>' : '<'} Analysis
      </button>

      {/* Panel */}
      <div className={`analysis-panel ${isVisible ? '' : 'collapsed'}`}>
        <h3>Motion Analysis</h3>

        {/* Joint selector */}
        <div className="analysis-section">
          <label>Joint</label>
          <Select
            value={String(selectedJointIndex)}
            options={JOINT_OPTIONS.map((j) => ({
              value: String(j.index),
              label: j.name,
            }))}
            onChange={(value) => onJointChange(Number(value))}
          />
        </div>

        {/* Joint angle chart */}
        <div className="analysis-section">
          <label>{selectedJoint?.name ?? 'Joint'} Angle</label>
          <div className="chart-container">
            <LineChart
              lines={jointAngleData}
              width={200}
              height={80}
              yLabel="rad"
              showLegend={true}
            />
          </div>
        </div>

        {/* Touch sensors - piano roll style */}
        <div className="analysis-section">
          <label>Foot Contact</label>
          <div className="chart-container">
            <ContactPlot
              data={contactData}
              labels={contactLabels}
              colors={contactColors}
              width={200}
              height={56}
            />
          </div>
        </div>

        {/* Latent space scatter - hidden for now */}
        {false && (
          <div className="analysis-section">
            <label>Latent Space</label>
            <div className="chart-container">
              <ScatterPlot
                points={latentPoints}
                width={200}
                height={150}
                xLabel={`dim ${latentXDim}`}
                yLabel={`dim ${latentYDim}`}
                xDim={latentXDim}
                yDim={latentYDim}
                onXDimChange={setLatentXDim}
                onYDimChange={setLatentYDim}
                numDims={16}
              />
            </div>
          </div>
        )}
      </div>
    </>
  )
}
