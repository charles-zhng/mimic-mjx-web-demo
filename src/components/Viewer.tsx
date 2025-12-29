import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { MainModule, MjModel, MjData } from '../types'
import type { AnimalConfig } from '../types/animal-config'
import type { InferenceMode } from '../hooks/useSimulation'
import { MuJoCoRenderer } from '../lib/mujoco-renderer'

interface ViewerProps {
  mujoco: MainModule | null
  model: MjModel | null
  data: MjData | null
  ghostData: MjData | null
  isReady: boolean
  config: AnimalConfig
  inferenceMode: InferenceMode
}

export default function Viewer({ mujoco, model, data, ghostData, isReady, config, inferenceMode }: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const mujocoRendererRef = useRef<MuJoCoRenderer | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const hasCenteredRef = useRef(false)

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

    const { rendering } = config

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a1a)
    sceneRef.current = scene

    // Camera - MuJoCo uses Z-up, so we position camera accordingly
    const camera = new THREE.PerspectiveCamera(
      50,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.01,
      100
    )
    camera.position.set(...rendering.cameraPosition)
    camera.up.set(0, 0, 1) // Z-up to match MuJoCo
    camera.lookAt(...rendering.cameraTarget)
    cameraRef.current = camera

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    containerRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(...rendering.cameraTarget)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.minDistance = rendering.orbitMinDistance
    controls.maxDistance = rendering.orbitMaxDistance
    controls.update()
    controlsRef.current = controls

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(...rendering.lightPosition)
    directionalLight.castShadow = true
    directionalLight.shadow.mapSize.width = 2048
    directionalLight.shadow.mapSize.height = 2048
    directionalLight.shadow.camera.near = 0.1
    directionalLight.shadow.camera.far = 20
    directionalLight.shadow.camera.left = -rendering.shadowBounds
    directionalLight.shadow.camera.right = rendering.shadowBounds
    directionalLight.shadow.camera.top = rendering.shadowBounds
    directionalLight.shadow.camera.bottom = -rendering.shadowBounds
    scene.add(directionalLight)

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current || !camera || !renderer) return
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    }
    window.addEventListener('resize', handleResize)

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize)
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (mujocoRendererRef.current) {
        mujocoRendererRef.current.dispose()
      }
      renderer.dispose()
      containerRef.current?.removeChild(renderer.domElement)
    }
  }, [config])

  // Create MuJoCo renderer when model is ready
  useEffect(() => {
    if (!isReady || !mujoco || !model || !sceneRef.current) return

    // Clean up old renderer
    if (mujocoRendererRef.current) {
      mujocoRendererRef.current.dispose()
    }

    // Create new MuJoCo renderer
    mujocoRendererRef.current = new MuJoCoRenderer(mujoco, model, sceneRef.current)

    // Create ghost meshes for reference visualization
    mujocoRendererRef.current.createGhostGeometries()

    console.log('MuJoCo renderer created with ghost reference')
  }, [isReady, mujoco, model])

  // Toggle ghost visibility based on inference mode
  useEffect(() => {
    if (mujocoRendererRef.current) {
      mujocoRendererRef.current.setGhostVisible(inferenceMode === 'tracking')
    }
  }, [inferenceMode])

  // Center camera on midpoint between rodent and ghost torsos on initial load
  useEffect(() => {
    if (!isReady || !data || !cameraRef.current || !controlsRef.current || hasCenteredRef.current) return

    const { rendering, body } = config
    const xpos = (data as unknown as Record<string, Float64Array>).xpos

    // Get torso position (body index * 3 for x, y, z)
    const torsoX = xpos[body.torsoIndex * 3]
    const torsoY = xpos[body.torsoIndex * 3 + 1]
    const torsoZ = xpos[body.torsoIndex * 3 + 2]

    // In tracking mode, center between rodent and ghost torsos
    // In latent walk mode, center on rodent only (no ghost)
    const ghostOffsetForCamera = inferenceMode === 'tracking' ? rendering.ghostOffset / 2 : 0
    const centerX = torsoX + ghostOffsetForCamera
    const centerY = torsoY
    const centerZ = torsoZ

    // Update camera target to center on both bodies
    controlsRef.current.target.set(centerX, centerY, centerZ)

    // Position camera relative to the new center
    const [camOffsetX, camOffsetY, camOffsetZ] = rendering.cameraPosition
    const [targetOffsetX, targetOffsetY, targetOffsetZ] = rendering.cameraTarget
    cameraRef.current.position.set(
      centerX + (camOffsetX - targetOffsetX),
      centerY + (camOffsetY - targetOffsetY),
      centerZ + (camOffsetZ - targetOffsetZ)
    )

    controlsRef.current.update()
    hasCenteredRef.current = true
  }, [isReady, data, config, inferenceMode])

  // Render loop
  useEffect(() => {
    const render = () => {
      if (controlsRef.current) {
        controlsRef.current.update()
      }

      // Sync MuJoCo to Three.js
      if (mujocoRendererRef.current && data) {
        mujocoRendererRef.current.sync(data)
      }

      // Sync ghost reference
      if (mujocoRendererRef.current && ghostData) {
        mujocoRendererRef.current.syncGhost(ghostData)
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current)
      }

      animationFrameRef.current = requestAnimationFrame(render)
    }

    animationFrameRef.current = requestAnimationFrame(render)

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [data, ghostData])

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
      }}
    />
  )
}
