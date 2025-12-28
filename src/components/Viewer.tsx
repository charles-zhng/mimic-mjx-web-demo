import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { MainModule, MjModel, MjData } from '../types'
import { MuJoCoRenderer } from '../lib/mujoco-renderer'

interface ViewerProps {
  mujoco: MainModule | null
  model: MjModel | null
  data: MjData | null
  ghostData: MjData | null
  isReady: boolean
}

export default function Viewer({ mujoco, model, data, ghostData, isReady }: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const mujocoRendererRef = useRef<MuJoCoRenderer | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return

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
    camera.position.set(0.5, -0.5, 0.3)
    camera.up.set(0, 0, 1) // Z-up to match MuJoCo
    camera.lookAt(0, 0, 0.05)
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
    controls.target.set(0, 0, 0.05)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.minDistance = 0.1
    controls.maxDistance = 5
    controls.update()
    controlsRef.current = controls

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(2, -2, 5) // Adjusted for Z-up
    directionalLight.castShadow = true
    directionalLight.shadow.mapSize.width = 2048
    directionalLight.shadow.mapSize.height = 2048
    directionalLight.shadow.camera.near = 0.1
    directionalLight.shadow.camera.far = 20
    directionalLight.shadow.camera.left = -2
    directionalLight.shadow.camera.right = 2
    directionalLight.shadow.camera.top = 2
    directionalLight.shadow.camera.bottom = -2
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
  }, [])

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
