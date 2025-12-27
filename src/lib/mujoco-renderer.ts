import * as THREE from 'three'
import type { MainModule, MjModel, MjData } from '../types'

// MuJoCo geom types
const GEOM_PLANE = 0
const GEOM_SPHERE = 2
const GEOM_CAPSULE = 3
const GEOM_ELLIPSOID = 4
const GEOM_CYLINDER = 5
const GEOM_BOX = 6

interface GeomInfo {
  mesh: THREE.Mesh
  geomId: number
}

/**
 * Helper to convert MuJoCo array property to typed array.
 * MuJoCo WASM returns Float64Array views for most properties.
 */
function toFloat32Array(arr: unknown): Float32Array {
  if (arr instanceof Float32Array) return arr
  if (arr instanceof Float64Array) return new Float32Array(arr)
  if (Array.isArray(arr)) return new Float32Array(arr)
  return new Float32Array(0)
}

function toInt32Array(arr: unknown): Int32Array {
  if (arr instanceof Int32Array) return arr
  if (arr instanceof Float64Array) return new Int32Array(arr)
  if (Array.isArray(arr)) return new Int32Array(arr)
  return new Int32Array(0)
}

/**
 * MuJoCo to Three.js renderer.
 * Creates Three.js geometry from MuJoCo model geoms and syncs transforms each frame.
 */
export class MuJoCoRenderer {
  private geoms: GeomInfo[] = []
  private model: MjModel
  private scene: THREE.Scene

  constructor(_mujoco: MainModule, model: MjModel, scene: THREE.Scene) {
    this.model = model
    this.scene = scene
    this.createGeometries()
  }

  /**
   * Create Three.js geometries for all MuJoCo geoms.
   */
  private createGeometries(): void {
    const { model, scene } = this

    // Access model arrays - these are direct typed array views in mujoco-js
    const geomTypes = toInt32Array(model.geom_type)
    const geomSizes = toFloat32Array(model.geom_size)
    const geomRgba = toFloat32Array(model.geom_rgba)
    const geomGroup = toInt32Array(model.geom_group)

    for (let i = 0; i < model.ngeom; i++) {
      const type = geomTypes[i]
      const size = [geomSizes[i * 3], geomSizes[i * 3 + 1], geomSizes[i * 3 + 2]]
      const rgba = [geomRgba[i * 4], geomRgba[i * 4 + 1], geomRgba[i * 4 + 2], geomRgba[i * 4 + 3]]
      const group = geomGroup[i]

      // Skip certain groups or invisible geoms
      if (rgba[3] < 0.01) continue
      if (group > 2) continue // Skip collision-only geoms

      let geometry: THREE.BufferGeometry | null = null

      switch (type) {
        case GEOM_PLANE:
          // Create a large plane for the ground
          // MuJoCo plane normal is +Z, Three.js PlaneGeometry faces +Z
          geometry = new THREE.PlaneGeometry(10, 10)
          break

        case GEOM_SPHERE:
          geometry = new THREE.SphereGeometry(size[0], 16, 16)
          break

        case GEOM_CAPSULE:
          // Capsule: size[0] = radius, size[1] = half-length
          // MuJoCo capsules are along Z-axis, Three.js along Y-axis
          // Rotate geometry to align with MuJoCo convention
          geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2, 8, 16)
          geometry.rotateX(Math.PI / 2) // Rotate from Y-up to Z-up
          break

        case GEOM_ELLIPSOID:
          // Approximate ellipsoid with scaled sphere
          geometry = new THREE.SphereGeometry(1, 16, 16)
          // MuJoCo size is (x, y, z) radii
          geometry.scale(size[0], size[1], size[2])
          break

        case GEOM_CYLINDER:
          // size[0] = radius, size[1] = half-height
          // MuJoCo cylinders are along Z-axis, Three.js along Y-axis
          geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 16)
          geometry.rotateX(Math.PI / 2) // Rotate from Y-up to Z-up
          break

        case GEOM_BOX:
          // size = half-sizes (x, y, z)
          geometry = new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2)
          break

        default:
          // Skip unsupported geom types (mesh, hfield, etc.)
          continue
      }

      if (!geometry) continue

      // Create material
      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        transparent: rgba[3] < 1,
        opacity: rgba[3],
        side: type === GEOM_PLANE ? THREE.DoubleSide : THREE.FrontSide,
      })

      const mesh = new THREE.Mesh(geometry, material)
      mesh.castShadow = type !== GEOM_PLANE
      mesh.receiveShadow = true

      scene.add(mesh)
      this.geoms.push({ mesh, geomId: i })
    }

    console.log(`Created ${this.geoms.length} geom meshes`)
  }

  /**
   * Sync Three.js mesh transforms from MuJoCo simulation data.
   */
  private debugCount = 0

  sync(data: MjData): void {
    // Get geom world transforms directly from simulation
    const geomXpos = toFloat32Array(data.geom_xpos)
    const geomXmat = toFloat32Array(data.geom_xmat)

    // Debug: log first few syncs
    if (this.debugCount < 3) {
      console.log('geomXpos length:', geomXpos.length, 'geomXmat length:', geomXmat.length)
      console.log('First geom pos:', geomXpos[0], geomXpos[1], geomXpos[2])
      console.log('First geom mat:',
        geomXmat[0].toFixed(3), geomXmat[1].toFixed(3), geomXmat[2].toFixed(3),
        geomXmat[3].toFixed(3), geomXmat[4].toFixed(3), geomXmat[5].toFixed(3),
        geomXmat[6].toFixed(3), geomXmat[7].toFixed(3), geomXmat[8].toFixed(3)
      )
      this.debugCount++
    }

    const mat4 = new THREE.Matrix4()

    for (const geom of this.geoms) {
      const { mesh, geomId } = geom

      // Set position directly from geom_xpos
      mesh.position.set(
        geomXpos[geomId * 3],
        geomXpos[geomId * 3 + 1],
        geomXpos[geomId * 3 + 2]
      )

      // Convert rotation matrix to quaternion
      // MuJoCo geom_xmat is a 3x3 rotation matrix in row-major order
      const o = geomId * 9
      mat4.set(
        geomXmat[o + 0], geomXmat[o + 1], geomXmat[o + 2], 0,
        geomXmat[o + 3], geomXmat[o + 4], geomXmat[o + 5], 0,
        geomXmat[o + 6], geomXmat[o + 7], geomXmat[o + 8], 0,
        0, 0, 0, 1
      )

      // Extract quaternion from rotation matrix
      mesh.quaternion.setFromRotationMatrix(mat4)
    }
  }

  /**
   * Clean up Three.js resources.
   */
  dispose(): void {
    for (const geom of this.geoms) {
      geom.mesh.geometry.dispose()
      if (geom.mesh.material instanceof THREE.Material) {
        geom.mesh.material.dispose()
      }
      this.scene.remove(geom.mesh)
    }
    this.geoms = []
  }
}
