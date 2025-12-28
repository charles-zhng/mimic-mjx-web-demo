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

interface SkinInfo {
  mesh: THREE.Mesh
  geometry: THREE.BufferGeometry
  vertexStart: number
  vertexCount: number
  boneStart: number
  boneCount: number
  // Per-vertex bone data
  boneBodyIds: Int32Array // body id for each bone
  boneBindPos: Float32Array // bind position for each bone (3 floats)
  boneBindQuat: Float32Array // bind quaternion for each bone (4 floats)
  vertBoneId: Int32Array // bone indices for each vertex (up to 4)
  vertBoneWeight: Float32Array // bone weights for each vertex (up to 4)
  // Bind pose normals for smooth shading
  bindNormals: Float32Array
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
  private skins: SkinInfo[] = []
  private model: MjModel
  private scene: THREE.Scene

  constructor(_mujoco: MainModule, model: MjModel, scene: THREE.Scene) {
    this.model = model
    this.scene = scene
    this.createGeometries()
    // Skin rendering disabled - using geoms only
    void this._createSkins
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
        case GEOM_PLANE: {
          // Create a large plane for the ground with MuJoCo-style checkerboard
          // MuJoCo plane normal is +Z, Three.js PlaneGeometry faces +Z
          geometry = new THREE.PlaneGeometry(10, 10)

          // Create checkerboard texture
          const checkerSize = 64 // pixels per square
          const numSquares = 16 // number of squares per side
          const texSize = checkerSize * numSquares
          const canvas = document.createElement('canvas')
          canvas.width = texSize
          canvas.height = texSize
          const ctx = canvas.getContext('2d')!

          // MuJoCo default checkerboard colors (light grey and dark grey)
          const color1 = '#999999'
          const color2 = '#666666'

          for (let y = 0; y < numSquares; y++) {
            for (let x = 0; x < numSquares; x++) {
              ctx.fillStyle = (x + y) % 2 === 0 ? color1 : color2
              ctx.fillRect(x * checkerSize, y * checkerSize, checkerSize, checkerSize)
            }
          }

          const texture = new THREE.CanvasTexture(canvas)
          texture.wrapS = THREE.RepeatWrapping
          texture.wrapT = THREE.RepeatWrapping
          texture.repeat.set(1, 1)

          const planeMaterial = new THREE.MeshStandardMaterial({
            map: texture,
            side: THREE.DoubleSide,
            roughness: 0.8,
            metalness: 0.0,
          })

          const planeMesh = new THREE.Mesh(geometry, planeMaterial)
          planeMesh.receiveShadow = true
          scene.add(planeMesh)
          this.geoms.push({ mesh: planeMesh, geomId: i })
          continue // Skip the default material creation below
        }

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

      // Create material (planes are handled separately above)
      const material = new THREE.MeshPhongMaterial({
        color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        transparent: rgba[3] < 1,
        opacity: rgba[3],
        side: THREE.FrontSide,
      })

      const mesh = new THREE.Mesh(geometry, material)
      mesh.castShadow = true
      mesh.receiveShadow = true

      scene.add(mesh)
      this.geoms.push({ mesh, geomId: i })
    }

    console.log(`Created ${this.geoms.length} geom meshes`)
  }

  /**
   * Create Three.js meshes for MuJoCo skins.
   * Skins are deformable meshes attached to bones (bodies).
   * Currently disabled - using geoms only.
   */
  private _createSkins(): void {
    const { model, scene } = this

    const nskin = model.nskin as number
    if (nskin === 0) {
      console.log('No skins in model')
      return
    }

    console.log(`Creating ${nskin} skin meshes`)

    // Get skin data arrays from model
    const skinVertAdr = toInt32Array(model.skin_vertadr)
    const skinVertNum = toInt32Array(model.skin_vertnum)
    const skinFaceAdr = toInt32Array(model.skin_faceadr)
    const skinFaceNum = toInt32Array(model.skin_facenum)
    const skinBoneAdr = toInt32Array(model.skin_boneadr)
    const skinBoneNum = toInt32Array(model.skin_bonenum)
    const skinVert = toFloat32Array(model.skin_vert)
    const skinFace = toInt32Array(model.skin_face)
    const skinBoneBodyId = toInt32Array(model.skin_bonebodyid)
    const skinBoneBindPos = toFloat32Array(model.skin_bonebindpos)
    const skinBoneBindQuat = toFloat32Array(model.skin_bonebindquat)
    const skinBoneVertAdr = toInt32Array(model.skin_bonevertadr)
    const skinBoneVertNum = toInt32Array(model.skin_bonevertnum)
    const skinBoneVertId = toInt32Array(model.skin_bonevertid)
    const skinBoneVertWeight = toFloat32Array(model.skin_bonevertweight)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const skinNormal = toFloat32Array((model as any).skin_normal)

    for (let skinId = 0; skinId < nskin; skinId++) {
      const vertStart = skinVertAdr[skinId]
      const vertCount = skinVertNum[skinId]
      const faceStart = skinFaceAdr[skinId]
      const faceCount = skinFaceNum[skinId]
      const boneStart = skinBoneAdr[skinId]
      const boneCount = skinBoneNum[skinId]

      console.log(`Skin ${skinId}: ${vertCount} vertices, ${faceCount} faces, ${boneCount} bones, skinNormal length: ${skinNormal.length}`)

      // Create geometry
      const geometry = new THREE.BufferGeometry()

      // Copy vertex positions (will be updated each frame)
      const positions = new Float32Array(vertCount * 3)
      for (let i = 0; i < vertCount * 3; i++) {
        positions[i] = skinVert[vertStart * 3 + i]
      }
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

      // Copy face indices (adjusted for local vertex indices)
      const indices = new Uint32Array(faceCount * 3)
      for (let i = 0; i < faceCount * 3; i++) {
        indices[i] = skinFace[faceStart * 3 + i]
      }
      geometry.setIndex(new THREE.BufferAttribute(indices, 1))

      // Compute smooth vertex normals by averaging face normals at each position
      const normals = new Float32Array(vertCount * 3)
      const bindNormals = new Float32Array(vertCount * 3)
      const hasNormals = skinNormal.length >= (vertStart + vertCount) * 3

      if (hasNormals) {
        for (let i = 0; i < vertCount * 3; i++) {
          const n = skinNormal[vertStart * 3 + i]
          normals[i] = n
          bindNormals[i] = n
        }
        console.log(`Using model normals`)
      } else {
        // Compute smooth normals by averaging face normals at each vertex position
        // First, build a map from position -> accumulated normal
        const posToNormal = new Map<string, THREE.Vector3>()

        const v0 = new THREE.Vector3()
        const v1 = new THREE.Vector3()
        const v2 = new THREE.Vector3()
        const faceNormal = new THREE.Vector3()
        const edge1 = new THREE.Vector3()
        const edge2 = new THREE.Vector3()

        // Compute face normals and accumulate at each vertex position
        for (let f = 0; f < faceCount; f++) {
          const i0 = indices[f * 3]
          const i1 = indices[f * 3 + 1]
          const i2 = indices[f * 3 + 2]

          v0.set(positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2])
          v1.set(positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2])
          v2.set(positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2])

          edge1.subVectors(v1, v0)
          edge2.subVectors(v2, v0)
          faceNormal.crossVectors(edge1, edge2).normalize()

          // Accumulate face normal at each vertex position
          for (const v of [v0, v1, v2]) {
            // Use rounded position as key to group nearby vertices
            const key = `${v.x.toFixed(5)},${v.y.toFixed(5)},${v.z.toFixed(5)}`
            if (!posToNormal.has(key)) {
              posToNormal.set(key, new THREE.Vector3())
            }
            posToNormal.get(key)!.add(faceNormal)
          }
        }

        // Normalize accumulated normals
        for (const normal of posToNormal.values()) {
          normal.normalize()
        }

        // Assign smooth normals to each vertex
        for (let v = 0; v < vertCount; v++) {
          const px = positions[v * 3]
          const py = positions[v * 3 + 1]
          const pz = positions[v * 3 + 2]
          const key = `${px.toFixed(5)},${py.toFixed(5)},${pz.toFixed(5)}`
          const normal = posToNormal.get(key)
          if (normal) {
            normals[v * 3] = normal.x
            normals[v * 3 + 1] = normal.y
            normals[v * 3 + 2] = normal.z
            bindNormals[v * 3] = normal.x
            bindNormals[v * 3 + 1] = normal.y
            bindNormals[v * 3 + 2] = normal.z
          }
        }
        console.log(`Computed smooth normals for ${vertCount} vertices`)
      }
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))

      // Create material with light grey color and nice shading
      const material = new THREE.MeshStandardMaterial({
        color: 0xb8b8b8, // Light grey
        roughness: 0.6,
        metalness: 0.1,
        side: THREE.DoubleSide,
      })

      const mesh = new THREE.Mesh(geometry, material)
      mesh.castShadow = true
      mesh.receiveShadow = true
      scene.add(mesh)

      // Store bone data for vertex skinning
      const boneBodyIds = new Int32Array(boneCount)
      const boneBindPos = new Float32Array(boneCount * 3)
      const boneBindQuat = new Float32Array(boneCount * 4)

      for (let b = 0; b < boneCount; b++) {
        const boneIdx = boneStart + b
        boneBodyIds[b] = skinBoneBodyId[boneIdx]
        boneBindPos[b * 3] = skinBoneBindPos[boneIdx * 3]
        boneBindPos[b * 3 + 1] = skinBoneBindPos[boneIdx * 3 + 1]
        boneBindPos[b * 3 + 2] = skinBoneBindPos[boneIdx * 3 + 2]
        boneBindQuat[b * 4] = skinBoneBindQuat[boneIdx * 4]
        boneBindQuat[b * 4 + 1] = skinBoneBindQuat[boneIdx * 4 + 1]
        boneBindQuat[b * 4 + 2] = skinBoneBindQuat[boneIdx * 4 + 2]
        boneBindQuat[b * 4 + 3] = skinBoneBindQuat[boneIdx * 4 + 3]
      }

      // Build per-vertex bone indices and weights (up to 4 bones per vertex)
      const vertBoneId = new Int32Array(vertCount * 4).fill(-1)
      const vertBoneWeight = new Float32Array(vertCount * 4).fill(0)

      for (let b = 0; b < boneCount; b++) {
        const boneIdx = boneStart + b
        const bvStart = skinBoneVertAdr[boneIdx]
        const bvCount = skinBoneVertNum[boneIdx]

        for (let bv = 0; bv < bvCount; bv++) {
          const vertId = skinBoneVertId[bvStart + bv]
          const weight = skinBoneVertWeight[bvStart + bv]

          // Find empty slot for this vertex (up to 4 bones)
          for (let slot = 0; slot < 4; slot++) {
            if (vertBoneId[vertId * 4 + slot] === -1) {
              vertBoneId[vertId * 4 + slot] = b
              vertBoneWeight[vertId * 4 + slot] = weight
              break
            }
          }
        }
      }

      this.skins.push({
        mesh,
        geometry,
        vertexStart: vertStart,
        vertexCount: vertCount,
        boneStart,
        boneCount,
        boneBodyIds,
        boneBindPos,
        boneBindQuat,
        vertBoneId,
        vertBoneWeight,
        bindNormals,
      })
    }

    console.log(`Created ${this.skins.length} skin meshes`)
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

    // Update skin vertices
    this.updateSkins(data)
  }

  /**
   * Update skin mesh vertices based on bone transforms.
   */
  private updateSkins(data: MjData): void {
    const bodyXpos = toFloat32Array(data.xpos)
    const bodyXquat = toFloat32Array(data.xquat)

    // Temporary vectors for skinning calculations
    const bindPos = new THREE.Vector3()
    const bindQuat = new THREE.Quaternion()
    const currentPos = new THREE.Vector3()
    const currentQuat = new THREE.Quaternion()
    const vertexBind = new THREE.Vector3()
    const vertexLocal = new THREE.Vector3()
    const vertexWorld = new THREE.Vector3()
    const vertexResult = new THREE.Vector3()
    const normalBind = new THREE.Vector3()
    const normalLocal = new THREE.Vector3()
    const normalWorld = new THREE.Vector3()
    const normalResult = new THREE.Vector3()

    for (const skin of this.skins) {
      const positions = skin.geometry.attributes.position.array as Float32Array
      const normals = skin.geometry.attributes.normal.array as Float32Array
      const { boneBodyIds, boneBindPos, boneBindQuat, vertBoneId, vertBoneWeight, vertexCount, bindNormals } = skin

      // Transform each vertex
      const skinVert = toFloat32Array(this.model.skin_vert)
      for (let v = 0; v < vertexCount; v++) {
        vertexResult.set(0, 0, 0)
        normalResult.set(0, 0, 0)
        let totalWeight = 0

        // Get original vertex position and normal in bind pose
        const vIdx = skin.vertexStart + v
        vertexBind.set(skinVert[vIdx * 3], skinVert[vIdx * 3 + 1], skinVert[vIdx * 3 + 2])
        normalBind.set(bindNormals[v * 3], bindNormals[v * 3 + 1], bindNormals[v * 3 + 2])

        // Accumulate weighted bone transforms
        for (let slot = 0; slot < 4; slot++) {
          const boneIdx = vertBoneId[v * 4 + slot]
          if (boneIdx === -1) continue

          const weight = vertBoneWeight[v * 4 + slot]
          if (Math.abs(weight) < 0.0001) continue

          const bodyId = boneBodyIds[boneIdx]

          // Get bind pose for this bone
          bindPos.set(boneBindPos[boneIdx * 3], boneBindPos[boneIdx * 3 + 1], boneBindPos[boneIdx * 3 + 2])
          bindQuat.set(boneBindQuat[boneIdx * 4 + 1], boneBindQuat[boneIdx * 4 + 2], boneBindQuat[boneIdx * 4 + 3], boneBindQuat[boneIdx * 4])

          // Get current pose for this bone's body
          currentPos.set(bodyXpos[bodyId * 3], bodyXpos[bodyId * 3 + 1], bodyXpos[bodyId * 3 + 2])
          currentQuat.set(bodyXquat[bodyId * 4 + 1], bodyXquat[bodyId * 4 + 2], bodyXquat[bodyId * 4 + 3], bodyXquat[bodyId * 4])

          // Transform vertex from bind pose to local bone space, then to world
          // v_local = bindQuat^-1 * (v_bind - bindPos)
          // v_world = currentQuat * v_local + currentPos
          vertexLocal.copy(vertexBind).sub(bindPos)
          vertexLocal.applyQuaternion(bindQuat.clone().invert())
          vertexWorld.copy(vertexLocal).applyQuaternion(currentQuat).add(currentPos)

          // Transform normal (rotation only, no translation)
          // n_local = bindQuat^-1 * n_bind
          // n_world = currentQuat * n_local
          normalLocal.copy(normalBind).applyQuaternion(bindQuat.clone().invert())
          normalWorld.copy(normalLocal).applyQuaternion(currentQuat)

          vertexResult.addScaledVector(vertexWorld, weight)
          normalResult.addScaledVector(normalWorld, weight)
          totalWeight += weight
        }

        // Normalize by total weight (handle edge case)
        if (totalWeight > 0.0001) {
          vertexResult.divideScalar(totalWeight)
          normalResult.normalize() // Normals should be normalized, not scaled
        } else {
          vertexResult.copy(vertexBind)
          normalResult.copy(normalBind)
        }

        // Update position and normal buffers
        positions[v * 3] = vertexResult.x
        positions[v * 3 + 1] = vertexResult.y
        positions[v * 3 + 2] = vertexResult.z
        normals[v * 3] = normalResult.x
        normals[v * 3 + 1] = normalResult.y
        normals[v * 3 + 2] = normalResult.z
      }

      // Mark geometry for update
      skin.geometry.attributes.position.needsUpdate = true
      skin.geometry.attributes.normal.needsUpdate = true
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

    for (const skin of this.skins) {
      skin.mesh.geometry.dispose()
      if (skin.mesh.material instanceof THREE.Material) {
        skin.mesh.material.dispose()
      }
      this.scene.remove(skin.mesh)
    }
    this.skins = []
  }
}
