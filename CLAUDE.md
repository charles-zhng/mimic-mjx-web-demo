# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev      # Start dev server (http://localhost:5173)
npm run build    # TypeScript check + Vite build
npm run lint     # ESLint
npm run preview  # Preview production build
```

Python scripts (for data prep):
```bash
/Users/charles/MIMIC-MJX/track-mjx/.venv/bin/python scripts/export_motions.py --input /path/to/clips.h5 --output public/motions/clips.json --num-clips 10
/Users/charles/MIMIC-MJX/track-mjx/.venv/bin/python scripts/convert_checkpoint.py  # JAX checkpoint to ONNX
```

## Architecture

Browser-based neural network physics simulation: A trained rodent locomotion policy runs in WebAssembly via MuJoCo (physics) + ONNX Runtime (neural network inference).

### Core Loop (`src/hooks/useSimulation.ts`)

1. **Observation** (`src/lib/observation.ts`): Build 917-dim observation vector
   - Reference obs (640): Future trajectory targets (5 frames) in egocentric frame
   - Proprioceptive obs (277): Joint angles/velocities, actuator forces, sensors, previous action
2. **Inference** (`src/hooks/useONNX.ts`): ONNX model outputs 76-dim logits (38 means + 38 log_stds)
3. **Action**: `tanh(mean)` applied to MuJoCo ctrl
4. **Physics**: Multiple substeps per control step (100Hz control, physics timestep from model)

### Key Hooks

- `useMuJoCo`: Loads MuJoCo WASM, model XML, skin file
- `useONNX`: Loads ONNX policy network (normalization baked into model)
- `useMotionClips`: Loads reference motion clips (gzipped JSON)
- `useSimulation`: Runs the simulation loop with requestAnimationFrame

### Rendering (`src/lib/mujoco-renderer.ts`)

Converts MuJoCo geoms to Three.js meshes, syncs transforms each frame.

### Public Assets

- `/models/rodent_scaled.xml` - MuJoCo model (0.9x scale, includes arena)
- `/nn/intention_network.onnx` - Policy network with embedded normalization
- `/motions/clips.json.gz` - Reference motion clips (qpos, xpos per frame)

## Technical Notes

- Requires SharedArrayBuffer (COOP/COEP headers in vite.config.ts and `_headers`)
- ONNX WASM loaded from CDN in production
- Motion clips at 50Hz, control at 100Hz
- Body indices offset by +1 in model vs reference clips (due to floor body)
