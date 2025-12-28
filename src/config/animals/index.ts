import type { AnimalConfig } from '../../types/animal-config'
import { rodentConfig } from './rodent'

/** Registry of all available animal configurations */
export const animalRegistry: Record<string, AnimalConfig> = {
  rodent: rodentConfig,
}

/** Default animal to load */
export const defaultAnimalId = 'rodent'

/** Get animal config by id, throws if not found */
export function getAnimalConfig(id: string): AnimalConfig {
  const config = animalRegistry[id]
  if (!config) {
    throw new Error(`Unknown animal: ${id}. Available: ${Object.keys(animalRegistry).join(', ')}`)
  }
  return config
}

/** List all available animal ids */
export function getAvailableAnimals(): string[] {
  return Object.keys(animalRegistry)
}

// Re-export configs for direct access
export { rodentConfig }
