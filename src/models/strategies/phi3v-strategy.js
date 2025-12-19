import { BaseModelStrategy } from './base-model-strategy.js';
import { multimodal_text_to_text_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for Phi3V (Phi-3 Vision) models.
 * These models override their forward() method for custom vision integration.
 */
export class Phi3VStrategy extends BaseModelStrategy {
    getForwardFunction() {
        // Phi3V models override forward() in their class
        return null;
    }

    getPrepareInputsForGeneration() {
        return multimodal_text_to_text_prepare_inputs_for_generation;
    }

    canGenerate() {
        return true;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                prepare_inputs_embeds: 'prepare_inputs_embeds',
                model: 'model',
                vision_encoder: 'vision_encoder',
            },
            configs: ['generation_config.json'],
            primary_session: 'model',
        };
    }

    modifyForwardParams(forward_params) {
        return [...forward_params, 'past_key_values'];
    }
}
