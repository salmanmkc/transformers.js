import { BaseModelStrategy } from './base-model-strategy.js';
import { multimodality_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for MultiModality models that can generate both text and images.
 * These models typically override their forward() method.
 */
export class MultiModalityStrategy extends BaseModelStrategy {
    getForwardFunction() {
        // MultiModality models override forward() in their class
        return null;
    }

    getPrepareInputsForGeneration() {
        return multimodality_prepare_inputs_for_generation;
    }

    canGenerate() {
        return true;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                prepare_inputs_embeds: 'prepare_inputs_embeds',
                model: 'language_model',
                lm_head: 'lm_head',
                gen_head: 'gen_head',
                gen_img_embeds: 'gen_img_embeds',
                image_decode: 'image_decode',
            },
            configs: ['generation_config.json'],
            primary_session: 'model',
        };
    }

    modifyForwardParams(forward_params) {
        return [...forward_params, 'past_key_values'];
    }
}
