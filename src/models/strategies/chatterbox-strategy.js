import { BaseModelStrategy } from './base-model-strategy.js';
import { chatterbox_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for Chatterbox models (speech-to-speech).
 * These models override their forward() method for custom processing.
 */
export class ChatterboxStrategy extends BaseModelStrategy {
    getForwardFunction() {
        // Chatterbox models override forward() in their class
        return null;
    }

    getPrepareInputsForGeneration() {
        return chatterbox_prepare_inputs_for_generation;
    }

    canGenerate() {
        return true;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                embed_tokens: 'embed_tokens',
                speech_encoder: 'speech_encoder',
                model: 'language_model',
                conditional_decoder: 'conditional_decoder',
            },
            configs: ['generation_config.json'],
            primary_session: 'model',
        };
    }

    modifyForwardParams(forward_params) {
        return [...forward_params, 'past_key_values'];
    }
}
