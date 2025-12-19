import { BaseModelStrategy } from './base-model-strategy.js';
import { decoderForward, decoder_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for decoder-only language models (e.g., GPT, Llama, Mistral).
 * These models generate text autoregressively using only a decoder.
 */
export class DecoderOnlyStrategy extends BaseModelStrategy {
    getForwardFunction() {
        return decoderForward;
    }

    getPrepareInputsForGeneration() {
        return decoder_prepare_inputs_for_generation;
    }

    canGenerate() {
        return true;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                model: options.model_file_name ?? 'model',
            },
            configs: ['generation_config.json'],
            primary_session: 'model',
        };
    }

    modifyForwardParams(forward_params) {
        // Decoder-only models need past_key_values for caching
        return [...forward_params, 'past_key_values'];
    }
}
