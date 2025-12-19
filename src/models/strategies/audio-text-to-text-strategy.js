import { BaseModelStrategy } from './base-model-strategy.js';
import { audioTextToTextForward, multimodal_text_to_text_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for audio-text-to-text models (e.g., Ultravox, Voxtral).
 * These models take audio + text inputs and generate text outputs.
 */
export class AudioTextToTextStrategy extends BaseModelStrategy {
    getForwardFunction() {
        return audioTextToTextForward;
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
                embed_tokens: 'embed_tokens',
                audio_encoder: 'audio_encoder',
                decoder_model_merged: 'decoder_model_merged',
            },
            configs: ['generation_config.json'],
            primary_session: 'decoder_model_merged',
        };
    }

    modifyForwardParams(forward_params) {
        return [...forward_params, 'past_key_values'];
    }
}
