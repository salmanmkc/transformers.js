import { BaseModelStrategy } from './base-model-strategy.js';
import { multimodal_text_to_text_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for image-audio-text-to-text models (e.g., Gemma3n).
 * These models take image + audio + text inputs and generate text outputs.
 */
export class ImageAudioTextToTextStrategy extends BaseModelStrategy {
    getForwardFunction() {
        // Note: These models typically override forward() in their class
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
                embed_tokens: 'embed_tokens',
                audio_encoder: 'audio_encoder',
                vision_encoder: 'vision_encoder',
                decoder_model_merged: 'decoder_model_merged',
            },
            configs: ['generation_config.json'],
        };
    }

    modifyForwardParams(forward_params) {
        return [...forward_params, 'past_key_values'];
    }
}
