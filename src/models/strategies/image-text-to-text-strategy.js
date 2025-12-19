import { BaseModelStrategy } from './base-model-strategy.js';
import { imageTextToTextForward, multimodal_text_to_text_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for image-text-to-text models (e.g., LLaVA, Idefics, PaliGemma).
 * These models take image + text inputs and generate text outputs.
 */
export class ImageTextToTextStrategy extends BaseModelStrategy {
    getForwardFunction() {
        return imageTextToTextForward;
    }

    getPrepareInputsForGeneration() {
        return multimodal_text_to_text_prepare_inputs_for_generation;
    }

    canGenerate() {
        return true;
    }

    getSessionConfig(options, config) {
        const sessions = {
            embed_tokens: 'embed_tokens',
            vision_encoder: 'vision_encoder',
            decoder_model_merged: 'decoder_model_merged',
        };

        // Some models (like Florence2) have an encoder
        if (config?.is_encoder_decoder) {
            sessions.model = 'encoder_model';
        }

        return {
            sessions,
            configs: ['generation_config.json'],
            primary_session: 'decoder_model_merged',
        };
    }

    modifyForwardParams(forward_params) {
        return [...forward_params, 'past_key_values'];
    }
}
