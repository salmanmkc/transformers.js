import { BaseModelStrategy } from './base-model-strategy.js';

/**
 * Strategy for mask generation models (e.g., SAM, SAM2).
 * These models generate segmentation masks from images and prompts.
 */
export class MaskGenerationStrategy extends BaseModelStrategy {
    getForwardFunction() {
        // Mask generation models typically override forward()
        return null;
    }

    getPrepareInputsForGeneration() {
        // These models don't use autoregressive generation
        return null;
    }

    canGenerate() {
        return false;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                model: 'vision_encoder',
                prompt_encoder_mask_decoder: 'prompt_encoder_mask_decoder',
            },
            configs: [],
        };
    }

    modifyForwardParams(forward_params) {
        return forward_params;
    }
}
