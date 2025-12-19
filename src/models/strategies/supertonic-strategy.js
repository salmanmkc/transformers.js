import { BaseModelStrategy } from './base-model-strategy.js';

/**
 * Strategy for Supertonic models (text-to-speech generation).
 * Uses a custom pipeline with text encoder, latent denoiser, and voice decoder.
 */
export class SupertonicStrategy extends BaseModelStrategy {
    getForwardFunction() {
        // Supertonic models typically override forward()
        return null;
    }

    getPrepareInputsForGeneration() {
        return null;
    }

    canGenerate() {
        return false;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                text_encoder: 'text_encoder',
                latent_denoiser: 'latent_denoiser',
                voice_decoder: 'voice_decoder',
            },
            configs: [],
        };
    }

    modifyForwardParams(forward_params) {
        return forward_params;
    }
}
