import { BaseModelStrategy } from './base-model-strategy.js';
import { encoderForward } from '../utils.js';

/**
 * Strategy for encoder-only models (e.g., BERT, RoBERTa, ViT).
 * These models don't generate text - they're used for classification,
 * embeddings, and feature extraction tasks.
 */
export class EncoderOnlyStrategy extends BaseModelStrategy {
    getForwardFunction() {
        return encoderForward;
    }

    getPrepareInputsForGeneration() {
        // Encoder-only models don't generate
        return null;
    }

    canGenerate() {
        return false;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                model: options.model_file_name ?? 'model',
            },
            configs: [],
        };
    }

    modifyForwardParams(forward_params) {
        // No modifications needed - encoder-only models don't use caching
        return forward_params;
    }
}
