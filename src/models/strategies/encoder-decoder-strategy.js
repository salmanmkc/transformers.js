import { BaseModelStrategy } from './base-model-strategy.js';
import { seq2seqForward } from '../utils.js';

/**
 * Strategy for encoder-decoder models that don't generate autoregressively.
 * These models use encoder-decoder architecture but aren't used for generation
 * (e.g., VisionEncoderDecoder for document understanding).
 */
export class EncoderDecoderStrategy extends BaseModelStrategy {
    getForwardFunction() {
        return seq2seqForward;
    }

    getPrepareInputsForGeneration() {
        // These models don't generate autoregressively
        return null;
    }

    canGenerate() {
        return false;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                model: 'encoder_model',
                decoder_model_merged: 'decoder_model_merged',
            },
            configs: [],
        };
    }

    modifyForwardParams(forward_params) {
        return forward_params;
    }
}
