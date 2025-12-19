import { BaseModelStrategy } from './base-model-strategy.js';
import { autoEncoderForward } from '../utils.js';

/**
 * Strategy for auto-encoder models (e.g., Mimi, DAC, SNAC).
 * These models encode and decode audio/data without text generation.
 */
export class AutoEncoderStrategy extends BaseModelStrategy {
    getForwardFunction() {
        return autoEncoderForward;
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
                encoder_model: 'encoder_model',
                decoder_model: 'decoder_model',
            },
            configs: [],
        };
    }

    modifyForwardParams(forward_params) {
        return forward_params;
    }
}
