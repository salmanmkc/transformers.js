import { BaseModelStrategy } from './base-model-strategy.js';
import { seq2seqForward, encoder_decoder_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for Musicgen models (text-to-music generation).
 * Uses seq2seq architecture with special encodec decoder.
 */
export class MusicgenStrategy extends BaseModelStrategy {
    getForwardFunction() {
        return seq2seqForward;
    }

    getPrepareInputsForGeneration() {
        return encoder_decoder_prepare_inputs_for_generation;
    }

    canGenerate() {
        return true;
    }

    getSessionConfig(options) {
        return {
            sessions: {
                model: 'text_encoder',
                decoder_model_merged: 'decoder_model_merged',
                encodec_decode: 'encodec_decode',
            },
            configs: ['generation_config.json'],
            primary_session: 'decoder_model_merged',
        };
    }

    modifyForwardParams(forward_params) {
        return [...forward_params, 'past_key_values'];
    }
}
