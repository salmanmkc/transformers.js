import { BaseModelStrategy } from './base-model-strategy.js';
import { seq2seqForward, encoder_decoder_prepare_inputs_for_generation } from '../utils.js';

/**
 * Strategy for sequence-to-sequence models (e.g., T5, BART, Whisper).
 * These models use an encoder-decoder architecture for tasks like translation,
 * summarization, and speech-to-text.
 *
 * This strategy is also used for Vision2Seq models (same architecture).
 */
export class Seq2SeqStrategy extends BaseModelStrategy {
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
                model: 'encoder_model',
                decoder_model_merged: 'decoder_model_merged',
            },
            configs: ['generation_config.json'],
            primary_session: 'decoder_model_merged',
        };
    }

    modifyForwardParams(forward_params) {
        // Seq2Seq models need past_key_values for decoder caching
        return [...forward_params, 'past_key_values'];
    }
}
