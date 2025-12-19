/**
 * Strategy system for PreTrainedModel.
 *
 * This module provides a plugin-style architecture for adding model type support.
 * Each MODEL_TYPE has a corresponding strategy that defines:
 * - Forward function (how the model performs inference)
 * - Input preparation for generation
 * - Session/config loading configuration
 * - Whether the model can generate text
 *
 * To add a new model type, create a strategy class extending BaseModelStrategy
 * and register it with the strategyRegistry.
 */

import { MODEL_TYPES } from '../model-types.js';
import { strategyRegistry } from './strategy-registry.js';

// Import all strategies
import { DecoderOnlyStrategy } from './decoder-only-strategy.js';
import { Seq2SeqStrategy } from './seq2seq-strategy.js';
import { EncoderOnlyStrategy } from './encoder-only-strategy.js';
import { ImageTextToTextStrategy } from './image-text-to-text-strategy.js';
import { AudioTextToTextStrategy } from './audio-text-to-text-strategy.js';
import { ImageAudioTextToTextStrategy } from './image-audio-text-to-text-strategy.js';
import { MultiModalityStrategy } from './multimodality-strategy.js';
import { Phi3VStrategy } from './phi3v-strategy.js';
import { MaskGenerationStrategy } from './mask-generation-strategy.js';
import { AutoEncoderStrategy } from './auto-encoder-strategy.js';
import { ChatterboxStrategy } from './chatterbox-strategy.js';
import { MusicgenStrategy } from './musicgen-strategy.js';
import { SupertonicStrategy } from './supertonic-strategy.js';
import { EncoderDecoderStrategy } from './encoder-decoder-strategy.js';

/**
 * Register all default strategies.
 * This is called automatically when this module is imported.
 */
function registerDefaultStrategies() {
    // Generative text models
    strategyRegistry.register(MODEL_TYPES.DecoderOnly, new DecoderOnlyStrategy(MODEL_TYPES.DecoderOnly));
    strategyRegistry.register(MODEL_TYPES.Seq2Seq, new Seq2SeqStrategy(MODEL_TYPES.Seq2Seq));
    strategyRegistry.register(MODEL_TYPES.Vision2Seq, new Seq2SeqStrategy(MODEL_TYPES.Vision2Seq));

    // Multimodal generative models
    strategyRegistry.register(MODEL_TYPES.ImageTextToText, new ImageTextToTextStrategy(MODEL_TYPES.ImageTextToText));
    strategyRegistry.register(MODEL_TYPES.AudioTextToText, new AudioTextToTextStrategy(MODEL_TYPES.AudioTextToText));
    strategyRegistry.register(
        MODEL_TYPES.ImageAudioTextToText,
        new ImageAudioTextToTextStrategy(MODEL_TYPES.ImageAudioTextToText),
    );
    strategyRegistry.register(MODEL_TYPES.MultiModality, new MultiModalityStrategy(MODEL_TYPES.MultiModality));
    strategyRegistry.register(MODEL_TYPES.Phi3V, new Phi3VStrategy(MODEL_TYPES.Phi3V));

    // Non-generative models
    strategyRegistry.register(MODEL_TYPES.EncoderOnly, new EncoderOnlyStrategy(MODEL_TYPES.EncoderOnly));
    strategyRegistry.register(MODEL_TYPES.EncoderDecoder, new EncoderDecoderStrategy(MODEL_TYPES.EncoderDecoder));
    strategyRegistry.register(MODEL_TYPES.MaskGeneration, new MaskGenerationStrategy(MODEL_TYPES.MaskGeneration));
    strategyRegistry.register(MODEL_TYPES.AutoEncoder, new AutoEncoderStrategy(MODEL_TYPES.AutoEncoder));

    // Specialized models
    strategyRegistry.register(MODEL_TYPES.Musicgen, new MusicgenStrategy(MODEL_TYPES.Musicgen));
    strategyRegistry.register(MODEL_TYPES.Chatterbox, new ChatterboxStrategy(MODEL_TYPES.Chatterbox));
    strategyRegistry.register(MODEL_TYPES.Supertonic, new SupertonicStrategy(MODEL_TYPES.Supertonic));
}

// Auto-register on import
registerDefaultStrategies();

// Export registry and base class for external use
export { strategyRegistry } from './strategy-registry.js';
export { BaseModelStrategy } from './base-model-strategy.js';

// Export individual strategies for testing or custom use
export { DecoderOnlyStrategy } from './decoder-only-strategy.js';
export { Seq2SeqStrategy } from './seq2seq-strategy.js';
export { EncoderOnlyStrategy } from './encoder-only-strategy.js';
export { ImageTextToTextStrategy } from './image-text-to-text-strategy.js';
export { AudioTextToTextStrategy } from './audio-text-to-text-strategy.js';
export { ImageAudioTextToTextStrategy } from './image-audio-text-to-text-strategy.js';
export { MultiModalityStrategy } from './multimodality-strategy.js';
export { Phi3VStrategy } from './phi3v-strategy.js';
export { MaskGenerationStrategy } from './mask-generation-strategy.js';
export { AutoEncoderStrategy } from './auto-encoder-strategy.js';
export { ChatterboxStrategy } from './chatterbox-strategy.js';
export { MusicgenStrategy } from './musicgen-strategy.js';
export { SupertonicStrategy } from './supertonic-strategy.js';
export { EncoderDecoderStrategy } from './encoder-decoder-strategy.js';
