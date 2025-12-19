/**
 * Model type constants and mappings.
 * Separated from PreTrainedModel to avoid circular dependencies with strategies.
 */

export const MODEL_TYPES = {
    EncoderOnly: 0,
    EncoderDecoder: 1,
    Seq2Seq: 2,
    Vision2Seq: 3,
    DecoderOnly: 4,
    MaskGeneration: 5,
    ImageTextToText: 6,
    Musicgen: 7,
    MultiModality: 8,
    Phi3V: 9,
    AudioTextToText: 10,
    AutoEncoder: 11,
    ImageAudioTextToText: 12,
    Supertonic: 13,
    Chatterbox: 14,
};

export const MODEL_TYPE_MAPPING = new Map();
export const MODEL_NAME_TO_CLASS_MAPPING = new Map();
export const MODEL_CLASS_TO_NAME_MAPPING = new Map();
