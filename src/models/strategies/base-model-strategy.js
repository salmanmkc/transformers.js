/**
 * Base strategy interface for model type behaviors.
 * Each MODEL_TYPE gets its own strategy instance that defines how the model:
 * - Performs forward passes
 * - Prepares inputs for generation
 * - Loads sessions and configs
 *
 * This enables a plugin-style architecture where new model types can be added
 * by registering strategies without modifying PreTrainedModel.
 */
export class BaseModelStrategy {
    /**
     * @param {number} modelType - The MODEL_TYPES enum value
     */
    constructor(modelType) {
        this.modelType = modelType;
    }

    /**
     * Returns the forward function for this model type.
     * The forward function handles the model's inference logic.
     *
     * @returns {Function|null} Forward function (e.g., decoderForward, seq2seqForward)
     */
    getForwardFunction() {
        return null;
    }

    /**
     * Returns the prepare_inputs_for_generation function for this model type.
     * This function prepares model inputs before each generation step.
     *
     * @returns {Function|null} Prepare inputs function
     */
    getPrepareInputsForGeneration() {
        return null;
    }

    /**
     * Indicates if this model type can generate text/tokens.
     *
     * @returns {boolean} True if the model supports generation
     */
    canGenerate() {
        return false;
    }

    /**
     * Returns session configuration for loading model files.
     *
     * @param {Object} options - Loading options from from_pretrained
     * @returns {Object} Session configuration:
     *   - sessions: {session_name: file_name} mapping
     *   - configs: array of config file names to load (e.g., ['generation_config.json'])
     *   - primary_session: optional name of the primary session (for cache shapes)
     */
    getSessionConfig(options) {
        throw new Error('getSessionConfig must be implemented by subclass');
    }

    /**
     * Hook to modify forward_params after construction.
     * For example, generative models add 'past_key_values' to forward params.
     *
     * @param {string[]} forward_params - Current forward params
     * @returns {string[]} Modified forward params
     */
    modifyForwardParams(forward_params) {
        return forward_params;
    }
}
