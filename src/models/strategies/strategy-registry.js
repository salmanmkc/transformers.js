/**
 * Central registry for model type strategies.
 * Allows plugins to register new strategies without modifying PreTrainedModel.
 *
 * This singleton instance is used by PreTrainedModel to look up the appropriate
 * strategy for each model type during construction and loading.
 */
class StrategyRegistry {
    constructor() {
        /** @type {Map<number, BaseModelStrategy>} */
        this.strategies = new Map();
    }

    /**
     * Register a strategy for a model type.
     * This allows external code to add support for new model types.
     *
     * @param {number} modelType - MODEL_TYPES enum value
     * @param {Object} strategy - Strategy instance (should extend BaseModelStrategy)
     */
    register(modelType, strategy) {
        this.strategies.set(modelType, strategy);
    }

    /**
     * Get strategy for a model type.
     *
     * @param {number} modelType - MODEL_TYPES enum value
     * @returns {BaseModelStrategy|undefined} The strategy, or undefined if not registered
     */
    getStrategy(modelType) {
        return this.strategies.get(modelType);
    }

    /**
     * Check if a strategy exists for a model type.
     *
     * @param {number} modelType - MODEL_TYPES enum value
     * @returns {boolean} True if a strategy is registered for this type
     */
    hasStrategy(modelType) {
        return this.strategies.has(modelType);
    }

    /**
     * Get all registered model types.
     *
     * @returns {number[]} Array of registered MODEL_TYPES values
     */
    getRegisteredTypes() {
        return Array.from(this.strategies.keys());
    }
}

/**
 * Singleton instance of the strategy registry.
 * Import this to register or look up strategies.
 */
export const strategyRegistry = new StrategyRegistry();

// Export the class for testing or custom registry creation
export { StrategyRegistry };
