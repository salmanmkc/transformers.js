import { constructSessions } from '../session.js';
import { getOptionalConfigs } from '../utils.js';
import { strategyRegistry } from '../strategies/index.js';
import { AutoConfig } from '../../configs.js';
import { GITHUB_ISSUE_URL } from '../../utils/constants.js';
import { MODEL_TYPES, MODEL_CLASS_TO_NAME_MAPPING, MODEL_TYPE_MAPPING } from '../model-types.js';

/**
 * Handles loading of pretrained models by coordinating strategy-based session loading.
 * Extracts the complex from_pretrained logic from PreTrainedModel.
 */
export class ModelLoader {
    /**
     * Load a pretrained model using the appropriate strategy.
     *
     * @param {Function} modelClass - The model class constructor
     * @param {string} pretrained_model_name_or_path - Model identifier or path
     * @param {Object} options - Loading options
     * @returns {Promise<[Object, Object]>} [sessions, configs]
     */
    static async load(modelClass, pretrained_model_name_or_path, options) {
        // Load config first
        const config = (options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options));

        // Determine model type and get strategy
        const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(modelClass);
        const modelType = MODEL_TYPE_MAPPING.get(modelName);

        const strategy = strategyRegistry.getStrategy(modelType);

        if (!strategy) {
            // Fallback to EncoderOnly with warning
            return this._loadEncoderOnly(pretrained_model_name_or_path, options, modelType, modelName, config);
        }

        // Get session configuration from strategy
        const sessionConfig = strategy.getSessionConfig(options, config);

        // Load sessions and optional configs in parallel
        const info = await Promise.all([
            constructSessions(
                pretrained_model_name_or_path,
                sessionConfig.sessions,
                options,
                sessionConfig.primary_session,
            ),
            sessionConfig.configs && sessionConfig.configs.length > 0
                ? getOptionalConfigs(
                      pretrained_model_name_or_path,
                      // Convert array ['generation_config.json'] to object {generation_config: 'generation_config.json'}
                      Object.fromEntries(sessionConfig.configs.map((c) => [c.replace('.json', ''), c])),
                      options,
                  )
                : Promise.resolve({}),
        ]);

        return info;
    }

    /**
     * Fallback loader for unknown model types.
     * Assumes encoder-only architecture.
     *
     * @private
     */
    static async _loadEncoderOnly(pretrained_model_name_or_path, options, modelType, modelName, config) {
        if (modelType !== MODEL_TYPES.EncoderOnly) {
            const type = modelName ?? config?.model_type;
            if (type !== 'custom') {
                console.warn(
                    `Model type for '${type}' not found, assuming encoder-only architecture. Please report this at ${GITHUB_ISSUE_URL}.`,
                );
            }
        }

        return await Promise.all([
            constructSessions(
                pretrained_model_name_or_path,
                {
                    model: options.model_file_name ?? 'model',
                },
                options,
            ),
            Promise.resolve({}),
        ]);
    }
}
