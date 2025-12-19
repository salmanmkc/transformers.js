import { Tensor, ones, cat } from '../utils/tensor.js';
import { LogitsSampler } from './logits_sampler.js';
import { pick } from '../utils/core.js';

/**
 * Encapsulates the text generation loop and state management.
 * Separates generation logic from model inference logic, making it:
 * - Easier to test in isolation
 * - Reusable across different model types
 * - Clearer to understand and modify
 */
export class GenerationPipeline {
    /**
     * @param {Object} model - The PreTrainedModel instance
     * @param {import('./configuration_utils.js').GenerationConfig} generation_config - Generation configuration
     * @param {import('./logits_process.js').LogitsProcessorList} logits_processor - Logits processors
     * @param {import('./stopping_criteria.js').StoppingCriteriaList} stopping_criteria - Stopping criteria
     * @param {Object|null} streamer - Optional streamer for real-time output
     */
    constructor(model, generation_config, logits_processor, stopping_criteria, streamer = null) {
        this.model = model;
        this.generation_config = generation_config;
        this.logits_processor = logits_processor;
        this.stopping_criteria = stopping_criteria;
        this.streamer = streamer;
        this.is_encoder_decoder = model.config.is_encoder_decoder;
        this.sampler = LogitsSampler.getSampler(generation_config);
    }

    /**
     * Run the generation loop.
     *
     * @param {Tensor} input_ids - Starting input IDs
     * @param {Object} model_inputs - Model inputs (includes past_key_values, attention_mask, etc.)
     * @returns {Promise<Object|Tensor>} Generation results (sequences or full dict)
     */
    async generate(input_ids, model_inputs) {
        const numInputs = model_inputs[this.model.main_input_name].dims.at(0);
        const scores = new Array(numInputs).fill(0);
        const all_input_ids = input_ids.tolist();

        if (this.streamer) {
            this.streamer.put(all_input_ids);
        }

        let outputs;
        let attentions = {};
        let return_dict_items = {};

        // Main generation loop
        while (true) {
            // Prepare inputs for this iteration
            model_inputs = this.model.prepare_inputs_for_generation(
                all_input_ids,
                model_inputs,
                this.generation_config,
            );

            // Forward pass
            outputs = await this.model.forward(model_inputs);

            // Collect additional outputs if requested
            if (this.generation_config.return_dict_in_generate) {
                this._collectOutputs(outputs, attentions, return_dict_items);
            }

            // Get next tokens
            const logits = outputs.logits.slice(null, -1, null).to('float32');
            const next_tokens_scores = this.logits_processor(all_input_ids, logits);

            // Sample tokens
            const generated_input_ids = await this._sampleNextTokens(next_tokens_scores, all_input_ids, scores);

            if (this.streamer) {
                this.streamer.put(generated_input_ids);
            }

            // Check stopping criteria
            const stop = this.stopping_criteria(all_input_ids);
            if (stop.every((x) => x)) {
                break;
            }

            // Update model inputs for next iteration
            model_inputs = this.model._update_model_kwargs_for_generation({
                generated_input_ids,
                outputs,
                model_inputs,
                is_encoder_decoder: this.is_encoder_decoder,
            });
        }

        if (this.streamer) {
            this.streamer.end();
        }

        return this._buildOutput(all_input_ids, outputs, model_inputs, attentions, return_dict_items);
    }

    /**
     * Sample next tokens from logits.
     *
     * @param {Tensor} next_tokens_scores - Processed logits
     * @param {bigint[][]} all_input_ids - Current input IDs
     * @param {number[]} scores - Current scores
     * @returns {Promise<[bigint][]>} Generated token IDs
     * @private
     */
    async _sampleNextTokens(next_tokens_scores, all_input_ids, scores) {
        const generated_input_ids = [];

        for (let batch_idx = 0; batch_idx < next_tokens_scores.dims.at(0); ++batch_idx) {
            const logs = next_tokens_scores[batch_idx];
            const sampledTokens = await this.sampler(logs);

            for (const [newTokenId, logProb] of sampledTokens) {
                const bigint = BigInt(newTokenId);
                scores[batch_idx] += logProb;
                all_input_ids[batch_idx].push(bigint);
                generated_input_ids.push([bigint]);
                break; // TODO: Support beam search
            }
        }

        return generated_input_ids;
    }

    /**
     * Collect attentions and other outputs during generation.
     *
     * @param {Object} outputs - Model outputs
     * @param {Object} attentions - Accumulated attentions
     * @param {Object} return_dict_items - Accumulated return dict items
     * @private
     */
    _collectOutputs(outputs, attentions, return_dict_items) {
        if (this.generation_config.output_attentions) {
            const token_attentions = this.model.getAttentions(outputs);
            for (const key in token_attentions) {
                if (!(key in attentions)) {
                    attentions[key] = [];
                }
                attentions[key].push(token_attentions[key]);
            }
        } else if (this.model._return_dict_in_generate_keys) {
            Object.assign(return_dict_items, pick(outputs, this.model._return_dict_in_generate_keys));
        }
    }

    /**
     * Build final output from generation results.
     *
     * @param {bigint[][]} all_input_ids - All generated IDs
     * @param {Object} outputs - Last model outputs
     * @param {Object} model_inputs - Final model inputs
     * @param {Object} attentions - Collected attentions
     * @param {Object} return_dict_items - Collected return dict items
     * @returns {Object|Tensor} Final output (dict or sequences)
     * @private
     */
    _buildOutput(all_input_ids, outputs, model_inputs, attentions, return_dict_items) {
        // Retrieve and dispose all final past key values (including encoder attentions)
        const past_key_values = this.model.getPastKeyValues(outputs, model_inputs.past_key_values, true);

        // Convert to tensor
        const sequences = new Tensor('int64', all_input_ids.flat(), [all_input_ids.length, all_input_ids[0].length]);

        if (this.generation_config.return_dict_in_generate) {
            return {
                sequences,
                past_key_values,
                ...attentions,
                ...return_dict_items,
                // TODO:
                // scores,
                // logits,
            };
        } else {
            // Dispose GPU tensors
            for (const tensor of Object.values(outputs)) {
                if (tensor.location === 'gpu-buffer') {
                    tensor.dispose();
                }
            }
            return sequences;
        }
    }
}
