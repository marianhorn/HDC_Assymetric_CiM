/**
 * @file encoder.c
 * @brief Implements functionality for encoding EMG signals into hypervectors.
 *
 * @details
 * The encoder maps raw EMG signals into high-dimensional representations (hypervectors) 
 * for further processing in Hyperdimensional Computing (HDC). It supports both spatial 
 * and temporal encoding. The encoded hypervectors are used for tasks such as classification and evaluation.
 *
 * Key features include:
 * - Conversion of continuous EMG signals into discrete levels.
 * - Encoding of individual timestamps (spatial encoding).
 * - Aggregation of multiple timestamps into N-gram hypervectors (temporal encoding).
 * - Compatibility with both bipolar and binary modes.
 *
 * @note 
 * This file supports configurations for both precomputed and dynamically generated 
 * item memories through the `PRECOMPUTED_ITEM_MEMORY` macro.
 *
 * @author Marian Horn
 */
#include "encoder.h"
#include "operations.h"
#include "quantizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#if PRECOMPUTED_ITEM_MEMORY
/**
 * @brief Initializes the encoder with the provided item memory.
 *
 * This function sets up the encoder for use with precomputed item memory. 
 * It directly links the item memory to the encoder.
 *
 * @param enc A pointer to the encoder structure to initialize.
 * @param itemMem A pointer to the precomputed item memory.
 *
 * @note Only used when `PRECOMPUTED_ITEM_MEMORY` is enabled.
 */
void init_encoder(struct encoder *enc, struct item_memory *itemMem) {
    enc->item_mem = itemMem;
}
#else
/**
 * @brief Initializes the encoder with channel and signal memory.
 *
 * This function sets up the encoder for use with dynamically generated 
 * channel and signal item memories.
 *
 * @param enc A pointer to the encoder structure to initialize.
 * @param channel_memory A pointer to the item memory for channels.
 * @param signal_memory A pointer to the item memory for signal levels.
 *
 * @note Only used when `PRECOMPUTED_ITEM_MEMORY` is disabled.
 */
void init_encoder(struct encoder *enc, struct item_memory *channel_memory, struct item_memory *signal_memory) {
    enc->channel_memory = channel_memory;
    enc->signal_memory = signal_memory;
}
#endif
/**
 * @brief Encodes a single timestamp of data into a hypervector.
 *
 * This function performs spatial encoding by binding channel and signal vectors
 * for each feature and bundling them into a single hypervector.
 *
 * @param enc A pointer to the encoder structure.
 * @param emg_sample An array of EMG data for a single timestamp.
 * @param result A pointer to the resulting hypervector.
 */
void encode_timestamp(struct encoder *enc, double *emg_sample, Vector *result) {
    if (enc == NULL || emg_sample == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to encode_timestamp\n");
        return;
    }

#if PRECOMPUTED_ITEM_MEMORY
    Vector* bound_vectors[NUM_FEATURES];
    for (int channel = 0; channel < NUM_FEATURES; channel++) {
        int signal_level = get_signal_level(channel, emg_sample[channel]);
        bound_vectors[channel] = enc->item_mem->base_vectors[(signal_level * NUM_FEATURES) + channel];
    }
    bundle_multi(bound_vectors, NUM_FEATURES, result);
#else
#if BIPOLAR_MODE
    for (int d = 0; d < VECTOR_DIMENSION; d++) {
        result->data[d] = 0;
    }
    for (int channel = 0; channel < NUM_FEATURES; channel++) {
        int signal_level = get_signal_level(channel, emg_sample[channel]);
        Vector *channel_vec = enc->channel_memory->base_vectors[channel];
        Vector *signal_vec = enc->signal_memory->base_vectors[signal_level];
        for (int d = 0; d < VECTOR_DIMENSION; d++) {
            result->data[d] += channel_vec->data[d] * signal_vec->data[d];
        }
    }
#else
    int threshold = NUM_FEATURES / 2;
    int nbits = 0;
    while ((1 << nbits) <= NUM_FEATURES && nbits < 31) {
        nbits++;
    }
    if (nbits < 1) {
        nbits = 1;
    }

    uint64_t *planes = (uint64_t *)calloc((size_t)nbits, sizeof(uint64_t));
    if (!planes) {
        fprintf(stderr, "Error: Failed to allocate bit-sliced counters\n");
        return;
    }

    size_t words = vector_storage_count();
    for (size_t w = 0; w < words; w++) {
        memset(planes, 0, (size_t)nbits * sizeof(uint64_t));
        for (int channel = 0; channel < NUM_FEATURES; channel++) {
            int signal_level = get_signal_level(channel, emg_sample[channel]);
            Vector *channel_vec = enc->channel_memory->base_vectors[channel];
            Vector *signal_vec = enc->signal_memory->base_vectors[signal_level];
            uint64_t carry = channel_vec->data[w] ^ signal_vec->data[w];
            for (int b = 0; b < nbits; b++) {
                uint64_t t = planes[b];
                planes[b] = t ^ carry;
                carry = t & carry;
            }
        }

        uint64_t out_word = 0ull;
        for (int bit = 0; bit < 64; bit++) {
            int count = 0;
            for (int b = 0; b < nbits; b++) {
                count |= (int)(((planes[b] >> bit) & 1ull) << b);
            }
            if (count >= threshold) {
                out_word |= (1ull << bit);
            }
        }
        result->data[w] = out_word;
    }
    free(planes);
    vector_mask_tail(result);
#endif
#endif
}

/**
 * @brief Checks if a sliding window of labels is stable.
 *
 * This function verifies if all labels in the current N-gram window 
 * are identical, indicating a stable state.
 *
 * @param labels A pointer to the array of labels.
 * @return `true` if the window is stable, `false` otherwise.
 */
bool is_window_stable(int* labels){
    // Check if labels for the entire n-gram are the same
    if (labels[0] != labels[N_GRAM_SIZE-1]) {
        return false; 
    }
    return true;
}
/**
 * @brief Encodes a sequence of data into a single hypervector (N-gram).
 *
 * This function performs temporal encoding by iteratively encoding each 
 * timestamp and applying binding and permutation for N-gram aggregation.
 *
 * @param enc A pointer to the encoder structure.
 * @param emg_data A 2D array of EMG data to encode.
 * @param result A pointer to the resulting hypervector.
 * @return 0 on success, -1 if any pointer is NULL.
 */
int encode_timeseries(struct encoder *enc, double **emg_data, Vector *result) {
    if (enc == NULL || emg_data == NULL || result == NULL) {
        fprintf(stdout, "Error: NULL pointer passed to encode_timeseries\n");
        return -1;
    }

    #if BIPOLAR_MODE

    encode_timestamp(enc, emg_data[0], result);

    Vector* encoded = create_vector();
    Vector* result_permuted = create_vector();
    for (size_t i = 1; i < N_GRAM_SIZE; i++) {
        encode_timestamp(enc, emg_data[i], encoded);
        permute(result,1,result_permuted);
        bind(result_permuted,encoded,result);
    }
    free_vector(encoded);
    free_vector(result_permuted);

    if (output_mode >= OUTPUT_DEBUG) {

        bool vectorContainsOnlyZeroEntries = true;
        for(int z = 0; z<VECTOR_DIMENSION; z++){
            if(result->data[z]!=0){
                vectorContainsOnlyZeroEntries = false;
                break;
            }
        }
        if(vectorContainsOnlyZeroEntries){
            print_vector(result);
            fprintf(stdout,"Encoding Error: This vector is zero\n");
        }
    }
    #else
#if MODEL_VARIANT == MODEL_VARIANT_KRISCHAN
    // Rolling-style temporal composition: XOR over slot-rotated timestamp HVs.
    vector_zero(result);

    Vector* encoded = create_vector();
    Vector* encoded_permuted = create_vector();
    for (size_t i = 0; i < N_GRAM_SIZE; i++) {
        encode_timestamp(enc, emg_data[i], encoded);
        permute(encoded, (int)i, encoded_permuted);
        size_t words = vector_storage_count();
        for (size_t w = 0; w < words; w++) {
            result->data[w] ^= encoded_permuted->data[w];
        }
    }
    free_vector(encoded);
    free_vector(encoded_permuted);
#else
    encode_timestamp(enc, emg_data[0], result);

    Vector* encoded = create_vector();
    Vector* result_permuted = create_vector();
    for (size_t i = 1; i < N_GRAM_SIZE; i++) {
        encode_timestamp(enc, emg_data[i], encoded);
        permute(result,1,result_permuted);
        bind(result_permuted,encoded,result);
    }
    free_vector(encoded);
    free_vector(result_permuted);
#endif
    if (output_mode >= OUTPUT_DEBUG) {
        bool vectorContainsOnlyZeroEntries = true;
        for(int z = 0; z<VECTOR_DIMENSION; z++){
            if(vector_get_bit(result, z)!=0){
                vectorContainsOnlyZeroEntries = false;
                break;
            }
        }
        if(vectorContainsOnlyZeroEntries){
            print_vector(result);
            fprintf(stdout,"Encoding Error: This vector is zero\n");
        }
    }
#endif
    return 0;

}

/**
 * @brief Encodes a single EMG data point into a hypervector.
 *
 * This function performs spatial encoding for general data by binding 
 * channel and signal vectors and bundling them into a single hypervector.
 *
 * @param enc A pointer to the encoder structure.
 * @param emg_data An array representing the EMG data point to encode.
 * @param result A pointer to the resulting hypervector.
 * @return 0 on success, -1 if any pointer is NULL or memory allocation fails.
 * @note This function can be used for other applications than timeseries classification.
 */
int encode_general_data(struct encoder *enc, double *emg_data, Vector *result) {
    if (enc == NULL || emg_data == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to encode_timestamp\n");
        return -1;
    }

    encode_timestamp(enc, emg_data, result);
    return 0;
}
