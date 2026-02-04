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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

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
 * @brief Converts an EMG value to a discrete signal level.
 *
 * This function maps a continuous EMG signal value to a discrete 
 * level based on the predefined minimum and maximum levels (MIN_LEVEL, MAX_LEVEL, NUM_LEVELS).
 *
 * @param emg_value The EMG signal value to be converted.
 * @return The discrete signal level (integer) corresponding to the EMG value.
 */
int get_signal_level(double emg_value) {

    if (emg_value <= MIN_LEVEL) {
        return 0;
    }
    if (emg_value >= MAX_LEVEL) {
        return NUM_LEVELS - 1;
    }
    double normalized_value = (emg_value - MIN_LEVEL) / (MAX_LEVEL - MIN_LEVEL);

    return (int)(normalized_value * (NUM_LEVELS - 1));
    /*
    int64_t key = (int64_t)round(emg_value);
    return key;*/
}

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
    Vector** bound_vectors = (Vector**)malloc(NUM_FEATURES * sizeof(Vector*));
    if (bound_vectors == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for bound_vectors\n");
        return;
    }
     if (enc == NULL || emg_sample == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to encode_timestamp\n");
        return;
    }

    for (int channel = 0; channel < NUM_FEATURES; channel++) {
          
        int signal_level = get_signal_level(emg_sample[channel]);
        #if PRECOMPUTED_ITEM_MEMORY
            bound_vectors[channel] = enc->item_mem->base_vectors[(signal_level*NUM_FEATURES)+channel];
        #else
        bound_vectors[channel] = create_vector();
        bind(enc->channel_memory->base_vectors[channel], enc->signal_memory->base_vectors[signal_level], bound_vectors[channel]);
        #endif
       
    }
    
    bundle_multi(bound_vectors, NUM_FEATURES, result);

    #if PRECOMPUTED_ITEM_MEMORY
    #else
    for (int channel = 0; channel < NUM_FEATURES; channel++) {
        free_vector(bound_vectors[channel]);
    }
    #endif
    free(bound_vectors);
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

    for (size_t i = 1; i < N_GRAM_SIZE; i++) {
        Vector* encoded = create_vector();
        Vector* result_permuted = create_vector();
        encode_timestamp(enc, emg_data[i], encoded);
        permute(result,1,result_permuted);
        bind(result_permuted,encoded,result);

        free_vector(encoded);
        free_vector(result_permuted);
    }

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
    encode_timestamp(enc, emg_data[0], result);

    for (size_t i = 1; i < N_GRAM_SIZE; i++) {
        Vector* encoded = create_vector();
        Vector* result_permuted = create_vector();
        encode_timestamp(enc, emg_data[i], encoded);
        permute(result,1,result_permuted);
        bind(result_permuted,encoded,result);

        free_vector(encoded);
        free_vector(result_permuted);
    }
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
    Vector** bound_vectors = (Vector**)malloc(NUM_FEATURES * sizeof(Vector*));
    if (bound_vectors == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for bound_vectors\n");
        return -1;
    }
     if (enc == NULL || emg_data == NULL || result == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to encode_timestamp\n");
        return -1;
    }

    for (int channel = 0; channel < NUM_FEATURES; channel++) {
          
        int signal_level = get_signal_level(emg_data[channel]);
        #if PRECOMPUTED_ITEM_MEMORY
            bound_vectors[channel] = enc->item_mem->base_vectors[(signal_level*NUM_FEATURES)+channel];
        #else
        bound_vectors[channel] = create_vector();
        bind(enc->channel_memory->base_vectors[channel], enc->signal_memory->base_vectors[signal_level], bound_vectors[channel]);
        #endif
       
    }
    
    bundle_multi(bound_vectors, NUM_FEATURES, result);

    #if PRECOMPUTED_ITEM_MEMORY
    #else
    for (int channel = 0; channel < NUM_FEATURES; channel++) {
        free_vector(bound_vectors[channel]);
    }
    #endif
    free(bound_vectors);
    return 0;
}
