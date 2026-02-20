/**
 * @file trainer.c
 * @brief Implementation of training functions for hyperdimensional computing (HDC) models.
 *
 * This file contains the implementation of functions for training HDC models
 * with timeseries or general data. The training process involves encoding
 * data into hypervectors and updating associative memory with class-specific
 * bundled hypervectors.
 *
 * @details
 * Training is performed either using:
 * - Timeseries data, encoded with n-grams and handled as temporal sequences.
 * - General data, where each sample is encoded independently.
 * 
 * The implementation supports both bipolar and binary modes:
 * - **Bipolar Mode:** Incremental bundling of hypervectors is supported.
 * - **Binary Mode:** Hypervectors are pre-bundled for each class before updating
 *   associative memory due to the nature of majority voting.
 *
 * The file includes mechanisms for handling stability checks on n-grams and
 * ensures proper memory management for dynamically allocated hypervectors.
 */

#include "trainer.h"
#include "vector.h"
#include "assoc_mem.h"
#include "encoder.h"
#include "operations.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief Trains the HDC model using timeseries data.
 *
 * This function processes timeseries data, encoding n-grams into hypervectors
 * and updating the associative memory with class-specific bundled hypervectors.
 *
 * @param training_data A 2D array of training data with dimensions `[training_samples][NUM_FEATURES]`.
 * @param training_labels An array of class labels corresponding to the training data.
 * @param training_samples The number of samples in the training data.
 * @param assoc_mem A pointer to the associative memory structure for storing class-specific hypervectors.
 * @param enc A pointer to the encoder structure for encoding the training data.
 *
 * @note 
 * - In **bipolar mode**, incremental bundling is supported, allowing real-time updates.
 * - In **binary mode**, the hypervectors are pre-bundled for each class and then added.
 *
 * @warning 
 * - The function assumes proper memory allocation for all input pointers.
 *
 * @details 
 * - This function first verifies label stability within n-grams. 
 * - Stable n-grams are encoded into hypervectors, which are added to the associative memory.
 * - In bipolar mode, associative memory updates occur incrementally, while in binary mode, updates are applied after bundling.
 */
void train_model_timeseries(double **training_data, int *training_labels, int training_samples, struct associative_memory *assoc_mem, struct encoder *enc) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Training HDC-Model for %d training samples.\n",training_samples);
        fflush(stdout);
    }
    #if BIPOLAR_MODE

    for (int j = 0; j < training_samples - N_GRAM_SIZE; j++) {
        Vector* sample_hv = create_vector();
        if (is_window_stable(&training_labels[j])) {
            encode_timeseries(enc, &training_data[j], sample_hv);
            add_to_assoc_mem(assoc_mem, sample_hv, training_labels[j]);

        }
        free_vector(sample_hv);

    }
    if (NORMALIZE) {
        normalize(assoc_mem);
    }
    
#else
#if ENCODER_ROLLING
    int window_size = N_GRAM_SIZE;
    Vector *rolling_acc = create_vector();
    Vector **window_vectors = (Vector **)malloc((size_t)window_size * sizeof(Vector *));
    int **class_bit_counts = (int **)malloc((size_t)NUM_CLASSES * sizeof(int *));
    int *class_counts = (int *)calloc((size_t)NUM_CLASSES, sizeof(int));

    if (!rolling_acc || !window_vectors || !class_bit_counts || !class_counts) {
        fprintf(stderr, "Failed to allocate rolling training buffers.\n");
        if (rolling_acc) {
            free_vector(rolling_acc);
        }
        free(window_vectors);
        free(class_bit_counts);
        free(class_counts);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < window_size; i++) {
        window_vectors[i] = create_vector();
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
        class_bit_counts[c] = (int *)calloc((size_t)VECTOR_DIMENSION, sizeof(int));
        if (!class_bit_counts[c]) {
            fprintf(stderr, "Failed to allocate class bit counters.\n");
            exit(EXIT_FAILURE);
        }
    }

    int window_filled = 0;
    int window_pos = 0;
    for (int i = 0; i < training_samples; i++) {
        Vector *sample_hv = create_vector();
        Vector *rotated_hv = create_vector();
        encode_timestamp(enc, training_data[i], sample_hv);
        permute(sample_hv, window_pos, rotated_hv);

        if (window_filled < window_size) {
            bind(rolling_acc, rotated_hv, rolling_acc);
            memcpy(window_vectors[window_pos]->data,
                   rotated_hv->data,
                   VECTOR_DIMENSION * sizeof(vector_element));
            window_filled++;
        } else {
            bind(rolling_acc, window_vectors[window_pos], rolling_acc);
            bind(rolling_acc, rotated_hv, rolling_acc);
            memcpy(window_vectors[window_pos]->data,
                   rotated_hv->data,
                   VECTOR_DIMENSION * sizeof(vector_element));
        }

        window_pos = (window_pos + 1) % window_size;

        if (i >= window_size - 1) {
            int label = training_labels[i];
            if (label >= 0 && label < NUM_CLASSES) {
                for (int d = 0; d < VECTOR_DIMENSION; d++) {
                    class_bit_counts[label][d] += rolling_acc->data[d] ? 1 : 0;
                }
                class_counts[label]++;
            }
        }

        free_vector(rotated_hv);
        free_vector(sample_hv);
    }

    for (int class_id = 0; class_id < NUM_CLASSES; class_id++) {
        Vector *bundled_hv = create_vector();
        int thr = class_counts[class_id] / 2;
        for (int d = 0; d < VECTOR_DIMENSION; d++) {
            bundled_hv->data[d] = (class_bit_counts[class_id][d] > thr) ? 1 : 0;
        }
        add_to_assoc_mem(assoc_mem, bundled_hv, class_id);
        assoc_mem->counts[class_id] = class_counts[class_id];
        free_vector(bundled_hv);
    }

    for (int c = 0; c < NUM_CLASSES; c++) {
        free(class_bit_counts[c]);
    }
    for (int i = 0; i < window_size; i++) {
        free_vector(window_vectors[i]);
    }
    free(class_counts);
    free(class_bit_counts);
    free(window_vectors);
    free_vector(rolling_acc);
#else
    Vector*** encoded_vectors = (Vector***)malloc((training_samples - N_GRAM_SIZE) * NUM_CLASSES * sizeof(Vector*));
    int* vector_counts = (int*)calloc(NUM_CLASSES, sizeof(int)); // Array to keep track of vector count for each class
    // Allocate memory for each class' vector array
    for (int class_id = 0; class_id < NUM_CLASSES; class_id++) {
        encoded_vectors[class_id] = (Vector**)malloc((training_samples - N_GRAM_SIZE) * sizeof(Vector*));
        
        // Initialize each vector to NULL or allocate the Vector objects
        for (int i = 0; i < training_samples - N_GRAM_SIZE; i++) {
            encoded_vectors[class_id][i] = create_vector();  // Allocate each Vector for the class
        }
    }
    for (int j = 0; j < training_samples - N_GRAM_SIZE; j++) {
     
        if (is_window_stable(&training_labels[j])) {  // Ensure the window is stable
            encode_timeseries(enc, &training_data[j], encoded_vectors[training_labels[j]][vector_counts[training_labels[j]]]);  // Encode the data
            if(training_labels[j]==1){
              //  printf("%d ",j+1);
            }

            vector_counts[training_labels[j]]++;
        }else{
            j+=(N_GRAM_SIZE-1);
        }
    }

    for (int class_id = 0; class_id < NUM_CLASSES; class_id++) {
        Vector* bundled_hv = create_vector();  // Create a bundled vector to store the final result
        bundle_multi(encoded_vectors[class_id], vector_counts[class_id], bundled_hv);  // Bundle all vectors for this class
        
        // Add the bundled vector to the associative memory for this class
        add_to_assoc_mem(assoc_mem, bundled_hv, class_id);
        assoc_mem->counts[class_id] = vector_counts[class_id];
        
        free_vector(bundled_hv);  // Free the bundled vector
        
        for (int i = 0; i < training_samples - N_GRAM_SIZE; i++) {
            free_vector(encoded_vectors[class_id][i]);  // Free each stored vector
        }

        free(encoded_vectors[class_id]);  // Free the array of vectors for each class
    }
    free(encoded_vectors);
    free(vector_counts);
#endif
#endif

    if (output_mode >= OUTPUT_DETAILED) {
        print_class_vectors(assoc_mem);
    }
}
/**
 * @brief Trains the HDC model using general (non-timeseries) data.
 *
 * This function encodes each training sample into a hypervector and updates
 * the associative memory with class-specific bundled hypervectors.
 *
 * @param training_data A 2D array of training data with dimensions `[training_samples][NUM_FEATURES]`.
 * @param training_labels An array of class labels corresponding to the training data.
 * @param training_samples The number of samples in the training data.
 * @param assoc_mem A pointer to the associative memory structure for storing class-specific hypervectors.
 * @param enc A pointer to the encoder structure for encoding the training data.
 *
 * @note 
 * - In **bipolar mode**, incremental bundling of hypervectors is supported.
 * - In **binary mode**, pre-bundling of class hypervectors is necessary before updates.
 *
 * @details 
 * - Each sample is individually encoded into a hypervector.
 * - Encoded hypervectors are bundled together for each class and added to associative memory.
 * - In binary mode, majority voting influences how the bundling is performed.
 *
 * @warning The caller is responsible for proper memory allocation and cleanup of all input data.
 */

void train_model_general_data(double **training_data, int *training_labels, int training_samples, struct associative_memory *assoc_mem, struct encoder *enc) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Training HDC-Model for %d training samples.",training_samples);
        fflush(stdout);
    }
    #if BIPOLAR_MODE
    // Training loop
    for (int j = 0; j < training_samples; j++) {
        Vector* sample_hv = create_vector();
        
        encode_general_data(enc, training_data[j], sample_hv);
        add_to_assoc_mem(assoc_mem, sample_hv, training_labels[j]);
        free_vector(sample_hv);

    }
    if (NORMALIZE) {
        normalize(assoc_mem);
    }
    
    #else
    Vector*** encoded_vectors = (Vector***)malloc((training_samples) * NUM_CLASSES * sizeof(Vector*));
    int* vector_counts = (int*)calloc(NUM_CLASSES, sizeof(int)); // Array to keep track of vector count for each class
    // Allocate memory for each class' vector array
    for (int class_id = 0; class_id < NUM_CLASSES; class_id++) {
        encoded_vectors[class_id] = (Vector**)malloc((training_samples) * sizeof(Vector*));
        
        // Initialize each vector to NULL or allocate the Vector objects
        for (int i = 0; i < training_samples; i++) {
            encoded_vectors[class_id][i] = create_vector();  // Allocate each Vector for the class
        }
    }
    for (int j = 0; j < training_samples; j++) {
            encode_general_data(enc, training_data[j], encoded_vectors[training_labels[j]][vector_counts[training_labels[j]]]); 
            vector_counts[training_labels[j]]++;
        
    }

    for (int class_id = 0; class_id < NUM_CLASSES; class_id++) {
        Vector* bundled_hv = create_vector();  // Create a bundled vector to store the final result
        bundle_multi(encoded_vectors[class_id], vector_counts[class_id], bundled_hv);  // Bundle all vectors for this class
        
        // Add the bundled vector to the associative memory for this class
        add_to_assoc_mem(assoc_mem, bundled_hv, class_id);
        assoc_mem->counts[class_id] = vector_counts[class_id];
        
        free_vector(bundled_hv);  // Free the bundled vector
        
        for (int i = 0; i < training_samples; i++) {
            free_vector(encoded_vectors[class_id][i]);  // Free each stored vector
        }

        free(encoded_vectors[class_id]);  // Free the array of vectors for each class
    }
    free(encoded_vectors);
    free(vector_counts);
    #endif

    if (output_mode >= OUTPUT_DETAILED) {
        print_class_vectors(assoc_mem);
    }
}
