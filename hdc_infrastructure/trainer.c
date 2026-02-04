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
