/**
 * @file preprocessor.c
 * @brief Contains functions for preprocessing data before hyperdimensional encoding.
 *
 * This file includes utilities to preprocess input datasets, such as:
 * - Downsampling data to reduce its size while preserving structure.
 *
 * @details
 * The preprocessing steps are crucial for optimizing computational efficiency and ensuring compatibility
 * with the hyperdimensional computing pipeline. Downsampling reduces the data size while maintaining
 * temporal resolution, based on the `DOWNSAMPLE` factor.
 */
#include "preprocessor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/**
 * @brief Downsamples the input data by a specified factor.
 *
 * This function reduces the size of the input dataset by selecting every `DOWNSAMPLE`-th sample.
 * It adjusts both the data and their corresponding labels accordingly.
 *
 * @param data The original 2D array of input data with dimensions `[original_size][NUM_FEATURES]`.
 * @param labels The array of labels corresponding to the input data, with `original_size` elements.
 * @param original_size The total number of samples in the input data.
 * @param downsampled_data Pointer to the output 2D array for the downsampled data.
 * @param downsampled_labels Pointer to the output array for the downsampled labels.
 * @param newSize Pointer to a variable to store the new size of the downsampled data.
 *
 * @note The output arrays (`downsampled_data` and `downsampled_labels`) are dynamically allocated
 * and must be freed by the caller.
 *
 * @warning If memory allocation fails at any point, the program will terminate with an error message.
 */
void down_sample(double** data, int* labels, size_t original_size, double*** downsampled_data, int** downsampled_labels, size_t *newSize) {
    size_t new_length = original_size / DOWNSAMPLE;
    *downsampled_data = (double **)malloc(new_length * sizeof(double*));
    if(*downsampled_data == NULL) {
        perror("Malloc failed for downsampled_data");
        exit(EXIT_FAILURE);
    }

    *downsampled_labels = (int *)malloc(new_length * sizeof(int));
    if(*downsampled_labels == NULL) {
        perror("Malloc failed for downsampled_labels");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < new_length; i++) {
        (*downsampled_data)[i] = (double *)malloc(NUM_FEATURES * sizeof(double));
        if ((*downsampled_data)[i] == NULL) {
            perror("Malloc failed for downsampled_data[i]");
            exit(EXIT_FAILURE);
        }
        memcpy((*downsampled_data)[i], data[i * DOWNSAMPLE], NUM_FEATURES * sizeof(double));
        (*downsampled_labels)[i] = labels[i * DOWNSAMPLE];
    }

    *newSize = new_length;

    if(*downsampled_data == NULL) {
        perror("Error already in downsampling");
        exit(EXIT_FAILURE);
    }
}
