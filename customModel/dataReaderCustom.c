/**
 * @file dataReaderCustom.c
 * @brief This file provides functions for reading, allocating, and managing training and testing data.
 *
 * @details
 * The file defines functions for:
 * - Freeing dynamically allocated 2D arrays.
 * - Preparing training and testing datasets for an example problem.
 * 
 * This implementation uses predefined static arrays for demonstration and simulates data loading.
 *
 * @author Marian Horn
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dataReaderCustom.h"
#include "../hdc_infrastructure/preprocessor.h"


/**
 * @brief Frees a dynamically allocated 2D array.
 *
 * This function releases memory allocated for a 2D array of doubles.
 *
 * @param data A pointer to the 2D array to be freed.
 * @param rows The number of rows in the 2D array.
 */
void freeData(double** data, size_t rows) {
    for (size_t i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
}

/**
 * @brief Allocates and initializes training and testing datasets.
 *
 * This function initializes training and testing datasets using predefined static arrays and assigns appropriate labels.
 *
 * @param[out] trainingData Pointer to the allocated training data.
 * @param[out] testingData Pointer to the allocated testing data.
 * @param[out] trainingLabels Pointer to the allocated training labels.
 * @param[out] testingLabels Pointer to the allocated testing labels.
 * @param[out] trainingSamples Pointer to the number of training samples.
 * @param[out] testingSamples Pointer to the number of testing samples.
 *
 * @note
 * The input data and labels must follow this structure:
 *
 * - **Data (`trainingData`, `testingData`):** 
 *   A 2D array of type `double`, where:
 *   - Each row represents a sample.
 *   - Each column represents a feature.
 *   - The dimensions are `numSamples x numFeatures`.
 *
 *   Example:
 *   @code
 *   trainingData = {
 *       {feature_1, feature_2, ..., feature_n},  // Sample 1
 *       {feature_1, feature_2, ..., feature_n},  // Sample 2
 *       ...
 *   };
 *   @endcode
 *
 * - **Labels (`trainingLabels`, `testingLabels`):**
 *   A 1D array of type `int`, where:
 *   - Each element corresponds to the class label of a sample.
 *
 *   Example:
 *   @code
 *   trainingLabels = {0, 1, ..., k}; // Class labels for each sample
 *   @endcode
 *
 * Ensure that:
 * - The number of samples in the label array matches the number of rows in the data array.
 * - The data is correctly preprocessed to match the requirements of the model, including 
 *   the number of features and the label space.
 */

void getData(double*** trainingData, double*** testingData, int** trainingLabels, int** testingLabels, int* trainingSamples, int* testingSamples) {
    
    *trainingSamples = 3;
    double trainingArray[3][NUM_FEATURES] = {
        {3, 0, 0, 0},
        {0, 3, 0, 0},
        {0, 0, 3, 0}
    };

    *testingSamples = 3;
    double testingArray[3][NUM_FEATURES] = {
        {2, 0, 0, 0},
        {0, 4, 0, 0},
        {0, 0, 3, 0}
    };

    static int trainingArrayLabels[3] = {0, 1, 2};

    static int testingArrayLabels[3] = {0, 1, 2};

    *trainingData = (double**)malloc(*trainingSamples * sizeof(double*));
    *testingData = (double**)malloc(*testingSamples * sizeof(double*));

    for (int i = 0; i < 3; i++) {
        (*trainingData)[i] = (double*)malloc(NUM_FEATURES * sizeof(double));
        (*testingData)[i] = (double*)malloc(NUM_FEATURES * sizeof(double));
        memcpy((*trainingData)[i], trainingArray[i], NUM_FEATURES * sizeof(double));
        memcpy((*testingData)[i], testingArray[i], NUM_FEATURES * sizeof(double));
    }

    *trainingLabels = (int*)malloc(*trainingSamples * sizeof(int));
    *testingLabels = (int*)malloc(*testingSamples * sizeof(int));
    memcpy(*trainingLabels, trainingArrayLabels, *trainingSamples * sizeof(int));
    memcpy(*testingLabels, testingArrayLabels, *testingSamples * sizeof(int));


    printf("Data initialized successfully!\n");
}