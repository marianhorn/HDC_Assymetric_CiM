/**
 * @file modelCustom.c
 * @brief Main file for implementing new HDC-Models. Should be replaced or extended.
 * 
 * This program implements the training and evaluation pipeline for hyperdimensional 
 * classification of a general example problem (no timeseries data). It includes data 
 * preprocessing, model training, and evaluation stages.
 * 
 * @details The program supports two configurations:
 * - Precomputed item memory (`PRECOMPUTED_ITEM_MEMORY` enabled)
 * - Dynamic item memory initialization (`PRECOMPUTED_ITEM_MEMORY` disabled)
 */
#include <stdio.h>
#include <stdlib.h>
#include "configCustom.h"
#include "dataReaderCustom.h"
#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/item_mem.h"
#include "../hdc_infrastructure/encoder.h"
#include "../hdc_infrastructure/operations.h"
#include "../hdc_infrastructure/evaluator.h"
#include "../hdc_infrastructure/ResultManager.h"
#include "../hdc_infrastructure/quantizer.h"
#include "../hdc_infrastructure/vector.h"
#include "../hdc_infrastructure/trainer.h"
/**
 * @brief Main entry point of the program.
 * 
 * @return Returns 1 on successful execution.
 */
int output_mode = OUTPUT_MODE;

int main(){
    quantizer_clear();
    result_manager_init();
    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n\n");
    }

    struct item_memory features;
    struct item_memory values;
    init_item_memory(&features,NUM_FEATURES);
    init_continuous_item_memory(&values,NUM_LEVELS);

    struct encoder enc;
    init_encoder(&enc,&features,&values);

    double** trainingData;
    double** testingData;
    int* trainingLabels;
    int* testingLabels;
    int trainingSamples, testingSamples;

    struct associative_memory assMem;
    init_assoc_mem(&assMem);

    getData(&trainingData,&testingData,&trainingLabels,&testingLabels,&trainingSamples,&testingSamples);

    if (quantizer_fit_from_training(trainingData,
                                    trainingLabels,
                                    trainingSamples,
                                    NUM_FEATURES,
                                    NUM_LEVELS) != 0) {
        fprintf(stderr, "Error: Failed to initialize quantizer.\n");
        return EXIT_FAILURE;
    }

    train_model_general_data(trainingData, trainingLabels, trainingSamples, &assMem, &enc);

    struct timeseries_eval_result eval_result =
        evaluate_model_general_direct(&enc, &assMem, testingData, testingLabels, testingSamples);
    addResult(&eval_result, "custom");

    if (output_mode == OUTPUT_BASIC) {
        printf("Test accuracy: %.2f%%\n", eval_result.overall_accuracy * 100.0);
        printf("Class-average accuracy: %.2f%%\n", eval_result.class_average_accuracy * 100.0);
        printf("Class vector similarity: %.3f\n", eval_result.class_vector_similarity);
    }

    // Free allocated memory
    freeData(trainingData, trainingSamples);
    freeData(testingData, testingSamples);
    free(trainingLabels);
    free(testingLabels);
    free_assoc_mem(&assMem);

    free_item_memory(&features);
    free_item_memory(&values);
    quantizer_clear();
    result_manager_close();
    return 0;
}
