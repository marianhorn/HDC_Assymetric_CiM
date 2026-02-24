//This serves as main file.
//Classifies Dietmars foot movements

#include <stdio.h>
#include <stdlib.h>
#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/item_mem.h"
#include "../hdc_infrastructure/asymItemMemory.h"
#include "../hdc_infrastructure/encoder.h"
#include "../hdc_infrastructure/operations.h"
#include "dataReaderFootEMG.h"
#include "configFoot.h"
#include "../hdc_infrastructure/evaluator.h"
#include "../hdc_infrastructure/ResultManager.h"
#include "../hdc_infrastructure/vector.h"
#include "../hdc_infrastructure/trainer.h"

int output_mode = OUTPUT_MODE;

#define TEMP_KRISCHAN_IM_CSV "analysis/big_test/krischan_position_vectors.csv"
#define TEMP_KRISCHAN_CM_CSV "analysis/big_test/krischan_value_vectors.csv"

int main(){
    result_manager_init();
    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n\n");
    }

    double mean_overall_accuracy = 0.0;
    double mean_class_average_accuracy = 0.0;
    double mean_class_vector_similarity = 0.0;
    size_t sum_correct = 0;
    size_t sum_not_correct = 0;
    size_t sum_transition_error = 0;
    size_t sum_total = 0;

    for(int dataset = 0; dataset<4;dataset++){

        if (output_mode >= OUTPUT_BASIC) {
            printf("\n\nModel for dataset #%d\n",dataset);
        }
        #if PRECOMPUTED_ITEM_MEMORY
        struct item_memory itemMem;
        init_precomp_item_memory(&itemMem,NUM_LEVELS,NUM_FEATURES);

        struct encoder enc;
        init_encoder(&enc,&itemMem);
        #else
        struct item_memory electrodes;
        struct item_memory intensityLevels;
        if (output_mode >= OUTPUT_BASIC) {
            printf("Loading Krischan IM/CM from CSV:\n");
            printf("  IM: %s\n", TEMP_KRISCHAN_IM_CSV);
            printf("  CM: %s\n", TEMP_KRISCHAN_CM_CSV);
        }
        load_item_mem_from_csv(&electrodes, TEMP_KRISCHAN_IM_CSV, NUM_FEATURES);
        load_item_mem_from_csv(&intensityLevels, TEMP_KRISCHAN_CM_CSV, NUM_LEVELS);

        struct encoder enc;
        init_encoder(&enc,&electrodes,&intensityLevels);
        #endif

        double** trainingData;
        double** validationData;
        double** testingData;
        int* trainingLabels;
        int* validationLabels;
        int* testingLabels;
        int trainingSamples, validationSamples, testingSamples;

        struct associative_memory assMem;
        init_assoc_mem(&assMem);

        double validationRatio = VALIDATION_RATIO;
        getDataWithValSet(dataset,
                          &trainingData,
                          &validationData,
                          &testingData,
                          &trainingLabels,
                          &validationLabels,
                          &testingLabels,
                          &trainingSamples,
                          &validationSamples,
                          &testingSamples,
                          validationRatio);

        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);

        #if USE_GENETIC_ITEM_MEMORY
        #if PRECOMPUTED_ITEM_MEMORY
        optimize_item_memory(&itemMem,
                             trainingData,
                             trainingLabels,
                             trainingSamples,
                             validationData,
                             validationLabels,
                             validationSamples);
        #else
        optimize_item_memory(&intensityLevels,
                             &electrodes,
                             trainingData,
                             trainingLabels,
                             trainingSamples,
                             validationData,
                             validationLabels,
                             validationSamples);
        #endif
        free_assoc_mem(&assMem);
        init_assoc_mem(&assMem);
        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);
        #endif

        struct timeseries_eval_result eval_test =
            evaluate_model_timeseries_direct(&enc, &assMem, testingData, testingLabels, testingSamples);

        if (output_mode >= OUTPUT_BASIC) {
            printf("Dataset %02d accuracy: %.2f%%\n", dataset, eval_test.overall_accuracy * 100.0);
        }

        mean_overall_accuracy += eval_test.overall_accuracy;
        mean_class_average_accuracy += eval_test.class_average_accuracy;
        mean_class_vector_similarity += eval_test.class_vector_similarity;
        sum_correct += eval_test.correct;
        sum_not_correct += eval_test.not_correct;
        sum_transition_error += eval_test.transition_error;
        sum_total += eval_test.total;

        char result_info[128];
        #if USE_GENETIC_ITEM_MEMORY
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=postopt-test", dataset);
        #else
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=test", dataset);
        #endif
        addResult(&eval_test, result_info);

        // Free allocated memory
        freeData(trainingData, trainingSamples);
        if (validationData && validationSamples > 0) {
            freeData(validationData, validationSamples);
        }
        freeData(testingData, testingSamples);
        free(trainingLabels);
        free(validationLabels);
        free(testingLabels);
        free_assoc_mem(&assMem);

        #if PRECOMPUTED_ITEM_MEMORY
        free_item_memory(&itemMem);
        #else
        free_item_memory(&electrodes);
        free_item_memory(&intensityLevels);
        #endif
    }

    struct timeseries_eval_result overall_result = {0};
    overall_result.correct = sum_correct;
    overall_result.not_correct = sum_not_correct;
    overall_result.transition_error = sum_transition_error;
    overall_result.total = sum_total;
    overall_result.overall_accuracy = mean_overall_accuracy / 4.0;
    overall_result.class_average_accuracy = mean_class_average_accuracy / 4.0;
    overall_result.class_vector_similarity = mean_class_vector_similarity / 4.0;

    if (output_mode >= OUTPUT_BASIC) {
        printf("Accuracy: %.2f%%\n", overall_result.overall_accuracy * 100.0);
    }

    #if USE_GENETIC_ITEM_MEMORY
    addResult(&overall_result, "model=mine,scope=overall,phase=postopt-test");
    #else
    addResult(&overall_result, "model=mine,scope=overall,phase=test");
    #endif

    result_manager_close();
    return 0;
}
