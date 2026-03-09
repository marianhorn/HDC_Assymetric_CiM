//This serves as main file.
//Classifies Dietmars foot movements

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
#ifdef _OPENMP
#include <omp.h>
#endif

int output_mode = OUTPUT_MODE;

static double now_ms(void) {
#ifdef _OPENMP
    return omp_get_wtime() * 1000.0;
#else
    return 1000.0 * (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

int main(){
    result_manager_init();
    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n\n");
    }

    double mean_post_test_overall_accuracy = 0.0;
    double mean_post_test_class_average_accuracy = 0.0;
    double mean_post_test_class_vector_similarity = 0.0;
    double mean_pre_test_overall_accuracy = 0.0;
    double mean_pre_test_class_average_accuracy = 0.0;
    double mean_pre_test_class_vector_similarity = 0.0;
    double mean_pre_val_overall_accuracy = 0.0;
    double mean_pre_val_class_average_accuracy = 0.0;
    double mean_pre_val_class_vector_similarity = 0.0;
    double mean_post_val_overall_accuracy = 0.0;
    double mean_post_val_class_average_accuracy = 0.0;
    double mean_post_val_class_vector_similarity = 0.0;
    size_t sum_correct = 0;
    size_t sum_not_correct = 0;
    size_t sum_transition_error = 0;
    size_t sum_total = 0;
    double sum_training_time_ms = 0.0;

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
        init_item_memory(&electrodes, NUM_FEATURES);
        init_continuous_item_memory(&intensityLevels, NUM_LEVELS);

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

        double train_start_ms = now_ms();
        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);
        double train_end_ms = now_ms();
        double training_time_ms = train_end_ms - train_start_ms;

        if (output_mode >= OUTPUT_BASIC) {
            printf("Dataset %02d initial training time: %.3f ms\n", dataset, training_time_ms);
        }

        #if USE_GENETIC_ITEM_MEMORY
        #else
        sum_training_time_ms += training_time_ms;
        #endif

        struct timeseries_eval_result eval_pre_val = {0};
        struct timeseries_eval_result eval_pre_test = {0};
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Evaluating pre-optimization model on validation set.\n");
        }
        if (validationData && validationLabels && validationSamples > 0) {
            eval_pre_val = evaluate_model_timeseries_direct(&enc, &assMem, validationData, validationLabels, validationSamples);
        }
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Evaluating pre-optimization model on test set.\n");
        }
        eval_pre_test = evaluate_model_timeseries_direct(&enc, &assMem, testingData, testingLabels, testingSamples);

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
        #endif

        if (output_mode >= OUTPUT_BASIC) {
            printf("Dataset %02d pre-optimization test accuracy: %.2f%%\n", dataset, eval_pre_test.overall_accuracy * 100.0);
        }

        char result_info[160];
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=preopt-validation", dataset);
        addResult(&eval_pre_val, result_info);
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=preopt-test", dataset);
        addResult(&eval_pre_test, result_info);

        mean_pre_val_overall_accuracy += eval_pre_val.overall_accuracy;
        mean_pre_val_class_average_accuracy += eval_pre_val.class_average_accuracy;
        mean_pre_val_class_vector_similarity += eval_pre_val.class_vector_similarity;
        mean_pre_test_overall_accuracy += eval_pre_test.overall_accuracy;
        mean_pre_test_class_average_accuracy += eval_pre_test.class_average_accuracy;
        mean_pre_test_class_vector_similarity += eval_pre_test.class_vector_similarity;

        #if USE_GENETIC_ITEM_MEMORY
        sum_training_time_ms += training_time_ms;
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Re-training post-optimization model.\n");
        }
        train_start_ms = now_ms();
        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);
        train_end_ms = now_ms();
        training_time_ms = train_end_ms - train_start_ms;
        if (output_mode >= OUTPUT_BASIC) {
            printf("Dataset %02d post-optimization training time: %.3f ms\n", dataset, training_time_ms);
        }
        #endif

        struct timeseries_eval_result eval_post_val = {0};
        struct timeseries_eval_result eval_post_test =
            evaluate_model_timeseries_direct(&enc, &assMem, testingData, testingLabels, testingSamples);

        if (validationData && validationLabels && validationSamples > 0) {
            if (output_mode >= OUTPUT_DETAILED) {
                printf("Evaluating post-optimization model on validation set.\n");
            }
            eval_post_val = evaluate_model_timeseries_direct(&enc, &assMem, validationData, validationLabels, validationSamples);
        }

        mean_post_val_overall_accuracy += eval_post_val.overall_accuracy;
        mean_post_val_class_average_accuracy += eval_post_val.class_average_accuracy;
        mean_post_val_class_vector_similarity += eval_post_val.class_vector_similarity;
        mean_post_test_overall_accuracy += eval_post_test.overall_accuracy;
        mean_post_test_class_average_accuracy += eval_post_test.class_average_accuracy;
        mean_post_test_class_vector_similarity += eval_post_test.class_vector_similarity;

        if (output_mode >= OUTPUT_BASIC) {
            printf("Dataset %02d pre-opt test accuracy: %.2f%%\n", dataset, eval_pre_test.overall_accuracy * 100.0);
            printf("Dataset %02d post-opt test accuracy: %.2f%%\n", dataset, eval_post_test.overall_accuracy * 100.0);
        }

        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=postopt-validation", dataset);
        addResult(&eval_post_val, result_info);
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=postopt-test", dataset);
        addResult(&eval_post_test, result_info);

        sum_correct += eval_post_test.correct;
        sum_not_correct += eval_post_test.not_correct;
        sum_transition_error += eval_post_test.transition_error;
        sum_total += eval_post_test.total;

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
    overall_result.overall_accuracy = mean_post_test_overall_accuracy / 4.0;
    overall_result.class_average_accuracy = mean_post_test_class_average_accuracy / 4.0;
    overall_result.class_vector_similarity = mean_post_test_class_vector_similarity / 4.0;

    if (output_mode >= OUTPUT_BASIC) {
        printf("Accuracy: %.2f%%\n", overall_result.overall_accuracy * 100.0);
        printf("Mean training time per dataset: %.3f ms\n", sum_training_time_ms / 4.0);
    }

    if (output_mode >= OUTPUT_BASIC) {
        printf("Overall pre-optimization (test): %.2f%%\n", (mean_pre_test_overall_accuracy / 4.0) * 100.0);
        printf("Overall post-optimization (test): %.2f%%\n", (mean_post_test_overall_accuracy / 4.0) * 100.0);
        printf("Overall pre-optimization (validation): %.2f%%\n", (mean_pre_val_overall_accuracy / 4.0) * 100.0);
        printf("Overall post-optimization (validation): %.2f%%\n", (mean_post_val_overall_accuracy / 4.0) * 100.0);
    }

    struct timeseries_eval_result overall_pre_val = {0};
    overall_pre_val.overall_accuracy = mean_pre_val_overall_accuracy / 4.0;
    overall_pre_val.class_average_accuracy = mean_pre_val_class_average_accuracy / 4.0;
    overall_pre_val.class_vector_similarity = mean_pre_val_class_vector_similarity / 4.0;
    addResult(&overall_pre_val, "model=mine,scope=overall,phase=preopt-validation");

    struct timeseries_eval_result overall_pre_test = {0};
    overall_pre_test.overall_accuracy = mean_pre_test_overall_accuracy / 4.0;
    overall_pre_test.class_average_accuracy = mean_pre_test_class_average_accuracy / 4.0;
    overall_pre_test.class_vector_similarity = mean_pre_test_class_vector_similarity / 4.0;
    addResult(&overall_pre_test, "model=mine,scope=overall,phase=preopt-test");

    addResult(&overall_result, "model=mine,scope=overall,phase=postopt-test");

    struct timeseries_eval_result overall_post_val = {0};
    overall_post_val.overall_accuracy = mean_post_val_overall_accuracy / 4.0;
    overall_post_val.class_average_accuracy = mean_post_val_class_average_accuracy / 4.0;
    overall_post_val.class_vector_similarity = mean_post_val_class_vector_similarity / 4.0;
    addResult(&overall_post_val, "model=mine,scope=overall,phase=postopt-validation");

    result_manager_close();
    return 0;
}
