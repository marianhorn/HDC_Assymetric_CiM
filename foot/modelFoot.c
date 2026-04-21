// Main file for foot EMG classification.

#include <stdio.h>
#include <stdlib.h>

#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/encoder.h"
#include "../hdc_infrastructure/evaluator.h"
#include "../hdc_infrastructure/item_mem.h"
#include "../hdc_infrastructure/quantizer.h"
#include "../hdc_infrastructure/ResultManager.h"
#include "../hdc_infrastructure/trainer.h"
#include "configFoot.h"
#include "dataReaderFootEMG.h"

int output_mode = OUTPUT_MODE;

int main(void) {
    result_manager_init();

    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n\n");
    }

    double mean_validation_accuracy = 0.0;
    double mean_validation_class_average = 0.0;
    double mean_validation_class_similarity = 0.0;
    double mean_test_accuracy = 0.0;
    double mean_test_class_average = 0.0;
    double mean_test_class_similarity = 0.0;
    size_t sum_correct = 0;
    size_t sum_not_correct = 0;
    size_t sum_transition_error = 0;
    size_t sum_total = 0;
    int processed_datasets = 0;

    for (int dataset = 0; dataset < 4; dataset++) {
        double **training_data = NULL;
        double **validation_data = NULL;
        double **testing_data = NULL;
        int *training_labels = NULL;
        int *validation_labels = NULL;
        int *testing_labels = NULL;
        int training_samples = 0;
        int validation_samples = 0;
        int testing_samples = 0;

        quantizer_clear();

        if (output_mode >= OUTPUT_BASIC) {
            printf("Model for dataset #%d\n", dataset);
        }

#if PRECOMPUTED_ITEM_MEMORY
        struct item_memory item_mem;
        init_precomp_item_memory(&item_mem, NUM_LEVELS, NUM_FEATURES);

        struct encoder enc;
        init_encoder(&enc, &item_mem);
#else
        struct item_memory electrodes;
        struct item_memory intensity_levels;
        init_item_memory(&electrodes, NUM_FEATURES);
        init_continuous_item_memory(&intensity_levels, NUM_LEVELS);

        struct encoder enc;
        init_encoder(&enc, &electrodes, &intensity_levels);
#endif

        struct associative_memory assoc_mem;
        init_assoc_mem(&assoc_mem);

        getDataWithValSet(dataset,
                          &training_data,
                          &validation_data,
                          &testing_data,
                          &training_labels,
                          &validation_labels,
                          &testing_labels,
                          &training_samples,
                          &validation_samples,
                          &testing_samples,
                          VALIDATION_RATIO);

        if (quantizer_fit_from_training(training_data,
                                        training_labels,
                                        training_samples,
                                        NUM_FEATURES,
                                        NUM_LEVELS) != 0) {
            fprintf(stderr, "Error: Failed to initialize quantizer for dataset %d.\n", dataset);
            return EXIT_FAILURE;
        }

        train_model_timeseries(training_data, training_labels, training_samples, &assoc_mem, &enc);

        struct timeseries_eval_result validation_result = {0};
        struct timeseries_eval_result test_result = {0};

        if (validation_data && validation_labels && validation_samples > 0) {
            if (output_mode >= OUTPUT_DETAILED) {
                printf("Evaluating validation set.\n");
            }
            validation_result =
                evaluate_model_timeseries_direct(&enc, &assoc_mem, validation_data, validation_labels, validation_samples);
        }

        if (output_mode >= OUTPUT_DETAILED) {
            printf("Evaluating test set.\n");
        }
        test_result =
            evaluate_model_timeseries_direct(&enc, &assoc_mem, testing_data, testing_labels, testing_samples);

        if (output_mode >= OUTPUT_BASIC) {
            printf("Dataset %02d validation accuracy: ", dataset);
            if (validation_data && validation_labels && validation_samples > 0) {
                printf("%.2f%%\n", validation_result.overall_accuracy * 100.0);
            } else {
                printf("n/a\n");
            }
            printf("Dataset %02d test accuracy: %.2f%%\n\n", dataset, test_result.overall_accuracy * 100.0);
        }

        char result_info[128];
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=validation", dataset);
        addResult(&validation_result, result_info);
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=test", dataset);
        addResult(&test_result, result_info);

        mean_validation_accuracy += validation_result.overall_accuracy;
        mean_validation_class_average += validation_result.class_average_accuracy;
        mean_validation_class_similarity += validation_result.class_vector_similarity;
        mean_test_accuracy += test_result.overall_accuracy;
        mean_test_class_average += test_result.class_average_accuracy;
        mean_test_class_similarity += test_result.class_vector_similarity;

        sum_correct += test_result.correct;
        sum_not_correct += test_result.not_correct;
        sum_transition_error += test_result.transition_error;
        sum_total += test_result.total;
        processed_datasets++;

        freeData(training_data, training_samples);
        if (validation_data && validation_samples > 0) {
            freeData(validation_data, validation_samples);
        }
        freeData(testing_data, testing_samples);
        free(training_labels);
        free(validation_labels);
        free(testing_labels);
        free_assoc_mem(&assoc_mem);

#if PRECOMPUTED_ITEM_MEMORY
        free_item_memory(&item_mem);
#else
        free_item_memory(&electrodes);
        free_item_memory(&intensity_levels);
#endif
    }

    if (processed_datasets > 0) {
        double dataset_count = (double)processed_datasets;
        struct timeseries_eval_result overall_validation = {0};
        struct timeseries_eval_result overall_test = {0};

        overall_validation.overall_accuracy = mean_validation_accuracy / dataset_count;
        overall_validation.class_average_accuracy = mean_validation_class_average / dataset_count;
        overall_validation.class_vector_similarity = mean_validation_class_similarity / dataset_count;

        overall_test.correct = sum_correct;
        overall_test.not_correct = sum_not_correct;
        overall_test.transition_error = sum_transition_error;
        overall_test.total = sum_total;
        overall_test.overall_accuracy = mean_test_accuracy / dataset_count;
        overall_test.class_average_accuracy = mean_test_class_average / dataset_count;
        overall_test.class_vector_similarity = mean_test_class_similarity / dataset_count;

        if (output_mode >= OUTPUT_BASIC) {
            printf("Overall validation accuracy: %.2f%%\n", overall_validation.overall_accuracy * 100.0);
            printf("Overall test accuracy: %.2f%%\n", overall_test.overall_accuracy * 100.0);
        }

        addResult(&overall_validation, "model=mine,scope=overall,phase=validation");
        addResult(&overall_test, "model=mine,scope=overall,phase=test");
    }

    quantizer_clear();
    result_manager_close();
    return 0;
}
