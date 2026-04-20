// Main entry point for foot EMG classification.

#include <stdio.h>
#include <stdlib.h>

#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/asymItemMemory.h"
#include "../hdc_infrastructure/encoder.h"
#include "../hdc_infrastructure/evaluator.h"
#include "../hdc_infrastructure/item_mem.h"
#include "../hdc_infrastructure/quantizer.h"
#include "../hdc_infrastructure/trainer.h"
#include "configFoot.h"
#include "dataReaderFootEMG.h"

int output_mode = OUTPUT_MODE;

int main(void) {
    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n");
    }

    double mean_pre_validation_accuracy = 0.0;
    double mean_pre_test_accuracy = 0.0;
#if USE_GENETIC_ITEM_MEMORY
    double mean_post_validation_accuracy = 0.0;
    double mean_post_test_accuracy = 0.0;
#endif
    int processed_datasets = 0;
//Loop over datasets
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
            printf("\nDataset %02d\n", dataset);
        }
//Initialize Item Memory and Encoder
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
//Fit quantizer
        if (quantizer_fit_from_training(training_data,
                                        training_labels,
                                        training_samples,
                                        NUM_FEATURES,
                                        NUM_LEVELS) != 0) {
            fprintf(stderr, "Error: Failed to initialize quantizer for dataset %d.\n", dataset);
            return EXIT_FAILURE;
        }
//Branch only for GA-Binning
#if BINNING_MODE == GA_REFINED_BINNING
        {
            const int genome_length = NUM_FEATURES * (NUM_LEVELS - 1);
            uint16_t *flip_counts = NULL;

            if (genome_length <= 0) {
                fprintf(stderr, "Error: Invalid GA-refined quantizer genome length for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }

            flip_counts = (uint16_t *)calloc((size_t)genome_length, sizeof(uint16_t));
            if (!flip_counts) {
                fprintf(stderr, "Error: Failed to allocate preprocessing GA flip counts for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }

            if (output_mode >= OUTPUT_DETAILED) {
                printf("Running preprocessing GA for quantizer refinement.\n");
            }

            if (optimize_item_memory_get_flip_counts(training_data,
                                                     training_labels,
                                                     training_samples,
                                                     validation_data,
                                                     validation_labels,
                                                     validation_samples,
                                                     flip_counts) != 0) {
                free(flip_counts);
                fprintf(stderr, "Error: Failed to run preprocessing GA for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }

            if (quantizer_refine_from_flip_counts(flip_counts, genome_length) != 0) {
                free(flip_counts);
                fprintf(stderr, "Error: Failed to refine quantizer for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }

            free(flip_counts);
        }
#endif

        struct timeseries_eval_result pre_validation_result = {0};
        struct timeseries_eval_result pre_test_result = {0};
        struct timeseries_eval_result post_validation_result = {0};
        struct timeseries_eval_result post_test_result = {0};
//Train model
        train_model_timeseries(training_data, training_labels, training_samples, &assoc_mem, &enc);
//Evaluate model before GA optimization
        pre_validation_result =
            evaluate_model_timeseries_direct(&enc, &assoc_mem, validation_data, validation_labels, validation_samples);

        pre_test_result =
            evaluate_model_timeseries_direct(&enc, &assoc_mem, testing_data, testing_labels, testing_samples);
//Run GA if enabled
#if USE_GENETIC_ITEM_MEMORY
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Running item-memory GA.\n");
        }

#if PRECOMPUTED_ITEM_MEMORY
        optimize_item_memory(&item_mem,
                             training_data,
                             training_labels,
                             training_samples,
                             validation_data,
                             validation_labels,
                             validation_samples);
#else
        optimize_item_memory(&intensity_levels,
                             &electrodes,
                             training_data,
                             training_labels,
                             training_samples,
                             validation_data,
                             validation_labels,
                             validation_samples);
#endif

        free_assoc_mem(&assoc_mem);
        init_assoc_mem(&assoc_mem);
        train_model_timeseries(training_data, training_labels, training_samples, &assoc_mem, &enc);

    //Evaluate model after GA optimization
        post_validation_result =
            evaluate_model_timeseries_direct(&enc, &assoc_mem, validation_data, validation_labels, validation_samples);

        post_test_result =
            evaluate_model_timeseries_direct(&enc, &assoc_mem, testing_data, testing_labels, testing_samples);
#endif
//Output results
        if (output_mode >= OUTPUT_BASIC) {
            printf("\n  Pre-Optimization\n");
            if (validation_data && validation_labels && validation_samples > 0) {
                printf("    validation accuracy: %.2f%%\n", pre_validation_result.overall_accuracy * 100.0);
            } else {
                printf("    validation accuracy: n/a\n");
            }
            printf("    test accuracy: %.2f%%\n", pre_test_result.overall_accuracy * 100.0);
#if USE_GENETIC_ITEM_MEMORY
            printf("  Post-Optimization\n");
            if (validation_data && validation_labels && validation_samples > 0) {
                printf("    validation accuracy: %.2f%%\n", post_validation_result.overall_accuracy * 100.0);
            } else {
                printf("    validation accuracy: n/a\n");
            }
            printf("    test accuracy: %.2f%%\n", post_test_result.overall_accuracy * 100.0);
#endif
        }

        mean_pre_validation_accuracy += pre_validation_result.overall_accuracy;
        mean_pre_test_accuracy += pre_test_result.overall_accuracy;
#if USE_GENETIC_ITEM_MEMORY
        mean_post_validation_accuracy += post_validation_result.overall_accuracy;
        mean_post_test_accuracy += post_test_result.overall_accuracy;
#endif
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

    {
        const double dataset_count = processed_datasets > 0 ? (double)processed_datasets : 1.0;

        if (output_mode >= OUTPUT_BASIC && processed_datasets > 0) {
            printf("\nOverall\n");
            printf("  Pre-Optimization\n");
            printf("    validation accuracy: %.2f%%\n", (mean_pre_validation_accuracy / dataset_count) * 100.0);
            printf("    test accuracy: %.2f%%\n", (mean_pre_test_accuracy / dataset_count) * 100.0);
#if USE_GENETIC_ITEM_MEMORY
            printf("  Post-Optimization\n");
            printf("    validation accuracy: %.2f%%\n", (mean_post_validation_accuracy / dataset_count) * 100.0);
            printf("    test accuracy: %.2f%%\n", (mean_post_test_accuracy / dataset_count) * 100.0);
#endif
        }

    }

    quantizer_clear();
    return 0;
}
