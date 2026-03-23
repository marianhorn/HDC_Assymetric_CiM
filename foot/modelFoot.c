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
#include "../hdc_infrastructure/quantizer.h"
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

int main(void){
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
#if BINNING_MODE != UNIFORM_BINNING
    double mean_uniform_test_overall_accuracy = 0.0;
    double mean_uniform_test_class_average_accuracy = 0.0;
    double mean_uniform_test_class_vector_similarity = 0.0;
    double mean_uniform_val_overall_accuracy = 0.0;
    double mean_uniform_val_class_average_accuracy = 0.0;
    double mean_uniform_val_class_vector_similarity = 0.0;
#endif
#if BINNING_MODE == GA_REFINED_BINNING
#if USE_GENETIC_ITEM_MEMORY
    double mean_uniform_ga_test_overall_accuracy = 0.0;
    double mean_uniform_ga_test_class_average_accuracy = 0.0;
    double mean_uniform_ga_test_class_vector_similarity = 0.0;
    double mean_uniform_ga_val_overall_accuracy = 0.0;
    double mean_uniform_ga_val_class_average_accuracy = 0.0;
    double mean_uniform_ga_val_class_vector_similarity = 0.0;
#endif
#endif
    size_t sum_correct = 0;
    size_t sum_not_correct = 0;
    size_t sum_transition_error = 0;
    size_t sum_total = 0;
    double sum_training_time_ms = 0.0;

    for(int dataset = 0; dataset<4;dataset++){
        quantizer_clear();

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

        if (quantizer_fit_from_training(trainingData,
                                        trainingLabels,
                                        trainingSamples,
                                        NUM_FEATURES,
                                        NUM_LEVELS) != 0) {
            fprintf(stderr, "Error: Failed to initialize quantizer for dataset %d.\n", dataset);
            return EXIT_FAILURE;
        }

#if BINNING_MODE != UNIFORM_BINNING
        struct timeseries_eval_result eval_uniform_val = {0};
        struct timeseries_eval_result eval_uniform_test = {0};
        int has_uniform_baseline = 0;
#endif

#if BINNING_MODE == GA_REFINED_BINNING
        struct timeseries_eval_result eval_uniform_ga_val = {0};
        struct timeseries_eval_result eval_uniform_ga_test = {0};
        int has_uniform_ga_stage = 0;
        {
            struct associative_memory uniformAssMem;
            init_assoc_mem(&uniformAssMem);
            int saved_output_mode = output_mode;
            output_mode = OUTPUT_NONE;
            train_model_timeseries(trainingData, trainingLabels, trainingSamples, &uniformAssMem, &enc);
            if (validationData && validationLabels && validationSamples > 0) {
                eval_uniform_val =
                    evaluate_model_timeseries_direct(&enc, &uniformAssMem, validationData, validationLabels, validationSamples);
            }
            eval_uniform_test =
                evaluate_model_timeseries_direct(&enc, &uniformAssMem, testingData, testingLabels, testingSamples);
            has_uniform_baseline = 1;
            free_assoc_mem(&uniformAssMem);

#if USE_GENETIC_ITEM_MEMORY
            struct item_memory uniformGaItemMem;
            init_precomp_item_memory(&uniformGaItemMem, NUM_LEVELS, NUM_FEATURES);
            struct encoder uniformGaEnc;
            init_encoder(&uniformGaEnc, &uniformGaItemMem);
            struct associative_memory uniformGaAssMem;
            init_assoc_mem(&uniformGaAssMem);
            output_mode = OUTPUT_NONE;
            optimize_item_memory(&uniformGaItemMem,
                                 trainingData,
                                 trainingLabels,
                                 trainingSamples,
                                 validationData,
                                 validationLabels,
                                 validationSamples);
            train_model_timeseries(trainingData, trainingLabels, trainingSamples, &uniformGaAssMem, &uniformGaEnc);
            if (validationData && validationLabels && validationSamples > 0) {
                eval_uniform_ga_val =
                    evaluate_model_timeseries_direct(&uniformGaEnc, &uniformGaAssMem, validationData, validationLabels, validationSamples);
            }
            eval_uniform_ga_test =
                evaluate_model_timeseries_direct(&uniformGaEnc, &uniformGaAssMem, testingData, testingLabels, testingSamples);
            has_uniform_ga_stage = 1;
            free_assoc_mem(&uniformGaAssMem);
            free_item_memory(&uniformGaItemMem);
#endif
            output_mode = saved_output_mode;

            int genome_length = NUM_FEATURES * (NUM_LEVELS - 1);
            if (genome_length <= 0) {
                fprintf(stderr, "Error: Invalid GA-refined quantizer genome length for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }
            uint16_t *flip_counts = (uint16_t *)calloc((size_t)genome_length, sizeof(uint16_t));
            if (!flip_counts) {
                fprintf(stderr, "Error: Failed to allocate preprocessing GA flip counts for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }

            if (output_mode >= OUTPUT_BASIC) {
                printf("Dataset %02d running preprocessing GA for quantizer refinement.\n", dataset);
            }

            if (optimize_item_memory_get_flip_counts(trainingData,
                                                     trainingLabels,
                                                     trainingSamples,
                                                     validationData,
                                                     validationLabels,
                                                     validationSamples,
                                                     flip_counts) != 0) {
                free(flip_counts);
                fprintf(stderr, "Error: Failed to run preprocessing GA for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }

            if (quantizer_refine_from_flip_counts(flip_counts, genome_length) != 0) {
                free(flip_counts);
                fprintf(stderr, "Error: Failed to refine quantizer from GA output for dataset %d.\n", dataset);
                return EXIT_FAILURE;
            }

            free(flip_counts);
        }
#elif BINNING_MODE != UNIFORM_BINNING
        {
            struct associative_memory uniformAssMem;
            init_assoc_mem(&uniformAssMem);
            int saved_output_mode = output_mode;
            output_mode = OUTPUT_NONE;
            quantizer_set_force_uniform_lookup(1);
            train_model_timeseries(trainingData, trainingLabels, trainingSamples, &uniformAssMem, &enc);
            if (validationData && validationLabels && validationSamples > 0) {
                eval_uniform_val =
                    evaluate_model_timeseries_direct(&enc, &uniformAssMem, validationData, validationLabels, validationSamples);
            }
            eval_uniform_test =
                evaluate_model_timeseries_direct(&enc, &uniformAssMem, testingData, testingLabels, testingSamples);
            quantizer_set_force_uniform_lookup(0);
            output_mode = saved_output_mode;
            has_uniform_baseline = 1;
            free_assoc_mem(&uniformAssMem);
        }
#endif

        if (quantizer_export_cuts_csv_for_dataset(dataset) != 0) {
            fprintf(stderr, "Error: Failed to export quantizer cuts for dataset %d.\n", dataset);
            return EXIT_FAILURE;
        }

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
#if BINNING_MODE != GA_REFINED_BINNING
            printf("Dataset %02d pre-optimization test accuracy: %.2f%%\n", dataset, eval_pre_test.overall_accuracy * 100.0);
#endif
        }

        char result_info[160];
#if BINNING_MODE != UNIFORM_BINNING
        if (has_uniform_baseline) {
            snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=uniform-validation", dataset);
            addResult(&eval_uniform_val, result_info);
            snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=uniform-test", dataset);
            addResult(&eval_uniform_test, result_info);

            mean_uniform_val_overall_accuracy += eval_uniform_val.overall_accuracy;
            mean_uniform_val_class_average_accuracy += eval_uniform_val.class_average_accuracy;
            mean_uniform_val_class_vector_similarity += eval_uniform_val.class_vector_similarity;
            mean_uniform_test_overall_accuracy += eval_uniform_test.overall_accuracy;
            mean_uniform_test_class_average_accuracy += eval_uniform_test.class_average_accuracy;
            mean_uniform_test_class_vector_similarity += eval_uniform_test.class_vector_similarity;
        }
#endif

#if BINNING_MODE == GA_REFINED_BINNING
#if USE_GENETIC_ITEM_MEMORY
        if (has_uniform_ga_stage) {
            snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=uniform-ga-validation", dataset);
            addResult(&eval_uniform_ga_val, result_info);
            snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=uniform-ga-test", dataset);
            addResult(&eval_uniform_ga_test, result_info);

            mean_uniform_ga_val_overall_accuracy += eval_uniform_ga_val.overall_accuracy;
            mean_uniform_ga_val_class_average_accuracy += eval_uniform_ga_val.class_average_accuracy;
            mean_uniform_ga_val_class_vector_similarity += eval_uniform_ga_val.class_vector_similarity;
            mean_uniform_ga_test_overall_accuracy += eval_uniform_ga_test.overall_accuracy;
            mean_uniform_ga_test_class_average_accuracy += eval_uniform_ga_test.class_average_accuracy;
            mean_uniform_ga_test_class_vector_similarity += eval_uniform_ga_test.class_vector_similarity;
        }
#endif
#endif
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=advanced-validation", dataset);
        addResult(&eval_pre_val, result_info);
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=advanced-test", dataset);
        addResult(&eval_pre_test, result_info);

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
#if BINNING_MODE == GA_REFINED_BINNING
            if (validationData && validationLabels && validationSamples > 0) {
                printf("Dataset %02d uniform-quantizer validation accuracy: %.2f%%\n",
                       dataset,
                       eval_uniform_val.overall_accuracy * 100.0);
            }
            printf("Dataset %02d uniform-quantizer test accuracy: %.2f%%\n",
                   dataset,
                   eval_uniform_test.overall_accuracy * 100.0);
            if (has_uniform_ga_stage) {
                if (validationData && validationLabels && validationSamples > 0) {
                    printf("Dataset %02d uniform-quantizer + CiM GA validation accuracy: %.2f%%\n",
                           dataset,
                           eval_uniform_ga_val.overall_accuracy * 100.0);
                }
                printf("Dataset %02d uniform-quantizer + CiM GA test accuracy: %.2f%%\n",
                       dataset,
                       eval_uniform_ga_test.overall_accuracy * 100.0);
            }
            if (validationData && validationLabels && validationSamples > 0) {
                printf("Dataset %02d GA-refined-quantizer validation accuracy (no item-memory GA): %.2f%%\n",
                       dataset,
                       eval_pre_val.overall_accuracy * 100.0);
            }
            printf("Dataset %02d GA-refined-quantizer test accuracy (no item-memory GA): %.2f%%\n",
                   dataset,
                   eval_pre_test.overall_accuracy * 100.0);
#if USE_GENETIC_ITEM_MEMORY
            if (validationData && validationLabels && validationSamples > 0) {
                printf("Dataset %02d final validation accuracy (with item-memory GA): %.2f%%\n",
                       dataset,
                       eval_post_val.overall_accuracy * 100.0);
            }
            printf("Dataset %02d final test accuracy (with item-memory GA): %.2f%%\n",
                   dataset,
                   eval_post_test.overall_accuracy * 100.0);
#else
            if (validationData && validationLabels && validationSamples > 0) {
                printf("Dataset %02d final validation accuracy: %.2f%%\n",
                       dataset,
                       eval_post_val.overall_accuracy * 100.0);
            }
            printf("Dataset %02d final test accuracy: %.2f%%\n",
                   dataset,
                   eval_post_test.overall_accuracy * 100.0);
#endif
#else
#if BINNING_MODE != UNIFORM_BINNING
            if (has_uniform_baseline) {
                if (validationData && validationLabels && validationSamples > 0) {
                    printf("Dataset %02d uniform-quantizer validation accuracy: %.2f%%\n",
                           dataset,
                           eval_uniform_val.overall_accuracy * 100.0);
                }
                printf("Dataset %02d uniform-quantizer test accuracy: %.2f%%\n",
                       dataset,
                       eval_uniform_test.overall_accuracy * 100.0);
            }
            if (validationData && validationLabels && validationSamples > 0) {
                printf("Dataset %02d %s validation accuracy: %.2f%%\n",
                       dataset,
                       quantizer_get_mode_name(),
                       eval_pre_val.overall_accuracy * 100.0);
            }
            printf("Dataset %02d %s test accuracy: %.2f%%\n",
                   dataset,
                   quantizer_get_mode_name(),
                   eval_pre_test.overall_accuracy * 100.0);
#else
            printf("Dataset %02d pre-opt test accuracy: %.2f%%\n", dataset, eval_pre_test.overall_accuracy * 100.0);
            printf("Dataset %02d post-opt test accuracy: %.2f%%\n", dataset, eval_post_test.overall_accuracy * 100.0);
#endif
#endif
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
#if BINNING_MODE != UNIFORM_BINNING
        printf("Overall uniform quantizer (test): %.2f%%\n", (mean_uniform_test_overall_accuracy / 4.0) * 100.0);
        printf("Overall uniform quantizer (validation): %.2f%%\n", (mean_uniform_val_overall_accuracy / 4.0) * 100.0);
#endif
#if BINNING_MODE == GA_REFINED_BINNING
#if USE_GENETIC_ITEM_MEMORY
        printf("Overall uniform quantizer + CiM GA (test): %.2f%%\n", (mean_uniform_ga_test_overall_accuracy / 4.0) * 100.0);
        printf("Overall uniform quantizer + CiM GA (validation): %.2f%%\n", (mean_uniform_ga_val_overall_accuracy / 4.0) * 100.0);
#endif
#endif
        printf("Overall pre-optimization (test): %.2f%%\n", (mean_pre_test_overall_accuracy / 4.0) * 100.0);
        printf("Overall post-optimization (test): %.2f%%\n", (mean_post_test_overall_accuracy / 4.0) * 100.0);
        printf("Overall pre-optimization (validation): %.2f%%\n", (mean_pre_val_overall_accuracy / 4.0) * 100.0);
        printf("Overall post-optimization (validation): %.2f%%\n", (mean_post_val_overall_accuracy / 4.0) * 100.0);
    }

#if BINNING_MODE != UNIFORM_BINNING
    struct timeseries_eval_result overall_uniform_val = {0};
    overall_uniform_val.overall_accuracy = mean_uniform_val_overall_accuracy / 4.0;
    overall_uniform_val.class_average_accuracy = mean_uniform_val_class_average_accuracy / 4.0;
    overall_uniform_val.class_vector_similarity = mean_uniform_val_class_vector_similarity / 4.0;
    addResult(&overall_uniform_val, "model=mine,scope=overall,phase=uniform-validation");

    struct timeseries_eval_result overall_uniform_test = {0};
    overall_uniform_test.overall_accuracy = mean_uniform_test_overall_accuracy / 4.0;
    overall_uniform_test.class_average_accuracy = mean_uniform_test_class_average_accuracy / 4.0;
    overall_uniform_test.class_vector_similarity = mean_uniform_test_class_vector_similarity / 4.0;
    addResult(&overall_uniform_test, "model=mine,scope=overall,phase=uniform-test");
#endif

#if BINNING_MODE == GA_REFINED_BINNING
#if USE_GENETIC_ITEM_MEMORY
    struct timeseries_eval_result overall_uniform_ga_val = {0};
    overall_uniform_ga_val.overall_accuracy = mean_uniform_ga_val_overall_accuracy / 4.0;
    overall_uniform_ga_val.class_average_accuracy = mean_uniform_ga_val_class_average_accuracy / 4.0;
    overall_uniform_ga_val.class_vector_similarity = mean_uniform_ga_val_class_vector_similarity / 4.0;
    addResult(&overall_uniform_ga_val, "model=mine,scope=overall,phase=uniform-ga-validation");

    struct timeseries_eval_result overall_uniform_ga_test = {0};
    overall_uniform_ga_test.overall_accuracy = mean_uniform_ga_test_overall_accuracy / 4.0;
    overall_uniform_ga_test.class_average_accuracy = mean_uniform_ga_test_class_average_accuracy / 4.0;
    overall_uniform_ga_test.class_vector_similarity = mean_uniform_ga_test_class_vector_similarity / 4.0;
    addResult(&overall_uniform_ga_test, "model=mine,scope=overall,phase=uniform-ga-test");
#endif
#endif

    struct timeseries_eval_result overall_advanced_val = {0};
    overall_advanced_val.overall_accuracy = mean_pre_val_overall_accuracy / 4.0;
    overall_advanced_val.class_average_accuracy = mean_pre_val_class_average_accuracy / 4.0;
    overall_advanced_val.class_vector_similarity = mean_pre_val_class_vector_similarity / 4.0;
    addResult(&overall_advanced_val, "model=mine,scope=overall,phase=advanced-validation");

    struct timeseries_eval_result overall_advanced_test = {0};
    overall_advanced_test.overall_accuracy = mean_pre_test_overall_accuracy / 4.0;
    overall_advanced_test.class_average_accuracy = mean_pre_test_class_average_accuracy / 4.0;
    overall_advanced_test.class_vector_similarity = mean_pre_test_class_vector_similarity / 4.0;
    addResult(&overall_advanced_test, "model=mine,scope=overall,phase=advanced-test");

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

    quantizer_clear();
    result_manager_close();
    return 0;
}
