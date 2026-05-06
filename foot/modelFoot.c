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
#include "../hdc_infrastructure/quantizer.h"
#include "../hdc_infrastructure/vector.h"
#include "../hdc_infrastructure/trainer.h"

int output_mode = OUTPUT_MODE;

int main(void) {
    result_manager_init();

    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n");
    }

    double mean_pre_val_accuracy = 0.0;
    double mean_pre_test_accuracy = 0.0;
    double mean_post_val_accuracy = 0.0;
    double mean_post_test_accuracy = 0.0;
    int processed_datasets = 0;

    for (int dataset = 0; dataset < 4; dataset++) {
        double **trainingData = NULL;
        double **validationData = NULL;
        double **testingData = NULL;
        int *trainingLabels = NULL;
        int *validationLabels = NULL;
        int *testingLabels = NULL;
        int trainingSamples = 0;
        int validationSamples = 0;
        int testingSamples = 0;
        struct timeseries_eval_result eval_pre_val = {0};
        struct timeseries_eval_result eval_pre_test = {0};
        struct timeseries_eval_result eval_post_val = {0};
        struct timeseries_eval_result eval_post_test = {0};
        char result_info[160];
        
        char quantizer_export_path[256];
        char cim_export_path[256];

        quantizer_clear();

        if (output_mode >= OUTPUT_BASIC) {
            printf("\n\nModel for dataset #%d\n", dataset);
        }

#if PRECOMPUTED_ITEM_MEMORY
        struct item_memory itemMem;
        init_precomp_item_memory(&itemMem, NUM_LEVELS, NUM_FEATURES);

        /* snprintf(cim_export_path,
                  sizeof(cim_export_path),
                  "systemc_hdc/import/cim_dataset%02d.txt",
                  dataset);
         store_precomp_item_mem_to_systemc_text(&itemMem,
                                                cim_export_path,
                                                NUM_LEVELS,
                                                NUM_FEATURES); */

        struct encoder enc;
        init_encoder(&enc, &itemMem);
#else
        struct item_memory electrodes;
        struct item_memory intensityLevels;
        init_item_memory(&electrodes, NUM_FEATURES);
        init_continuous_item_memory(&intensityLevels, NUM_LEVELS);

        struct encoder enc;
        init_encoder(&enc, &electrodes, &intensityLevels);
#endif

        struct associative_memory assMem;
        init_assoc_mem(&assMem);

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
                          VALIDATION_RATIO);

        if (quantizer_fit_from_training(trainingData,
                                        trainingLabels,
                                        trainingSamples,
                                        NUM_FEATURES,
                                        NUM_LEVELS) != 0) {
            fprintf(stderr, "Error: Failed to initialize quantizer for dataset %d.\n", dataset);
            return EXIT_FAILURE;
        }
        
        /* snprintf(quantizer_export_path,
                  sizeof(quantizer_export_path),
                  "systemc_hdc/import/quantizer_dataset%02d.txt",
                  dataset);
         if (quantizer_export_systemc_text(quantizer_export_path) != 0) {
             fprintf(stderr, "Error: Failed to export quantizer for dataset %d.\n", dataset);
             return EXIT_FAILURE;
         } */

        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);

        if (validationData && validationLabels && validationSamples > 0) {
            eval_pre_val = evaluate_model_timeseries_direct(&enc, &assMem, validationData, validationLabels, validationSamples);
        }
        eval_pre_test = evaluate_model_timeseries_direct(&enc, &assMem, testingData, testingLabels, testingSamples);

        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=preopt-validation", dataset);
        addResult(&eval_pre_val, result_info);
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=preopt-test", dataset);
        addResult(&eval_pre_test, result_info);

        mean_pre_val_accuracy += eval_pre_val.overall_accuracy;
        mean_pre_test_accuracy += eval_pre_test.overall_accuracy;

        if (output_mode >= OUTPUT_BASIC) {
            printf("  Pre-Optimization\n");
            printf("    validation accuracy: ");
            if (validationData && validationLabels && validationSamples > 0) {
                printf("%.2f%%\n", eval_pre_val.overall_accuracy * 100.0);
            } else {
                printf("n/a\n");
            }
            printf("    test accuracy: %.2f%%\n", eval_pre_test.overall_accuracy * 100.0);
        }

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

        if (validationData && validationLabels && validationSamples > 0) {
            eval_post_val = evaluate_model_timeseries_direct(&enc, &assMem, validationData, validationLabels, validationSamples);
        }
        eval_post_test = evaluate_model_timeseries_direct(&enc, &assMem, testingData, testingLabels, testingSamples);

        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=postopt-validation", dataset);
        addResult(&eval_post_val, result_info);
        snprintf(result_info, sizeof(result_info), "model=mine,scope=dataset,dataset=%d,phase=postopt-test", dataset);
        addResult(&eval_post_test, result_info);

        mean_post_val_accuracy += eval_post_val.overall_accuracy;
        mean_post_test_accuracy += eval_post_test.overall_accuracy;

        if (output_mode >= OUTPUT_BASIC) {
            printf("  Post-Optimization\n");
            printf("    validation accuracy: ");
            if (validationData && validationLabels && validationSamples > 0) {
                printf("%.2f%%\n", eval_post_val.overall_accuracy * 100.0);
            } else {
                printf("n/a\n");
            }
            printf("    test accuracy: %.2f%%\n", eval_post_test.overall_accuracy * 100.0);
        }
#endif

        free_assoc_mem(&assMem);
#if PRECOMPUTED_ITEM_MEMORY
        free_item_memory(&itemMem);
#else
        free_item_memory(&electrodes);
        free_item_memory(&intensityLevels);
#endif
        freeData(trainingData, trainingSamples);
        if (validationData != NULL) {
            freeData(validationData, validationSamples);
        }
        freeData(testingData, testingSamples);
        freeCSVLabels(trainingLabels);
        if (validationLabels != NULL) {
            freeCSVLabels(validationLabels);
        }
        freeCSVLabels(testingLabels);

        processed_datasets++;
    }

    if (processed_datasets > 0) {
        char result_info[96];
        struct timeseries_eval_result overall_pre_val = {0};
        struct timeseries_eval_result overall_pre_test = {0};

        overall_pre_val.overall_accuracy = mean_pre_val_accuracy / (double)processed_datasets;
        snprintf(result_info, sizeof(result_info), "model=mine,scope=overall,phase=preopt-validation");
        addResult(&overall_pre_val, result_info);

        overall_pre_test.overall_accuracy = mean_pre_test_accuracy / (double)processed_datasets;
        snprintf(result_info, sizeof(result_info), "model=mine,scope=overall,phase=preopt-test");
        addResult(&overall_pre_test, result_info);

#if USE_GENETIC_ITEM_MEMORY
        struct timeseries_eval_result overall_post_val = {0};
        struct timeseries_eval_result overall_post_test = {0};

        overall_post_val.overall_accuracy = mean_post_val_accuracy / (double)processed_datasets;
        snprintf(result_info, sizeof(result_info), "model=mine,scope=overall,phase=postopt-validation");
        addResult(&overall_post_val, result_info);

        overall_post_test.overall_accuracy = mean_post_test_accuracy / (double)processed_datasets;
        snprintf(result_info, sizeof(result_info), "model=mine,scope=overall,phase=postopt-test");
        addResult(&overall_post_test, result_info);
#endif

        if (output_mode >= OUTPUT_BASIC) {
            printf("\nOverall\n");
            printf("  Pre-Optimization\n");
            printf("    validation accuracy: %.2f%%\n", 100.0 * mean_pre_val_accuracy / (double)processed_datasets);
            printf("    test accuracy: %.2f%%\n", 100.0 * mean_pre_test_accuracy / (double)processed_datasets);
#if USE_GENETIC_ITEM_MEMORY
            printf("  Post-Optimization\n");
            printf("    validation accuracy: %.2f%%\n", 100.0 * mean_post_val_accuracy / (double)processed_datasets);
            printf("    test accuracy: %.2f%%\n", 100.0 * mean_post_test_accuracy / (double)processed_datasets);
#endif
        }
    }

    result_manager_close();
    return 0;
}
