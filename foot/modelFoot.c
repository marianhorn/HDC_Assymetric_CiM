//This serves as main file.

#include <stdio.h>
#include <stdlib.h>
#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/item_mem.h"
#include "../hdc_infrastructure/encoder.h"
#include "../hdc_infrastructure/operations.h"
#include "dataReaderFootEMG.h"
#include "configFoot.h"
#include "../hdc_infrastructure/evaluator.h"
#include "../hdc_infrastructure/quantizer.h"
#include "../hdc_infrastructure/vector.h"
#include "../hdc_infrastructure/trainer.h"

int output_mode = OUTPUT_MODE;

int main(void) {
    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n");
    }

    double mean_test_accuracy = 0.0;
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
        struct timeseries_eval_result eval_test = {0};

        quantizer_clear();

        if (output_mode >= OUTPUT_BASIC) {
            printf("\n\nModel for dataset #%d\n", dataset);
        }

        struct item_memory itemMem;
        {
            char cim_import_path[128];
            snprintf(cim_import_path,
                     sizeof(cim_import_path),
                     "%s/cim_dataset%02d.csv",
                     CIM_EXPORT_DIR,
                     dataset);
            load_precomp_item_mem_from_csv(&itemMem,
                                           cim_import_path,
                                           NUM_LEVELS,
                                           NUM_FEATURES);
        }

        struct encoder enc;
        init_encoder(&enc, &itemMem);

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

        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);

        eval_test = evaluate_model_timeseries_direct(&enc, &assMem, testingData, testingLabels, testingSamples);

        mean_test_accuracy += eval_test.overall_accuracy;

        if (output_mode >= OUTPUT_BASIC) {
            printf("  test accuracy: %.2f%%\n", eval_test.overall_accuracy * 100.0);
        }

        free_assoc_mem(&assMem);
        free_item_memory(&itemMem);
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
        if (output_mode >= OUTPUT_BASIC) {
            printf("\nOverall\n");
            printf("  test accuracy: %.2f%%\n", 100.0 * mean_test_accuracy / (double)processed_datasets);
        }
    }

    return 0;
}
