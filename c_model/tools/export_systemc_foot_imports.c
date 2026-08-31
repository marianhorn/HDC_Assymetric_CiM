#include <stdio.h>
#include <stdlib.h>

#include "../include/config.h"
#include "../include/data_reader.h"
#include "../include/item_mem.h"
#include "../include/quantizer.h"

#ifndef NUM_DATASETS
#define NUM_DATASETS 4
#endif

int output_mode = 0;

static int export_dataset(int dataset, const char *out_dir) {
    double **training_data = NULL;
    double **validation_data = NULL;
    double **testing_data = NULL;
    int *training_labels = NULL;
    int *validation_labels = NULL;
    int *testing_labels = NULL;
    int training_samples = 0;
    int validation_samples = 0;
    int testing_samples = 0;
    char cim_path[512];
    char quantizer_path[512];

    quantizer_clear();

    struct item_memory item_mem;
    init_precomp_item_memory(&item_mem, NUM_LEVELS, NUM_FEATURES);

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
        fprintf(stderr, "failed to fit quantizer for dataset %d\n", dataset);
        return -1;
    }

    snprintf(cim_path, sizeof(cim_path), "%s/cim_dataset%02d.txt", out_dir, dataset);
    snprintf(quantizer_path, sizeof(quantizer_path), "%s/quantizer_dataset%02d.txt", out_dir, dataset);

    store_precomp_item_mem_to_systemc_text(&item_mem, cim_path, NUM_LEVELS, NUM_FEATURES);
    if (quantizer_export_systemc_text(quantizer_path) != 0) {
        fprintf(stderr, "failed to export quantizer for dataset %d\n", dataset);
        return -1;
    }

    free_item_memory(&item_mem);
    freeData(training_data, training_samples);
    if (validation_data != NULL) {
        freeData(validation_data, validation_samples);
    }
    freeData(testing_data, testing_samples);
    freeCSVLabels(training_labels);
    if (validation_labels != NULL) {
        freeCSVLabels(validation_labels);
    }
    freeCSVLabels(testing_labels);
    return 0;
}

int main(int argc, char **argv) {
    const char *out_dir = (argc > 1) ? argv[1] : "systemc_accelerator/partially_parallelized/import";

    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        if (export_dataset(dataset, out_dir) != 0) {
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
