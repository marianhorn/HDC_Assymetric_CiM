// Temporary SystemC parity main.
// Loads the SystemC-exported CiM and quantizer text files, trains on the C data,
// and prints only the test metrics needed for comparison.

#include <stdio.h>
#include <stdlib.h>

#include "configFoot.h"
#include "dataReaderFootEMG.h"
#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/encoder.h"
#include "../hdc_infrastructure/evaluator.h"
#include "../hdc_infrastructure/item_mem.h"
#include "../hdc_infrastructure/quantizer.h"
#include "../hdc_infrastructure/trainer.h"

int output_mode = OUTPUT_MODE;

static double accuracy_excluding_transitions(const struct timeseries_eval_result *result) {
    size_t non_transition_total = result->correct + result->not_correct;
    if (non_transition_total == 0) {
        return 0.0;
    }
    return (double)result->correct / (double)non_transition_total;
}

int main(void) {
    double mean_test_accuracy = 0.0;
    double mean_test_accuracy_no_transitions = 0.0;
    int processed_datasets = 0;

    for (int dataset = 0; dataset < 4; dataset++) {
        char cim_path[128];
        char quantizer_path[128];
        double **training_data = NULL;
        double **testing_data = NULL;
        int *training_labels = NULL;
        int *testing_labels = NULL;
        int training_samples = 0;
        int testing_samples = 0;

        snprintf(cim_path, sizeof(cim_path), "systemc_hdc/import/cim_dataset%02d.txt", dataset);
        snprintf(quantizer_path, sizeof(quantizer_path), "systemc_hdc/import/quantizer_dataset%02d.txt", dataset);

        struct item_memory item_mem;
        load_precomp_item_mem_from_systemc_text(&item_mem, cim_path, NUM_LEVELS, NUM_FEATURES);

        struct encoder enc;
        init_encoder(&enc, &item_mem);

        if (quantizer_import_systemc_text(quantizer_path) != 0) {
            fprintf(stderr, "Error: failed to import SystemC quantizer for dataset %d.\n", dataset);
            free_item_memory(&item_mem);
            return EXIT_FAILURE;
        }

        struct associative_memory assoc_mem;
        init_assoc_mem(&assoc_mem);

        getData(dataset,
                &training_data,
                &testing_data,
                &training_labels,
                &testing_labels,
                &training_samples,
                &testing_samples);

        train_model_timeseries(training_data, training_labels, training_samples, &assoc_mem, &enc);
        struct timeseries_eval_result test_result =
            evaluate_model_timeseries_direct(&enc, &assoc_mem, testing_data, testing_labels, testing_samples);
        double test_accuracy_no_transitions = accuracy_excluding_transitions(&test_result);

        if (dataset > 0) {
            printf("\n");
        }
        printf("Dataset %d\n", dataset);
        printf("Test accuracy: %.4f%%\n", test_result.overall_accuracy * 100.0);
        printf("Test accuracy excl. transitions: %.4f%%\n", test_accuracy_no_transitions * 100.0);
        printf("Test counts: correct=%zu, wrong=%zu, transitions=%zu, total=%zu\n",
               test_result.correct,
               test_result.not_correct,
               test_result.transition_error,
               test_result.total);

        mean_test_accuracy += test_result.overall_accuracy;
        mean_test_accuracy_no_transitions += test_accuracy_no_transitions;
        processed_datasets++;

        free_assoc_mem(&assoc_mem);
        free_item_memory(&item_mem);
        freeData(training_data, (size_t)training_samples);
        freeData(testing_data, (size_t)testing_samples);
        freeCSVLabels(training_labels);
        freeCSVLabels(testing_labels);
        quantizer_clear();
    }

    if (processed_datasets > 0) {
        printf("\nOverall\n");
        printf("Mean test accuracy: %.4f%%\n",
               100.0 * mean_test_accuracy / (double)processed_datasets);
        printf("Mean test accuracy excl. transitions: %.4f%%\n",
               100.0 * mean_test_accuracy_no_transitions / (double)processed_datasets);
        printf("Mean test counts: correct=0, wrong=0, transitions=0, total=0\n");
    }

    return EXIT_SUCCESS;
}
