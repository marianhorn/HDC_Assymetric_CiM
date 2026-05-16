#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "controller.h"
#include "foot_dataset_loader.h"

using namespace hdc_systemc;

namespace {

void print_eval_result(const char *name, const EvaluationResult &result) {
    std::cout << name << " accuracy: " << (result.overall_accuracy * 100.0) << "%" << std::endl;
    std::cout << name << " accuracy excl. transitions: "
              << (result.non_transition_accuracy * 100.0) << "%" << std::endl;
    std::cout << name << " counts: correct=" << result.correct
              << ", wrong=" << result.not_correct
              << ", transitions=" << result.transition_error
              << ", total=" << result.total << std::endl;
}

} // namespace

int sc_main(int, char *[]) {
    double mean_test_accuracy = 0.0;
    double mean_test_accuracy_no_transitions = 0.0;
    int processed_datasets = 0;
    FootDataset datasets[NUM_DATASETS];
    char cim_paths[NUM_DATASETS][128];
    char quantizer_paths[NUM_DATASETS][128];
    Controller controller("controller");

    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        std::snprintf(cim_paths[dataset], sizeof(cim_paths[dataset]), "import/cim_dataset%02d.txt", dataset);
        std::snprintf(quantizer_paths[dataset], sizeof(quantizer_paths[dataset]), "import/quantizer_dataset%02d.txt", dataset);
        datasets[dataset] = load_foot_dataset_by_id(dataset);
        controller.configure(dataset, cim_paths[dataset], quantizer_paths[dataset], &datasets[dataset]);
    }

    sc_core::sc_start();
    if (!controller.done()) {
        SC_REPORT_FATAL("tb_systemc", "controller did not finish");
    }

    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        const EvaluationResult &test_result = controller.test_result(dataset);

        std::cout << "\nDataset " << dataset << std::endl;
        print_eval_result("Test", test_result);

        mean_test_accuracy += test_result.overall_accuracy;
        mean_test_accuracy_no_transitions += test_result.non_transition_accuracy;
        ++processed_datasets;
    }

    if (processed_datasets > 0) {
        EvaluationResult overall_result;
        overall_result.correct = 0;
        overall_result.not_correct = 0;
        overall_result.transition_error = 0;
        overall_result.total = 0;
        overall_result.overall_accuracy = mean_test_accuracy / static_cast<double>(processed_datasets);
        overall_result.non_transition_accuracy =
            mean_test_accuracy_no_transitions / static_cast<double>(processed_datasets);
        for (int i = 0; i < NUM_CLASSES; ++i) {
            for (int j = 0; j < NUM_CLASSES; ++j) {
                overall_result.confusion_matrix[i][j] = 0;
            }
        }

        std::cout << "\nOverall" << std::endl;
        print_eval_result("Mean test", overall_result);
    }

    return EXIT_SUCCESS;
}
