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

    for (int dataset = 0; dataset < 4; ++dataset) {
        char cim_path[128];
        char quantizer_path[128];

        std::snprintf(cim_path, sizeof(cim_path), "import/cim_dataset%02d.txt", dataset);
        std::snprintf(quantizer_path, sizeof(quantizer_path), "import/quantizer_dataset%02d.txt", dataset);

        Controller controller(sc_core::sc_gen_unique_name("controller"));
        controller.load_cim_file(cim_path);
        controller.load_quantizer_file(quantizer_path);

        const FootDataset real_dataset = load_foot_dataset_by_id(dataset);
        controller.train_dataset(real_dataset.training.raw_data(),
                                 real_dataset.training.raw_labels(),
                                 real_dataset.training.samples);

        const EvaluationResult test_result =
            controller.evaluate_dataset(real_dataset.testing.raw_data(),
                                        real_dataset.testing.raw_labels(),
                                        real_dataset.testing.samples);

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
