#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "controller.h"
#include "foot_dataset_loader.h"

using namespace hdc_systemc;

namespace {

void write_eval_result(std::ostream &out, int dataset, const EvaluationResult &result) {
    out << "dataset=" << dataset << '\n';
    out << std::fixed << std::setprecision(12);
    out << "overall_accuracy=" << result.overall_accuracy << '\n';
    out << "non_transition_accuracy=" << result.non_transition_accuracy << '\n';
    out << "correct=" << result.correct << '\n';
    out << "not_correct=" << result.not_correct << '\n';
    out << "transition_error=" << result.transition_error << '\n';
    out << "total=" << result.total << '\n';
    out << "confusion_matrix=" << '\n';
    for (int actual = 0; actual < NUM_CLASSES; ++actual) {
        for (int predicted = 0; predicted < NUM_CLASSES; ++predicted) {
            if (predicted > 0) {
                out << ' ';
            }
            out << result.confusion_matrix[actual][predicted];
        }
        out << '\n';
    }
    out << "end_dataset=" << dataset << "\n\n";
}

} // namespace

int sc_main(int argc, char *argv[]) {
    const char *output_path = "golden_regression_current.txt";
    if (argc > 1) {
        output_path = argv[1];
    }

    std::ofstream output(output_path);
    if (!output.is_open()) {
        SC_REPORT_FATAL("golden_regression", "failed to open regression output file");
    }

    std::ostringstream regression_text;
    regression_text << "# SystemC HDC golden functional regression\n";
    regression_text << "# Deterministic prediction output for datasets 0..3\n";
    regression_text << "# Fields must stay unchanged after architecture refactors.\n\n";

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
        SC_REPORT_FATAL("golden_regression", "controller did not finish");
    }

    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        const EvaluationResult &result = controller.test_result(dataset);
        write_eval_result(regression_text, dataset, result);
    }

    output << regression_text.str();
    if (!output.good()) {
        SC_REPORT_FATAL("golden_regression", "failed while writing regression output file");
    }

    std::cout << regression_text.str();
    std::cout << "Wrote golden regression output to " << output_path << std::endl;
    return EXIT_SUCCESS;
}
