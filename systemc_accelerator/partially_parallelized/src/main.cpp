// SIMULATION / SOFTWARE ONLY: This file is not part of the HLS synthesis target.
// It is the SystemC testbench entry point and prints functional results.
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
    FootDataset datasets[NUM_DATASETS];
    char cim_paths[NUM_DATASETS][128];
    char quantizer_paths[NUM_DATASETS][128];
    sc_core::sc_clock clk("clk", sc_core::sc_time(10, sc_core::SC_NS));
    sc_core::sc_signal<bool> rst("rst");
    Controller controller("controller");
    controller.clk(clk);
    controller.rst(rst);

    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        std::snprintf(cim_paths[dataset], sizeof(cim_paths[dataset]), "import/cim_dataset%02d.txt", dataset);
        std::snprintf(quantizer_paths[dataset], sizeof(quantizer_paths[dataset]), "import/quantizer_dataset%02d.txt", dataset);
        datasets[dataset] = load_foot_dataset_by_id(dataset);
        controller.configure(dataset, cim_paths[dataset], quantizer_paths[dataset], &datasets[dataset]);
    }

    rst.write(true);
    sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_NS));
    rst.write(false);
    sc_core::sc_start();
    if (!controller.done()) {
        SC_REPORT_FATAL("main", "controller did not finish");
    }

    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        const EvaluationResult &test_result = controller.test_result(dataset);

        std::cout << "\nDataset " << dataset << std::endl;
        print_eval_result("Test", test_result);
    }

    return EXIT_SUCCESS;
}
