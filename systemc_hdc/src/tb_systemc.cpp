#include <cstdlib>
#include <cstdio>
#include <cstdint>
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

void print_memory_stats(const MemoryStats &stats) {
    const std::uint64_t total_read_accesses =
        stats.quantizer_row_reads + stats.cim_reads + stats.assoc_reads;
    const std::uint64_t total_read_bytes =
        stats.quantizer_row_read_bytes + stats.cim_read_bytes + stats.assoc_read_bytes;
    const std::uint64_t total_write_accesses = stats.assoc_writes;
    const std::uint64_t total_write_bytes = stats.assoc_write_bytes;

    std::cout << "Memory stats:" << std::endl;
    std::cout << "  quantizer row reads: " << stats.quantizer_row_reads
              << ", bytes=" << stats.quantizer_row_read_bytes << std::endl;
    std::cout << "  CiM reads: " << stats.cim_reads
              << ", bytes=" << stats.cim_read_bytes << std::endl;
    std::cout << "  associative reads: " << stats.assoc_reads
              << ", bytes=" << stats.assoc_read_bytes << std::endl;
    std::cout << "  associative writes: " << stats.assoc_writes
              << ", bytes=" << stats.assoc_write_bytes << std::endl;
    std::cout << "  total reads: " << total_read_accesses
              << ", bytes=" << total_read_bytes << std::endl;
    std::cout << "  total writes: " << total_write_accesses
              << ", bytes=" << total_write_bytes << std::endl;
}

void print_accelerator_stats(const AcceleratorStats &stats) {
    std::cout << "Accelerator stats:" << std::endl;
    std::cout << "  commands: " << stats.command_count << std::endl;
    std::cout << "  train samples: " << stats.train_samples << std::endl;
    std::cout << "  inference samples: " << stats.infer_samples << std::endl;
    std::cout << "  encoded samples: " << stats.encoded_samples << std::endl;
    std::cout << "  ngram samples: " << stats.ngram_samples << std::endl;
    std::cout << "  valid ngrams: " << stats.valid_ngrams << std::endl;
    std::cout << "  bundled ngrams: " << stats.bundled_ngrams << std::endl;
    std::cout << "  bundle flushes: " << stats.bundle_flushes << std::endl;
    std::cout << "  distance requests: " << stats.distance_requests << std::endl;
    std::cout << "  valid distance requests: " << stats.valid_distance_requests << std::endl;
}

} // namespace

int sc_main(int, char *[]) {
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
        std::cout << "Simulation time: " << controller.dataset_sim_time(dataset) << std::endl;
        print_memory_stats(controller.memory_stats(dataset));
        print_accelerator_stats(controller.accelerator_stats(dataset));
    }

    return EXIT_SUCCESS;
}
