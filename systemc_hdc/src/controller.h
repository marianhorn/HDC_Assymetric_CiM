#ifndef SYSTEMC_HDC_CONTROLLER_H
#define SYSTEMC_HDC_CONTROLLER_H

#include <systemc>
#include "systemc_types.h"
#include "hdc_transactions.h"
#include "hdc_memory.h"
#include "hdc_accelerator.h"
#include "foot_dataset_loader.h"

namespace hdc_systemc {

SC_MODULE(Controller) {
public:
    SC_CTOR(Controller);

    void configure(int dataset_id,
                   const char *cim_path,
                   const char *quantizer_path,
                   const FootDataset *dataset);
    bool done() const;
    const EvaluationResult &test_result(int dataset_id) const;
    const MemoryStats &memory_stats(int dataset_id) const;
    const AcceleratorStats &accelerator_stats(int dataset_id) const;
    const sc_core::sc_time &dataset_sim_time(int dataset_id) const;

private:
    struct DatasetConfig {
        int dataset_id;
        const char *cim_path;
        const char *quantizer_path;
        const FootDataset *dataset;
        bool configured;
    };

    void main_thread();
    void load_cim(const char *path);
    void load_quantizer(const char *path);
    void train_dataset(const double *raw_data, const int *labels, int num_samples);
    EvaluationResult evaluate_dataset(const double *raw_data, const int *labels, int num_samples);
    level_t quantize_value(unsigned feature, double value) const;
    void quantize_sample(const double *raw_sample, level_t *quantized_sample) const;
    int get_ngram_real_label(const int *labels, int size) const;
    void copy_quantized_sample(const level_t *levels, QuantizedSample &sample) const;
    void send_command(const AccelCommand &command);
    AccelResponse send_inference_command(const AccelCommand &command);

    DatasetConfig m_dataset_configs[NUM_DATASETS];
    EvaluationResult m_test_results[NUM_DATASETS];
    MemoryStats m_memory_stats[NUM_DATASETS];
    AcceleratorStats m_accelerator_stats[NUM_DATASETS];
    sc_core::sc_time m_dataset_sim_times[NUM_DATASETS];
    bool m_done;
    HDC_Memory m_memory;
    sc_core::sc_fifo<AccelCommand> m_cmd_fifo;
    sc_core::sc_fifo<AccelResponse> m_rsp_fifo;
    HDC_Accelerator m_accelerator;
};

} // namespace hdc_systemc

#endif
