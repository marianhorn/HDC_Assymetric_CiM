// SIMULATION / SOFTWARE ONLY: This file is not part of the HLS synthesis target.
// It models the RISC-V/software side that loads files, quantizes raw EMG, and drives the accelerator.
#ifndef SYSTEMC_HDC_CONTROLLER_H
#define SYSTEMC_HDC_CONTROLLER_H

#include <systemc>
#include "systemc_types.h"
#include "evaluation_result.h"
#include "hdc_transactions.h"
#include "hdc_memory.h"
#include "hdc_accelerator.h"
#include "foot_dataset_loader.h"

namespace hdc_systemc {

SC_MODULE(Controller) {
public:
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    SC_CTOR(Controller);

    void configure(int dataset_id,
                   const char *cim_path,
                   const char *quantizer_path,
                   const FootDataset *dataset);
    bool done() const;
    const EvaluationResult &test_result(int dataset_id) const;

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
    AccelResponse read_response();

    DatasetConfig m_dataset_configs[NUM_DATASETS];
    EvaluationResult m_test_results[NUM_DATASETS];
    bool m_done;
    HDC_Memory m_memory;
    sc_core::sc_signal<bool> m_cmd_valid;
    sc_core::sc_signal<bool> m_cmd_ready;
    sc_core::sc_signal<command_kind_t> m_cmd_kind;
    sc_core::sc_signal<class_t> m_cmd_class_id;
    sc_core::sc_signal<level_t> m_cmd_sample_levels[NUM_FEATURES];
    sc_core::sc_signal<bool> m_rsp_valid;
    sc_core::sc_signal<bool> m_rsp_ready;
    sc_core::sc_signal<bool> m_rsp_valid_prediction;
    sc_core::sc_signal<distance_counter_t> m_rsp_distances[NUM_CLASSES];
    HDC_Accelerator m_accelerator;
};

} // namespace hdc_systemc

#endif
