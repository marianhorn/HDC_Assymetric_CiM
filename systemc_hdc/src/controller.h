#ifndef SYSTEMC_HDC_CONTROLLER_H
#define SYSTEMC_HDC_CONTROLLER_H

#include <systemc>
#include "systemc_types.h"
#include "hdc_memory.h"
#include "hdc_accelerator.h"

namespace hdc_systemc {

SC_MODULE(Controller) {
public:
    SC_CTOR(Controller);

    void load_cim(const hv_t *flat_cim);
    void load_cim_file(const char *path);
    void load_quantizer_boundaries(const double *flat_boundaries);
    void load_quantizer_file(const char *path);
    void reset_assoc_mem();
    void quantize_window(const double *raw_window, level_t *quantized_window) const;
    void encode_window(const double *raw_window, hv_t &encoded) const;
    void classify_window(const double *raw_window, distance_counter_t *distances) const;
    int predict_window(const double *raw_window) const;
    void train_dataset(const double *raw_data, const int *labels, int num_samples);
    EvaluationResult evaluate_dataset(const double *raw_data, const int *labels, int num_samples) const;
    const hv_t &get_class_vector(unsigned class_id) const;
    const HDC_Memory &get_memory() const;

private:
    bool is_window_stable(const int *labels) const;
    void reset_training_buffers();
    void accumulate_class_vector(int class_id, const hv_t &encoded_ngram);
    void finalize_assoc_mem();
    int mode_smallest_tie(const int *labels, int size) const;

    HDC_Memory m_memory;
    HDC_Accelerator m_accelerator;
    train_counter_t m_class_bit_counts[NUM_CLASSES][VECTOR_DIMENSION];
    train_counter_t m_class_counts[NUM_CLASSES];
};

} // namespace hdc_systemc

#endif
