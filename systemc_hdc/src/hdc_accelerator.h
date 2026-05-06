#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc>
#include "systemc_types.h"
#include "hdc_memory.h"

namespace hdc_systemc {

SC_MODULE(HDC_Accelerator) {
public:
    SC_CTOR(HDC_Accelerator);

    void bind_memory(HDC_Memory *memory);
    void encode(const level_t *quantized_window, hv_t &encoded) const;
    void classify(const level_t *quantized_window, distance_counter_t *distances) const;
    void reset_training_state();
    void accumulate_class_vector(int class_id, const hv_t &encoded_ngram);
    void finalize_assoc_mem();

private:
    void encode_timestamp(const level_t *quantized_sample, hv_t &encoded_timestamp) const;
    void compute_hamming_distances(const hv_t &query, distance_counter_t *distances) const;

    HDC_Memory *m_memory;
    train_counter_t m_class_bit_counts[NUM_CLASSES][VECTOR_DIMENSION];
    train_counter_t m_class_counts[NUM_CLASSES];
};

} // namespace hdc_systemc

#endif
