#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc>
#include "systemc_types.h"
#include "hdc_memory.h"

namespace hdc_systemc {

SC_MODULE(HDC_Accelerator) {
public:
    SC_CTOR(HDC_Accelerator);

    void bind_memory(const HDC_Memory *memory);
    void encode(const level_t *quantized_window, hv_t &encoded) const;
    void classify(const level_t *quantized_window, distance_counter_t *distances) const;

private:
    void encode_timestamp(const level_t *quantized_sample, hv_t &encoded_timestamp) const;
    void compute_hamming_distances(const hv_t &query, distance_counter_t *distances) const;

    const HDC_Memory *m_memory;
};

} // namespace hdc_systemc

#endif
