#ifndef SYSTEMC_HDC_HDC_MEMORY_H
#define SYSTEMC_HDC_HDC_MEMORY_H

#include <systemc>
#include "systemc_types.h"

namespace hdc_systemc {

SC_MODULE(HDC_Memory) {
public:
    SC_CTOR(HDC_Memory);

    void clear_all();
    void load_cim_flat(const hv_t *flat_cim);
    void load_quantizer_boundaries(const double *flat_boundaries);
    const hv_t &read_cim(level_t level, unsigned feature) const;
    level_t quantize_value(unsigned feature, double value) const;
    void quantize_sample(const double *raw_sample, level_t *quantized_sample) const;
    void clear_assoc_mem();
    void write_assoc_class(unsigned class_id, const hv_t &class_hv);
    const hv_t &read_assoc_class(unsigned class_id) const;

private:
    hv_t m_cim[NUM_LEVELS * NUM_FEATURES];
    double m_quantizer_boundaries[NUM_FEATURES * ((NUM_LEVELS > 1) ? (NUM_LEVELS - 1) : 1)];
    hv_t m_assoc_mem[NUM_CLASSES];
    bool m_cim_loaded;
    bool m_quantizer_loaded;
};

} // namespace hdc_systemc

#endif
