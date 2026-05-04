#include "hdc_accelerator.h"

namespace hdc_systemc {

namespace {

bool get_bit(const hv_t &hv, int index) {
    return hv[index].to_bool();
}

void set_bit(hv_t &hv, int index, bool value) {
    hv[index] = value ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;
}

void xor_hv(const hv_t &lhs, const hv_t &rhs, hv_t &dst) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        set_bit(dst, d, get_bit(lhs, d) ^ get_bit(rhs, d));
    }
}

void permute_right(const hv_t &src, unsigned shift, hv_t &dst) {
    const unsigned effective_shift = shift % VECTOR_DIMENSION;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        const int out_index = (d + static_cast<int>(effective_shift)) % VECTOR_DIMENSION;
        dst[out_index] = src[d];
    }
}

} // namespace

HDC_Accelerator::HDC_Accelerator(sc_core::sc_module_name name)
    : sc_module(name), m_memory(0) {}

void HDC_Accelerator::bind_memory(const HDC_Memory *memory) {
    m_memory = memory;
}

void HDC_Accelerator::encode_timestamp(const level_t *quantized_sample, hv_t &encoded_timestamp) const {
    if (m_memory == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "memory not bound");
    }
    if (quantized_sample == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "quantized_sample must not be null");
    }

    const feature_counter_t threshold = NUM_FEATURES / 2;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        feature_counter_t ones = 0;
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            const hv_t &feature_hv = m_memory->read_cim(quantized_sample[feature], static_cast<unsigned>(feature));
            if (get_bit(feature_hv, d)) {
                ones = ones + 1;
            }
        }
        set_bit(encoded_timestamp, d, ones >= threshold);
    }
}

void HDC_Accelerator::encode(const level_t *quantized_window, hv_t &encoded) const {
    if (quantized_window == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "quantized_window must not be null");
    }

    encode_timestamp(quantized_window, encoded);

    hv_t encoded_timestamp;
    hv_t permuted_result;
    for (int i = 1; i < N_GRAM_SIZE; ++i) {
        encode_timestamp(&quantized_window[i * NUM_FEATURES], encoded_timestamp);
        permute_right(encoded, 1u, permuted_result);
        xor_hv(permuted_result, encoded_timestamp, encoded);
    }
}

void HDC_Accelerator::compute_hamming_distances(const hv_t &query, distance_counter_t *distances) const {
    if (m_memory == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "memory not bound");
    }
    if (distances == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "distances must not be null");
    }

    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        distance_counter_t distance = 0;
        const hv_t &class_vector = m_memory->read_assoc_class(static_cast<unsigned>(class_id));
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(query, d) != get_bit(class_vector, d)) {
                distance = distance + 1;
            }
        }
        distances[class_id] = distance;
    }
}

void HDC_Accelerator::classify(const level_t *quantized_window, distance_counter_t *distances) const {
    hv_t encoded;
    encode(quantized_window, encoded);
    compute_hamming_distances(encoded, distances);
}

} // namespace hdc_systemc
