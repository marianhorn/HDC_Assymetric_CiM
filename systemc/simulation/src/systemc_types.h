#ifndef SYSTEMC_HDC_SYSTEMC_TYPES_H
#define SYSTEMC_HDC_SYSTEMC_TYPES_H

#include <cstdint>
#include <systemc>
#include "config_systemc.h"

namespace hdc_systemc {

constexpr unsigned required_bits(unsigned max_value_exclusive) {
    unsigned bits = 0;
    unsigned limit = 1;
    while (limit < max_value_exclusive) {
        limit <<= 1;
        ++bits;
    }
    return bits == 0 ? 1u : bits;
}

static constexpr unsigned LEVEL_BITS = required_bits(NUM_LEVELS);
static constexpr unsigned CLASS_BITS = required_bits(NUM_CLASSES);
static constexpr unsigned FEATURE_COUNT_BITS = required_bits(NUM_FEATURES + 1);
static constexpr unsigned DISTANCE_BITS = required_bits(VECTOR_DIMENSION + 1);
static constexpr unsigned TRAIN_COUNT_BITS = 32;

typedef sc_dt::sc_uint<LEVEL_BITS> level_t;
typedef sc_dt::sc_uint<CLASS_BITS> class_t;
typedef sc_dt::sc_uint<FEATURE_COUNT_BITS> feature_counter_t;
typedef sc_dt::sc_int<FEATURE_COUNT_BITS + 1> feature_score_t;
typedef sc_dt::sc_uint<DISTANCE_BITS> distance_counter_t;
typedef sc_dt::sc_uint<TRAIN_COUNT_BITS> train_counter_t;
typedef sc_dt::sc_int<TRAIN_COUNT_BITS + 1> train_score_t;

static constexpr unsigned HV_WORD_BITS = 64;
static_assert((VECTOR_DIMENSION % HV_WORD_BITS) == 0,
              "VECTOR_DIMENSION must be divisible by HV_WORD_BITS for simulation");
static constexpr unsigned HV_WORDS = VECTOR_DIMENSION / HV_WORD_BITS;

typedef sc_dt::sc_uint<HV_WORD_BITS> hv_word_t;

struct hv_t {
    hv_word_t words[HV_WORDS];
};

inline bool hv_get_bit(const hv_t &hv, unsigned bit_index) {
    const unsigned word = bit_index >> 6;
    const unsigned shift = bit_index & 63u;
    return ((hv.words[word] >> shift) & hv_word_t(1)) != 0;
}

inline void hv_set_bit(hv_t &hv, unsigned bit_index, bool value) {
    const unsigned word = bit_index >> 6;
    const unsigned shift = bit_index & 63u;
    const hv_word_t mask = hv_word_t(1) << shift;
    if (value) {
        hv.words[word] = hv.words[word] | mask;
    } else {
        hv.words[word] = hv.words[word] & ~mask;
    }
}

inline void hv_clear(hv_t &hv) {
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        hv.words[word] = 0;
    }
}

inline void hv_copy(hv_t &dst, const hv_t &src) {
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        dst.words[word] = src.words[word];
    }
}

struct EvaluationResult {
    unsigned correct;
    unsigned not_correct;
    unsigned transition_error;
    unsigned total;
    double overall_accuracy;
    double non_transition_accuracy;
    unsigned confusion_matrix[NUM_CLASSES][NUM_CLASSES];
};

struct MemoryStats {
    std::uint64_t quantizer_row_reads;
    std::uint64_t quantizer_row_read_bytes;
    std::uint64_t cim_reads;
    std::uint64_t cim_read_bytes;
    std::uint64_t assoc_reads;
    std::uint64_t assoc_read_bytes;
    std::uint64_t assoc_writes;
    std::uint64_t assoc_write_bytes;
};

struct AcceleratorStats {
    std::uint64_t command_count;
    std::uint64_t train_samples;
    std::uint64_t infer_samples;
    std::uint64_t encoded_samples;
    std::uint64_t ngram_samples;
    std::uint64_t valid_ngrams;
    std::uint64_t bundled_ngrams;
    std::uint64_t bundle_flushes;
    std::uint64_t distance_requests;
    std::uint64_t valid_distance_requests;
};

} // namespace hdc_systemc

#endif
