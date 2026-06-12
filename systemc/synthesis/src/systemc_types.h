// SHARED TYPE DEFINITIONS: Used by both the accelerator model and simulation harness.
// Keep this header limited to fixed-width datapath and boundary types.
#ifndef SYSTEMC_HDC_SYSTEMC_TYPES_H
#define SYSTEMC_HDC_SYSTEMC_TYPES_H

#ifndef STRATUS_HLS
#include <sstream>
#endif
#include <systemc.h>
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
static constexpr unsigned COMMAND_KIND_BITS = required_bits(5);
static constexpr unsigned FEATURE_COUNT_BITS = required_bits(NUM_FEATURES + 1);
static constexpr unsigned DISTANCE_BITS = required_bits(VECTOR_DIMENSION + 1);
static constexpr unsigned TRAIN_COUNT_BITS = 32;

typedef sc_dt::sc_uint<LEVEL_BITS> level_t;
typedef sc_dt::sc_uint<CLASS_BITS> class_t;
typedef sc_dt::sc_uint<COMMAND_KIND_BITS> command_kind_t;
typedef sc_dt::sc_uint<FEATURE_COUNT_BITS> feature_counter_t;
typedef sc_dt::sc_int<FEATURE_COUNT_BITS + 1> feature_score_t;
typedef sc_dt::sc_uint<DISTANCE_BITS> distance_counter_t;
typedef sc_dt::sc_uint<TRAIN_COUNT_BITS> train_counter_t;
typedef sc_dt::sc_int<TRAIN_COUNT_BITS + 1> train_score_t;

static constexpr unsigned HV_WORD_BITS = 64;
static_assert((VECTOR_DIMENSION % HV_WORD_BITS) == 0,
              "VECTOR_DIMENSION must be divisible by HV_WORD_BITS for synthesis");
static constexpr unsigned HV_WORDS = VECTOR_DIMENSION / HV_WORD_BITS;

typedef sc_dt::sc_uint<HV_WORD_BITS> hv_word_t;

struct hv_t {
    hv_word_t words[HV_WORDS];

    bool operator==(const hv_t &other) const {
        for (unsigned word = 0; word < HV_WORDS; ++word) {
            if (words[word] != other.words[word]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const hv_t &other) const {
        return !(*this == other);
    }
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

inline void sc_trace(sc_core::sc_trace_file *tf, const hv_t &hv, const std::string &name) {
#ifndef STRATUS_HLS
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        std::ostringstream signal_name;
        signal_name << name << ".word" << word;
        sc_core::sc_trace(tf, hv.words[word], signal_name.str());
    }
#else
    (void)tf;
    (void)hv;
    (void)name;
#endif
}

#ifndef STRATUS_HLS
inline std::ostream &operator<<(std::ostream &os, const hv_t &hv) {
    os << "hv_t{";
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        if (word != 0) {
            os << ',';
        }
        os << hv.words[word];
    }
    os << '}';
    return os;
}
#endif

} // namespace hdc_systemc

#endif
