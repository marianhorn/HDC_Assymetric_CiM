#ifndef SYSTEMC_HDC_SYSTEMC_TYPES_H
#define SYSTEMC_HDC_SYSTEMC_TYPES_H

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
typedef sc_dt::sc_uint<DISTANCE_BITS> distance_counter_t;
typedef sc_dt::sc_uint<TRAIN_COUNT_BITS> train_counter_t;
typedef sc_dt::sc_bv<VECTOR_DIMENSION> hv_t;

struct EvaluationResult {
    unsigned correct;
    unsigned not_correct;
    unsigned transition_error;
    unsigned total;
    double overall_accuracy;
    double non_transition_accuracy;
    unsigned confusion_matrix[NUM_CLASSES][NUM_CLASSES];
};

} // namespace hdc_systemc

#endif
