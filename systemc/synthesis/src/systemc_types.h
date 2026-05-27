// SHARED TYPE DEFINITIONS: Used by both the accelerator model and simulation harness.
// Keep this header limited to fixed-width datapath and boundary types.
#ifndef SYSTEMC_HDC_SYSTEMC_TYPES_H
#define SYSTEMC_HDC_SYSTEMC_TYPES_H

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
static constexpr unsigned SAMPLE_LEVELS_BITS = NUM_FEATURES * LEVEL_BITS;
static constexpr unsigned RESPONSE_DISTANCES_BITS = NUM_CLASSES * DISTANCE_BITS;

typedef sc_dt::sc_uint<LEVEL_BITS> level_t;
typedef sc_dt::sc_uint<CLASS_BITS> class_t;
typedef sc_dt::sc_uint<COMMAND_KIND_BITS> command_kind_t;
typedef sc_dt::sc_uint<FEATURE_COUNT_BITS> feature_counter_t;
typedef sc_dt::sc_int<FEATURE_COUNT_BITS + 1> feature_score_t;
typedef sc_dt::sc_uint<DISTANCE_BITS> distance_counter_t;
typedef sc_dt::sc_uint<TRAIN_COUNT_BITS> train_counter_t;
typedef sc_dt::sc_int<TRAIN_COUNT_BITS + 1> train_score_t;
typedef sc_dt::sc_bv<VECTOR_DIMENSION> hv_t;
typedef sc_dt::sc_bv<SAMPLE_LEVELS_BITS> sample_levels_packed_t;
typedef sc_dt::sc_bv<RESPONSE_DISTANCES_BITS> distances_packed_t;

} // namespace hdc_systemc

#endif
