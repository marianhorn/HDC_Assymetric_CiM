// SHARED TYPE DEFINITIONS: Used by both the accelerator model and simulation harness.
// Keep this header limited to fixed-width datapath and boundary types.
#ifndef SYSTEMC_HDC_SYSTEMC_TYPES_H
#define SYSTEMC_HDC_SYSTEMC_TYPES_H

#include <systemc.h>
#include "config_systemc.h"

#if NUM_LEVELS <= 2
#define LEVEL_BITS 1
#elif NUM_LEVELS <= 4
#define LEVEL_BITS 2
#elif NUM_LEVELS <= 8
#define LEVEL_BITS 3
#elif NUM_LEVELS <= 16
#define LEVEL_BITS 4
#elif NUM_LEVELS <= 32
#define LEVEL_BITS 5
#elif NUM_LEVELS <= 64
#define LEVEL_BITS 6
#elif NUM_LEVELS <= 128
#define LEVEL_BITS 7
#elif NUM_LEVELS <= 256
#define LEVEL_BITS 8
#else
#error "NUM_LEVELS too large for configured LEVEL_BITS range"
#endif

#if NUM_CLASSES <= 2
#define CLASS_BITS 1
#elif NUM_CLASSES <= 4
#define CLASS_BITS 2
#elif NUM_CLASSES <= 8
#define CLASS_BITS 3
#elif NUM_CLASSES <= 16
#define CLASS_BITS 4
#elif NUM_CLASSES <= 32
#define CLASS_BITS 5
#elif NUM_CLASSES <= 64
#define CLASS_BITS 6
#else
#error "NUM_CLASSES too large for configured CLASS_BITS range"
#endif

#define COMMAND_KIND_BITS 3

#if (NUM_FEATURES + 1) <= 2
#define FEATURE_COUNT_BITS 1
#elif (NUM_FEATURES + 1) <= 4
#define FEATURE_COUNT_BITS 2
#elif (NUM_FEATURES + 1) <= 8
#define FEATURE_COUNT_BITS 3
#elif (NUM_FEATURES + 1) <= 16
#define FEATURE_COUNT_BITS 4
#elif (NUM_FEATURES + 1) <= 32
#define FEATURE_COUNT_BITS 5
#elif (NUM_FEATURES + 1) <= 64
#define FEATURE_COUNT_BITS 6
#elif (NUM_FEATURES + 1) <= 128
#define FEATURE_COUNT_BITS 7
#else
#error "NUM_FEATURES too large for configured FEATURE_COUNT_BITS range"
#endif

#if (VECTOR_DIMENSION + 1) <= 2
#define DISTANCE_BITS 1
#elif (VECTOR_DIMENSION + 1) <= 4
#define DISTANCE_BITS 2
#elif (VECTOR_DIMENSION + 1) <= 8
#define DISTANCE_BITS 3
#elif (VECTOR_DIMENSION + 1) <= 16
#define DISTANCE_BITS 4
#elif (VECTOR_DIMENSION + 1) <= 32
#define DISTANCE_BITS 5
#elif (VECTOR_DIMENSION + 1) <= 64
#define DISTANCE_BITS 6
#elif (VECTOR_DIMENSION + 1) <= 128
#define DISTANCE_BITS 7
#elif (VECTOR_DIMENSION + 1) <= 256
#define DISTANCE_BITS 8
#elif (VECTOR_DIMENSION + 1) <= 512
#define DISTANCE_BITS 9
#elif (VECTOR_DIMENSION + 1) <= 1024
#define DISTANCE_BITS 10
#elif (VECTOR_DIMENSION + 1) <= 2048
#define DISTANCE_BITS 11
#elif (VECTOR_DIMENSION + 1) <= 4096
#define DISTANCE_BITS 12
#elif (VECTOR_DIMENSION + 1) <= 8192
#define DISTANCE_BITS 13
#else
#error "VECTOR_DIMENSION too large for configured DISTANCE_BITS range"
#endif

#define TRAIN_COUNT_BITS 32
#define SAMPLE_LEVELS_BITS (NUM_FEATURES * LEVEL_BITS)
#define RESPONSE_DISTANCES_BITS (NUM_CLASSES * DISTANCE_BITS)

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

#endif
