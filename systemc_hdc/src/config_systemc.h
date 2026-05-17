#ifndef SYSTEMC_HDC_CONFIG_SYSTEMC_H
#define SYSTEMC_HDC_CONFIG_SYSTEMC_H

#ifndef VECTOR_DIMENSION
#define VECTOR_DIMENSION 1024
#endif

#ifndef NUM_FEATURES
#define NUM_FEATURES 32
#endif

#ifndef NUM_LEVELS
#define NUM_LEVELS 30
#endif

#ifndef NUM_CLASSES
#define NUM_CLASSES 5
#endif

#ifndef N_GRAM_SIZE
#define N_GRAM_SIZE 3
#endif

#ifndef NUM_DATASETS
#define NUM_DATASETS 4
#endif

#ifndef ENCODER_PES
#define ENCODER_PES 8
#endif

#ifndef NGRAM_PES
#define NGRAM_PES 8
#endif

// Distance computation is parallelized across classes.
// The accelerator instantiates one distance PE per class.

#ifndef ACCEL_LATENCY_ENCODE_NS
#define ACCEL_LATENCY_ENCODE_NS 1
#endif

#ifndef ACCEL_LATENCY_NGRAM_NS
#define ACCEL_LATENCY_NGRAM_NS 1
#endif

#ifndef ACCEL_LATENCY_BUNDLE_NS
#define ACCEL_LATENCY_BUNDLE_NS 1
#endif

#ifndef ACCEL_LATENCY_DISTANCE_NS
#define ACCEL_LATENCY_DISTANCE_NS 1
#endif

#endif
