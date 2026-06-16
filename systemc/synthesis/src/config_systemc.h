// SIMULATION / SYNTHESIS CONFIGURATION: Shared constants for the current SystemC model.
// Only constants needed by HDC_Accelerator should be treated as part of the synthesis boundary.
#ifndef SYSTEMC_HDC_CONFIG_SYSTEMC_H
#define SYSTEMC_HDC_CONFIG_SYSTEMC_H

#ifndef VECTOR_DIMENSION
#define VECTOR_DIMENSION 128
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

#ifndef MAX_SAMPLES_IN_PIPELINE
#define MAX_SAMPLES_IN_PIPELINE 32
#endif


#endif
