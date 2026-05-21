#ifndef CONFIGFOOT_H
#define CONFIGFOOT_H // include guard

#ifndef VECTOR_DIMENSION
#define VECTOR_DIMENSION 1024 // hypervector dimension
#endif
#ifndef NUM_LEVELS
#define NUM_LEVELS 35 // number of quantization levels
#endif
#ifndef MIN_LEVEL
#define MIN_LEVEL -1 // min input level
#endif
#ifndef MAX_LEVEL
#define MAX_LEVEL 1 // max input level
#endif

#ifndef WINDOW
#define WINDOW 100 // sliding window length
#endif
#ifndef N_GRAM_SIZE
#define N_GRAM_SIZE 3 // n-gram size
#endif
#ifndef DOWNSAMPLE
#define DOWNSAMPLE 1 // downsample factor
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 5 // number of classes
#endif

#ifndef NUM_FEATURES
#define NUM_FEATURES 32 // number of input features
#endif
#ifndef NORMALIZE
#define NORMALIZE 1 // normalize assoc memory
#endif

#ifndef CUTTING_ANGLE_THRESHOLD
#define CUTTING_ANGLE_THRESHOLD 0.9 // cutting angle threshold
#endif
#ifndef OUTPUT_MODE
#define OUTPUT_MODE OUTPUT_BASIC // output verbosity level
#endif
#ifndef CIM_EXPORT_DIR
#define CIM_EXPORT_DIR "CiMs/preoptimized" // default CiM import directory
#endif
#ifndef ITEM_MEM_SEED
#define ITEM_MEM_SEED 1 // seed for deterministic item-memory initialization
#endif
#ifndef VALIDATION_RATIO
#define VALIDATION_RATIO 0.3 // validation split ratio
#endif
#ifndef ITEM_MEM_TOTAL_FLIPS
#define ITEM_MEM_TOTAL_FLIPS VECTOR_DIMENSION // CiM flip budget across levels
#endif

extern int output_mode;

//*************DONT CHANGE ANYTHING below this line */
#define OUTPUT_NONE 0        // No printing
#define OUTPUT_BASIC 1       // Print results
#define OUTPUT_DETAILED 2    // Print intermediate information
#define OUTPUT_DEBUG 3       // Print everything

#endif
