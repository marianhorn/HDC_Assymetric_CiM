#ifndef CONFIGFOOT_H
#define CONFIGFOOT_H // include guard

#ifndef VECTOR_DIMENSION
#define VECTOR_DIMENSION 10000 // hypervector dimension
#endif
#ifndef NUM_LEVELS
#define NUM_LEVELS 100 // number of quantization levels
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
#define N_GRAM_SIZE 5 // n-gram size
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
#ifndef PRECOMPUTED_ITEM_MEMORY
#define PRECOMPUTED_ITEM_MEMORY 0 // use precomputed item memory
#endif
#ifndef USE_GENETIC_ITEM_MEMORY
#define USE_GENETIC_ITEM_MEMORY 0 // enable GA item memory
#endif
#ifndef OUTPUT_MODE
#define OUTPUT_MODE OUTPUT_BASIC // output verbosity level
#endif
#ifndef RESULT_CSV_PATH
#define RESULT_CSV_PATH "analysis/results.csv" // results CSV output path
#endif
#ifndef VALIDATION_RATIO
#define VALIDATION_RATIO 0 // validation split ratio
#endif

#ifndef BIPOLAR_MODE
#define BIPOLAR_MODE 0 // use bipolar vectors
#endif

#ifndef GA_DEFAULT_POPULATION_SIZE
#define GA_DEFAULT_POPULATION_SIZE 32 // GA population size
#endif
#ifndef GA_DEFAULT_GENERATIONS
#define GA_DEFAULT_GENERATIONS 64 // GA generations
#endif
#ifndef GA_DEFAULT_CROSSOVER_RATE
#define GA_DEFAULT_CROSSOVER_RATE 0.0 // GA crossover rate
#endif
#ifndef GA_DEFAULT_MUTATION_RATE
#define GA_DEFAULT_MUTATION_RATE 0.8 // GA mutation rate
#endif
#ifndef GA_DEFAULT_TOURNAMENT_SIZE
#define GA_DEFAULT_TOURNAMENT_SIZE 3 // GA tournament size
#endif
#ifndef GA_DEFAULT_LOG_EVERY
#define GA_DEFAULT_LOG_EVERY 0 // GA log frequency
#endif
#ifndef GA_DEFAULT_SEED
#define GA_DEFAULT_SEED 1u // GA RNG seed
#endif
#ifndef GA_MAX_FLIPS_CIM
#define GA_MAX_FLIPS_CIM VECTOR_DIMENSION // CiM max flips budget
#endif
#ifndef GA_INIT_UNIFORM
#define GA_INIT_UNIFORM 1 // GA init uniform vs equal
#endif

#endif
#ifndef GA_SELECTION_MODE
#define GA_SELECTION_MODE GA_SELECTION_PARETO // GA selection mode
#endif


extern int output_mode;

//*************DONT CHANGE ANYTHING below this line */
#define OUTPUT_NONE 0        // No printing
#define OUTPUT_BASIC 1       // Print results
#define OUTPUT_DETAILED 2    // Print intermediate information


#define OUTPUT_DEBUG 3         // Print everything


#define GA_SELECTION_PARETO 0 // GA selection: NSGA-II Pareto
#define GA_SELECTION_MULTI 1 // GA selection: accuracy minus similarity
#define GA_SELECTION_ACCURACY 2 // GA selection: accuracy only



