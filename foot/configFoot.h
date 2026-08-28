#ifndef CONFIGFOOT_H
#define CONFIGFOOT_H // include guard

#ifndef VECTOR_DIMENSION
#define VECTOR_DIMENSION 5000 // hypervector dimension
#endif
#ifndef NUM_LEVELS
#define NUM_LEVELS 40 // number of quantization levels
#endif
#ifndef MIN_LEVEL
#define MIN_LEVEL -1 // min input level
#endif
#ifndef MAX_LEVEL
#define MAX_LEVEL 1 // max input level
#endif

#ifndef N_GRAM_SIZE
#define N_GRAM_SIZE 3 // n-gram size
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 5 // number of classes
#endif

#ifndef NUM_FEATURES
#define NUM_FEATURES 32 // number of input features
#endif
#ifndef PRECOMPUTED_ITEM_MEMORY
#define PRECOMPUTED_ITEM_MEMORY 1 // use precomputed item memory
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
#ifndef ITEM_MEM_SEED
#define ITEM_MEM_SEED 1 // seed for deterministic item-memory initialization
#endif
#ifndef VALIDATION_RATIO
#define VALIDATION_RATIO 0.3 // validation split ratio
#endif
#ifndef DATASET_START
#define DATASET_START 0 // first dataset index to run
#endif
#ifndef DATASET_END
#define DATASET_END 3 // last dataset index to run, inclusive
#endif

#ifndef GA_DEFAULT_POPULATION_SIZE
#define GA_DEFAULT_POPULATION_SIZE 128 // GA population size
#endif
#ifndef GA_DEFAULT_GENERATIONS
#define GA_DEFAULT_GENERATIONS 64 // GA generations
#endif
#ifndef GA_DEFAULT_CROSSOVER_RATE
#define GA_DEFAULT_CROSSOVER_RATE 0.7 // GA crossover rate
#endif
#ifndef GA_DEFAULT_MUTATION_RATE
#define GA_DEFAULT_MUTATION_RATE 0.2 // GA mutation rate
#endif
#ifndef GA_DEFAULT_TOURNAMENT_SIZE
#define GA_DEFAULT_TOURNAMENT_SIZE 3 // GA tournament size
#endif
#ifndef GA_DEFAULT_LOG_EVERY
#define GA_DEFAULT_LOG_EVERY 0 // GA log frequency
#endif
#ifndef GA_DEFAULT_SEED
#define GA_DEFAULT_SEED 45 // GA RNG seed
#endif
#ifndef GA_CIM_EXPORT_LABEL
#define GA_CIM_EXPORT_LABEL "final_precomputed_ga" // folder label for exported GA CiMs
#endif
#ifndef GA_MAX_FLIPS_CIM
#define GA_MAX_FLIPS_CIM VECTOR_DIMENSION // CiM max flips budget
#endif
#ifndef GA_INIT_UNIFORM
#define GA_INIT_UNIFORM 1 // GA init uniform vs equal
#endif
#ifndef BINNING_MODE
#define BINNING_MODE UNIFORM_BINNING // active value-to-level binning mode
#endif
#ifndef GA_BINNING_EPSILON
#define GA_BINNING_EPSILON 1.0 // smoothing for GA-refined transition weights
#endif
#ifndef GA_BINNING_ALPHA
#define GA_BINNING_ALPHA 1.0 // inverse-width strength for GA-refined quantizer
#endif
#ifndef GA_CROSSOVER_ALPHA
#define GA_CROSSOVER_ALPHA 1 // schedule curvature for custom crossover chunk size
#endif
#ifndef GA_CROSSOVER_CHUNK_WIDTH
#define GA_CROSSOVER_CHUNK_WIDTH 0.2 // relative random width around scheduled chunk size
#endif
#ifndef GA_MUTATION_BETA
#define GA_MUTATION_BETA 0 // schedule curvature for custom mutation step size
#endif

extern int output_mode;

//*************DONT CHANGE ANYTHING below this line */
#define OUTPUT_NONE 0        // No printing
#define OUTPUT_BASIC 1       // Print results
#define OUTPUT_DETAILED 2    // Print intermediate information
#define OUTPUT_DEBUG 3       // Print everything

#define UNIFORM_BINNING 0    // use existing uniform value-to-level mapping
#define QUANTILE_BINNING 1   // use per-feature quantile value-to-level mapping
#define KMEANS_1D_BINNING 2  // use per-feature 1D k-means value-to-level mapping
#define DECISION_TREE_1D_BINNING 3  // use per-feature supervised 1D decision-tree value-to-level mapping
#define CHIMERGE_BINNING 4  // use per-feature supervised ChiMerge value-to-level mapping
#define GA_REFINED_BINNING 5  // use one preprocessing GA run to refine per-feature thresholds


#endif
