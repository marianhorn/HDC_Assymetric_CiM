/**
 * @file configCustom.h
 * @brief Configuration parameters for a model.
 * 
 * This file contains macros defining various parameters and settings needed for an HDC-Model.
 * 
 * @note This file should only be included when CUSTOM is defined via makefile.
 */
#ifndef CONFIGCUSTOM_H
#define CONFIGCUSTOM_H // include guard
/** 
 * @brief Enables bipolar mode for vector representation.
 * 
 * If set to 1, vectors are represented as -1 and 1. Otherwise, they are binary (0 and 1).
 */
#ifndef BIPOLAR_MODE
#define BIPOLAR_MODE 1 // use bipolar vectors
#endif
/** 
 * @brief Dimension of hyperdimensional vectors.
 * 
 * Defines the number of elements in each hyperdimensional vector.
 */
#ifndef VECTOR_DIMENSION
#define VECTOR_DIMENSION 10000 // hypervector dimension
#endif

/** 
 * @brief Number of signal levels for discretization.
 * 
 * Determines the granularity of the signal level quantization.
 */
#ifndef NUM_LEVELS
#define NUM_LEVELS 5 // number of quantization levels
#endif
/** 
 * @brief Minimum level for signal normalization.
 */
#ifndef MIN_LEVEL
#define MIN_LEVEL 0 // min input level
#endif

/** 
 * @brief Maximum level for signal normalization.
 */
#ifndef MAX_LEVEL
#define MAX_LEVEL 4 // max input level
#endif
/** 
 * @brief Size of the sliding window for evaluate_model_timeseries_with_window.
 * 
 * Specifies how many samples are processed together in a single window.
 */
#ifndef WINDOW
#define WINDOW 100 // sliding window length
#endif
/** 
 * @brief Size of the n-gram for encoding temporal information.
 * 
 * Determines the number of timestamps bundled together to one timeseries in encoding.
 */
#ifndef N_GRAM_SIZE
#define N_GRAM_SIZE 3 // n-gram size
#endif
#define MODEL_VARIANT_MARIAN 0 // Marian baseline path
#define MODEL_VARIANT_KRISCHAN 1 // Krischan-compatible rolling path
#define MODEL_VARIANT_FUSION 2 // Marian temporal path + Krischan quantization
#ifndef MODEL_VARIANT
#define MODEL_VARIANT MODEL_VARIANT_FUSION // active default model path
#endif
/** 
 * @brief Downsampling rate for the data.
 * 
 * Defines how much the data is reduced during preprocessing. A value of 1 means no downsampling.
 */
#ifndef DOWNSAMPLE
#define DOWNSAMPLE 1 // downsample factor
#endif
/** 
 * @brief Number of distinct classes in the dataset.
 * 
 * Determines the number of different labels/classes used in the model.
 */
#ifndef NUM_CLASSES
#define NUM_CLASSES 3 // number of classes
#endif
/** 
 * @brief Number of features (channels) in the input data.
 */
#ifndef NUM_FEATURES
#define NUM_FEATURES 4 // number of input features
#endif
/** 
 * @brief Flag to enable or disable normalization.
 * 
 * If set to 1, the associative memory will be normalized after training.
 * Only relevant in bipolar Mode.
 */
#ifndef NORMALIZE
#define NORMALIZE 1 // normalize assoc memory
#endif
/** 
 * @brief Threshold for cutting angle during data preprocessing.
 * Only relevant in bipolar Mode.
 */
#ifndef CUTTING_ANGLE_THRESHOLD
#define CUTTING_ANGLE_THRESHOLD 0.9 // cutting angle threshold
#endif
/** 
 * @brief Enables precomputed item memory optimization.
 * 
 * If set to 1, precomputes the item memory for faster encoding.
 */
#ifndef PRECOMPUTED_ITEM_MEMORY
#define PRECOMPUTED_ITEM_MEMORY 0 // use precomputed item memory
#endif
/** 
 * @brief Enables GA-based asymmetric item memory generation.
 * 
 * If set to 1, uses genetic optimization for item memory generation.
 */
#ifndef USE_GENETIC_ITEM_MEMORY
#define USE_GENETIC_ITEM_MEMORY 0 // enable GA item memory
#endif
/** 
 * @brief Sets the output verbosity level.
 * 
 * The `OUTPUT_MODE` can be one of the following:
 * - `OUTPUT_NONE`: No printing.
 * - `OUTPUT_BASIC`: Print results.
 * - `OUTPUT_DETAILED`: Print intermediate information.
 * - `OUTPUT_DEBUG`: Print all debugging information.
 */
#ifndef OUTPUT_MODE
#define OUTPUT_MODE OUTPUT_DEBUG // output verbosity level
#endif
#ifndef RESULT_CSV_PATH
#define RESULT_CSV_PATH "analysis/results.csv" // results CSV output path
#endif
#ifndef VALIDATION_RATIO
#define VALIDATION_RATIO 0.5 // validation split ratio
#endif

#ifndef GA_DEFAULT_POPULATION_SIZE
#define GA_DEFAULT_POPULATION_SIZE 64 // GA population size
#endif
#ifndef GA_DEFAULT_GENERATIONS
#define GA_DEFAULT_GENERATIONS 256 // GA generations
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
#define GA_MAX_FLIPS_CIM (VECTOR_DIMENSION / 2) // CiM max flips budget
#endif
#ifndef GA_INIT_UNIFORM
#define GA_INIT_UNIFORM 0 // GA init uniform vs equal
#endif
#ifndef GA_SELECTION_PARETO
#define GA_SELECTION_PARETO 0 // GA selection: NSGA-II Pareto
#endif
#ifndef GA_SELECTION_MULTI
#define GA_SELECTION_MULTI 1 // GA selection: accuracy minus similarity
#endif
#ifndef GA_SELECTION_ACCURACY
#define GA_SELECTION_ACCURACY 2 // GA selection: accuracy only
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

#endif
