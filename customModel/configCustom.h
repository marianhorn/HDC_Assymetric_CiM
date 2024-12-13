/**
 * @file configCustom.h
 * @brief Configuration parameters for a model.
 * 
 * This file contains macros defining various parameters and settings needed for an HDC-Model.
 * 
 * @note This file should only be included when CUSTOM is defined via makefile.
 */
#ifndef CONFIGCUSTOM_H
#define CONFIGCUSTOM_H
/** 
 * @brief Enables bipolar mode for vector representation.
 * 
 * If set to 1, vectors are represented as -1 and 1. Otherwise, they are binary (0 and 1).
 */
#define BIPOLAR_MODE 1
/** 
 * @brief Dimension of hyperdimensional vectors.
 * 
 * Defines the number of elements in each hyperdimensional vector.
 */
#define VECTOR_DIMENSION 10000

/** 
 * @brief Number of signal levels for discretization.
 * 
 * Determines the granularity of the signal level quantization.
 */
#define NUM_LEVELS 5
/** 
 * @brief Minimum level for signal normalization.
 */
#define MIN_LEVEL 0

/** 
 * @brief Maximum level for signal normalization.
 */
#define MAX_LEVEL 4
/** 
 * @brief Size of the sliding window for evaluate_model_timeseries_with_window.
 * 
 * Specifies how many samples are processed together in a single window.
 */
#define WINDOW 100
/** 
 * @brief Size of the n-gram for encoding temporal information.
 * 
 * Determines the number of timestamps bundled together to one timeseries in encoding.
 */
#define N_GRAM_SIZE 3
/** 
 * @brief Downsampling rate for the data.
 * 
 * Defines how much the data is reduced during preprocessing. A value of 1 means no downsampling.
 */
#define DOWNSAMPLE 1
/** 
 * @brief Number of distinct classes in the dataset.
 * 
 * Determines the number of different labels/classes used in the model.
 */
#define NUM_CLASSES 3
/** 
 * @brief Number of features (channels) in the input data.
 */
#define NUM_FEATURES 4
/** 
 * @brief Flag to enable or disable normalization.
 * 
 * If set to 1, the associative memory will be normalized after training.
 * Only relevant in bipolar Mode.
 */
#define NORMALIZE 1
/** 
 * @brief Threshold for cutting angle during data preprocessing.
 * Only relevant in bipolar Mode.
 */
#define CUTTING_ANGLE_THRESHOLD 0.9
/** 
 * @brief Enables precomputed item memory optimization.
 * 
 * If set to 1, precomputes the item memory for faster encoding.
 */
#define PRECOMPUTED_ITEM_MEMORY 1
/** 
 * @brief Sets the output verbosity level.
 * 
 * The `OUTPUT_MODE` can be one of the following:
 * - `OUTPUT_NONE`: No printing.
 * - `OUTPUT_BASIC`: Print results.
 * - `OUTPUT_DETAILED`: Print intermediate information.
 * - `OUTPUT_DEBUG`: Print all debugging information.
 */
#define OUTPUT_MODE OUTPUT_DEBUG




//*************DONT CHANGE ANYTHING below this line */
#define OUTPUT_NONE 0        // No printing
#define OUTPUT_BASIC 1       // Print results
#define OUTPUT_DETAILED 2    // Print intermediate information
#define OUTPUT_DEBUG 3         // Print everything

#endif
