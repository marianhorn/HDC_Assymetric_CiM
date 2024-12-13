#ifndef ASSOC_MEM_H
#define ASSOC_MEM_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include <stdbool.h>
#include "vector.h"
/**
 * @brief Represents the associative memory used for HDC.
 *
 * This structure holds the class vectors and their associated counts.
 * - **num_classes**: The total number of classes in the associative memory.
 * - **class_vectors**: Array of hypervectors, one for each class.
 * - **counts**: Array of integers tracking the number of samples per class.
 */
struct associative_memory {
    int num_classes;
    Vector **class_vectors;
    int *counts;
};

// Initialize associative memory
void init_assoc_mem(struct associative_memory *assoc_mem);

// Add a sample hypervector to the associative memory
int add_to_assoc_mem(struct associative_memory *assoc_mem, Vector *sample_hv, int class_id);

// Classify a given hypervector
int classify(struct associative_memory *assoc_mem, Vector *sample_hv);

Vector* get_class_vector(struct associative_memory *assoc_mem, int class_id);
void free_assoc_mem(struct associative_memory *assoc_mem);
void print_class_vectors(struct associative_memory *assoc_mem);
void normalize(struct associative_memory *assoc_mem);
void store_assoc_mem_to_bin(struct associative_memory *assoc_mem, const char *file_path);
void load_assoc_mem_from_bin(struct associative_memory *assoc_mem, const char *filepath);
#endif // ASSOC_MEM_H
