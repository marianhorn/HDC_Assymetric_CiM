#ifndef ITEM_MEMORY_H
#define ITEM_MEMORY_H

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
#include <stdint.h>
struct item_memory {
    int num_vectors;/**< Number of base vectors in the item memory. */
    Vector **base_vectors;/**< Array of pointers to the base hypervectors. */
};

// Initialize item memory for discrete items
void init_item_memory(struct item_memory *item_mem, int num_items);
void init_precomp_item_memory(struct item_memory *item_mem, int num_levels, int num_features);
// Initialize continuous item memory for signal intensities
void init_continuous_item_memory(struct item_memory *item_mem, int num_levels);

// Free item memory
void free_item_memory(struct item_memory *item_mem);

// Get the vector for a specific item
Vector* get_item_vector(struct item_memory *item_mem, int item_id);

void print_item_memory(struct item_memory *item_mem);
void store_item_mem_to_bin(struct item_memory *item_mem, const char *filepath);
void load_item_mem_from_bin(struct item_memory *item_mem, const char *filepath, int num_items);

#endif // ITEM_MEMORY_H
