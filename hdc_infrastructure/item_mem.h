#ifndef ITEM_MEMORY_H
#define ITEM_MEMORY_H

#include "../foot/configFoot.h"
#include "vector.h"

struct item_memory {
    int num_vectors;
    Vector **base_vectors;
};

void init_precomp_item_memory(struct item_memory *item_mem, int num_levels, int num_features);
void init_precomp_item_memory_with_B(struct item_memory *item_mem,
                                     int num_levels,
                                     int num_features,
                                     const int *B,
                                     const int *permutations);
void free_item_memory(struct item_memory *item_mem);
Vector *get_item_vector(struct item_memory *item_mem, int item_id);
void print_item_memory(struct item_memory *item_mem);
void store_precomp_item_mem_to_bin(struct item_memory *item_mem,
                                   const char *filepath,
                                   int num_levels,
                                   int num_features);
void load_precomp_item_mem_from_bin(struct item_memory *item_mem,
                                    const char *filepath,
                                    int num_levels,
                                    int num_features);
void store_precomp_item_mem_to_csv(struct item_memory *item_mem,
                                   const char *filepath,
                                   int num_levels,
                                   int num_features);
void store_precomp_item_mem_to_systemc_text(struct item_memory *item_mem,
                                            const char *filepath,
                                            int num_levels,
                                            int num_features);
void load_precomp_item_mem_from_csv(struct item_memory *item_mem,
                                    const char *filepath,
                                    int num_levels,
                                    int num_features);

#endif // ITEM_MEMORY_H