#ifndef ASYM_ITEM_MEMORY_H
#define ASYM_ITEM_MEMORY_H

#include <stdint.h>
#include "item_mem.h"

#include "../foot/configFoot.h"
struct item_memory;

struct ga_params {
    int population_size;
    int generations;
    double crossover_rate;
    double mutation_rate;
    int tournament_size;
    int log_every;
    unsigned int seed;
};


// Initializes GA parameters with module defaults.
void init_ga_params(struct ga_params *params);

// Optimizes the precomputed item memory using GA with caller-provided data.
void optimize_item_memory(
    struct item_memory *item_mem,
    double **training_data,
    int *training_labels,
    int training_samples,
    double **testing_data,
    int *testing_labels,
    int testing_samples);
int optimize_item_memory_get_flip_counts(
    double **training_data,
    int *training_labels,
    int training_samples,
    double **testing_data,
    int *testing_labels,
    int testing_samples,
    uint16_t *flip_counts_out);

#endif
