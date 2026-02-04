#ifndef ASYM_ITEM_MEMORY_H
#define ASYM_ITEM_MEMORY_H

#include <stdint.h>
#include "item_mem.h"

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif
struct item_memory;

struct ga_params {
    int population_size;
    int generations;
    double crossover_rate;
    double mutation_rate;
    int tournament_size;
    int log_every;
    unsigned int seed;
    uint16_t max_flip;
};


// Initializes GA parameters with module defaults.
void init_ga_params(struct ga_params *params);

// Optimizes the item memory using GA with caller-provided data.
#if PRECOMPUTED_ITEM_MEMORY
void optimize_item_memory(
    struct item_memory *item_mem,
    double **training_data,
    int *training_labels,
    int training_samples,
    double **testing_data,
    int *testing_labels,
    int testing_samples);
#else
void optimize_item_memory(
    struct item_memory *signal_mem,
    struct item_memory *channel_mem,
    double **training_data,
    int *training_labels,
    int training_samples,
    double **testing_data,
    int *testing_labels,
    int testing_samples);
#endif

#endif
