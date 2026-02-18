#ifndef BLOCK_ACCUMULATOR_H
#define BLOCK_ACCUMULATOR_H

#include "hdc_types.h"

// Window size für Rolling Bundling
#define BLOCK_WINDOW 5

// Muss einmal beim Programmstart aufgerufen werden
void block_init(void);

// Reset des accumulators (für neues Signal / neue Sequenz)
void block_reset(void);

// Berechnet Rolling-Block-Bundling:
// - input_sample: HV eines einzelnen Samples (aus encode_sample())
// - out: aktueller Rolling-HV (wird aktualisiert)
void block_accumulate(hv_t out, hv_t input_sample);

#endif
