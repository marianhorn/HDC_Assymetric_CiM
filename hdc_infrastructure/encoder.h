#ifndef ENCODER_H
#define ENCODER_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include "item_mem.h"
#include "vector.h"


#if PRECOMPUTED_ITEM_MEMORY
/**
 * @brief Encoder structure for precomputed item memory.
 *
 * This structure uses a single item memory that combines signal levels and features.
 * - **item_mem**: Pointer to the precomputed item memory.
 */
struct encoder {
    struct item_memory *item_mem;/**< Pointer to precomputed item memory. */
};

// Initialize the encoder
void init_encoder(struct encoder *enc, struct item_memory *item_mem);
#else
/**
 * @brief Encoder structure for non-precomputed item memory.
 *
 * This structure maintains separate item memories for signal levels and features.
 * - **channel_memory**: Pointer to the item memory for features.
 * - **signal_memory**: Pointer to the item memory for signal levels.
 */
struct encoder {
    struct item_memory *channel_memory;/**< Item memory for features. */
    struct item_memory *signal_memory;/**< Item memory for signal levels. */
};

// Initialize the encoder
void init_encoder(struct encoder *enc, struct item_memory *channel_memory, struct item_memory *signal_memory);
#endif
void encode_timestamp(struct encoder *enc, double *emg_sample, Vector *result);
int encode_timeseries(struct encoder *enc, double **emg_data, Vector *result);
bool is_window_stable(int* labels);
int encode_general_data(struct encoder *enc, double *emg_data, Vector *result);

#endif // ENCODER_H
