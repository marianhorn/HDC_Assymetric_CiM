#ifndef ENCODER_H
#define ENCODER_H

#include "../foot/configFoot.h"

#include "item_mem.h"
#include "vector.h"


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
struct ngram_encoder_state {
    Vector *encoded_samples[N_GRAM_SIZE];
    Vector *permuted_result;
    int write_pos;
    int fill_count;
};

void encode_timestamp(struct encoder *enc, double *emg_sample, Vector *result);
int encode_timeseries(struct encoder *enc, double **emg_data, Vector *result);
void init_ngram_encoder_state(struct ngram_encoder_state *state);
void reset_ngram_encoder_state(struct ngram_encoder_state *state);
void free_ngram_encoder_state(struct ngram_encoder_state *state);
int push_ngram_encoder_sample(struct encoder *enc,
                              struct ngram_encoder_state *state,
                              double *emg_sample,
                              Vector *result);
bool is_window_stable(int* labels);
int encode_general_data(struct encoder *enc, double *emg_data, Vector *result);

#endif // ENCODER_H
