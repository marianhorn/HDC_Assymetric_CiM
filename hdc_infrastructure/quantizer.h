#ifndef QUANTIZER_H
#define QUANTIZER_H

#include <stdint.h>

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No model type defined. Please define HAND_EMG, FOOT_EMG, or CUSTOM."
#endif

int quantizer_fit_from_training(double **training_data,
                                const int *training_labels,
                                int training_samples,
                                int num_features,
                                int num_levels);
#if BINNING_MODE == GA_REFINED_BINNING
int quantizer_refine_from_flip_counts(const uint16_t *flip_counts, int genome_length);
#endif
int get_signal_level(int feature_idx, double emg_value);
const char *quantizer_get_mode_name(void);
int quantizer_export_cuts_csv_for_dataset(int dataset);
int quantizer_export_cuts_csv(const char *filepath);
int quantizer_export_systemc_text(const char *filepath);
void quantizer_clear(void);
int quantizer_is_fitted(void);

#endif
