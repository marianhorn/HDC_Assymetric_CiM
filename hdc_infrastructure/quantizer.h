#ifndef QUANTIZER_H
#define QUANTIZER_H

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
                                int training_samples,
                                int num_features,
                                int num_levels);
int get_signal_level(int feature_idx, double emg_value);
int quantizer_export_cuts_csv_for_dataset(int dataset);
int quantizer_export_cuts_csv(const char *filepath);
void quantizer_clear(void);
int quantizer_is_fitted(void);

#endif
