#ifndef TRAINER_H
#define TRAINER_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include "assoc_mem.h"
#include "encoder.h"

// Function to train the model
void train_model_timeseries(double **trainingData, int *trainingLabels, int trainingSamples, struct associative_memory *assMem, struct encoder *enc);
void train_model_general_data(double **training_data, int *training_labels, int training_samples, struct associative_memory *assoc_mem, struct encoder *enc);

#endif // TRAINER_H