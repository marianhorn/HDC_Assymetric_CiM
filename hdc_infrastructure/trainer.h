#ifndef TRAINER_H
#define TRAINER_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#else
#error "No EMG type defined. Please define FOOT_EMG."
#endif

#include "assoc_mem.h"
#include "encoder.h"

void train_model_timeseries(double **trainingData, int *trainingLabels, int trainingSamples, struct associative_memory *assMem, struct encoder *enc);

#endif // TRAINER_H
