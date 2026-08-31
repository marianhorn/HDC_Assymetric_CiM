#ifndef TRAINER_H
#define TRAINER_H

#include "../foot/configFoot.h"

#include "assoc_mem.h"
#include "encoder.h"

// Function to train the model
void train_model_timeseries(double **trainingData, int *trainingLabels, int trainingSamples, struct associative_memory *assMem, struct encoder *enc);

#endif // TRAINER_H
