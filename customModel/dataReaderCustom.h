#ifndef DATAREADERCUSTOM_H
#define DATAREADERCUSTOM_H

#include <stdlib.h>
#include "configCustom.h"
void getData(double*** trainingData, double*** testingData, int** trainingLabels, int** testingLabels, int* trainingSamples, int* testingSamples);
void freeData(double** data, size_t rows);

#endif
