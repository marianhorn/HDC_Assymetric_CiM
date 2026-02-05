#ifndef DATAREADERFOOTEMG_H
#define DATAREADERFOOTEMG_H

#include <stdlib.h>

void getData(int dataset,double*** trainingData, double*** testingData, int** trainingLabels, int** testingLabels, int* trainingSamples, int* testingSamples);
void getDataWithValSet(int dataset,
                       double*** trainingData,
                       double*** validationData,
                       double*** testingData,
                       int** trainingLabels,
                       int** validationLabels,
                       int** testingLabels,
                       int* trainingSamples,
                       int* validationSamples,
                       int* testingSamples,
                       double validationRatio);
void getTestingData(int dataset, double*** testingData, int** testingLabels, int* testingSamples);
void freeData(double** data, size_t rows);
void freeCSVLabels(int* labels);

#endif // DATAREADERFOOTEMG_H
