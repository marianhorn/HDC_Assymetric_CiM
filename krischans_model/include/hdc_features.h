#ifndef HDC_FEATURES_H
#define HDC_FEATURES_H

#include <stdio.h>

#define FEATURE_DIM 32

int load_csv_labels(const char* path, int** out_labels, int* out_count);
int load_csv_features(const char* path, float*** out_feats, int* out_count);

#endif
