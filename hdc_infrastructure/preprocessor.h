#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif
#include <stddef.h>


// Function to downsample the data
void down_sample(double** data, int* labels, size_t original_size, double*** downsampled_data, int** downsampled_labels, size_t* new_size);

#endif // PREPROCESSOR_H
