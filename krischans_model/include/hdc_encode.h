#ifndef HDC_ENCODE_H
#define HDC_ENCODE_H

#include "hdc_types.h"

void encode_one_feature(hv_t out, int f, float val);
void encode_sample(hv_t out, float features[N]);
void encode_sample_debug(hv_t out, float features[N]);
#endif
