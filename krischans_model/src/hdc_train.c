#include "hdc_train.h"
#include "hdc_utils.h"
#include <stdlib.h>

void train_class(hv_t class_out, hv_t* samples, int num_samples)
{
    uint16_t* cnt = calloc(D, sizeof(uint16_t));

    for(int s = 0; s < num_samples; s++)
        for(int bit = 0; bit < D; bit++)
            cnt[bit] += get_bit(samples[s], bit);

    int thr = num_samples / 2;

    for(int bit = 0; bit < D; bit++)
        set_bit(class_out, bit, cnt[bit] > thr);

    free(cnt);
}
