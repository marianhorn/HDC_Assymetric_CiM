#include "hdc_encode.h"
#include "hdc_memory.h"
#include "hdc_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

void encode_one_feature(hv_t out, int feature_idx, float feature_val)
{
    float scaled = (int)(ceil(feature_val * 10000.0f + 10000.0f));

    if(scaled < 0.0f) scaled = 0.0f;
    if(scaled > 20000.0f) scaled = 20000.0f;

    int level = ((int)scaled * (M - 1) + 10000) / 20000;

    if(level < 0) level = 0;
    if(level >= M) level = M - 1;

    memset(out, 0, chunks_per_vec()*sizeof(uint32_t));

    // printf("\n=====================================\n");
    // printf("FEATURE %d\n", feature_idx);
    // printf(" value       = %.6f\n", feature_val);
    // printf(" scaled      = %.2f\n", scaled);
    // printf(" level       = %d\n", level);
    // printf("=====================================\n");

    // printf("IM[%d]:\n", feature_idx);
    // hv_print_hex(IM[feature_idx]);

    // printf("CM[%d]:\n", level);
    // hv_print_hex(CM[level]);

    hv_xor(out, IM[feature_idx], CM[level]);

    // printf("XOR(IM[%d], CM[%d]):\n", feature_idx, level);
    // hv_print_hex(out);
}



void encode_sample(hv_t out, float features[N])
{
    memset(out, 0, chunks_per_vec()*sizeof(uint32_t));

    hv_t temp[N];
    for(int f = 0; f < N; f++)
        temp[f] = hv_alloc();

    for(int f = 0; f < N; f++){
        encode_one_feature(temp[f], f, features[f]);
    } 

    // majority over N temporary vectors
    uint16_t* cnt = calloc(D, sizeof(uint16_t));

    for(int f = 0; f < N; f++)
        for(int bit = 0; bit < D; bit++)
            cnt[bit] += get_bit(temp[f], bit);
           

    for(int bit = 0; bit < D; bit++)
        set_bit(out, bit, cnt[bit] >= (N/2));
    free(cnt);
    for(int f = 0; f < N; f++)
        hv_free(temp[f]);
}
