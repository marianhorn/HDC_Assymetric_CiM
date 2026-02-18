#include "hdc_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int get_bit(const hv_t hv, int i) {
    int w = i / 32;
    int b = i % 32;
    return (hv[w] >> b) & 1;
}

void set_bit(hv_t hv, int i, int value) {
    int w = i / 32;
    int b = i % 32;
    if(value)
        hv[w] |= (1u << b);
    else
        hv[w] &= ~(1u << b);
}

void hv_xor(hv_t out, const hv_t a, const hv_t b) {
    int chunks = chunks_per_vec();
    for(int i = 0; i < chunks; i++)
        out[i] = a[i] ^ b[i];
}

int hamming_distance(const hv_t a, const hv_t b) {
    int sum = 0;
    int chunks = chunks_per_vec();
    for(int i = 0; i < chunks; i++)
        sum += __builtin_popcount(a[i] ^ b[i]);
    return sum;
}

void hv_copy(hv_t dst, const hv_t src) {
    memcpy(dst, src, chunks_per_vec() * sizeof(uint32_t));
}

void hv_rotate_right(hv_t out, const hv_t in, int shift_bits)
{
    int chunks = chunks_per_vec();
    int word_shift = shift_bits / 32;
    int bit_shift  = shift_bits % 32;

    for(int i = 0; i < chunks; i++) {
        uint32_t a = in[(i + word_shift) % chunks];
        uint32_t b = in[(i + word_shift + 1) % chunks];
        out[i] = (a >> bit_shift) | (b << (32 - bit_shift));
    }
}

void hv_print_hex(hv_t hv) {
    int chunks = chunks_per_vec();
    for(int i = 0; i < chunks; i++) {
        printf("%08X ", hv[i]);
    }
    printf("\n");
}

