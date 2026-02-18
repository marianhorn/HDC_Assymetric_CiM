#ifndef HDC_UTILS_H
#define HDC_UTILS_H

#include "hdc_types.h"
#include <stdint.h>
#include <string.h>   // nötig für memcpy/memset

int get_bit(const hv_t hv, int i);
void set_bit(hv_t hv, int i, int value);

void hv_xor(hv_t out, const hv_t a, const hv_t b);

void hv_copy(hv_t dst, const hv_t src);

void hv_rotate_right(hv_t out, const hv_t in, int shift_bits);

int hamming_distance(const hv_t a, const hv_t b);

void hv_print_hex(hv_t hv);

#endif
