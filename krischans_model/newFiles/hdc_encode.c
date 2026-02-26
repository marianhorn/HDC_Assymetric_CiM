#include "hdc_encode.h"
#include "hdc_memory.h"
#include "hdc_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

static int ranges_set = 0;
static float g_minv[N];
static float g_maxv[N];

void set_quantization_ranges(const float minv[N], const float maxv[N])
{
    for (int i = 0; i < N; i++) {
        g_minv[i] = minv[i];
        g_maxv[i] = maxv[i];
    }
    ranges_set = 1;
}

static inline int quantize_level(int f, float x)
{
    const float DEFAULT_MINV = -1.0f;
    const float DEFAULT_MAXV = 1.0f;

    float minv = DEFAULT_MINV;
    float maxv = DEFAULT_MAXV;
    if (ranges_set) {
        minv = g_minv[f];
        maxv = g_maxv[f];
    }

    float denom = (maxv - minv);
    if (denom <= 0.0f) denom = 1.0f;

    float t = (x - minv) / denom;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    int level = (int)lroundf(t * (float)(M - 1));

    if (level < 0) level = 0;
    if (level >= M) level = M - 1;
    return level;
}



void encode_sample(hv_t out, float features[N])
{
    const int chunks = chunks_per_vec();

    // 1) Levels einmal berechnen (spart Arbeit im inneren Loop)
    int level_idx[N];
    for (int f = 0; f < N; f++) {
        level_idx[f] = quantize_level(f, features[f]);
    }

    // 2) Majority wortweise (N=32 => threshold >=16)
    for (int w = 0; w < chunks; w++) {
        uint32_t c0=0, c1=0, c2=0, c3=0, c4=0, c5=0;

        for (int f = 0; f < N; f++) {
            uint32_t x = IM[f][w] ^ CM[level_idx[f]][w];

            uint32_t carry = x, t;
            t = c0; c0 = t ^ carry; carry = t & carry;
            t = c1; c1 = t ^ carry; carry = t & carry;
            t = c2; c2 = t ^ carry; carry = t & carry;
            t = c3; c3 = t ^ carry; carry = t & carry;
            t = c4; c4 = t ^ carry; carry = t & carry;
            t = c5; c5 = t ^ carry; 
        }

        out[w] = (c4 | c5);
    }

    int r = D & 31;
    if (r != 0) {
        uint32_t mask = (1u << r) - 1u;
        out[chunks - 1] &= mask;
    }
}