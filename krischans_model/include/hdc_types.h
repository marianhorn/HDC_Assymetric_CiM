#ifndef HDC_TYPES_H
#define HDC_TYPES_H

#include <stdint.h>
#include <stdlib.h>

extern int D;     // Hypervector length (bits)
extern int M;     // Number of CM levels (dynamic)
#define N 32      // Number of features (fixed)
#define NUM_CLASSES 5

// dynamic HV type: pointer to uint32_t
typedef uint32_t* hv_t;

// compute #words needed for a hypervector
static inline int chunks_per_vec() {
    return (D + 31) / 32;
}

// allocate HV initialized with 0
hv_t hv_alloc(void);

// free HV
void hv_free(hv_t hv);

#endif
