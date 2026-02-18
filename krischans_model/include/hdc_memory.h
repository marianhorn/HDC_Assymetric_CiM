#ifndef HDC_MEMORY_H
#define HDC_MEMORY_H

#include "hdc_types.h"

extern hv_t IM[N];               // Identity Memory (32 entries)
extern hv_t* CM;                 // Continuous Memory (M entries)
extern hv_t AM[NUM_CLASSES];     // Associative Memory (5 entries)

int alloc_memory(void);
void free_memory(void);

int load_im(const char* path);
int load_cm(const char* path);

#endif
