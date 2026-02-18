#include "hdc_types.h"

int D = 10000;     // default (can be overwritten by argv)
int M = 32;        // default (continuous memory levels)

hv_t hv_alloc(void) {
    int chunks = chunks_per_vec();
    return calloc(chunks, sizeof(uint32_t));
}

void hv_free(hv_t hv) {
    free(hv);
}
