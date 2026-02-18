#include "hdc_classify.h"
#include "hdc_memory.h"
#include "hdc_utils.h"

int classify(const hv_t hv)
{
    int best_class = 0;
    int best_dist  = 1e9;

    for(int c = 0; c < NUM_CLASSES; c++) {
        int d = hamming_distance(hv, AM[c]);
        if(d < best_dist) {
            best_dist = d;
            best_class = c;
        }
    }
    return best_class;
}
