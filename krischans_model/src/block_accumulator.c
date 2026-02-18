#include "block_accumulator.h"
#include "hdc_utils.h"
#include <stdlib.h>
#include <string.h>

static hv_t window[BLOCK_WINDOW] = {0};
static int window_filled = 0;
static int window_pos = 0;

void block_init(void)
{
    for(int i = 0; i < BLOCK_WINDOW; i++)
        window[i] = hv_alloc();

    window_filled = 0;
    window_pos = 0;
}

void block_reset(void)
{
    int chunks = chunks_per_vec();

    for(int i = 0; i < BLOCK_WINDOW; i++)
        memset(window[i], 0, chunks * sizeof(uint32_t));

    window_filled = 0;
    window_pos = 0;
}

// input_sample = HV_single (Output von encode_sample())
// out          = Rolling-HV, wird von dir extern bereitgestellt
void block_accumulate(hv_t out, hv_t input_sample)
{
    hv_t rotated = hv_alloc();

    // Permutation abhängig vom Fensterindex
    hv_rotate_right(rotated, input_sample, window_pos);

    if(window_filled < BLOCK_WINDOW)
    {
        // Einfach hinzufügen
        hv_xor(out, out, rotated);
        hv_copy(window[window_pos], rotated);
        window_filled++;
    }
    else
    {
        // Ältesten entfernen
        hv_xor(out, out, window[window_pos]);
        // Neuen hinzufügen
        hv_xor(out, out, rotated);
        hv_copy(window[window_pos], rotated);
    }

    window_pos = (window_pos + 1) % BLOCK_WINDOW;

    hv_free(rotated);
}
