#include "hdc_memory.h"
#include "hdc_utils.h"
#include <stdio.h>
#include <string.h>

hv_t IM[N];
hv_t* CM = NULL;
hv_t AM[NUM_CLASSES];

int alloc_memory(void)
{
    // allocate IM
    for(int i = 0; i < N; i++)
        IM[i] = hv_alloc();

    // allocate AM
    for(int i = 0; i < NUM_CLASSES; i++)
        AM[i] = hv_alloc();

    // allocate CM (dynamic size M)
    CM = malloc(M * sizeof(hv_t));
    for(int i = 0; i < M; i++)
        CM[i] = hv_alloc();

    return 0;
}

void free_memory(void)
{
    for(int i = 0; i < N; i++) hv_free(IM[i]);
    for(int i = 0; i < NUM_CLASSES; i++) hv_free(AM[i]);

    for(int i = 0; i < M; i++) hv_free(CM[i]);
    free(CM);
}

static void hv_from_bitstring(hv_t out, const char* line)
{
    int chunks = chunks_per_vec();
    memset(out, 0, chunks * sizeof(uint32_t));

    for (int i = 0; i < D; i++)
    {
        if (line[i] != '1')
            continue;

        // Welches 32-Bit-Wort?
        int chunk = i / 32;

        // Position im Wort:
        // Datei ist MSB-first, also line[i] = Bit (31 - (i % 32))
        int bit_in_chunk = 31 - (i % 32);

        out[chunk] |= (1u << bit_in_chunk);
    }
}



int load_im(const char* path)
{
    FILE* f = fopen(path, "r");
    if(!f) return -1;

    char* line = malloc(D+5);

    for(int i = 0; i < N; i++) {
        if(!fgets(line, D+5, f)) { fclose(f); return -2; }
        hv_from_bitstring(IM[i], line);
    }

    free(line);
    fclose(f);
    return 0;
}

int load_cm(const char* path)
{
    FILE* f = fopen(path, "r");
    if(!f) return -1;

    char* line = malloc(D+5);

    for(int lvl = 0; lvl < M; lvl++) {
        if(!fgets(line, D+5, f)) { fclose(f); return -2; }
        hv_from_bitstring(CM[lvl], line);
    }

    free(line);
    fclose(f);
    return 0;
}
