#include <stdio.h>
#include <stdlib.h>

#include "../include/config.h"
#include "../include/item_mem.h"
#include "../include/vector.h"

int output_mode = OUTPUT_NONE;

static void write_naive_cim_csv(const struct item_memory *item_mem, const char *path) {
    FILE *file = fopen(path, "w");
    if (!file) {
        perror("Failed to open naive CiM export file");
        exit(EXIT_FAILURE);
    }

    const int num_vectors = NUM_LEVELS * NUM_FEATURES;
    fprintf(file,
            "#ga_cim_export,mode=precomputed,generation=0,candidate=0,num_levels=%d,num_features=%d,num_vectors=%d,dimension=%d,source=naive_unoptimized\n",
            NUM_LEVELS,
            NUM_FEATURES,
            num_vectors,
            VECTOR_DIMENSION);

    for (int level = 0; level < NUM_LEVELS; level++) {
        for (int feature = 0; feature < NUM_FEATURES; feature++) {
            const int index = level * NUM_FEATURES + feature;
            for (int bit = 0; bit < VECTOR_DIMENSION; bit++) {
                fprintf(file, "%d", vector_get_bit(item_mem->base_vectors[index], bit));
                if (bit + 1 < VECTOR_DIMENSION) {
                    fputc(',', file);
                }
            }
            fputc('\n', file);
        }
    }

    fclose(file);
}

int main(int argc, char **argv) {
    const char *out_path = argc > 1 ? argv[1] : "analysis/cim_data/naive_precomp_cim.csv";

    struct item_memory item_mem;
    init_precomp_item_memory(&item_mem, NUM_LEVELS, NUM_FEATURES);
    write_naive_cim_csv(&item_mem, out_path);
    free_item_memory(&item_mem);

    printf("Naive precomputed CiM exported to %s\n", out_path);
    return EXIT_SUCCESS;
}
