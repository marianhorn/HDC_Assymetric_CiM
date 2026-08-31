#include "item_mem.h"
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t item_mem_xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0u) {
        x = 0x6d2b79f5u;
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int item_mem_rand_range(uint32_t *state, int max) {
    if (max <= 0) {
        return 0;
    }
    return (int)(item_mem_xorshift32(state) % (uint32_t)max);
}

static uint32_t item_mem_seed_from_permutation(const int *perm, int length) {
    uint32_t hash = 2166136261u;
    if (!perm || length <= 0) {
        return 1u;
    }
    for (int i = 0; i < length; i++) {
        hash ^= (uint32_t)perm[i];
        hash *= 16777619u;
    }
    return hash == 0u ? 1u : hash;
}

static void generate_random_hv_with_rng(vector_element *data, int dimension, uint32_t *state) {
    for (size_t word = 0; word < vector_storage_count(); word++) {
        data[word] = 0ull;
    }
    for (int i = 0; i < dimension; i++) {
        int word = i >> 6;
        int bit = i & 63;
        if (item_mem_rand_range(state, 2)) {
            data[word] |= 1ull << bit;
        }
    }
    if ((dimension & 63) != 0) {
        data[(dimension + 63) / 64 - 1] &= ((1ull << (dimension & 63)) - 1ull);
    }
}

static void allocate_item_memory_vectors(struct item_memory *item_mem, int num_vectors) {
    item_mem->num_vectors = num_vectors;
    item_mem->base_vectors = (Vector **)malloc((size_t)num_vectors * sizeof(Vector *));
    if (!item_mem->base_vectors) {
        fprintf(stderr, "Failed to allocate item memory vector table.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_vectors; i++) {
        item_mem->base_vectors[i] = create_uninitialized_vector();
    }
}

void init_precomp_item_memory(struct item_memory *item_mem, int num_levels, int num_features) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing precomputed item memory with %d levels for %d features.\n", num_levels, num_features);
    }

    int total_vectors = num_levels * num_features;
    allocate_item_memory_vectors(item_mem, total_vectors);

    uint32_t rng_state = (uint32_t)ITEM_MEM_SEED;
    if (rng_state == 0u) {
        rng_state = 1u;
    }
    int total_flips = GA_MAX_FLIPS_CIM;

    for (int feature = 0; feature < num_features; feature++) {
        Vector *min_vector = create_uninitialized_vector();
        generate_random_hv_with_rng(min_vector->data, VECTOR_DIMENSION, &rng_state);

        int *perm = (int *)malloc((size_t)VECTOR_DIMENSION * sizeof(int));
        if (!perm) {
            fprintf(stderr, "Failed to allocate item-memory permutation.\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < VECTOR_DIMENSION; i++) {
            perm[i] = i;
        }
        for (int i = VECTOR_DIMENSION - 1; i > 0; i--) {
            int j = item_mem_rand_range(&rng_state, i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }

        vector_copy(item_mem->base_vectors[feature], min_vector);
        if (num_levels > 1) {
            int steps = num_levels - 1;
            int prev_target = 0;
            for (int level = 1; level < num_levels; level++) {
                double exact = ((double)level * (double)total_flips) / (double)steps;
                int target = (int)(exact + 0.5);
                if (target < 0) {
                    target = 0;
                } else if (target > total_flips) {
                    target = total_flips;
                }

                Vector *prev = item_mem->base_vectors[(level - 1) * num_features + feature];
                Vector *curr = item_mem->base_vectors[level * num_features + feature];
                vector_copy(curr, prev);
                for (int k = prev_target; k < target; k++) {
                    vector_flip_bit(curr, perm[k]);
                }
                prev_target = target;
            }
        }

        free(perm);
        free_vector(min_vector);
    }

    if (output_mode >= OUTPUT_DEBUG) {
        print_item_memory(item_mem);
        printf("\n");
    }
}

void init_precomp_item_memory_with_B(struct item_memory *item_mem,
                                     int num_levels,
                                     int num_features,
                                     const int *B,
                                     const int *permutations) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing precomputed item memory (B-driven) with %d levels for %d features.\n",
               num_levels,
               num_features);
    }
    if (!B || !permutations) {
        if (output_mode >= OUTPUT_BASIC) {
            fprintf(stderr, "init_precomp_item_memory_with_B: B or permutations is NULL.\n");
        }
        item_mem->num_vectors = 0;
        item_mem->base_vectors = NULL;
        return;
    }

    int total_vectors = num_levels * num_features;
    allocate_item_memory_vectors(item_mem, total_vectors);
    int max_flips = GA_MAX_FLIPS_CIM;

    for (int feature = 0; feature < num_features; feature++) {
        Vector *min_vector = create_uninitialized_vector();
        const int *perm = permutations + (size_t)feature * VECTOR_DIMENSION;
        uint32_t rng_state = item_mem_seed_from_permutation(perm, VECTOR_DIMENSION);
        generate_random_hv_with_rng(min_vector->data, VECTOR_DIMENSION, &rng_state);

        vector_copy(item_mem->base_vectors[feature], min_vector);
        if (num_levels > 1) {
            int prev_target = 0;
            for (int level = 1; level < num_levels; level++) {
                int flips = B[feature * (num_levels - 1) + (level - 1)];
                if (flips < 0) {
                    flips = 0;
                }
                int target = prev_target + flips;
                if (target > max_flips) {
                    target = max_flips;
                }

                Vector *prev = item_mem->base_vectors[(level - 1) * num_features + feature];
                Vector *curr = item_mem->base_vectors[level * num_features + feature];
                vector_copy(curr, prev);
                for (int k = prev_target; k < target; k++) {
                    vector_flip_bit(curr, perm[k]);
                }
                prev_target = target;
            }
        }

        free_vector(min_vector);
    }

    if (output_mode >= OUTPUT_DEBUG) {
        print_item_memory(item_mem);
        printf("\n");
    }
}

void free_item_memory(struct item_memory *item_mem) {
    if (!item_mem || !item_mem->base_vectors) {
        return;
    }
    for (int i = 0; i < item_mem->num_vectors; i++) {
        free_vector(item_mem->base_vectors[i]);
    }
    free(item_mem->base_vectors);
    item_mem->base_vectors = NULL;
    item_mem->num_vectors = 0;
}

Vector *get_item_vector(struct item_memory *item_mem, int item_id) {
    if (item_mem && item_id >= 0 && item_id < item_mem->num_vectors) {
        return item_mem->base_vectors[item_id];
    }
    return NULL;
}

void print_item_memory(struct item_memory *item_mem) {
    printf("Item memory contains %d vectors of dimension %d\n", item_mem->num_vectors, VECTOR_DIMENSION);
    for (int j = 0; j < VECTOR_DIMENSION; j += 1000) {
        for (int i = 0; i < item_mem->num_vectors; i++) {
            printf("%d ", vector_get_bit(item_mem->base_vectors[i], j));
        }
        printf("\n");
    }
}

void store_precomp_item_mem_to_bin(struct item_memory *item_mem,
                                   const char *filepath,
                                   int num_levels,
                                   int num_features) {
    if (!item_mem || num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "store_precomp_item_mem_to_bin: invalid arguments.\n");
        return;
    }
    int expected = num_levels * num_features;
    if (item_mem->num_vectors != expected && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "store_precomp_item_mem_to_bin: expected %d vectors, got %d.\n",
                expected,
                item_mem->num_vectors);
    }

    FILE *file = fopen(filepath, "wb");
    if (!file) {
        perror("Failed to open file for writing precomputed item memory");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < item_mem->num_vectors; i++) {
        fwrite(item_mem->base_vectors[i]->data, sizeof(vector_element), vector_storage_count(), file);
    }
    fclose(file);
    printf("Precomputed item memory successfully stored to %s\n", filepath);
}

void store_precomp_item_mem_to_csv(struct item_memory *item_mem,
                                   const char *filepath,
                                   int num_levels,
                                   int num_features) {
    if (!item_mem || num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "store_precomp_item_mem_to_csv: invalid arguments.\n");
        return;
    }
    int expected = num_levels * num_features;
    if (item_mem->num_vectors != expected && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "store_precomp_item_mem_to_csv: expected %d vectors, got %d.\n",
                expected,
                item_mem->num_vectors);
    }

    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("Failed to open file for writing precomputed item memory CSV");
        exit(EXIT_FAILURE);
    }
    fprintf(file,
            "#precomp_item_mem,num_levels=%d,num_features=%d,num_vectors=%d,dimension=%d\n",
            num_levels,
            num_features,
            expected,
            VECTOR_DIMENSION);
    for (int i = 0; i < item_mem->num_vectors; i++) {
        for (int bit = 0; bit < VECTOR_DIMENSION; bit++) {
            fprintf(file, "%d", vector_get_bit(item_mem->base_vectors[i], bit));
            if (bit < VECTOR_DIMENSION - 1) {
                fputc(',', file);
            }
        }
        fputc('\n', file);
    }
    fclose(file);
    printf("Precomputed item memory successfully stored to %s\n", filepath);
}

void store_precomp_item_mem_to_systemc_text(struct item_memory *item_mem,
                                            const char *filepath,
                                            int num_levels,
                                            int num_features) {
    if (!item_mem || !filepath || filepath[0] == '\0' || num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "store_precomp_item_mem_to_systemc_text: invalid arguments.\n");
        return;
    }
    int expected = num_levels * num_features;
    if (item_mem->num_vectors != expected && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "store_precomp_item_mem_to_systemc_text: expected %d vectors, got %d.\n",
                expected,
                item_mem->num_vectors);
    }

    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("Failed to open file for writing SystemC precomputed item memory text");
        exit(EXIT_FAILURE);
    }
    fprintf(file,
            "#systemc_precomp_cim num_levels=%d num_features=%d num_vectors=%d dimension=%d layout=level_major_feature_minor\n",
            num_levels,
            num_features,
            expected,
            VECTOR_DIMENSION);
    for (int level = 0; level < num_levels; level++) {
        for (int feature = 0; feature < num_features; feature++) {
            int index = level * num_features + feature;
            fprintf(file, "%d %d ", level, feature);
            for (int bit = 0; bit < VECTOR_DIMENSION; bit++) {
                fputc(vector_get_bit(item_mem->base_vectors[index], bit) ? '1' : '0', file);
            }
            fputc('\n', file);
        }
    }
    fclose(file);
    printf("SystemC precomputed item memory successfully stored to %s\n", filepath);
}

static char *trim_in_place(char *s) {
    while (*s && isspace((unsigned char)*s)) {
        s++;
    }
    if (*s == '\0') {
        return s;
    }
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end-- = '\0';
    }
    return s;
}

static int parse_csv_header(FILE *file,
                            int *num_vectors,
                            int *num_levels,
                            int *num_features,
                            int *dimension) {
    long pos = ftell(file);
    char line[512];
    if (!fgets(line, sizeof(line), file)) {
        fseek(file, pos, SEEK_SET);
        return 0;
    }
    if (line[0] != '#') {
        fseek(file, pos, SEEK_SET);
        return 0;
    }

    char *cursor = line + 1;
    char *token = strtok(cursor, ",");
    while (token) {
        char *entry = trim_in_place(token);
        char *eq = strchr(entry, '=');
        if (eq) {
            *eq = '\0';
            char *key = trim_in_place(entry);
            char *value = trim_in_place(eq + 1);
            int parsed = atoi(value);
            if (strcmp(key, "num_vectors") == 0 && num_vectors) {
                *num_vectors = parsed;
            } else if (strcmp(key, "num_levels") == 0 && num_levels) {
                *num_levels = parsed;
            } else if (strcmp(key, "num_features") == 0 && num_features) {
                *num_features = parsed;
            } else if (strcmp(key, "dimension") == 0 && dimension) {
                *dimension = parsed;
            }
        }
        token = strtok(NULL, ",");
    }
    return 1;
}

static void load_item_mem_from_csv_stream(struct item_memory *item_mem, FILE *file, int num_items) {
    allocate_item_memory_vectors(item_mem, num_items);
    for (int i = 0; i < num_items; i++) {
        for (int bit = 0; bit < VECTOR_DIMENSION; bit++) {
            int value = 0;
            if (fscanf(file, "%d", &value) != 1) {
                fprintf(stderr, "Error: Incomplete vector data at row %d, col %d\n", i, bit);
                exit(EXIT_FAILURE);
            }
            vector_set_bit(item_mem->base_vectors[i], bit, value);
            if (bit < VECTOR_DIMENSION - 1) {
                int ch = fgetc(file);
                if (ch != ',') {
                    fprintf(stderr, "Error: Expected ',' at row %d, col %d\n", i, bit);
                    exit(EXIT_FAILURE);
                }
            }
        }
        int ch = fgetc(file);
        if (ch != '\n' && ch != EOF) {
            fprintf(stderr, "Error: Expected end of line at row %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
}

void load_precomp_item_mem_from_bin(struct item_memory *item_mem,
                                    const char *filepath,
                                    int num_levels,
                                    int num_features) {
    if (num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "load_precomp_item_mem_from_bin: invalid dimensions.\n");
        return;
    }
    int total = num_levels * num_features;
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        perror("Failed to open file for reading precomputed item memory");
        exit(EXIT_FAILURE);
    }
    allocate_item_memory_vectors(item_mem, total);
    for (int i = 0; i < total; i++) {
        size_t items_read = fread(item_mem->base_vectors[i]->data,
                                  sizeof(vector_element),
                                  vector_storage_count(),
                                  file);
        if (items_read != vector_storage_count()) {
            fprintf(stderr, "Error: Incomplete vector data at row %d with only %ld elements\n", i, items_read);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
    printf("Precomputed item memory successfully loaded from %s\n", filepath);
}

void load_precomp_item_mem_from_csv(struct item_memory *item_mem,
                                    const char *filepath,
                                    int num_levels,
                                    int num_features) {
    FILE *file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open file for reading precomputed item memory CSV");
        exit(EXIT_FAILURE);
    }

    int header_vectors = 0;
    int header_levels = 0;
    int header_features = 0;
    int header_dim = 0;
    int has_header = parse_csv_header(file, &header_vectors, &header_levels, &header_features, &header_dim);
    if (has_header) {
        if (header_levels > 0) {
            num_levels = header_levels;
        }
        if (header_features > 0) {
            num_features = header_features;
        }
    }
    if (num_levels <= 0 || num_features <= 0) {
        fprintf(stderr, "load_precomp_item_mem_from_csv: invalid dimensions.\n");
        fclose(file);
        return;
    }

    int total = num_levels * num_features;
    if (header_vectors > 0 && header_vectors != total && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "load_precomp_item_mem_from_csv: header vectors %d override derived %d.\n",
                header_vectors,
                total);
        total = header_vectors;
    }
    if (header_dim > 0 && header_dim != VECTOR_DIMENSION && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr, "load_precomp_item_mem_from_csv: file dimension %d differs from build dimension %d.\n",
                header_dim,
                VECTOR_DIMENSION);
    }

    load_item_mem_from_csv_stream(item_mem, file, total);
    fclose(file);
    printf("Precomputed item memory successfully loaded from %s\n", filepath);
}