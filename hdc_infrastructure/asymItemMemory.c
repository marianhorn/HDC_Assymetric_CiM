#include "asymItemMemory.h"
#include "assoc_mem.h"
#include "encoder.h"
#include "item_mem.h"
#include "operations.h"
#include "vector.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define GA_DEFAULT_POPULATION_SIZE 4 //Default: 12
#define GA_DEFAULT_GENERATIONS 4 //Default: 10
#define GA_DEFAULT_CROSSOVER_RATE 0.7 //Default 0.7
#define GA_DEFAULT_MUTATION_RATE 0.02 //Default 0.02
#define GA_DEFAULT_TOURNAMENT_SIZE 3 //Default 3
#define GA_DEFAULT_LOG_EVERY 0 //Default 0
#define GA_DEFAULT_SEED 1u
#define GA_DEFAULT_MAX_FLIP 0u

void init_ga_params(struct ga_params *params) {
    if (!params) {
        return;
    }
    params->population_size = GA_DEFAULT_POPULATION_SIZE;
    params->generations = GA_DEFAULT_GENERATIONS;
    params->crossover_rate = GA_DEFAULT_CROSSOVER_RATE;
    params->mutation_rate = GA_DEFAULT_MUTATION_RATE;
    params->tournament_size = GA_DEFAULT_TOURNAMENT_SIZE;
    params->log_every = GA_DEFAULT_LOG_EVERY;
    params->seed = GA_DEFAULT_SEED;
    params->max_flip = GA_DEFAULT_MAX_FLIP;
}

static uint32_t xorshift32(uint32_t *state) {
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

static double rng_uniform(uint32_t *state) {
    return (double)xorshift32(state) / (double)UINT32_MAX;
}

static int rng_range(uint32_t *state, int max) {
    if (max <= 0) {
        return 0;
    }
    return (int)(xorshift32(state) % (uint32_t)max);
}

static void fill_random_vector(vector_element *data, int dimension, uint32_t *state) {
    for (int i = 0; i < dimension; i++) {
#if BIPOLAR_MODE
        data[i] = (xorshift32(state) & 1u) ? 1 : -1;
#else
        data[i] = (vector_element)((xorshift32(state) & 1u) ? 1 : 0);
#endif
    }
}

static void flip_vector_element(vector_element *value) {
#if BIPOLAR_MODE
    *value = -(*value);
#else
    *value = !(*value);
#endif
}

static void build_asym_item_memory_from_B(
    struct item_memory *item_mem,
    const uint16_t *B,
    int num_levels,
    int num_features,
    int vector_dimension,
    unsigned int seed) {
    if (!item_mem || !B || num_levels <= 0 || num_features <= 0) {
        return;
    }

    int dimension = vector_dimension > 0 ? vector_dimension : VECTOR_DIMENSION;
    if (dimension != VECTOR_DIMENSION) {
#if OUTPUT_MODE >= OUTPUT_BASIC
        fprintf(stderr,
                "Warning: vector_dimension (%d) != VECTOR_DIMENSION (%d). Using VECTOR_DIMENSION.\n",
                vector_dimension,
                VECTOR_DIMENSION);
#endif
        dimension = VECTOR_DIMENSION;
    }

    int total_vectors = num_levels * num_features;
    item_mem->num_vectors = total_vectors;
    item_mem->base_vectors = (Vector **)malloc(total_vectors * sizeof(Vector *));
    if (!item_mem->base_vectors) {
        fprintf(stderr, "Failed to allocate asymmetric item memory pointers.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < total_vectors; i++) {
        item_mem->base_vectors[i] = create_uninitialized_vector();
        if (!item_mem->base_vectors[i]) {
            fprintf(stderr, "Failed to allocate asymmetric item memory vector.\n");
            exit(EXIT_FAILURE);
        }
    }

    uint32_t base_state = seed ? seed : 1u;

    for (int feature = 0; feature < num_features; feature++) {
        Vector *level0 = item_mem->base_vectors[feature];
        fill_random_vector(level0->data, dimension, &base_state);

        for (int level = 1; level < num_levels; level++) {
            Vector *prev = item_mem->base_vectors[(level - 1) * num_features + feature];
            Vector *curr = item_mem->base_vectors[level * num_features + feature];
            memcpy(curr->data, prev->data, dimension * sizeof(vector_element));

            uint16_t flips = B[feature * num_levels + level];
            if (flips > (uint16_t)dimension) {
                flips = (uint16_t)dimension;
            }

            uint32_t flip_state = seed ? seed : 1u;
            flip_state ^= (uint32_t)(feature + 1) * 0x9e3779b9u;
            flip_state ^= (uint32_t)(level + 1) * 0x85ebca6bu;

            for (uint16_t k = 0; k < flips; k++) {
                int idx = rng_range(&flip_state, dimension);
                flip_vector_element(&curr->data[idx]);
            }
        }
    }
}

static int mode_value(const int *array, int size) {
    int max_value = 0;
    int max_count = 0;

    for (int i = 0; i < size; i++) {
        int count = 0;
        for (int j = 0; j < size; j++) {
            if (array[j] == array[i]) {
                ++count;
            }
        }
        if (count > max_count) {
            max_count = count;
            max_value = array[i];
        } else if (count == max_count) {
            if (array[i] < max_value) {
                max_value = array[i];
            }
        }
    }
    return max_value;
}

static void init_assoc_mem_silent(struct associative_memory *assoc_mem) {
    assoc_mem->num_classes = NUM_CLASSES;
    assoc_mem->class_vectors = (Vector **)malloc(NUM_CLASSES * sizeof(Vector *));
    assoc_mem->counts = (int *)malloc(NUM_CLASSES * sizeof(int));
    if (!assoc_mem->class_vectors || !assoc_mem->counts) {
        fprintf(stderr, "Failed to allocate associative memory.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        assoc_mem->class_vectors[i] = create_vector();
        memset(assoc_mem->class_vectors[i]->data, 0, VECTOR_DIMENSION * sizeof(vector_element));
        assoc_mem->counts[i] = 0;
    }
}

static void normalize_silent(struct associative_memory *assoc_mem) {
    for (int i = 0; i < assoc_mem->num_classes; i++) {
        int count = assoc_mem->counts[i];
        if (count > 0) {
            for (int j = 0; j < VECTOR_DIMENSION; j++) {
                assoc_mem->class_vectors[i]->data[j] /= count;
            }
        }
    }
}

static void train_timeseries_silent(double **training_data,
                                    int *training_labels,
                                    int training_samples,
                                    struct associative_memory *assoc_mem,
                                    struct encoder *enc) {
#if BIPOLAR_MODE
    for (int j = 0; j < training_samples - N_GRAM_SIZE; j++) {
        Vector *sample_hv = create_vector();
        if (is_window_stable(&training_labels[j])) {
            encode_timeseries(enc, &training_data[j], sample_hv);
            add_to_assoc_mem(assoc_mem, sample_hv, training_labels[j]);
        }
        free_vector(sample_hv);
    }
    if (NORMALIZE) {
        normalize_silent(assoc_mem);
    }
#else
    int max_samples = training_samples - N_GRAM_SIZE;
    Vector ***encoded_vectors = (Vector ***)malloc(NUM_CLASSES * sizeof(Vector **));
    int *vector_counts = (int *)calloc(NUM_CLASSES, sizeof(int));
    if (!encoded_vectors || !vector_counts) {
        fprintf(stderr, "Failed to allocate binary training buffers.\n");
        exit(EXIT_FAILURE);
    }

    for (int class_id = 0; class_id < NUM_CLASSES; class_id++) {
        encoded_vectors[class_id] = (Vector **)malloc(max_samples * sizeof(Vector *));
        for (int i = 0; i < max_samples; i++) {
            encoded_vectors[class_id][i] = create_vector();
        }
    }

    for (int j = 0; j < training_samples - N_GRAM_SIZE; j++) {
        if (is_window_stable(&training_labels[j])) {
            encode_timeseries(enc,
                              &training_data[j],
                              encoded_vectors[training_labels[j]][vector_counts[training_labels[j]]]);
            vector_counts[training_labels[j]]++;
        } else {
            j += (N_GRAM_SIZE - 1);
        }
    }

    for (int class_id = 0; class_id < NUM_CLASSES; class_id++) {
        Vector *bundled_hv = create_vector();
        bundle_multi(encoded_vectors[class_id], vector_counts[class_id], bundled_hv);
        add_to_assoc_mem(assoc_mem, bundled_hv, class_id);
        assoc_mem->counts[class_id] = vector_counts[class_id];

        free_vector(bundled_hv);

        for (int i = 0; i < max_samples; i++) {
            free_vector(encoded_vectors[class_id][i]);
        }
        free(encoded_vectors[class_id]);
    }

    free(encoded_vectors);
    free(vector_counts);
#endif
}

static double evaluate_timeseries_direct_accuracy(struct encoder *enc,
                                                  struct associative_memory *assoc_mem,
                                                  double **testing_data,
                                                  int *testing_labels,
                                                  int testing_samples) {
    size_t correct = 0;
    size_t not_correct = 0;
    size_t transition_error = 0;

    for (int j = 0; j < testing_samples - N_GRAM_SIZE + 1; j += N_GRAM_SIZE) {
        int actual_label = mode_value(testing_labels + j, N_GRAM_SIZE);
        Vector *sample_hv = create_vector();
        int encoding_result = encode_timeseries(enc, &(testing_data[j]), sample_hv);
        int predicted_label = classify(assoc_mem, sample_hv);
        if (predicted_label == -1) {
            fprintf(stderr, "GA evaluation: invalid label (encoding=%d).\n", encoding_result);
            exit(EXIT_FAILURE);
        }
        double confidence = similarity_check(sample_hv, get_class_vector(assoc_mem, predicted_label));
        if (confidence == -2) {
            fprintf(stderr, "GA evaluation: invalid similarity.\n");
            exit(EXIT_FAILURE);
        }

        free_vector(sample_hv);

        if (predicted_label == actual_label) {
            correct++;
        } else if (testing_labels[j] != testing_labels[j + N_GRAM_SIZE - 1]) {
            transition_error++;
        } else {
            not_correct++;
        }
    }

    size_t total = correct + not_correct + transition_error;
    if (total == 0) {
        return 0.0;
    }
    return (double)correct / (double)total;
}

struct ga_eval_context {
    int num_features;
    int num_levels;
    int vector_dimension;
    unsigned int seed;
    struct item_memory *channel_memory;
    double **training_data;
    int *training_labels;
    int training_samples;
    double **testing_data;
    int *testing_labels;
    int testing_samples;
};

static double evaluate_candidate(const uint16_t *B,
                                 const struct ga_eval_context *ctx) {
    if (!ctx || !ctx->training_data || !ctx->training_labels || ctx->training_samples <= N_GRAM_SIZE) {
        return 0.0;
    }

    struct encoder enc;
    struct associative_memory assoc_mem;
    init_assoc_mem_silent(&assoc_mem);

#if PRECOMPUTED_ITEM_MEMORY
    struct item_memory tmp_im;
    build_asym_item_memory_from_B(&tmp_im,
                                  B,
                                  ctx->num_levels,
                                  ctx->num_features,
                                  ctx->vector_dimension,
                                  ctx->seed);
    init_encoder(&enc, &tmp_im);
    train_timeseries_silent(ctx->training_data,
                            ctx->training_labels,
                            ctx->training_samples,
                            &assoc_mem,
                            &enc);
#else
    if (!ctx->channel_memory) {
        free_assoc_mem(&assoc_mem);
        return 0.0;
    }
    struct item_memory signal_mem;
    build_asym_item_memory_from_B(&signal_mem,
                                  B,
                                  ctx->num_levels,
                                  ctx->num_features,
                                  ctx->vector_dimension,
                                  ctx->seed);
    init_encoder(&enc, ctx->channel_memory, &signal_mem);
    train_timeseries_silent(ctx->training_data,
                            ctx->training_labels,
                            ctx->training_samples,
                            &assoc_mem,
                            &enc);
#endif

    double **eval_data = ctx->training_data;
    int *eval_labels = ctx->training_labels;
    int eval_samples = ctx->training_samples;
    if (ctx->testing_data && ctx->testing_labels && ctx->testing_samples > 0) {
        eval_data = ctx->testing_data;
        eval_labels = ctx->testing_labels;
        eval_samples = ctx->testing_samples;
    }

    double accuracy = evaluate_timeseries_direct_accuracy(&enc, &assoc_mem, eval_data, eval_labels, eval_samples);

    free_assoc_mem(&assoc_mem);
#if PRECOMPUTED_ITEM_MEMORY
    free_item_memory(&tmp_im);
#else
    free_item_memory(&signal_mem);
#endif
    return accuracy;
}

static double evaluate_fitness(const uint16_t *B, const struct ga_eval_context *ctx) {
    if (!ctx) {
        return 0.0;
    }
    return evaluate_candidate(B, ctx);
}

static void init_individual(uint16_t *individual,
                            int num_features,
                            int num_levels,
                            uint16_t max_flip,
                            uint32_t *rng_state) {
    for (int feature = 0; feature < num_features; feature++) {
        individual[feature * num_levels] = 0;
        for (int level = 1; level < num_levels; level++) {
            int value = max_flip > 0 ? rng_range(rng_state, (int)max_flip + 1) : 0;
            individual[feature * num_levels + level] = (uint16_t)value;
        }
    }
}

static void mutate_individual(uint16_t *individual,
                              int num_features,
                              int num_levels,
                              uint16_t max_flip,
                              double mutation_rate,
                              uint32_t *rng_state) {
    for (int feature = 0; feature < num_features; feature++) {
        for (int level = 1; level < num_levels; level++) {
            if (rng_uniform(rng_state) < mutation_rate) {
                int delta = rng_range(rng_state, 2) == 0 ? -1 : 1;
                int value = (int)individual[feature * num_levels + level] + delta;
                if (value < 0) {
                    value = 0;
                } else if (value > (int)max_flip) {
                    value = (int)max_flip;
                }
                individual[feature * num_levels + level] = (uint16_t)value;
            }
        }
    }
}

static void crossover_individual(const uint16_t *parent_a,
                                 const uint16_t *parent_b,
                                 uint16_t *child,
                                 int genome_length,
                                 int num_features,
                                 int num_levels,
                                 double crossover_rate,
                                 uint32_t *rng_state) {
    if (rng_uniform(rng_state) < crossover_rate) {
        for (int i = 0; i < genome_length; i++) {
            child[i] = rng_range(rng_state, 2) == 0 ? parent_a[i] : parent_b[i];
        }
    } else {
        memcpy(child, parent_a, genome_length * sizeof(uint16_t));
    }

    for (int feature = 0; feature < num_features; feature++) {
        child[feature * num_levels] = 0;
    }
}

static int tournament_select(const double *fitness,
                             int population_size,
                             int tournament_size,
                             uint32_t *rng_state) {
    int best_index = rng_range(rng_state, population_size);
    double best_fitness = fitness[best_index];

    for (int i = 1; i < tournament_size; i++) {
        int idx = rng_range(rng_state, population_size);
        if (fitness[idx] > best_fitness) {
            best_fitness = fitness[idx];
            best_index = idx;
        }
    }
    return best_index;
}

static void run_ga(const struct ga_eval_context *ctx_in,
                   struct ga_params *params,
                   uint16_t *B_out) {
    if (!ctx_in || !params || !B_out) {
        return;
    }
    if (ctx_in->num_features <= 0 || ctx_in->num_levels <= 0) {
        return;
    }

    struct ga_eval_context ctx = *ctx_in;
    int genome_length = ctx.num_features * ctx.num_levels;
    memset(B_out, 0, (size_t)genome_length * sizeof(uint16_t));

    if (!ctx.training_data || !ctx.training_labels || ctx.training_samples <= N_GRAM_SIZE) {
        return;
    }

    int dimension = ctx.vector_dimension > 0 ? ctx.vector_dimension : VECTOR_DIMENSION;
    if (dimension != VECTOR_DIMENSION) {
        dimension = VECTOR_DIMENSION;
    }

    if (params->population_size <= 0) {
        params->population_size = 8;
    }
    if (params->generations <= 0) {
        params->generations = 5;
    }
    if (params->tournament_size <= 0) {
        params->tournament_size = 3;
    }
    if (params->crossover_rate < 0.0 || params->crossover_rate > 1.0) {
        params->crossover_rate = 0.7;
    }
    if (params->mutation_rate < 0.0 || params->mutation_rate > 1.0) {
        params->mutation_rate = 0.02;
    }

    if (params->seed == 0) {
        params->seed = (unsigned int)time(NULL);
        if (params->seed == 0) {
            params->seed = 1;
        }
    }
    ctx.seed = params->seed;

    if (params->max_flip == 0) {
        if (ctx.num_levels > 1) {
            int default_max = dimension / (2 * (ctx.num_levels - 1));
            if (default_max < 1) {
                default_max = 1;
            } else if (default_max > dimension) {
                default_max = dimension;
            }
            if (default_max > (int)UINT16_MAX) {
                default_max = (int)UINT16_MAX;
            }
            params->max_flip = (uint16_t)default_max;
        } else {
            params->max_flip = 0;
        }
    }

    uint32_t ga_state = params->seed ^ 0xA3C59AC3u;
    if (ga_state == 0u) {
        ga_state = 1u;
    }

    int population_size = params->population_size;
    uint16_t *population = (uint16_t *)malloc((size_t)population_size * genome_length * sizeof(uint16_t));
    uint16_t *offspring = (uint16_t *)malloc((size_t)population_size * genome_length * sizeof(uint16_t));
    double *fitness = (double *)malloc((size_t)population_size * sizeof(double));
    uint16_t *best_individual = (uint16_t *)malloc((size_t)genome_length * sizeof(uint16_t));

    if (!population || !offspring || !fitness || !best_individual) {
        fprintf(stderr, "Failed to allocate GA buffers.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < population_size; i++) {
        init_individual(&population[i * genome_length],
                        ctx.num_features,
                        ctx.num_levels,
                        params->max_flip,
                        &ga_state);
    }

    double best_fitness = -1.0;

    for (int gen = 0; gen < params->generations; gen++) {
        for (int i = 0; i < population_size; i++) {
            fitness[i] = evaluate_fitness(&population[i * genome_length], &ctx);
#if OUTPUT_MODE >= OUTPUT_BASIC
            printf("GA gen %d/%d individual %d/%d accuracy: %.3f%%\n",
                   gen + 1,
                   params->generations,
                   i + 1,
                   population_size,
                   fitness[i] * 100.0);
#endif
            if (fitness[i] > best_fitness) {
                best_fitness = fitness[i];
                memcpy(best_individual,
                       &population[i * genome_length],
                       (size_t)genome_length * sizeof(uint16_t));
            }
        }

        if (params->log_every > 0 && (gen % params->log_every == 0)) {
#if OUTPUT_MODE >= OUTPUT_BASIC
            printf("GA generation %d best accuracy: %.3f%%\n", gen, best_fitness * 100.0);
#endif
        }

        if (gen == params->generations - 1) {
            break;
        }

        for (int i = 0; i < population_size; i++) {
            int parent_a = tournament_select(fitness, population_size, params->tournament_size, &ga_state);
            int parent_b = tournament_select(fitness, population_size, params->tournament_size, &ga_state);
            crossover_individual(&population[parent_a * genome_length],
                                 &population[parent_b * genome_length],
                                 &offspring[i * genome_length],
                                 genome_length,
                                 ctx.num_features,
                                 ctx.num_levels,
                                 params->crossover_rate,
                                 &ga_state);
            mutate_individual(&offspring[i * genome_length],
                              ctx.num_features,
                              ctx.num_levels,
                              params->max_flip,
                              params->mutation_rate,
                              &ga_state);
        }

        uint16_t *tmp = population;
        population = offspring;
        offspring = tmp;
    }

    memcpy(B_out, best_individual, (size_t)genome_length * sizeof(uint16_t));

    free(best_individual);
    free(fitness);
    free(offspring);
    free(population);
}

void optimize_item_memory(struct item_memory *signal_mem,
                          struct item_memory *channel_mem,
                          double **training_data,
                          int *training_labels,
                          int training_samples,
                          double **testing_data,
                          int *testing_labels,
                          int testing_samples) {
    if (!signal_mem || !training_data || !training_labels || training_samples <= N_GRAM_SIZE) {
        return;
    }

    int num_features = 1;
    int num_levels = 0;
#if PRECOMPUTED_ITEM_MEMORY
    num_features = NUM_FEATURES;
    if (signal_mem->num_vectors > 0 && num_features > 0) {
        num_levels = signal_mem->num_vectors / num_features;
    }
#else
    if (!channel_mem) {
        return;
    }
    if (signal_mem->num_vectors > 0) {
        num_levels = signal_mem->num_vectors;
    }
#endif
    if (num_levels <= 0) {
        return;
    }

    struct ga_params params;
    init_ga_params(&params);

    struct ga_eval_context ctx;
    ctx.num_features = num_features;
    ctx.num_levels = num_levels;
    ctx.vector_dimension = VECTOR_DIMENSION;
    ctx.seed = params.seed;
    ctx.channel_memory = channel_mem;
    ctx.training_data = training_data;
    ctx.training_labels = training_labels;
    ctx.training_samples = training_samples;
    ctx.testing_data = testing_data;
    ctx.testing_labels = testing_labels;
    ctx.testing_samples = testing_samples;

    int genome_length = num_features * num_levels;
    uint16_t *flip_counts = (uint16_t *)calloc((size_t)genome_length, sizeof(uint16_t));
    if (!flip_counts) {
        fprintf(stderr, "Failed to allocate GA flip matrix.\n");
        exit(EXIT_FAILURE);
    }

    run_ga(&ctx, &params, flip_counts);

    if (signal_mem->base_vectors && signal_mem->num_vectors > 0) {
        free_item_memory(signal_mem);
        signal_mem->base_vectors = NULL;
        signal_mem->num_vectors = 0;
    }

    build_asym_item_memory_from_B(signal_mem,
                                  flip_counts,
                                  num_levels,
                                  num_features,
                                  VECTOR_DIMENSION,
                                  params.seed);

    free(flip_counts);
}
