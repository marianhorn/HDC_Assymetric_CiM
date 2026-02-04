#include "asymItemMemory.h"
#include "assoc_mem.h"
#include "encoder.h"
#include "evaluator.h"
#include "item_mem.h"
#include "operations.h"
#include "trainer.h"
#include "vector.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GA_DEFAULT_POPULATION_SIZE 32 //Default: 12
#define GA_DEFAULT_GENERATIONS 32 //Default: 10
#define GA_DEFAULT_CROSSOVER_RATE 0.1 //Default 0.7
#define GA_DEFAULT_MUTATION_RATE 0.1 //Default 0.02
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

static void generate_permutation(int *perm, int length, uint32_t *rng_state) {
    if (!perm || length <= 0 || !rng_state) {
        return;
    }
    for (int i = 0; i < length; i++) {
        perm[i] = i;
    }
    for (int i = length - 1; i > 0; i--) {
        int j = rng_range(rng_state, i + 1);
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

static void init_individual(uint16_t *individual,
                            int transitions,
                            uint16_t max_flip,
                            int max_total,
                            uint32_t *rng_state,
                            const int *permutation,
                            int permutation_length) {
    if (!individual || transitions <= 0) {
        return;
    }
    if (max_total < 0) {
        max_total = 0;
    }

    for (int i = 0; i < transitions; i++) {
        individual[i] = 0;
    }

    int *order = NULL;
    if (permutation && permutation_length >= transitions) {
        order = (int *)malloc((size_t)transitions * sizeof(int));
        if (order) {
            int count = 0;
            for (int i = 0; i < permutation_length && count < transitions; i++) {
                int value = permutation[i];
                if (value >= 0 && value < transitions) {
                    order[count++] = value;
                }
            }
            if (count < transitions) {
                free(order);
                order = NULL;
            }
        }
    }

    int remaining = max_total;
    for (int i = 0; i < transitions; i++) {
        int level = order ? order[i] : i;
        int limit = (int)max_flip;
        if (limit > remaining) {
            limit = remaining;
        }
        int value = limit > 0 ? rng_range(rng_state, limit + 1) : 0;
        individual[level] = (uint16_t)value;
        remaining -= value;
        if (remaining <= 0) {
            break;
        }
    }

    free(order);
}


struct ga_eval_context {
    int num_features;
    int num_levels;
    int vector_dimension;
    unsigned int seed;
    const int *permutations;
    struct item_memory *channel_memory;
    double **training_data;
    int *training_labels;
    int training_samples;
    double **testing_data;
    int *testing_labels;
    int testing_samples;
};

static double evaluate_candidate(const uint16_t *B,
                                 const struct ga_eval_context *ctx,
                                 double *out_accuracy,
                                 double *out_similarity) {
    if (!ctx || !ctx->training_data || !ctx->training_labels || ctx->training_samples <= N_GRAM_SIZE) {
        if (out_accuracy) {
            *out_accuracy = 0.0;
        }
        if (out_similarity) {
            *out_similarity = 0.0;
        }
        return 0.0;
    }

#if PRECOMPUTED_ITEM_MEMORY
    int transitions = ctx->num_levels - 1;
    int genes = transitions * ctx->num_features;
    int *b_matrix = NULL;
    if (genes > 0) {
        b_matrix = (int *)malloc((size_t)genes * sizeof(int));
        if (!b_matrix) {
            fprintf(stderr, "Failed to allocate flip matrix.\n");
            if (out_accuracy) {
                *out_accuracy = 0.0;
            }
            if (out_similarity) {
                *out_similarity = 0.0;
            }
            return 0.0;
        }
        for (int i = 0; i < genes; i++) {
            b_matrix[i] = (int)B[i];
        }
    }

    if (!ctx->permutations) {
        free(b_matrix);
        if (out_accuracy) {
            *out_accuracy = 0.0;
        }
        if (out_similarity) {
            *out_similarity = 0.0;
        }
        return 0.0;
    }

    struct associative_memory assoc_mem;
    init_assoc_mem(&assoc_mem);
    struct item_memory item_mem;
    init_precomp_item_memory_with_B(&item_mem,
                                    ctx->num_levels,
                                    ctx->num_features,
                                    b_matrix,
                                    ctx->permutations);
    struct encoder enc;
    init_encoder(&enc, &item_mem);
    train_model_timeseries(ctx->training_data,
                           ctx->training_labels,
                           ctx->training_samples,
                           &assoc_mem,
                           &enc);

    double **eval_data = ctx->training_data;
    int *eval_labels = ctx->training_labels;
    int eval_samples = ctx->training_samples;
    if (ctx->testing_data && ctx->testing_labels && ctx->testing_samples > 0) {
        eval_data = ctx->testing_data;
        eval_labels = ctx->testing_labels;
        eval_samples = ctx->testing_samples;
    }

    struct timeseries_eval_result eval_result =
        evaluate_model_timeseries_direct(&enc, &assoc_mem, eval_data, eval_labels, eval_samples);
    if (out_accuracy) {
        *out_accuracy = eval_result.class_average_accuracy;
    }
    if (out_similarity) {
        *out_similarity = eval_result.class_vector_similarity;
    }
    double fitness = eval_result.class_average_accuracy - eval_result.class_vector_similarity;

    free_item_memory(&item_mem);
    free_assoc_mem(&assoc_mem);
    free(b_matrix);
    return fitness;
#else
    int transitions = ctx->num_levels - 1;
    int *b_levels = NULL;
    if (transitions > 0) {
        b_levels = (int *)malloc((size_t)transitions * sizeof(int));
        if (!b_levels) {
            fprintf(stderr, "Failed to allocate flip vector.\n");
            if (out_accuracy) {
                *out_accuracy = 0.0;
            }
            if (out_similarity) {
                *out_similarity = 0.0;
            }
            return 0.0;
        }
        for (int level = 0; level < transitions; level++) {
            b_levels[level] = (int)B[level];
        }
    }

    if (!ctx->channel_memory || !ctx->permutations) {
        free(b_levels);
        if (out_accuracy) {
            *out_accuracy = 0.0;
        }
        if (out_similarity) {
            *out_similarity = 0.0;
        }
        return 0.0;
    }
    struct associative_memory assoc_mem;
    init_assoc_mem(&assoc_mem);
    struct encoder enc;
    struct item_memory signal_mem;
    init_continuous_item_memory_with_B(&signal_mem,
                                       ctx->num_levels,
                                       b_levels,
                                       ctx->permutations);
    init_encoder(&enc, ctx->channel_memory, &signal_mem);
    train_model_timeseries(ctx->training_data,
                           ctx->training_labels,
                           ctx->training_samples,
                           &assoc_mem,
                           &enc);

    double **eval_data = ctx->training_data;
    int *eval_labels = ctx->training_labels;
    int eval_samples = ctx->training_samples;
    if (ctx->testing_data && ctx->testing_labels && ctx->testing_samples > 0) {
        eval_data = ctx->testing_data;
        eval_labels = ctx->testing_labels;
        eval_samples = ctx->testing_samples;
    }

    struct timeseries_eval_result eval_result =
        evaluate_model_timeseries_direct(&enc, &assoc_mem, eval_data, eval_labels, eval_samples);
    double accuracy = eval_result.class_average_accuracy;
    double similarity = eval_result.class_vector_similarity;
    if (out_accuracy) {
        *out_accuracy = accuracy;
    }
    if (out_similarity) {
        *out_similarity = similarity;
    }

    free(b_levels);
    free_assoc_mem(&assoc_mem);
    free_item_memory(&signal_mem);
    return accuracy - similarity;
#endif
}

static void mutate_individual(uint16_t *individual,
                              int gene_count,
                              uint16_t max_flip,
                              double mutation_rate,
                              uint32_t *rng_state) {
    for (int level = 0; level < gene_count; level++) {
        if (rng_uniform(rng_state) < mutation_rate) {
            int delta = rng_range(rng_state, 2) == 0 ? -1 : 1;
            int value = (int)individual[level] + delta;
            if (value < 0) {
                value = 0;
            } else if (value > (int)max_flip) {
                value = (int)max_flip;
            }
            individual[level] = (uint16_t)value;
        }
    }
}

static void crossover_individual(const uint16_t *parent_a,
                                 const uint16_t *parent_b,
                                 uint16_t *child,
                                 int genome_length,
                                 double crossover_rate,
                                 uint32_t *rng_state) {
    if (rng_uniform(rng_state) < crossover_rate) {
        for (int i = 0; i < genome_length; i++) {
            child[i] = rng_range(rng_state, 2) == 0 ? parent_a[i] : parent_b[i];
        }
    } else {
        memcpy(child, parent_a, genome_length * sizeof(uint16_t));
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
    if (ctx_in->num_levels <= 1) {
        return;
    }

    struct ga_eval_context ctx = *ctx_in;
    int ga_output_mode = output_mode;
    int genome_length = ctx.num_levels - 1;
#if PRECOMPUTED_ITEM_MEMORY
    genome_length *= ctx.num_features;
#endif
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
    double *accuracies = (double *)malloc((size_t)population_size * sizeof(double));
    double *similarities = (double *)malloc((size_t)population_size * sizeof(double));
    uint16_t *best_individual = (uint16_t *)malloc((size_t)genome_length * sizeof(uint16_t));

    if (!population || !offspring || !fitness || !best_individual || !accuracies || !similarities) {
        fprintf(stderr, "Failed to allocate GA buffers.\n");
        exit(EXIT_FAILURE);
    }

    int transitions = ctx.num_levels - 1;
    int max_total = dimension / 2;
    for (int i = 0; i < population_size; i++) {
        uint16_t *individual = &population[i * genome_length];
#if PRECOMPUTED_ITEM_MEMORY
        for (int feature = 0; feature < ctx.num_features; feature++) {
            init_individual(individual + feature * transitions,
                            transitions,
                            params->max_flip,
                            max_total,
                            &ga_state,
                            ctx.permutations + (size_t)feature * VECTOR_DIMENSION,
                            VECTOR_DIMENSION);
        }
#else
        init_individual(individual,
                        transitions,
                        params->max_flip,
                        max_total,
                        &ga_state,
                        ctx.permutations,
                        VECTOR_DIMENSION);
#endif
    }

    double best_fitness = -1.0;
    int best_gen = -1;
    int best_gen_index = -1;

    if (ga_output_mode >= OUTPUT_DETAILED) {
#ifdef _OPENMP
        printf("GA evaluating with %d threads\n", omp_get_max_threads());
#else
        printf("GA evaluating with 1 thread\n");
#endif
    }

    for (int gen = 0; gen < params->generations; gen++) {
        if (ga_output_mode >= OUTPUT_BASIC) {
            printf("GA generation %d/%d\n", gen + 1, params->generations);
        }

        output_mode = OUTPUT_NONE;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < population_size; i++) {
            fitness[i] = evaluate_candidate(&population[i * genome_length],
                                            &ctx,
                                            &accuracies[i],
                                            &similarities[i]
            );
        }
        output_mode = ga_output_mode;
        if (ga_output_mode >= OUTPUT_BASIC) {
            for (int i = 0; i < population_size; i++) {
                printf("  individual %d/%d accuracy: %.3f%%, similarity: %.3f, fitness: %.3f\n",
                       i + 1,
                       population_size,
                       accuracies[i] * 100.0,
                       similarities[i],
                       fitness[i]);
            }
        }

        double gen_best = -1.0;
        int gen_best_index = 0;
        for (int i = 0; i < population_size; i++) {
            if (fitness[i] > gen_best) {
                gen_best = fitness[i];
                gen_best_index = i;
            }
        }
        if (gen_best > best_fitness) {
            best_fitness = gen_best;
            best_gen = gen;
            best_gen_index = gen_best_index;
            memcpy(best_individual,
                   &population[gen_best_index * genome_length],
                   (size_t)genome_length * sizeof(uint16_t));
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
                                 params->crossover_rate,
                                 &ga_state);
            mutate_individual(&offspring[i * genome_length],
                              genome_length,
                              params->max_flip,
                              params->mutation_rate,
                              &ga_state);
        }

        uint16_t *tmp = population;
        population = offspring;
        offspring = tmp;
    }

    memcpy(B_out, best_individual, (size_t)genome_length * sizeof(uint16_t));

    if (ga_output_mode >= OUTPUT_DETAILED && best_gen >= 0 && best_gen_index >= 0) {
        printf("GA winner: generation %d, individual %d\n", best_gen + 1, best_gen_index + 1);
    }

    free(similarities);
    free(accuracies);
    free(best_individual);
    free(fitness);
    free(offspring);
    free(population);
}

#if PRECOMPUTED_ITEM_MEMORY
void optimize_item_memory(struct item_memory *item_mem,
                          double **training_data,
                          int *training_labels,
                          int training_samples,
                          double **testing_data,
                          int *testing_labels,
                          int testing_samples) {
    if (!item_mem || !training_data || !training_labels || training_samples <= N_GRAM_SIZE) {
        return;
    }

    int num_features = NUM_FEATURES;
    int num_levels = 0;
    if (item_mem->num_vectors > 0 && num_features > 0) {
        num_levels = item_mem->num_vectors / num_features;
    }
    if (num_levels <= 1) {
        return;
    }

    struct ga_params params;
    init_ga_params(&params);

    if (params.seed == 0) {
        params.seed = (unsigned int)time(NULL);
        if (params.seed == 0) {
            params.seed = 1;
        }
    }

    int *permutations = (int *)malloc((size_t)num_features * VECTOR_DIMENSION * sizeof(int));
    if (!permutations) {
        fprintf(stderr, "Failed to allocate permutations.\n");
        exit(EXIT_FAILURE);
    }
    uint32_t perm_state = params.seed ^ 0x9E3779B9u;
    if (perm_state == 0u) {
        perm_state = 1u;
    }
    for (int feature = 0; feature < num_features; feature++) {
        generate_permutation(permutations + (size_t)feature * VECTOR_DIMENSION,
                             VECTOR_DIMENSION,
                             &perm_state);
    }

    struct ga_eval_context ctx;
    ctx.num_features = num_features;
    ctx.num_levels = num_levels;
    ctx.vector_dimension = VECTOR_DIMENSION;
    ctx.seed = params.seed;
    ctx.permutations = permutations;
    ctx.channel_memory = NULL;
    ctx.training_data = training_data;
    ctx.training_labels = training_labels;
    ctx.training_samples = training_samples;
    ctx.testing_data = testing_data;
    ctx.testing_labels = testing_labels;
    ctx.testing_samples = testing_samples;

    int genome_length = (num_levels - 1) * num_features;
    uint16_t *flip_counts = (uint16_t *)calloc((size_t)genome_length, sizeof(uint16_t));
    if (!flip_counts) {
        fprintf(stderr, "Failed to allocate GA flip matrix.\n");
        exit(EXIT_FAILURE);
    }

    run_ga(&ctx, &params, flip_counts);

    if (item_mem->base_vectors && item_mem->num_vectors > 0) {
        free_item_memory(item_mem);
        item_mem->base_vectors = NULL;
        item_mem->num_vectors = 0;
    }

    int *b_matrix = NULL;
    if (genome_length > 0) {
        b_matrix = (int *)malloc((size_t)genome_length * sizeof(int));
        if (!b_matrix) {
            fprintf(stderr, "Failed to allocate flip matrix.\n");
            free(flip_counts);
            free(permutations);
            return;
        }
        for (int i = 0; i < genome_length; i++) {
            b_matrix[i] = (int)flip_counts[i];
        }
    }
    init_precomp_item_memory_with_B(item_mem,
                                    num_levels,
                                    num_features,
                                    b_matrix,
                                    permutations);
    free(b_matrix);
    free(flip_counts);
    free(permutations);
}
#else
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

    int num_levels = 0;
    if (!channel_mem) {
        return;
    }
    if (signal_mem->num_vectors > 0) {
        num_levels = signal_mem->num_vectors;
    }
    if (num_levels <= 1) {
        return;
    }

    struct ga_params params;
    init_ga_params(&params);

    if (params.seed == 0) {
        params.seed = (unsigned int)time(NULL);
        if (params.seed == 0) {
            params.seed = 1;
        }
    }

    int *permutation = (int *)malloc((size_t)VECTOR_DIMENSION * sizeof(int));
    if (!permutation) {
        fprintf(stderr, "Failed to allocate permutation.\n");
        exit(EXIT_FAILURE);
    }
    uint32_t perm_state = params.seed ^ 0x9E3779B9u;
    if (perm_state == 0u) {
        perm_state = 1u;
    }
    generate_permutation(permutation, VECTOR_DIMENSION, &perm_state);

    struct ga_eval_context ctx;
    ctx.num_features = 1;
    ctx.num_levels = num_levels;
    ctx.vector_dimension = VECTOR_DIMENSION;
    ctx.seed = params.seed;
    ctx.permutations = permutation;
    ctx.channel_memory = channel_mem;
    ctx.training_data = training_data;
    ctx.training_labels = training_labels;
    ctx.training_samples = training_samples;
    ctx.testing_data = testing_data;
    ctx.testing_labels = testing_labels;
    ctx.testing_samples = testing_samples;

    int genome_length = num_levels - 1;
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

    int transitions = num_levels - 1;
    int *b_levels = NULL;
    if (transitions > 0) {
        b_levels = (int *)malloc((size_t)transitions * sizeof(int));
        if (!b_levels) {
            fprintf(stderr, "Failed to allocate flip vector.\n");
            free(flip_counts);
            free(permutation);
            return;
        }
        for (int level = 0; level < transitions; level++) {
            b_levels[level] = (int)flip_counts[level];
        }
    }

    init_continuous_item_memory_with_B(signal_mem,
                                       num_levels,
                                       b_levels,
                                       permutation);

    free(b_levels);
    free(flip_counts);
    free(permutation);
}
#endif
