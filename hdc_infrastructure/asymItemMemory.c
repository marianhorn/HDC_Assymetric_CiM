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

#if GA_INIT_UNIFORM
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

    int total = max_total;
    if (total <= 0) {
        free(order);
        return;
    }

    double *weights = (double *)malloc((size_t)transitions * sizeof(double));
    int *values = (int *)malloc((size_t)transitions * sizeof(int));
    if (!weights || !values) {
        free(weights);
        free(values);
        free(order);
        return;
    }

    double sum_weights = 0.0;
    for (int i = 0; i < transitions; i++) {
        weights[i] = rng_uniform(rng_state);
        sum_weights += weights[i];
    }
    if (sum_weights <= 0.0) {
        weights[0] = 1.0;
        sum_weights = 1.0;
    }

    int assigned = 0;
    for (int i = 0; i < transitions; i++) {
        double scaled = (weights[i] / sum_weights) * (double)total;
        int value = (int)scaled;
        values[i] = value;
        assigned += value;
    }

    int remaining = total - assigned;
    while (remaining > 0) {
        int idx = rng_range(rng_state, transitions);
        values[idx] += 1;
        remaining--;
    }

    for (int i = 0; i < transitions; i++) {
        int level = order ? order[i] : i;
        individual[level] = (uint16_t)values[i];
    }

    free(values);
    free(weights);
    free(order);
#else
    (void)rng_state;
    (void)permutation;
    (void)permutation_length;

    int prev_target = 0;
    for (int level = 0; level < transitions; level++) {
        double exact = ((double)(level + 1) * (double)max_total) / (double)transitions;
        int target = (int)(exact + 0.5);
        if (target < 0) {
            target = 0;
        } else if (target > max_total) {
            target = max_total;
        }

        int flips = target - prev_target;
        if (flips < 0) {
            flips = 0;
        } else if (flips > (int)UINT16_MAX) {
            flips = (int)UINT16_MAX;
        }
        individual[level] = (uint16_t)flips;
        prev_target = target;
    }
#endif
}

static int dominates(double acc_a, double sim_a, double acc_b, double sim_b) {
    return (acc_a >= acc_b && sim_a <= sim_b) && (acc_a > acc_b || sim_a < sim_b);
}

static void sort_indices_by_value(int *indices, int count, const double *values) {
    for (int i = 1; i < count; i++) {
        int key = indices[i];
        int j = i - 1;
        while (j >= 0 && values[indices[j]] > values[key]) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

static void sort_indices_by_value_desc(int *indices, int count, const double *values) {
    for (int i = 1; i < count; i++) {
        int key = indices[i];
        int j = i - 1;
        while (j >= 0 && values[indices[j]] < values[key]) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

static void compute_crowding(const double *acc,
                             const double *sim,
                             const int *fronts,
                             int start,
                             int end,
                             double *crowd) {
    int size = end - start;
    if (size <= 0) {
        return;
    }

    for (int i = start; i < end; i++) {
        crowd[fronts[i]] = 0.0;
    }

    if (size <= 2) {
        for (int i = start; i < end; i++) {
            crowd[fronts[i]] = 1e9;
        }
        return;
    }

    int *indices = (int *)malloc((size_t)size * sizeof(int));
    if (!indices) {
        for (int i = start; i < end; i++) {
            crowd[fronts[i]] = 1e9;
        }
        return;
    }

    for (int i = 0; i < size; i++) {
        indices[i] = fronts[start + i];
    }
    sort_indices_by_value(indices, size, acc);
    crowd[indices[0]] = 1e9;
    crowd[indices[size - 1]] = 1e9;
    double min_val = acc[indices[0]];
    double max_val = acc[indices[size - 1]];
    if (max_val > min_val) {
        for (int i = 1; i < size - 1; i++) {
            crowd[indices[i]] += (acc[indices[i + 1]] - acc[indices[i - 1]]) / (max_val - min_val);
        }
    }

    for (int i = 0; i < size; i++) {
        indices[i] = fronts[start + i];
    }
    sort_indices_by_value(indices, size, sim);
    crowd[indices[0]] = 1e9;
    crowd[indices[size - 1]] = 1e9;
    min_val = sim[indices[0]];
    max_val = sim[indices[size - 1]];
    if (max_val > min_val) {
        for (int i = 1; i < size - 1; i++) {
            crowd[indices[i]] += (sim[indices[i + 1]] - sim[indices[i - 1]]) / (max_val - min_val);
        }
    }

    free(indices);
}

static void non_dominated_sort(const double *acc,
                               const double *sim,
                               int count,
                               int *rank,
                               int *fronts,
                               int *front_offsets,
                               int *num_fronts) {
    int *dom_count = (int *)calloc((size_t)count, sizeof(int));
    int *dominated_counts = (int *)calloc((size_t)count, sizeof(int));
    int *dominated = (int *)malloc((size_t)count * (size_t)count * sizeof(int));
    int *current = (int *)malloc((size_t)count * sizeof(int));
    int *next = (int *)malloc((size_t)count * sizeof(int));

    if (!dom_count || !dominated_counts || !dominated || !current || !next) {
        free(dom_count);
        free(dominated_counts);
        free(dominated);
        free(current);
        free(next);
        if (num_fronts) {
            *num_fronts = 0;
        }
        return;
    }

    for (int p = 0; p < count; p++) {
        for (int q = 0; q < count; q++) {
            if (p == q) {
                continue;
            }
            if (dominates(acc[p], sim[p], acc[q], sim[q])) {
                dominated[p * count + dominated_counts[p]++] = q;
            } else if (dominates(acc[q], sim[q], acc[p], sim[p])) {
                dom_count[p]++;
            }
        }
    }

    int current_size = 0;
    for (int p = 0; p < count; p++) {
        if (dom_count[p] == 0) {
            rank[p] = 0;
            current[current_size++] = p;
        }
    }

    int front_index = 0;
    int filled = 0;
    front_offsets[0] = 0;
    while (current_size > 0) {
        for (int i = 0; i < current_size; i++) {
            fronts[filled++] = current[i];
        }
        front_offsets[front_index + 1] = filled;

        int next_size = 0;
        for (int i = 0; i < current_size; i++) {
            int p = current[i];
            int dominated_count = dominated_counts[p];
            int *dominated_list = &dominated[p * count];
            for (int j = 0; j < dominated_count; j++) {
                int q = dominated_list[j];
                dom_count[q]--;
                if (dom_count[q] == 0) {
                    rank[q] = front_index + 1;
                    next[next_size++] = q;
                }
            }
        }

        current_size = next_size;
        int *tmp = current;
        current = next;
        next = tmp;
        front_index++;
    }

    if (num_fronts) {
        *num_fronts = front_index;
    }

    free(dom_count);
    free(dominated_counts);
    free(dominated);
    free(current);
    free(next);
}

static int nsga2_better(int a, int b, const int *rank, const double *crowd, uint32_t *rng_state) {
    if (rank[a] < rank[b]) {
        return a;
    }
    if (rank[a] > rank[b]) {
        return b;
    }
    if (crowd[a] > crowd[b]) {
        return a;
    }
    if (crowd[a] < crowd[b]) {
        return b;
    }
    return rng_range(rng_state, 2) == 0 ? a : b;
}

static int nsga2_tournament(const int *rank,
                            const double *crowd,
                            int population_size,
                            int tournament_size,
                            uint32_t *rng_state) {
    if (population_size <= 0) {
        return 0;
    }
    if (tournament_size <= 1) {
        tournament_size = 2;
    }
    int best = rng_range(rng_state, population_size);
    for (int i = 1; i < tournament_size; i++) {
        int challenger = rng_range(rng_state, population_size);
        best = nsga2_better(best, challenger, rank, crowd, rng_state);
    }
    return best;
}

static double compute_scalar_fitness(int selection_mode, double accuracy, double similarity) {
    if (selection_mode == GA_SELECTION_MULTI) {
        return accuracy - similarity;
    }
    return accuracy;
}

static int fitness_better(int a, int b, const double *fitness, uint32_t *rng_state) {
    if (fitness[a] > fitness[b]) {
        return a;
    }
    if (fitness[a] < fitness[b]) {
        return b;
    }
    return rng_range(rng_state, 2) == 0 ? a : b;
}

static int fitness_tournament(const double *fitness,
                              int population_size,
                              int tournament_size,
                              uint32_t *rng_state) {
    if (population_size <= 0) {
        return 0;
    }
    if (tournament_size <= 1) {
        tournament_size = 2;
    }
    int best = rng_range(rng_state, population_size);
    for (int i = 1; i < tournament_size; i++) {
        int challenger = rng_range(rng_state, population_size);
        best = fitness_better(best, challenger, fitness, rng_state);
    }
    return best;
}

static void select_next_population_pareto(const uint16_t *population,
                                          const uint16_t *offspring,
                                          int population_size,
                                          int genome_length,
                                          const double *accP,
                                          const double *simP,
                                          const double *accQ,
                                          const double *simQ,
                                          uint16_t *next_population,
                                          double *next_acc,
                                          double *next_sim,
                                          uint16_t *combined,
                                          double *accR,
                                          double *simR,
                                          int *rankR,
                                          double *crowdR,
                                          int *fronts,
                                          int *front_offsets) {
    int combined_count = population_size * 2;

    memcpy(combined,
           population,
           (size_t)population_size * (size_t)genome_length * sizeof(uint16_t));
    memcpy(combined + (size_t)population_size * (size_t)genome_length,
           offspring,
           (size_t)population_size * (size_t)genome_length * sizeof(uint16_t));
    for (int i = 0; i < population_size; i++) {
        accR[i] = accP[i];
        simR[i] = simP[i];
        accR[population_size + i] = accQ[i];
        simR[population_size + i] = simQ[i];
    }

    int num_fronts = 0;
    non_dominated_sort(accR, simR, combined_count, rankR, fronts, front_offsets, &num_fronts);
    for (int f = 0; f < num_fronts; f++) {
        compute_crowding(accR, simR, fronts, front_offsets[f], front_offsets[f + 1], crowdR);
    }

    int filled = 0;
    for (int f = 0; f < num_fronts && filled < population_size; f++) {
        int start = front_offsets[f];
        int end = front_offsets[f + 1];
        int front_size = end - start;
        if (filled + front_size <= population_size) {
            for (int i = start; i < end; i++) {
                int idx = fronts[i];
                memcpy(&next_population[(size_t)filled * genome_length],
                       &combined[(size_t)idx * genome_length],
                       (size_t)genome_length * sizeof(uint16_t));
                next_acc[filled] = accR[idx];
                next_sim[filled] = simR[idx];
                filled++;
            }
        } else {
            int remaining = population_size - filled;
            int *front_indices = (int *)malloc((size_t)front_size * sizeof(int));
            if (!front_indices) {
                for (int i = 0; i < remaining && (start + i) < end; i++) {
                    int idx = fronts[start + i];
                    memcpy(&next_population[(size_t)filled * genome_length],
                           &combined[(size_t)idx * genome_length],
                           (size_t)genome_length * sizeof(uint16_t));
                    next_acc[filled] = accR[idx];
                    next_sim[filled] = simR[idx];
                    filled++;
                }
                break;
            }
            for (int i = 0; i < front_size; i++) {
                front_indices[i] = fronts[start + i];
            }
            sort_indices_by_value_desc(front_indices, front_size, crowdR);
            for (int i = 0; i < remaining; i++) {
                int idx = front_indices[i];
                memcpy(&next_population[(size_t)filled * genome_length],
                       &combined[(size_t)idx * genome_length],
                       (size_t)genome_length * sizeof(uint16_t));
                next_acc[filled] = accR[idx];
                next_sim[filled] = simR[idx];
                filled++;
            }
            free(front_indices);
        }
    }
}

static void select_next_population_scalar(const uint16_t *population,
                                          const uint16_t *offspring,
                                          int population_size,
                                          int genome_length,
                                          const double *accP,
                                          const double *simP,
                                          const double *fitP,
                                          const double *accQ,
                                          const double *simQ,
                                          const double *fitQ,
                                          uint16_t *next_population,
                                          double *next_acc,
                                          double *next_sim,
                                          double *next_fit,
                                          uint16_t *combined,
                                          double *accR,
                                          double *simR,
                                          double *fitR,
                                          int *indices) {
    int combined_count = population_size * 2;

    memcpy(combined,
           population,
           (size_t)population_size * (size_t)genome_length * sizeof(uint16_t));
    memcpy(combined + (size_t)population_size * (size_t)genome_length,
           offspring,
           (size_t)population_size * (size_t)genome_length * sizeof(uint16_t));
    for (int i = 0; i < population_size; i++) {
        accR[i] = accP[i];
        simR[i] = simP[i];
        fitR[i] = fitP[i];
        accR[population_size + i] = accQ[i];
        simR[population_size + i] = simQ[i];
        fitR[population_size + i] = fitQ[i];
    }

    for (int i = 0; i < combined_count; i++) {
        indices[i] = i;
    }
    sort_indices_by_value_desc(indices, combined_count, fitR);
    for (int i = 0; i < population_size; i++) {
        int idx = indices[i];
        memcpy(&next_population[(size_t)i * genome_length],
               &combined[(size_t)idx * genome_length],
               (size_t)genome_length * sizeof(uint16_t));
        next_acc[i] = accR[idx];
        next_sim[i] = simR[idx];
        next_fit[i] = fitR[idx];
    }
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
                              double mutation_rate,
                              uint32_t *rng_state) {
    if (!individual || gene_count <= 1) {
        return;
    }

    for (int step = 0; step < gene_count; step++) {
        if (rng_uniform(rng_state) < mutation_rate) {
            int donor = -1;
            int max_tries = gene_count * 2;
            for (int tries = 0; tries < max_tries; tries++) {
                int idx = rng_range(rng_state, gene_count);
                if (individual[idx] > 0) {
                    donor = idx;
                    break;
                }
            }
            if (donor < 0) {
                continue;
            }

            int receiver = rng_range(rng_state, gene_count);
            if (receiver == donor && gene_count > 1) {
                receiver = (donor + 1 + rng_range(rng_state, gene_count - 1)) % gene_count;
            }

            individual[donor] -= 1;
            individual[receiver] += 1;
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
    int selection_mode = GA_SELECTION_MODE;
    if (selection_mode != GA_SELECTION_PARETO &&
        selection_mode != GA_SELECTION_MULTI &&
        selection_mode != GA_SELECTION_ACCURACY) {
        selection_mode = GA_SELECTION_PARETO;
    }
    int genome_length = ctx.num_levels - 1;
#if PRECOMPUTED_ITEM_MEMORY
    genome_length *= ctx.num_features;
#endif
    memset(B_out, 0, (size_t)genome_length * sizeof(uint16_t));

    if (!ctx.training_data || !ctx.training_labels || ctx.training_samples <= N_GRAM_SIZE) {
        return;
    }

    int dimension = VECTOR_DIMENSION;


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

    uint32_t ga_state = params->seed ^ 0xA3C59AC3u;
    if (ga_state == 0u) {
        ga_state = 1u;
    }

    int population_size = params->population_size;
    uint16_t *population = (uint16_t *)malloc((size_t)population_size * genome_length * sizeof(uint16_t));
    uint16_t *offspring = (uint16_t *)malloc((size_t)population_size * genome_length * sizeof(uint16_t));
    uint16_t *combined = (uint16_t *)malloc((size_t)(population_size * 2) * genome_length * sizeof(uint16_t));
    double *accP = (double *)malloc((size_t)population_size * sizeof(double));
    double *simP = (double *)malloc((size_t)population_size * sizeof(double));
    double *fitP = (double *)malloc((size_t)population_size * sizeof(double));
    int *rankP = (int *)malloc((size_t)population_size * sizeof(int));
    double *crowdP = (double *)malloc((size_t)population_size * sizeof(double));
    double *accQ = (double *)malloc((size_t)population_size * sizeof(double));
    double *simQ = (double *)malloc((size_t)population_size * sizeof(double));
    double *fitQ = (double *)malloc((size_t)population_size * sizeof(double));
    double *accR = (double *)malloc((size_t)(population_size * 2) * sizeof(double));
    double *simR = (double *)malloc((size_t)(population_size * 2) * sizeof(double));
    double *fitR = (double *)malloc((size_t)(population_size * 2) * sizeof(double));
    int *rankR = (int *)malloc((size_t)(population_size * 2) * sizeof(int));
    double *crowdR = (double *)malloc((size_t)(population_size * 2) * sizeof(double));
    int *fronts = (int *)malloc((size_t)(population_size * 2) * sizeof(int));
    int *front_offsets = (int *)malloc((size_t)(population_size * 2 + 1) * sizeof(int));

    if (!population || !offspring || !combined || !accP || !simP || !fitP || !rankP || !crowdP ||
        !accQ || !simQ || !fitQ || !accR || !simR || !fitR || !rankR || !crowdR || !fronts || !front_offsets) {
        fprintf(stderr, "Failed to allocate GA buffers.\n");
        exit(EXIT_FAILURE);
    }

    int transitions = ctx.num_levels - 1;
    int max_total = GA_MAX_FLIPS_CIM;
    for (int i = 0; i < population_size; i++) {
        uint16_t *individual = &population[i * genome_length];
#if PRECOMPUTED_ITEM_MEMORY
        for (int feature = 0; feature < ctx.num_features; feature++) {
            init_individual(individual + feature * transitions,
                            transitions,
                            max_total,
                            &ga_state,
                            ctx.permutations + (size_t)feature * VECTOR_DIMENSION,
                            VECTOR_DIMENSION);
        }
#else
        init_individual(individual,
                        transitions,
                        max_total,
                        &ga_state,
                        ctx.permutations,
                        VECTOR_DIMENSION);
#endif
    }

    double best_acc = -1.0;
    double best_sim = 0.0;
    double best_score = -1e9;
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
            (void)evaluate_candidate(&population[i * genome_length],
                                     &ctx,
                                     &accP[i],
                                     &simP[i]);
        }
        output_mode = ga_output_mode;
        if (ga_output_mode >= OUTPUT_BASIC) {
            for (int i = 0; i < population_size; i++) {
                printf("  individual %d/%d accuracy: %.3f%%, similarity: %.3f\n",
                       i + 1,
                       population_size,
                       accP[i] * 100.0,
                       simP[i]);
            }
        }

        if (selection_mode == GA_SELECTION_PARETO) {
            int num_fronts = 0;
            non_dominated_sort(accP, simP, population_size, rankP, fronts, front_offsets, &num_fronts);
            for (int f = 0; f < num_fronts; f++) {
                compute_crowding(accP, simP, fronts, front_offsets[f], front_offsets[f + 1], crowdP);
            }

            if (num_fronts > 0) {
                int start = front_offsets[0];
                int end = front_offsets[1];
                for (int i = start; i < end; i++) {
                    int idx = fronts[i];
                    if (accP[idx] > best_acc) {
                        best_acc = accP[idx];
                        best_sim = simP[idx];
                        best_gen = gen;
                        best_gen_index = idx;
                    }
                }
            }
        } else {
            for (int i = 0; i < population_size; i++) {
                fitP[i] = compute_scalar_fitness(selection_mode, accP[i], simP[i]);
                if (fitP[i] > best_score) {
                    best_score = fitP[i];
                    best_acc = accP[i];
                    best_sim = simP[i];
                    best_gen = gen;
                    best_gen_index = i;
                }
            }
        }

        for (int i = 0; i < population_size; i++) {
            int parent_a = 0;
            int parent_b = 0;
            if (selection_mode == GA_SELECTION_PARETO) {
                parent_a = nsga2_tournament(rankP, crowdP, population_size, params->tournament_size, &ga_state);
                parent_b = nsga2_tournament(rankP, crowdP, population_size, params->tournament_size, &ga_state);
            } else {
                parent_a = fitness_tournament(fitP, population_size, params->tournament_size, &ga_state);
                parent_b = fitness_tournament(fitP, population_size, params->tournament_size, &ga_state);
            }
            crossover_individual(&population[parent_a * genome_length],
                                 &population[parent_b * genome_length],
                                 &offspring[i * genome_length],
                                 genome_length,
                                 params->crossover_rate,
                                 &ga_state);
            mutate_individual(&offspring[i * genome_length],
                              genome_length,
                              params->mutation_rate,
                              &ga_state);
        }

        output_mode = OUTPUT_NONE;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < population_size; i++) {
            (void)evaluate_candidate(&offspring[i * genome_length],
                                     &ctx,
                                     &accQ[i],
                                     &simQ[i]);
        }
        output_mode = ga_output_mode;

        if (selection_mode == GA_SELECTION_PARETO) {
            select_next_population_pareto(population,
                                          offspring,
                                          population_size,
                                          genome_length,
                                          accP,
                                          simP,
                                          accQ,
                                          simQ,
                                          population,
                                          accP,
                                          simP,
                                          combined,
                                          accR,
                                          simR,
                                          rankR,
                                          crowdR,
                                          fronts,
                                          front_offsets);
        } else {
            for (int i = 0; i < population_size; i++) {
                fitQ[i] = compute_scalar_fitness(selection_mode, accQ[i], simQ[i]);
            }
            select_next_population_scalar(population,
                                          offspring,
                                          population_size,
                                          genome_length,
                                          accP,
                                          simP,
                                          fitP,
                                          accQ,
                                          simQ,
                                          fitQ,
                                          population,
                                          accP,
                                          simP,
                                          fitP,
                                          combined,
                                          accR,
                                          simR,
                                          fitR,
                                          fronts);
        }
    }

    int best_idx = 0;
    if (selection_mode == GA_SELECTION_PARETO) {
        int num_fronts = 0;
        non_dominated_sort(accP, simP, population_size, rankP, fronts, front_offsets, &num_fronts);
        double best_final_acc = -1.0;
        for (int i = 0; i < population_size; i++) {
            if (rankP[i] == 0 && accP[i] > best_final_acc) {
                best_final_acc = accP[i];
                best_idx = i;
            }
        }
    } else if (selection_mode == GA_SELECTION_MULTI) {
        double best_final_score = -1e9;
        for (int i = 0; i < population_size; i++) {
            if (fitP[i] > best_final_score) {
                best_final_score = fitP[i];
                best_idx = i;
            }
        }
    } else {
        double best_final_acc = -1.0;
        for (int i = 0; i < population_size; i++) {
            if (accP[i] > best_final_acc) {
                best_final_acc = accP[i];
                best_idx = i;
            }
        }
    }
    memcpy(B_out,
           &population[(size_t)best_idx * genome_length],
           (size_t)genome_length * sizeof(uint16_t));

    if (ga_output_mode >= OUTPUT_DETAILED && best_gen >= 0 && best_gen_index >= 0) {
        if (selection_mode == GA_SELECTION_PARETO) {
            printf("GA winner: generation %d, individual %d (acc %.3f%%, sim %.3f)\n",
                   best_gen + 1,
                   best_gen_index + 1,
                   best_acc * 100.0,
                   best_sim);
        } else {
            printf("GA winner: generation %d, individual %d (acc %.3f%%, sim %.3f, score %.3f)\n",
                   best_gen + 1,
                   best_gen_index + 1,
                   best_acc * 100.0,
                   best_sim,
                   best_score);
        }
    }

    free(front_offsets);
    free(fronts);
    free(crowdR);
    free(rankR);
    free(simR);
    free(accR);
    free(simQ);
    free(accQ);
    free(crowdP);
    free(rankP);
    free(fitR);
    free(fitQ);
    free(fitP);
    free(simP);
    free(accP);
    free(combined);
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
