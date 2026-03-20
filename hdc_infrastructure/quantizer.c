#include "quantizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QUANTIZER_EXPORT_ENABLED 0
#define QUANTIZER_EXPORT_CUTS_PATH_TEMPLATE "analysis/quantizer_cuts_dataset%02d.csv"
#define QUANTIZER_EXPORT_CENTERS_PATH_TEMPLATE "analysis/quantizer_centers_dataset%02d.csv"
#define KMEANS_1D_MAX_ITERATIONS 100
#define KMEANS_1D_TOLERANCE 1e-9
#define TREE_1D_MIN_SAMPLES_LEAF 10
#define TREE_1D_THRESHOLD_EPS 1e-12

typedef struct {
    double value;
    int label;
} feature_sample_t;

typedef struct {
    int start;
    int end;
} tree_leaf_t;

static double *g_boundaries = NULL;
static double *g_centers = NULL;
static int *g_refinement_counts = NULL;
static int *g_duplicate_center_counts = NULL;
static int *g_zero_width_interval_counts = NULL;
static int *g_empty_bin_counts = NULL;
static int *g_iteration_counts = NULL;
static int *g_training_occupancy = NULL;
static int *g_tree_split_counts = NULL;
static int *g_fallback_threshold_counts = NULL;
static int g_num_features = 0;
static int g_num_levels = 0;
static int g_fitted = 0;
static int g_non_finite_replacements = 0;
static int g_total_refinements = 0;
static int g_total_duplicate_centers = 0;
static int g_total_zero_width_intervals = 0;
static int g_total_empty_bins = 0;
static int g_total_tree_splits = 0;
static int g_total_fallback_thresholds = 0;

static size_t boundary_count_total_for(int num_features, int num_levels) {
    if (num_features <= 0 || num_levels <= 1) {
        return 0u;
    }
    return (size_t)num_features * (size_t)(num_levels - 1);
}

static size_t center_count_total_for(int num_features, int num_levels) {
    if (num_features <= 0 || num_levels <= 0) {
        return 0u;
    }
    return (size_t)num_features * (size_t)num_levels;
}

#if BINNING_MODE == KMEANS_1D_BINNING
static int center_index(int feature_idx, int center_idx) {
    return feature_idx * g_num_levels + center_idx;
}
#endif

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
static int occupancy_index(int feature_idx, int level_idx) {
    return feature_idx * g_num_levels + level_idx;
}
#endif

static void fill_with_nan(double *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = NAN;
    }
}

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
static int boundary_index(int feature_idx, int cut_idx) {
    return feature_idx * (g_num_levels - 1) + cut_idx;
}

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) {
        return -1;
    }
    if (da > db) {
        return 1;
    }
    return 0;
}

static double interpolate_sorted_value(const double *sorted_values, int sample_count, double q) {
    if (sample_count <= 0) {
        return 0.0;
    }
    if (sample_count == 1) {
        return sorted_values[0];
    }

    double clamped_q = q;
    if (clamped_q < 0.0) {
        clamped_q = 0.0;
    }
    if (clamped_q > 1.0) {
        clamped_q = 1.0;
    }

    double p = clamped_q * (double)(sample_count - 1);
    int left = (int)floor(p);
    int right = (int)ceil(p);
    double alpha = p - (double)left;
    return sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha;
}
#endif

static int allocate_quantizer_state(int num_features, int num_levels) {
    size_t boundary_count = boundary_count_total_for(num_features, num_levels);
    size_t center_count = center_count_total_for(num_features, num_levels);

    if (boundary_count > 0) {
        g_boundaries = (double *)malloc(boundary_count * sizeof(double));
    }
    if (center_count > 0) {
        g_centers = (double *)malloc(center_count * sizeof(double));
        g_training_occupancy = (int *)calloc(center_count, sizeof(int));
    }

    g_refinement_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_duplicate_center_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_zero_width_interval_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_empty_bin_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_iteration_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_tree_split_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_fallback_threshold_counts = (int *)calloc((size_t)num_features, sizeof(int));

    if ((boundary_count > 0 && g_boundaries == NULL) ||
        (center_count > 0 && g_centers == NULL) ||
        (center_count > 0 && g_training_occupancy == NULL) ||
        g_refinement_counts == NULL ||
        g_duplicate_center_counts == NULL ||
        g_zero_width_interval_counts == NULL ||
        g_empty_bin_counts == NULL ||
        g_iteration_counts == NULL ||
        g_tree_split_counts == NULL ||
        g_fallback_threshold_counts == NULL) {
        fprintf(stderr, "quantizer: failed to allocate state buffers.\n");
        return -1;
    }

    if (boundary_count > 0) {
        fill_with_nan(g_boundaries, boundary_count);
    }
    if (center_count > 0) {
        fill_with_nan(g_centers, center_count);
    }

    return 0;
}

static const char *quantizer_mode_name(void) {
#if BINNING_MODE == UNIFORM_BINNING
    return "uniform";
#elif BINNING_MODE == QUANTILE_BINNING
    return "quantile";
#elif BINNING_MODE == KMEANS_1D_BINNING
    return "kmeans-1d";
#elif BINNING_MODE == DECISION_TREE_1D_BINNING
    return "decision-tree-1d";
#else
    return "unknown";
#endif
}

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
static int map_value_with_boundaries_unchecked(int feature_idx, double x) {
    if (g_num_levels <= 1) {
        return 0;
    }
    if (g_boundaries == NULL) {
        fprintf(stderr, "quantizer: boundaries requested before allocation.\n");
        exit(EXIT_FAILURE);
    }

    int cut_count = g_num_levels - 1;
    const double *boundaries = &g_boundaries[feature_idx * cut_count];

    if (isnan(x) || x <= boundaries[0]) {
        return 0;
    }
    if (x > boundaries[cut_count - 1]) {
        return g_num_levels - 1;
    }

    int lo = 0;
    int hi = cut_count - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (x <= boundaries[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

static int map_value_with_boundaries_checked(int feature_idx, double x) {
    if (!g_fitted) {
        fprintf(stderr, "quantizer: map called before fit.\n");
        exit(EXIT_FAILURE);
    }
    if (feature_idx < 0 || feature_idx >= g_num_features) {
        fprintf(stderr, "quantizer: feature index %d out of range [0,%d).\n", feature_idx, g_num_features);
        exit(EXIT_FAILURE);
    }
    return map_value_with_boundaries_unchecked(feature_idx, x);
}
#endif

#if BINNING_MODE == UNIFORM_BINNING
#if BIPOLAR_MODE || MODEL_VARIANT == MODEL_VARIANT_MARIAN
static int get_signal_level_linear(double emg_value) {
    if (emg_value <= MIN_LEVEL) {
        return 0;
    }
    if (emg_value >= MAX_LEVEL) {
        return NUM_LEVELS - 1;
    }
    if (MAX_LEVEL == MIN_LEVEL) {
        return 0;
    }

    double normalized_value = (emg_value - MIN_LEVEL) / (MAX_LEVEL - MIN_LEVEL);
    int level = (int)(normalized_value * (NUM_LEVELS - 1));
    if (level < 0) {
        level = 0;
    }
    if (level >= NUM_LEVELS) {
        level = NUM_LEVELS - 1;
    }
    return level;
}
#endif

#if !BIPOLAR_MODE && (MODEL_VARIANT == MODEL_VARIANT_KRISCHAN || MODEL_VARIANT == MODEL_VARIANT_FUSION)
static int get_signal_level_krischan(double emg_value) {
    float value = (float)emg_value;
    int scaled = (int)ceilf(value * 10000.0f + 10000.0f);
    if (scaled < 0) {
        scaled = 0;
    }
    if (scaled > 20000) {
        scaled = 20000;
    }

    int level = (scaled * (NUM_LEVELS - 1) + 10000) / 20000;
    if (level < 0) {
        level = 0;
    }
    if (level >= NUM_LEVELS) {
        level = NUM_LEVELS - 1;
    }
    return level;
}
#endif

static int get_signal_level_uniform(double emg_value) {
#if !BIPOLAR_MODE
#if MODEL_VARIANT == MODEL_VARIANT_KRISCHAN || MODEL_VARIANT == MODEL_VARIANT_FUSION
    return get_signal_level_krischan(emg_value);
#else
    return get_signal_level_linear(emg_value);
#endif
#else
    return get_signal_level_linear(emg_value);
#endif
}
#endif

#if BINNING_MODE == QUANTILE_BINNING
static int fit_quantile_feature(int feature_idx, const double *sorted_values, int sample_count) {
    if (g_num_levels <= 1) {
        g_iteration_counts[feature_idx] = 1;
        return 0;
    }

    int cut_count = g_num_levels - 1;
    for (int k = 1; k < g_num_levels; k++) {
        double q = (double)k / (double)g_num_levels;
        g_boundaries[boundary_index(feature_idx, k - 1)] =
            interpolate_sorted_value(sorted_values, sample_count, q);
    }

    int refinements = 0;
    for (int k = 1; k < cut_count; k++) {
        int prev_idx = boundary_index(feature_idx, k - 1);
        int curr_idx = boundary_index(feature_idx, k);
        if (g_boundaries[curr_idx] <= g_boundaries[prev_idx]) {
            g_boundaries[curr_idx] = nextafter(g_boundaries[prev_idx], INFINITY);
            refinements++;
        }
    }

    g_refinement_counts[feature_idx] = refinements;
    g_total_refinements += refinements;
    g_iteration_counts[feature_idx] = 1;

    if (refinements > 0 && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr,
                "quantizer: feature %d required %d cut refinements to preserve strict bin order.\n",
                feature_idx,
                refinements);
    }

    return 0;
}
#endif
#if BINNING_MODE == KMEANS_1D_BINNING
static int nearest_center_index(const double *centers, int center_count, double value) {
    int best_idx = 0;
    double best_dist = fabs(value - centers[0]);
    for (int i = 1; i < center_count; i++) {
        double dist = fabs(value - centers[i]);
        if (dist + KMEANS_1D_TOLERANCE < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

static int select_repair_sample(const double *errors,
                                const unsigned char *claimed,
                                int sample_count) {
    int best_idx = -1;
    double best_error = -1.0;

    for (int i = 0; i < sample_count; i++) {
        if (claimed[i]) {
            continue;
        }
        if (errors[i] > best_error) {
            best_error = errors[i];
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        return best_idx;
    }

    for (int i = 0; i < sample_count; i++) {
        if (errors[i] > best_error) {
            best_error = errors[i];
            best_idx = i;
        }
    }

    if (best_idx >= 0) {
        return best_idx;
    }

    return 0;
}

static void analyze_kmeans_feature(int feature_idx) {
    int duplicate_count = 0;
    int zero_width_count = 0;

    for (int i = 0; i < g_num_levels - 1; i++) {
        double left = g_centers[center_index(feature_idx, i)];
        double right = g_centers[center_index(feature_idx, i + 1)];
        if (fabs(right - left) <= KMEANS_1D_TOLERANCE) {
            duplicate_count++;
        }
    }

    for (int i = 1; i < g_num_levels - 1; i++) {
        double left = g_boundaries[boundary_index(feature_idx, i - 1)];
        double right = g_boundaries[boundary_index(feature_idx, i)];
        if (fabs(right - left) <= KMEANS_1D_TOLERANCE) {
            zero_width_count++;
        }
    }

    g_duplicate_center_counts[feature_idx] = duplicate_count;
    g_zero_width_interval_counts[feature_idx] = zero_width_count;
    g_total_duplicate_centers += duplicate_count;
    g_total_zero_width_intervals += zero_width_count;
}

static int verify_kmeans_lookup_for_feature(int feature_idx,
                                            const double *values,
                                            int sample_count) {
    const double *centers = &g_centers[feature_idx * g_num_levels];
    for (int i = 0; i < sample_count; i++) {
        int boundary_level = map_value_with_boundaries_unchecked(feature_idx, values[i]);
        int center_level = nearest_center_index(centers, g_num_levels, values[i]);
        if (boundary_level != center_level) {
            fprintf(stderr,
                    "quantizer: k-means lookup mismatch for feature %d at sample %d (boundary=%d, center=%d).\n",
                    feature_idx,
                    i,
                    boundary_level,
                    center_level);
            return -1;
        }
    }
    return 0;
}

static int fit_kmeans_feature(int feature_idx, const double *sorted_values, int sample_count) {
    double *working_centers = NULL;
    double *next_centers = NULL;
    double *sums = NULL;
    double *errors = NULL;
    int *counts = NULL;
    unsigned char *claimed = NULL;
    int iterations_used = 0;

    if (g_num_levels <= 1) {
        double mean = 0.0;
        for (int i = 0; i < sample_count; i++) {
            mean += sorted_values[i];
        }
        mean /= (double)sample_count;
        g_centers[center_index(feature_idx, 0)] = mean;
        g_iteration_counts[feature_idx] = 1;
        return 0;
    }

    working_centers = (double *)malloc((size_t)g_num_levels * sizeof(double));
    next_centers = (double *)malloc((size_t)g_num_levels * sizeof(double));
    sums = (double *)malloc((size_t)g_num_levels * sizeof(double));
    errors = (double *)malloc((size_t)sample_count * sizeof(double));
    counts = (int *)malloc((size_t)g_num_levels * sizeof(int));
    claimed = (unsigned char *)malloc((size_t)sample_count * sizeof(unsigned char));

    if (!working_centers || !next_centers || !sums || !errors || !counts || !claimed) {
        fprintf(stderr, "quantizer: failed to allocate k-means work buffers.\n");
        free(working_centers);
        free(next_centers);
        free(sums);
        free(errors);
        free(counts);
        free(claimed);
        return -1;
    }

    for (int i = 0; i < g_num_levels; i++) {
        double q = ((double)i + 0.5) / (double)g_num_levels;
        working_centers[i] = interpolate_sorted_value(sorted_values, sample_count, q);
    }

    for (int iter = 0; iter < KMEANS_1D_MAX_ITERATIONS; iter++) {
        memset(counts, 0, (size_t)g_num_levels * sizeof(int));
        memset(sums, 0, (size_t)g_num_levels * sizeof(double));

        for (int i = 0; i < sample_count; i++) {
            int cluster_idx = nearest_center_index(working_centers, g_num_levels, sorted_values[i]);
            double delta = sorted_values[i] - working_centers[cluster_idx];
            counts[cluster_idx] += 1;
            sums[cluster_idx] += sorted_values[i];
            errors[i] = delta * delta;
        }

        memset(claimed, 0, (size_t)sample_count * sizeof(unsigned char));
        int had_empty_cluster = 0;
        double max_shift = 0.0;

        for (int cluster = 0; cluster < g_num_levels; cluster++) {
            if (counts[cluster] > 0) {
                next_centers[cluster] = sums[cluster] / (double)counts[cluster];
            } else {
                int repair_idx = select_repair_sample(errors, claimed, sample_count);
                next_centers[cluster] = sorted_values[repair_idx];
                claimed[repair_idx] = 1u;
                had_empty_cluster = 1;
            }

            double shift = fabs(next_centers[cluster] - working_centers[cluster]);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }

        for (int cluster = 0; cluster < g_num_levels; cluster++) {
            working_centers[cluster] = next_centers[cluster];
        }

        iterations_used = iter + 1;
        if (!had_empty_cluster && max_shift <= KMEANS_1D_TOLERANCE) {
            break;
        }
    }

    qsort(working_centers, (size_t)g_num_levels, sizeof(double), compare_doubles);
    for (int i = 0; i < g_num_levels; i++) {
        g_centers[center_index(feature_idx, i)] = working_centers[i];
    }
    for (int i = 0; i < g_num_levels - 1; i++) {
        double left = g_centers[center_index(feature_idx, i)];
        double right = g_centers[center_index(feature_idx, i + 1)];
        g_boundaries[boundary_index(feature_idx, i)] = 0.5 * (left + right);
    }

    g_iteration_counts[feature_idx] = iterations_used;
    analyze_kmeans_feature(feature_idx);
    if (verify_kmeans_lookup_for_feature(feature_idx, sorted_values, sample_count) != 0) {
        free(working_centers);
        free(next_centers);
        free(sums);
        free(errors);
        free(counts);
        free(claimed);
        return -1;
    }

    free(working_centers);
    free(next_centers);
    free(sums);
    free(errors);
    free(counts);
    free(claimed);
    return 0;
}
#endif
#if BINNING_MODE == DECISION_TREE_1D_BINNING
static int compare_feature_samples(const void *a, const void *b) {
    const feature_sample_t *sa = (const feature_sample_t *)a;
    const feature_sample_t *sb = (const feature_sample_t *)b;
    if (sa->value < sb->value) {
        return -1;
    }
    if (sa->value > sb->value) {
        return 1;
    }
    if (sa->label < sb->label) {
        return -1;
    }
    if (sa->label > sb->label) {
        return 1;
    }
    return 0;
}

static double gini_impurity(const int *counts, int total) {
    if (total <= 0) {
        return 0.0;
    }

    double sum_sq = 0.0;
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        double p = (double)counts[cls] / (double)total;
        sum_sq += p * p;
    }
    return 1.0 - sum_sq;
}

static void get_range_class_counts(const int *prefix_counts, int start, int end, int *out_counts) {
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        out_counts[cls] = prefix_counts[end * NUM_CLASSES + cls] -
                          prefix_counts[start * NUM_CLASSES + cls];
    }
}

static int contains_near_duplicate_threshold(const double *values,
                                             int count,
                                             double candidate,
                                             double eps) {
    for (int i = 0; i < count; i++) {
        if (fabs(values[i] - candidate) <= eps) {
            return 1;
        }
    }
    return 0;
}

static int finalize_decision_tree_boundaries(int feature_idx,
                                             const double *tree_thresholds,
                                             int tree_threshold_count,
                                             const double *sorted_values,
                                             int sample_count) {
    int cut_count = g_num_levels - 1;
    int fallback_count = 0;
    int refinements = 0;
    int unique_tree_count = 0;
    double *sorted_tree = NULL;
    double *quantile_thresholds = NULL;
    double *selected = NULL;

    if (cut_count <= 0) {
        g_tree_split_counts[feature_idx] = tree_threshold_count;
        g_fallback_threshold_counts[feature_idx] = 0;
        g_refinement_counts[feature_idx] = 0;
        g_total_tree_splits += tree_threshold_count;
        return 0;
    }

    sorted_tree = (double *)malloc((size_t)((tree_threshold_count > 0) ? tree_threshold_count : 1) * sizeof(double));
    quantile_thresholds = (double *)malloc((size_t)cut_count * sizeof(double));
    selected = (double *)malloc((size_t)cut_count * sizeof(double));
    if (!sorted_tree || !quantile_thresholds || !selected) {
        fprintf(stderr, "quantizer: failed to allocate decision-tree threshold buffers.\n");
        free(sorted_tree);
        free(quantile_thresholds);
        free(selected);
        return -1;
    }

    for (int i = 0; i < tree_threshold_count; i++) {
        sorted_tree[i] = tree_thresholds[i];
    }
    if (tree_threshold_count > 1) {
        qsort(sorted_tree, (size_t)tree_threshold_count, sizeof(double), compare_doubles);
    }

    for (int k = 1; k < g_num_levels; k++) {
        double q = (double)k / (double)g_num_levels;
        quantile_thresholds[k - 1] = interpolate_sorted_value(sorted_values, sample_count, q);
    }

    int selected_count = 0;
    for (int i = 0; i < tree_threshold_count && selected_count < cut_count; i++) {
        if (!contains_near_duplicate_threshold(selected,
                                               selected_count,
                                               sorted_tree[i],
                                               TREE_1D_THRESHOLD_EPS)) {
            selected[selected_count++] = sorted_tree[i];
            unique_tree_count++;
        }
    }

    for (int i = 0; i < cut_count && selected_count < cut_count; i++) {
        if (!contains_near_duplicate_threshold(selected,
                                               selected_count,
                                               quantile_thresholds[i],
                                               TREE_1D_THRESHOLD_EPS)) {
            selected[selected_count++] = quantile_thresholds[i];
        }
    }

    fallback_count = selected_count - unique_tree_count;
    if (fallback_count < 0) {
        fallback_count = 0;
    }

    int quantile_idx = 0;
    while (selected_count < cut_count) {
        selected[selected_count++] = quantile_thresholds[quantile_idx];
        fallback_count++;
        quantile_idx++;
        if (quantile_idx >= cut_count) {
            quantile_idx = 0;
        }
    }

    qsort(selected, (size_t)cut_count, sizeof(double), compare_doubles);
    for (int i = 1; i < cut_count; i++) {
        if (selected[i] <= selected[i - 1]) {
            selected[i] = nextafter(selected[i - 1], INFINITY);
            refinements++;
        }
    }

    for (int i = 0; i < cut_count; i++) {
        g_boundaries[boundary_index(feature_idx, i)] = selected[i];
    }

    g_tree_split_counts[feature_idx] = tree_threshold_count;
    g_fallback_threshold_counts[feature_idx] = fallback_count;
    g_refinement_counts[feature_idx] = refinements;
    g_total_tree_splits += tree_threshold_count;
    g_total_fallback_thresholds += fallback_count;
    g_total_refinements += refinements;

    free(sorted_tree);
    free(quantile_thresholds);
    free(selected);
    return 0;
}

static int evaluate_best_tree_split(const feature_sample_t *sorted_samples,
                                    const int *prefix_counts,
                                    int start,
                                    int end,
                                    int *best_split_pos,
                                    double *best_threshold,
                                    double *best_gain) {
    int leaf_size = end - start;
    if (leaf_size < 2 * TREE_1D_MIN_SAMPLES_LEAF) {
        return 0;
    }

    int total_counts[NUM_CLASSES];
    get_range_class_counts(prefix_counts, start, end, total_counts);
    double parent_weighted_impurity = (double)leaf_size * gini_impurity(total_counts, leaf_size);
    int found = 0;
    double local_best_gain = TREE_1D_THRESHOLD_EPS;
    int local_best_pos = -1;
    double local_best_threshold = 0.0;

    for (int split_pos = start + TREE_1D_MIN_SAMPLES_LEAF;
         split_pos <= end - TREE_1D_MIN_SAMPLES_LEAF;
         split_pos++) {
        if (fabs(sorted_samples[split_pos].value - sorted_samples[split_pos - 1].value) <=
            TREE_1D_THRESHOLD_EPS) {
            continue;
        }

        int left_size = split_pos - start;
        int right_size = end - split_pos;
        int left_counts[NUM_CLASSES];
        int right_counts[NUM_CLASSES];
        get_range_class_counts(prefix_counts, start, split_pos, left_counts);
        get_range_class_counts(prefix_counts, split_pos, end, right_counts);

        double children_weighted_impurity =
            (double)left_size * gini_impurity(left_counts, left_size) +
            (double)right_size * gini_impurity(right_counts, right_size);
        double gain = parent_weighted_impurity - children_weighted_impurity;

        if (gain > local_best_gain + TREE_1D_THRESHOLD_EPS) {
            local_best_gain = gain;
            local_best_pos = split_pos;
            local_best_threshold =
                0.5 * (sorted_samples[split_pos - 1].value + sorted_samples[split_pos].value);
            found = 1;
        }
    }

    if (!found) {
        return 0;
    }

    *best_split_pos = local_best_pos;
    *best_threshold = local_best_threshold;
    *best_gain = local_best_gain;
    return 1;
}

static int fit_decision_tree_feature(int feature_idx,
                                     const feature_sample_t *sorted_samples,
                                     const double *sorted_values,
                                     int sample_count) {
    int *prefix_counts = NULL;
    tree_leaf_t *leaves = NULL;
    double *tree_thresholds = NULL;
    int leaf_count = 1;
    int tree_threshold_count = 0;

    if (g_num_levels <= 1) {
        g_tree_split_counts[feature_idx] = 0;
        g_fallback_threshold_counts[feature_idx] = 0;
        g_refinement_counts[feature_idx] = 0;
        return 0;
    }

    prefix_counts = (int *)calloc((size_t)(sample_count + 1) * (size_t)NUM_CLASSES, sizeof(int));
    leaves = (tree_leaf_t *)malloc((size_t)g_num_levels * sizeof(tree_leaf_t));
    tree_thresholds = (double *)malloc((size_t)(g_num_levels - 1) * sizeof(double));
    if (!prefix_counts || !leaves || !tree_thresholds) {
        fprintf(stderr, "quantizer: failed to allocate decision-tree fit buffers.\n");
        free(prefix_counts);
        free(leaves);
        free(tree_thresholds);
        return -1;
    }

    for (int i = 0; i < sample_count; i++) {
        memcpy(&prefix_counts[(size_t)(i + 1) * (size_t)NUM_CLASSES],
               &prefix_counts[(size_t)i * (size_t)NUM_CLASSES],
               (size_t)NUM_CLASSES * sizeof(int));
        prefix_counts[(size_t)(i + 1) * (size_t)NUM_CLASSES + (size_t)sorted_samples[i].label] += 1;
    }

    leaves[0].start = 0;
    leaves[0].end = sample_count;

    while (leaf_count < g_num_levels) {
        int best_leaf_idx = -1;
        int best_split_pos = -1;
        double best_threshold = 0.0;
        double best_gain = TREE_1D_THRESHOLD_EPS;

        for (int leaf_idx = 0; leaf_idx < leaf_count; leaf_idx++) {
            int candidate_split_pos = -1;
            double candidate_threshold = 0.0;
            double candidate_gain = 0.0;

            if (evaluate_best_tree_split(sorted_samples,
                                         prefix_counts,
                                         leaves[leaf_idx].start,
                                         leaves[leaf_idx].end,
                                         &candidate_split_pos,
                                         &candidate_threshold,
                                         &candidate_gain)) {
                if (best_leaf_idx < 0 || candidate_gain > best_gain + TREE_1D_THRESHOLD_EPS) {
                    best_leaf_idx = leaf_idx;
                    best_split_pos = candidate_split_pos;
                    best_threshold = candidate_threshold;
                    best_gain = candidate_gain;
                }
            }
        }

        if (best_leaf_idx < 0) {
            break;
        }

        tree_leaf_t original_leaf = leaves[best_leaf_idx];
        leaves[best_leaf_idx].start = original_leaf.start;
        leaves[best_leaf_idx].end = best_split_pos;
        leaves[leaf_count].start = best_split_pos;
        leaves[leaf_count].end = original_leaf.end;
        leaf_count++;
        tree_thresholds[tree_threshold_count++] = best_threshold;
    }

    if (finalize_decision_tree_boundaries(feature_idx,
                                          tree_thresholds,
                                          tree_threshold_count,
                                          sorted_values,
                                          sample_count) != 0) {
        free(prefix_counts);
        free(leaves);
        free(tree_thresholds);
        return -1;
    }

    free(prefix_counts);
    free(leaves);
    free(tree_thresholds);
    return 0;
}
#endif
#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
static void compute_training_occupancy(double **training_data, int training_samples) {
    if (g_training_occupancy == NULL || g_num_features <= 0 || g_num_levels <= 0) {
        return;
    }

    memset(g_training_occupancy, 0, center_count_total_for(g_num_features, g_num_levels) * sizeof(int));
    memset(g_empty_bin_counts, 0, (size_t)g_num_features * sizeof(int));

    for (int sample = 0; sample < training_samples; sample++) {
        for (int feature = 0; feature < g_num_features; feature++) {
            int level = get_signal_level(feature, training_data[sample][feature]);
            g_training_occupancy[occupancy_index(feature, level)] += 1;
        }
    }

    g_total_empty_bins = 0;
    for (int feature = 0; feature < g_num_features; feature++) {
        int empty_bins = 0;
        for (int level = 0; level < g_num_levels; level++) {
            if (g_training_occupancy[occupancy_index(feature, level)] == 0) {
                empty_bins++;
            }
        }
        g_empty_bin_counts[feature] = empty_bins;
        g_total_empty_bins += empty_bins;
    }
}
#endif

#if BINNING_MODE == KMEANS_1D_BINNING
static void print_kmeans_diagnostics(void) {
    if (output_mode >= OUTPUT_DETAILED) {
        for (int feature = 0; feature < g_num_features; feature++) {
            fprintf(stdout,
                    "quantizer: kmeans feature %d: iter=%d, empty_bins=%d, duplicate_centers=%d, zero_width_intervals=%d\n",
                    feature,
                    g_iteration_counts[feature],
                    g_empty_bin_counts[feature],
                    g_duplicate_center_counts[feature],
                    g_zero_width_interval_counts[feature]);
        }
    }

    if (output_mode >= OUTPUT_DEBUG) {
        for (int feature = 0; feature < g_num_features; feature++) {
            fprintf(stdout, "quantizer: feature %d centers:", feature);
            for (int i = 0; i < g_num_levels; i++) {
                fprintf(stdout, " %.17g", g_centers[center_index(feature, i)]);
            }
            fprintf(stdout, "\n");

            if (g_num_levels > 1) {
                fprintf(stdout, "quantizer: feature %d boundaries:", feature);
                for (int i = 0; i < g_num_levels - 1; i++) {
                    fprintf(stdout, " %.17g", g_boundaries[boundary_index(feature, i)]);
                }
                fprintf(stdout, "\n");
            }

            fprintf(stdout, "quantizer: feature %d occupancy:", feature);
            for (int level = 0; level < g_num_levels; level++) {
                fprintf(stdout, " %d", g_training_occupancy[occupancy_index(feature, level)]);
            }
            fprintf(stdout, "\n");
        }
    }
}
#endif

#if BINNING_MODE == DECISION_TREE_1D_BINNING
static void print_decision_tree_diagnostics(void) {
    if (output_mode >= OUTPUT_DETAILED) {
        for (int feature = 0; feature < g_num_features; feature++) {
            fprintf(stdout,
                    "quantizer: tree feature %d: splits=%d, fallback_thresholds=%d, empty_bins=%d, refinements=%d\n",
                    feature,
                    g_tree_split_counts[feature],
                    g_fallback_threshold_counts[feature],
                    g_empty_bin_counts[feature],
                    g_refinement_counts[feature]);
        }
    }

    if (output_mode >= OUTPUT_DEBUG) {
        for (int feature = 0; feature < g_num_features; feature++) {
            if (g_num_levels > 1) {
                fprintf(stdout, "quantizer: feature %d boundaries:", feature);
                for (int i = 0; i < g_num_levels - 1; i++) {
                    fprintf(stdout, " %.17g", g_boundaries[boundary_index(feature, i)]);
                }
                fprintf(stdout, "\n");
            }

            fprintf(stdout, "quantizer: feature %d occupancy:", feature);
            for (int level = 0; level < g_num_levels; level++) {
                fprintf(stdout, " %d", g_training_occupancy[occupancy_index(feature, level)]);
            }
            fprintf(stdout, "\n");
        }
    }
}
#endif

void quantizer_clear(void) {
    free(g_boundaries);
    free(g_centers);
    free(g_refinement_counts);
    free(g_duplicate_center_counts);
    free(g_zero_width_interval_counts);
    free(g_empty_bin_counts);
    free(g_iteration_counts);
    free(g_training_occupancy);
    free(g_tree_split_counts);
    free(g_fallback_threshold_counts);
    g_boundaries = NULL;
    g_centers = NULL;
    g_refinement_counts = NULL;
    g_duplicate_center_counts = NULL;
    g_zero_width_interval_counts = NULL;
    g_empty_bin_counts = NULL;
    g_iteration_counts = NULL;
    g_training_occupancy = NULL;
    g_tree_split_counts = NULL;
    g_fallback_threshold_counts = NULL;
    g_num_features = 0;
    g_num_levels = 0;
    g_fitted = 0;
    g_non_finite_replacements = 0;
    g_total_refinements = 0;
    g_total_duplicate_centers = 0;
    g_total_zero_width_intervals = 0;
    g_total_empty_bins = 0;
    g_total_tree_splits = 0;
    g_total_fallback_thresholds = 0;
}

int quantizer_is_fitted(void) {
    return g_fitted;
}
int quantizer_fit_from_training(double **training_data,
                                const int *training_labels,
                                int training_samples,
                                int num_features,
                                int num_levels) {
    quantizer_clear();

    if (num_features <= 0 || num_levels <= 0) {
        fprintf(stderr, "quantizer: invalid fit input.\n");
        return -1;
    }

    g_num_features = num_features;
    g_num_levels = num_levels;

    if (allocate_quantizer_state(num_features, num_levels) != 0) {
        quantizer_clear();
        return -1;
    }

#if BINNING_MODE == UNIFORM_BINNING
    (void)training_data;
    (void)training_labels;
    (void)training_samples;
    g_fitted = 1;
    return 0;
#elif BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING
    (void)training_labels;
    if (!training_data || training_samples <= 0) {
        fprintf(stderr, "quantizer: invalid fit input.\n");
        quantizer_clear();
        return -1;
    }

    double *sorted_values = (double *)malloc((size_t)training_samples * sizeof(double));
    if (sorted_values == NULL) {
        fprintf(stderr, "quantizer: failed to allocate training buffer.\n");
        quantizer_clear();
        return -1;
    }

    for (int feature = 0; feature < num_features; feature++) {
        for (int sample = 0; sample < training_samples; sample++) {
            double value = training_data[sample][feature];
            if (!isfinite(value)) {
                value = 0.0;
                g_non_finite_replacements++;
            }
            sorted_values[sample] = value;
        }

        qsort(sorted_values, (size_t)training_samples, sizeof(double), compare_doubles);

#if BINNING_MODE == QUANTILE_BINNING
        if (fit_quantile_feature(feature, sorted_values, training_samples) != 0) {
            free(sorted_values);
            quantizer_clear();
            return -1;
        }
#elif BINNING_MODE == KMEANS_1D_BINNING
        if (fit_kmeans_feature(feature, sorted_values, training_samples) != 0) {
            free(sorted_values);
            quantizer_clear();
            return -1;
        }
#endif
    }

    free(sorted_values);
#elif BINNING_MODE == DECISION_TREE_1D_BINNING
    if (!training_data || !training_labels || training_samples <= 0) {
        fprintf(stderr, "quantizer: invalid fit input for decision-tree mode.\n");
        quantizer_clear();
        return -1;
    }

    feature_sample_t *sorted_samples = (feature_sample_t *)malloc((size_t)training_samples * sizeof(feature_sample_t));
    double *sorted_values = (double *)malloc((size_t)training_samples * sizeof(double));
    if (sorted_samples == NULL || sorted_values == NULL) {
        fprintf(stderr, "quantizer: failed to allocate decision-tree training buffers.\n");
        free(sorted_samples);
        free(sorted_values);
        quantizer_clear();
        return -1;
    }

    for (int feature = 0; feature < num_features; feature++) {
        for (int sample = 0; sample < training_samples; sample++) {
            double value = training_data[sample][feature];
            int label = training_labels[sample];
            if (label < 0 || label >= NUM_CLASSES) {
                fprintf(stderr,
                        "quantizer: label %d at sample %d is out of range [0,%d).\n",
                        label,
                        sample,
                        NUM_CLASSES);
                free(sorted_samples);
                free(sorted_values);
                quantizer_clear();
                return -1;
            }
            if (!isfinite(value)) {
                value = 0.0;
                g_non_finite_replacements++;
            }
            sorted_samples[sample].value = value;
            sorted_samples[sample].label = label;
        }

        qsort(sorted_samples, (size_t)training_samples, sizeof(feature_sample_t), compare_feature_samples);
        for (int sample = 0; sample < training_samples; sample++) {
            sorted_values[sample] = sorted_samples[sample].value;
        }

        if (fit_decision_tree_feature(feature, sorted_samples, sorted_values, training_samples) != 0) {
            free(sorted_samples);
            free(sorted_values);
            quantizer_clear();
            return -1;
        }
    }

    free(sorted_samples);
    free(sorted_values);
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING, QUANTILE_BINNING, KMEANS_1D_BINNING, or DECISION_TREE_1D_BINNING."
#endif

    g_fitted = 1;

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
    compute_training_occupancy(training_data, training_samples);
#endif

    if (g_non_finite_replacements > 0 && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr,
                "quantizer: replaced %d non-finite training values with 0.0 during fit.\n",
                g_non_finite_replacements);
    }

#if BINNING_MODE == QUANTILE_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (total refinements: %d).\n",
                quantizer_mode_name(),
                g_num_features,
                g_num_levels,
                g_total_refinements);
    }
#elif BINNING_MODE == KMEANS_1D_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (duplicate centers: %d, zero-width intervals: %d, empty bins: %d).\n",
                quantizer_mode_name(),
                g_num_features,
                g_num_levels,
                g_total_duplicate_centers,
                g_total_zero_width_intervals,
                g_total_empty_bins);
    }
    print_kmeans_diagnostics();
#elif BINNING_MODE == DECISION_TREE_1D_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (tree splits: %d, fallback thresholds: %d, refinements: %d, empty bins: %d).\n",
                quantizer_mode_name(),
                g_num_features,
                g_num_levels,
                g_total_tree_splits,
                g_total_fallback_thresholds,
                g_total_refinements,
                g_total_empty_bins);
    }
    print_decision_tree_diagnostics();
#endif

    return 0;
}

int get_signal_level(int feature_idx, double emg_value) {
#if BINNING_MODE == UNIFORM_BINNING
    (void)feature_idx;
    return get_signal_level_uniform(emg_value);
#elif BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
    return map_value_with_boundaries_checked(feature_idx, emg_value);
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING, QUANTILE_BINNING, KMEANS_1D_BINNING, or DECISION_TREE_1D_BINNING."
#endif
}
#if BINNING_MODE == KMEANS_1D_BINNING
static int quantizer_export_centers_csv(const char *filepath) {
    if (!filepath || filepath[0] == '\0') {
        return -1;
    }
    if (!g_fitted) {
        fprintf(stderr, "quantizer: center export requested before fit.\n");
        return -1;
    }

    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("quantizer: failed to open center export path");
        return -1;
    }

    fprintf(file,
            "#quantizer_centers,mode=%s,num_features=%d,num_levels=%d\n",
            quantizer_mode_name(),
            g_num_features,
            g_num_levels);
    fprintf(file, "feature");
    for (int k = 0; k < g_num_levels; k++) {
        fprintf(file, ",center_%03d", k);
    }
    fprintf(file, "\n");

    for (int feature = 0; feature < g_num_features; feature++) {
        fprintf(file, "%d", feature);
        for (int k = 0; k < g_num_levels; k++) {
            fprintf(file, ",%.17g", g_centers[center_index(feature, k)]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    return 0;
}
#endif

int quantizer_export_cuts_csv_for_dataset(int dataset) {
    if (!QUANTIZER_EXPORT_ENABLED) {
        return 0;
    }

    char cuts_filepath[512];
    int written = snprintf(cuts_filepath,
                           sizeof(cuts_filepath),
                           QUANTIZER_EXPORT_CUTS_PATH_TEMPLATE,
                           dataset);
    if (written < 0 || (size_t)written >= sizeof(cuts_filepath)) {
        fprintf(stderr, "quantizer: cuts export path is too long for dataset %d.\n", dataset);
        return -1;
    }

    if (quantizer_export_cuts_csv(cuts_filepath) != 0) {
        return -1;
    }

#if BINNING_MODE == KMEANS_1D_BINNING
    char centers_filepath[512];
    written = snprintf(centers_filepath,
                       sizeof(centers_filepath),
                       QUANTIZER_EXPORT_CENTERS_PATH_TEMPLATE,
                       dataset);
    if (written < 0 || (size_t)written >= sizeof(centers_filepath)) {
        fprintf(stderr, "quantizer: centers export path is too long for dataset %d.\n", dataset);
        return -1;
    }

    if (quantizer_export_centers_csv(centers_filepath) != 0) {
        return -1;
    }
#endif

    return 0;
}

int quantizer_export_cuts_csv(const char *filepath) {
    if (!filepath || filepath[0] == '\0') {
        return -1;
    }
    if (!g_fitted) {
        fprintf(stderr, "quantizer: export requested before fit.\n");
        return -1;
    }

    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("quantizer: failed to open export path");
        return -1;
    }

#if BINNING_MODE == UNIFORM_BINNING
    fprintf(file,
            "#quantizer,mode=%s,num_features=%d,num_levels=%d,total_refinements=0,non_finite_replacements=0,total_tree_splits=0,total_fallback_thresholds=0\n",
            quantizer_mode_name(),
            g_num_features,
            g_num_levels);
    fprintf(file, "feature,refinement_count,tree_split_count,fallback_threshold_count\n");
    for (int feature = 0; feature < g_num_features; feature++) {
        fprintf(file, "%d,0,0,0\n", feature);
    }
#elif BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
    int cut_count = g_num_levels - 1;
    fprintf(file,
            "#quantizer,mode=%s,num_features=%d,num_levels=%d,total_refinements=%d,non_finite_replacements=%d,total_tree_splits=%d,total_fallback_thresholds=%d\n",
            quantizer_mode_name(),
            g_num_features,
            g_num_levels,
            g_total_refinements,
            g_non_finite_replacements,
            g_total_tree_splits,
            g_total_fallback_thresholds);
    fprintf(file, "feature,refinement_count,tree_split_count,fallback_threshold_count");
    for (int k = 0; k < cut_count; k++) {
        fprintf(file, ",cut_%03d", k);
    }
    fprintf(file, "\n");

    for (int feature = 0; feature < g_num_features; feature++) {
        fprintf(file,
                "%d,%d,%d,%d",
                feature,
                g_refinement_counts[feature],
                g_tree_split_counts[feature],
                g_fallback_threshold_counts[feature]);
        for (int k = 0; k < cut_count; k++) {
            fprintf(file, ",%.17g", g_boundaries[boundary_index(feature, k)]);
        }
        fprintf(file, "\n");
    }
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING, QUANTILE_BINNING, KMEANS_1D_BINNING, or DECISION_TREE_1D_BINNING."
#endif

    fclose(file);
    return 0;
}
