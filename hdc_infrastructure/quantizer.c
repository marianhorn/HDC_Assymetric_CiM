#include "quantizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define QUANTIZER_EXPORT_ENABLED 0
#define QUANTIZER_EXPORT_CUTS_PATH_TEMPLATE "analysis/quantizer_cuts_dataset%02d.csv"
#define QUANTIZER_EXPORT_CENTERS_PATH_TEMPLATE "analysis/quantizer_centers_dataset%02d.csv"
#define KMEANS_1D_MAX_ITERATIONS 100
#define KMEANS_1D_TOLERANCE 1e-9
#define TREE_1D_MIN_SAMPLES_LEAF 10
#define TREE_1D_THRESHOLD_EPS 1e-12
#define CHIMERGE_THRESHOLD_EPS 1e-12

#if BIPOLAR_MODE != 0 || MODEL_VARIANT != MODEL_VARIANT_FUSION
#error "quantizer: only BIPOLAR_MODE=0 and MODEL_VARIANT_FUSION are supported in the current branch. For deprecated versions go back in git history to end of April."
#endif

#if BINNING_MODE == GA_REFINED_BINNING && (!defined(FOOT_EMG) || !PRECOMPUTED_ITEM_MEMORY)
#error "GA_REFINED_BINNING requires FOOT_EMG with PRECOMPUTED_ITEM_MEMORY=1."
#endif

typedef struct {
    double value;
    int label;
} feature_sample_t;

typedef struct {
    int start;
    int end;
} tree_leaf_t;

typedef struct {
    double left_value;
    double right_value;
    int counts[NUM_CLASSES];
    int sample_count;
} chimerge_interval_t;

typedef struct {
    double *boundaries;
    double *centers;
    int num_features;
    int num_levels;
    int fitted;
    int non_finite_replacements;
#if BINNING_MODE == GA_REFINED_BINNING
    uint16_t *ga_refined_flip_counts;
    double *ga_refined_transition_weights;
    double **training_data_ref;
    int training_samples_ref;
    int ga_refined_ready;
#endif
} quantizer_state_t;

typedef struct {
    int *refinement_counts;
    int *duplicate_center_counts;
    int *zero_width_interval_counts;
    int *empty_bin_counts;
    int *iteration_counts;
    int *training_occupancy;
    int *tree_split_counts;
    int *fallback_threshold_counts;
    int *initial_interval_counts;
    int total_refinements;
    int total_duplicate_centers;
    int total_zero_width_intervals;
    int total_empty_bins;
    int total_tree_splits;
    int total_fallback_thresholds;
} quantizer_statistics_t;

static quantizer_state_t g_quantizer_state = {0};
static quantizer_statistics_t g_quantizer_statistics = {0};


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
    return feature_idx * g_quantizer_state.num_levels + center_idx;
}
#endif

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING || BINNING_MODE == GA_REFINED_BINNING
static int occupancy_index(int feature_idx, int level_idx) {
    return feature_idx * g_quantizer_state.num_levels + level_idx;
}
#endif

static void fill_with_nan(double *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = NAN;
    }
}

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING || BINNING_MODE == GA_REFINED_BINNING
static int boundary_index(int feature_idx, int cut_idx) {
    return feature_idx * (g_quantizer_state.num_levels - 1) + cut_idx;
}
#endif

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING
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

#if BINNING_MODE == DECISION_TREE_1D_BINNING
static double interpolate_sorted_sample_value(const feature_sample_t *sorted_samples, int sample_count, double q) {
    if (sample_count <= 0) {
        return 0.0;
    }
    if (sample_count == 1) {
        return sorted_samples[0].value;
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
    return sorted_samples[left].value * (1.0 - alpha) + sorted_samples[right].value * alpha;
}
#endif

static int allocate_quantizer_state(int num_features, int num_levels) {
    size_t boundary_count = boundary_count_total_for(num_features, num_levels);
    size_t center_count = center_count_total_for(num_features, num_levels);

    if (boundary_count > 0) {
        g_quantizer_state.boundaries = (double *)malloc(boundary_count * sizeof(double));
    }
    if (center_count > 0) {
        g_quantizer_state.centers = (double *)malloc(center_count * sizeof(double));
        g_quantizer_statistics.training_occupancy = (int *)calloc(center_count, sizeof(int));
    }

    g_quantizer_statistics.refinement_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_quantizer_statistics.duplicate_center_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_quantizer_statistics.zero_width_interval_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_quantizer_statistics.empty_bin_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_quantizer_statistics.iteration_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_quantizer_statistics.tree_split_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_quantizer_statistics.fallback_threshold_counts = (int *)calloc((size_t)num_features, sizeof(int));
    g_quantizer_statistics.initial_interval_counts = (int *)calloc((size_t)num_features, sizeof(int));
#if BINNING_MODE == GA_REFINED_BINNING
    if (boundary_count > 0) {
        g_quantizer_state.ga_refined_flip_counts = (uint16_t *)calloc(boundary_count, sizeof(uint16_t));
        g_quantizer_state.ga_refined_transition_weights = (double *)malloc(boundary_count * sizeof(double));
    }
#endif

    if ((boundary_count > 0 && g_quantizer_state.boundaries == NULL) ||
        (center_count > 0 && g_quantizer_state.centers == NULL) ||
        (center_count > 0 && g_quantizer_statistics.training_occupancy == NULL) ||
        g_quantizer_statistics.refinement_counts == NULL ||
        g_quantizer_statistics.duplicate_center_counts == NULL ||
        g_quantizer_statistics.zero_width_interval_counts == NULL ||
        g_quantizer_statistics.empty_bin_counts == NULL ||
        g_quantizer_statistics.iteration_counts == NULL ||
        g_quantizer_statistics.tree_split_counts == NULL ||
        g_quantizer_statistics.fallback_threshold_counts == NULL ||
        g_quantizer_statistics.initial_interval_counts == NULL
#if BINNING_MODE == GA_REFINED_BINNING
        || (boundary_count > 0 && g_quantizer_state.ga_refined_flip_counts == NULL)
        || (boundary_count > 0 && g_quantizer_state.ga_refined_transition_weights == NULL)
#endif
        ) {
        fprintf(stderr, "quantizer: failed to allocate state buffers.\n");
        return -1;
    }

    if (boundary_count > 0) {
        fill_with_nan(g_quantizer_state.boundaries, boundary_count);
    }
    if (center_count > 0) {
        fill_with_nan(g_quantizer_state.centers, center_count);
    }
#if BINNING_MODE == GA_REFINED_BINNING
    if (boundary_count > 0) {
        fill_with_nan(g_quantizer_state.ga_refined_transition_weights, boundary_count);
    }
#endif

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
#elif BINNING_MODE == CHIMERGE_BINNING
    return "chimerge";
#elif BINNING_MODE == GA_REFINED_BINNING
    return "ga-refined";
#else
    return "unknown";
#endif
}

#if BINNING_MODE == QUANTILE_BINNING
static int fit_quantile_feature(int feature_idx, const double *sorted_values, int sample_count);
#endif
#if BINNING_MODE == KMEANS_1D_BINNING
static int fit_kmeans_feature(int feature_idx, const double *sorted_values, int sample_count);
static void print_kmeans_diagnostics(void);
#endif
#if BINNING_MODE == DECISION_TREE_1D_BINNING
static int fit_decision_tree_feature(int feature_idx,
                                     const feature_sample_t *sorted_samples,
                                     int sample_count);
static void print_decision_tree_diagnostics(void);
#endif
#if BINNING_MODE == CHIMERGE_BINNING
static int fit_chimerge_feature(int feature_idx,
                                const feature_sample_t *sorted_samples,
                                int sample_count);
static void print_chimerge_diagnostics(void);
#endif
#if BINNING_MODE == GA_REFINED_BINNING
static void print_ga_refined_diagnostics(void);
#endif
#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING || BINNING_MODE == GA_REFINED_BINNING
static void compute_training_occupancy(double **training_data, int training_samples);
#endif

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING || BINNING_MODE == GA_REFINED_BINNING
static int map_value_with_boundaries_unchecked(int feature_idx, double x) {
    if (g_quantizer_state.num_levels <= 1) {
        return 0;
    }
    if (g_quantizer_state.boundaries == NULL) {
        fprintf(stderr, "quantizer: boundaries requested before allocation.\n");
        exit(EXIT_FAILURE);
    }

    int cut_count = g_quantizer_state.num_levels - 1;
    const double *boundaries = &g_quantizer_state.boundaries[feature_idx * cut_count];

    if (isnan(x) || x <= boundaries[0]) {
        return 0;
    }
    if (x > boundaries[cut_count - 1]) {
        return g_quantizer_state.num_levels - 1;
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
    if (!g_quantizer_state.fitted) {
        fprintf(stderr, "quantizer: map called before fit.\n");
        exit(EXIT_FAILURE);
    }
    if (feature_idx < 0 || feature_idx >= g_quantizer_state.num_features) {
        fprintf(stderr, "quantizer: feature index %d out of range [0,%d).\n", feature_idx, g_quantizer_state.num_features);
        exit(EXIT_FAILURE);
    }
    return map_value_with_boundaries_unchecked(feature_idx, x);
}
#endif


static int get_signal_level_uniform(double emg_value) {
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

#if BINNING_MODE == GA_REFINED_BINNING
static void reset_ga_refined_feature_stats(void) {
    memset(g_quantizer_statistics.refinement_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    memset(g_quantizer_statistics.duplicate_center_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    memset(g_quantizer_statistics.zero_width_interval_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    memset(g_quantizer_statistics.empty_bin_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    memset(g_quantizer_statistics.iteration_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    memset(g_quantizer_statistics.tree_split_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    memset(g_quantizer_statistics.fallback_threshold_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    memset(g_quantizer_statistics.initial_interval_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));
    g_quantizer_statistics.total_refinements = 0;
    g_quantizer_statistics.total_duplicate_centers = 0;
    g_quantizer_statistics.total_zero_width_intervals = 0;
    g_quantizer_statistics.total_empty_bins = 0;
    g_quantizer_statistics.total_tree_splits = 0;
    g_quantizer_statistics.total_fallback_thresholds = 0;
}

static int fit_ga_refined_feature(int feature_idx, const uint16_t *flip_counts) {
    int transitions = g_quantizer_state.num_levels - 1;
    if (transitions <= 0) {
        return 0;
    }

    double sum_weights = 0.0;
    for (int level = 0; level < transitions; level++) {
        double weight = (double)flip_counts[level] + (double)GA_BINNING_EPSILON;
        if (!isfinite(weight) || weight <= 0.0) {
            weight = 1.0;
        }
        g_quantizer_state.ga_refined_transition_weights[boundary_index(feature_idx, level)] = weight;
        sum_weights += weight;
        g_quantizer_state.ga_refined_flip_counts[boundary_index(feature_idx, level)] = flip_counts[level];
    }
    if (sum_weights <= 0.0) {
        sum_weights = (double)transitions;
        for (int level = 0; level < transitions; level++) {
            g_quantizer_state.ga_refined_transition_weights[boundary_index(feature_idx, level)] = 1.0 / (double)transitions;
        }
    } else {
        for (int level = 0; level < transitions; level++) {
            g_quantizer_state.ga_refined_transition_weights[boundary_index(feature_idx, level)] /= sum_weights;
        }
    }

    double *bin_importance = (double *)malloc((size_t)g_quantizer_state.num_levels * sizeof(double));
    double *bin_widths = (double *)malloc((size_t)g_quantizer_state.num_levels * sizeof(double));
    if (!bin_importance || !bin_widths) {
        fprintf(stderr, "quantizer: failed to allocate GA-refined buffers.\n");
        free(bin_importance);
        free(bin_widths);
        return -1;
    }

    bin_importance[0] = g_quantizer_state.ga_refined_transition_weights[boundary_index(feature_idx, 0)];
    bin_importance[g_quantizer_state.num_levels - 1] =
        g_quantizer_state.ga_refined_transition_weights[boundary_index(feature_idx, transitions - 1)];
    for (int level = 1; level < g_quantizer_state.num_levels - 1; level++) {
        double left = g_quantizer_state.ga_refined_transition_weights[boundary_index(feature_idx, level - 1)];
        double right = g_quantizer_state.ga_refined_transition_weights[boundary_index(feature_idx, level)];
        bin_importance[level] = 0.5 * (left + right);
    }

    double width_sum = 0.0;
    for (int level = 0; level < g_quantizer_state.num_levels; level++) {
        double importance = bin_importance[level];
        if (!isfinite(importance) || importance <= 0.0) {
            importance = 1.0;
        }
        double width = 1.0 / pow(importance, (double)GA_BINNING_ALPHA);
        if (!isfinite(width) || width <= 0.0) {
            width = 1.0;
        }
        bin_widths[level] = width;
        width_sum += width;
    }

    double range = (double)MAX_LEVEL - (double)MIN_LEVEL;
    double scale = (width_sum > 0.0 && range > 0.0) ? (range / width_sum) : 0.0;
    double current = (double)MIN_LEVEL;
    int refinements = 0;
    int zero_width_intervals = 0;
    double previous_boundary = -INFINITY;
    for (int level = 0; level < transitions; level++) {
        current += bin_widths[level] * scale;
        double boundary = current;
        if (level > 0 && boundary <= previous_boundary) {
            zero_width_intervals++;
        }
        if (level > 0 && boundary <= previous_boundary) {
            boundary = nextafter(previous_boundary, INFINITY);
            refinements++;
        }
        g_quantizer_state.boundaries[boundary_index(feature_idx, level)] = boundary;
        previous_boundary = boundary;
    }

    g_quantizer_statistics.refinement_counts[feature_idx] = refinements;
    g_quantizer_statistics.zero_width_interval_counts[feature_idx] = zero_width_intervals;
    g_quantizer_statistics.total_refinements += refinements;
    g_quantizer_statistics.total_zero_width_intervals += zero_width_intervals;

    free(bin_widths);
    free(bin_importance);
    return 0;
}

static void print_ga_refined_diagnostics(void) {
    if (output_mode >= OUTPUT_DETAILED) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            fprintf(stdout,
                    "quantizer: ga-refined feature %d: refinements=%d, empty_bins=%d, zero_width_intervals=%d\n",
                    feature,
                    g_quantizer_statistics.refinement_counts[feature],
                    g_quantizer_statistics.empty_bin_counts[feature],
                    g_quantizer_statistics.zero_width_interval_counts[feature]);
        }
    }

    if (output_mode >= OUTPUT_DEBUG) {
        int transitions = g_quantizer_state.num_levels - 1;
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            fprintf(stdout, "quantizer: feature %d flip-counts:", feature);
            for (int level = 0; level < transitions; level++) {
                fprintf(stdout, " %u", (unsigned)g_quantizer_state.ga_refined_flip_counts[boundary_index(feature, level)]);
            }
            fprintf(stdout, "\n");

            fprintf(stdout, "quantizer: feature %d transition weights:", feature);
            for (int level = 0; level < transitions; level++) {
                fprintf(stdout, " %.17g", g_quantizer_state.ga_refined_transition_weights[boundary_index(feature, level)]);
            }
            fprintf(stdout, "\n");

            fprintf(stdout, "quantizer: feature %d boundaries:", feature);
            for (int level = 0; level < transitions; level++) {
                fprintf(stdout, " %.17g", g_quantizer_state.boundaries[boundary_index(feature, level)]);
            }
            fprintf(stdout, "\n");
        }
    }
}
#endif

#if BINNING_MODE == QUANTILE_BINNING
static int fit_quantile_feature(int feature_idx, const double *sorted_values, int sample_count) {
    if (g_quantizer_state.num_levels <= 1) {
        g_quantizer_statistics.iteration_counts[feature_idx] = 1;
        return 0;
    }

    int cut_count = g_quantizer_state.num_levels - 1;
    for (int k = 1; k < g_quantizer_state.num_levels; k++) {
        double q = (double)k / (double)g_quantizer_state.num_levels;
        g_quantizer_state.boundaries[boundary_index(feature_idx, k - 1)] =
            interpolate_sorted_value(sorted_values, sample_count, q);
    }

    int refinements = 0;
    for (int k = 1; k < cut_count; k++) {
        int prev_idx = boundary_index(feature_idx, k - 1);
        int curr_idx = boundary_index(feature_idx, k);
        if (g_quantizer_state.boundaries[curr_idx] <= g_quantizer_state.boundaries[prev_idx]) {
            g_quantizer_state.boundaries[curr_idx] = nextafter(g_quantizer_state.boundaries[prev_idx], INFINITY);
            refinements++;
        }
    }

    g_quantizer_statistics.refinement_counts[feature_idx] = refinements;
    g_quantizer_statistics.total_refinements += refinements;
    g_quantizer_statistics.iteration_counts[feature_idx] = 1;

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
        if (dist < best_dist) {
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

    for (int i = 0; i < g_quantizer_state.num_levels - 1; i++) {
        double left = g_quantizer_state.centers[center_index(feature_idx, i)];
        double right = g_quantizer_state.centers[center_index(feature_idx, i + 1)];
        if (fabs(right - left) <= KMEANS_1D_TOLERANCE) {
            duplicate_count++;
        }
    }

    for (int i = 1; i < g_quantizer_state.num_levels - 1; i++) {
        double left = g_quantizer_state.boundaries[boundary_index(feature_idx, i - 1)];
        double right = g_quantizer_state.boundaries[boundary_index(feature_idx, i)];
        if (fabs(right - left) <= KMEANS_1D_TOLERANCE) {
            zero_width_count++;
        }
    }

    g_quantizer_statistics.duplicate_center_counts[feature_idx] = duplicate_count;
    g_quantizer_statistics.zero_width_interval_counts[feature_idx] = zero_width_count;
    g_quantizer_statistics.total_duplicate_centers += duplicate_count;
    g_quantizer_statistics.total_zero_width_intervals += zero_width_count;
}

static int verify_kmeans_lookup_for_feature(int feature_idx,
                                            const double *values,
                                            int sample_count) {
    const double *centers = &g_quantizer_state.centers[feature_idx * g_quantizer_state.num_levels];
    for (int i = 0; i < sample_count; i++) {
        int boundary_level = map_value_with_boundaries_unchecked(feature_idx, values[i]);
        int center_level = nearest_center_index(centers, g_quantizer_state.num_levels, values[i]);
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
    double *working_quantizer_state.centers = NULL;
    double *next_centers = NULL;
    double *sums = NULL;
    double *errors = NULL;
    int *counts = NULL;
    unsigned char *claimed = NULL;
    int iterations_used = 0;

    if (g_quantizer_state.num_levels <= 1) {
        double mean = 0.0;
        for (int i = 0; i < sample_count; i++) {
            mean += sorted_values[i];
        }
        mean /= (double)sample_count;
        g_quantizer_state.centers[center_index(feature_idx, 0)] = mean;
        g_quantizer_statistics.iteration_counts[feature_idx] = 1;
        return 0;
    }

    working_quantizer_state.centers = (double *)malloc((size_t)g_quantizer_state.num_levels * sizeof(double));
    next_centers = (double *)malloc((size_t)g_quantizer_state.num_levels * sizeof(double));
    sums = (double *)malloc((size_t)g_quantizer_state.num_levels * sizeof(double));
    errors = (double *)malloc((size_t)sample_count * sizeof(double));
    counts = (int *)malloc((size_t)g_quantizer_state.num_levels * sizeof(int));
    claimed = (unsigned char *)malloc((size_t)sample_count * sizeof(unsigned char));

    if (!working_quantizer_state.centers || !next_centers || !sums || !errors || !counts || !claimed) {
        fprintf(stderr, "quantizer: failed to allocate k-means work buffers.\n");
        free(working_quantizer_state.centers);
        free(next_centers);
        free(sums);
        free(errors);
        free(counts);
        free(claimed);
        return -1;
    }

    for (int i = 0; i < g_quantizer_state.num_levels; i++) {
        double q = ((double)i + 0.5) / (double)g_quantizer_state.num_levels;
        working_quantizer_state.centers[i] = interpolate_sorted_value(sorted_values, sample_count, q);
    }

    for (int iter = 0; iter < KMEANS_1D_MAX_ITERATIONS; iter++) {
        memset(counts, 0, (size_t)g_quantizer_state.num_levels * sizeof(int));
        memset(sums, 0, (size_t)g_quantizer_state.num_levels * sizeof(double));

        for (int i = 0; i < sample_count; i++) {
            int cluster_idx = nearest_center_index(working_quantizer_state.centers, g_quantizer_state.num_levels, sorted_values[i]);
            double delta = sorted_values[i] - working_quantizer_state.centers[cluster_idx];
            counts[cluster_idx] += 1;
            sums[cluster_idx] += sorted_values[i];
            errors[i] = delta * delta;
        }

        memset(claimed, 0, (size_t)sample_count * sizeof(unsigned char));
        int had_empty_cluster = 0;
        double max_shift = 0.0;

        for (int cluster = 0; cluster < g_quantizer_state.num_levels; cluster++) {
            if (counts[cluster] > 0) {
                next_centers[cluster] = sums[cluster] / (double)counts[cluster];
            } else {
                int repair_idx = select_repair_sample(errors, claimed, sample_count);
                next_centers[cluster] = sorted_values[repair_idx];
                claimed[repair_idx] = 1u;
                had_empty_cluster = 1;
            }

            double shift = fabs(next_centers[cluster] - working_quantizer_state.centers[cluster]);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }

        for (int cluster = 0; cluster < g_quantizer_state.num_levels; cluster++) {
            working_quantizer_state.centers[cluster] = next_centers[cluster];
        }

        iterations_used = iter + 1;
        if (!had_empty_cluster && max_shift <= KMEANS_1D_TOLERANCE) {
            break;
        }
    }

    qsort(working_quantizer_state.centers, (size_t)g_quantizer_state.num_levels, sizeof(double), compare_doubles);
    for (int i = 0; i < g_quantizer_state.num_levels; i++) {
        g_quantizer_state.centers[center_index(feature_idx, i)] = working_quantizer_state.centers[i];
    }
    for (int i = 0; i < g_quantizer_state.num_levels - 1; i++) {
        double left = g_quantizer_state.centers[center_index(feature_idx, i)];
        double right = g_quantizer_state.centers[center_index(feature_idx, i + 1)];
        g_quantizer_state.boundaries[boundary_index(feature_idx, i)] = 0.5 * (left + right);
    }

    g_quantizer_statistics.iteration_counts[feature_idx] = iterations_used;
    analyze_kmeans_feature(feature_idx);
    if (verify_kmeans_lookup_for_feature(feature_idx, sorted_values, sample_count) != 0) {
        free(working_quantizer_state.centers);
        free(next_centers);
        free(sums);
        free(errors);
        free(counts);
        free(claimed);
        return -1;
    }

    free(working_quantizer_state.centers);
    free(next_centers);
    free(sums);
    free(errors);
    free(counts);
    free(claimed);
    return 0;
}
#endif
#if BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING
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
#endif

#if BINNING_MODE == DECISION_TREE_1D_BINNING
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
                                             const feature_sample_t *sorted_samples,
                                             int sample_count) {
    int cut_count = g_quantizer_state.num_levels - 1;
    int fallback_count = 0;
    int refinements = 0;
    int unique_tree_count = 0;
    double *sorted_tree = NULL;
    double *quantile_thresholds = NULL;
    double *selected = NULL;

    if (cut_count <= 0) {
        g_quantizer_statistics.tree_split_counts[feature_idx] = tree_threshold_count;
        g_quantizer_statistics.fallback_threshold_counts[feature_idx] = 0;
        g_quantizer_statistics.refinement_counts[feature_idx] = 0;
        g_quantizer_statistics.total_tree_splits += tree_threshold_count;
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

    for (int k = 1; k < g_quantizer_state.num_levels; k++) {
        double q = (double)k / (double)g_quantizer_state.num_levels;
        quantile_thresholds[k - 1] = interpolate_sorted_sample_value(sorted_samples, sample_count, q);
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
        g_quantizer_state.boundaries[boundary_index(feature_idx, i)] = selected[i];
    }

    g_quantizer_statistics.tree_split_counts[feature_idx] = tree_threshold_count;
    g_quantizer_statistics.fallback_threshold_counts[feature_idx] = fallback_count;
    g_quantizer_statistics.refinement_counts[feature_idx] = refinements;
    g_quantizer_statistics.total_tree_splits += tree_threshold_count;
    g_quantizer_statistics.total_fallback_thresholds += fallback_count;
    g_quantizer_statistics.total_refinements += refinements;

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
                                     int sample_count) {
    int *prefix_counts = NULL;
    tree_leaf_t *leaves = NULL;
    double *tree_thresholds = NULL;
    int leaf_count = 1;
    int tree_threshold_count = 0;

    if (g_quantizer_state.num_levels <= 1) {
        g_quantizer_statistics.tree_split_counts[feature_idx] = 0;
        g_quantizer_statistics.fallback_threshold_counts[feature_idx] = 0;
        g_quantizer_statistics.refinement_counts[feature_idx] = 0;
        return 0;
    }

    prefix_counts = (int *)calloc((size_t)(sample_count + 1) * (size_t)NUM_CLASSES, sizeof(int));
    leaves = (tree_leaf_t *)malloc((size_t)g_quantizer_state.num_levels * sizeof(tree_leaf_t));
    tree_thresholds = (double *)malloc((size_t)(g_quantizer_state.num_levels - 1) * sizeof(double));
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

    while (leaf_count < g_quantizer_state.num_levels) {
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
                                          sorted_samples,
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
#if BINNING_MODE == CHIMERGE_BINNING
static double compute_chimerge_score(const chimerge_interval_t *left,
                                     const chimerge_interval_t *right) {
    double grand_total = (double)(left->sample_count + right->sample_count);
    if (grand_total <= 0.0) {
        return 0.0;
    }

    double score = 0.0;
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        double column_total = (double)(left->counts[cls] + right->counts[cls]);
        if (column_total <= 0.0) {
            continue;
        }

        double expected_left = ((double)left->sample_count * column_total) / grand_total;
        double expected_right = ((double)right->sample_count * column_total) / grand_total;

        if (expected_left > 0.0) {
            double diff_left = (double)left->counts[cls] - expected_left;
            score += (diff_left * diff_left) / expected_left;
        }
        if (expected_right > 0.0) {
            double diff_right = (double)right->counts[cls] - expected_right;
            score += (diff_right * diff_right) / expected_right;
        }
    }

    return score;
}

static int finalize_chimerge_boundaries(int feature_idx,
                                        const chimerge_interval_t *intervals,
                                        int interval_count,
                                        const feature_sample_t *sorted_samples,
                                        int sample_count) {
    int cut_count = g_quantizer_state.num_levels - 1;
    int meaningful_count = (interval_count > 0) ? (interval_count - 1) : 0;
    int fallback_count = 0;
    int refinements = 0;
    double *selected = NULL;

    if (cut_count <= 0) {
        g_quantizer_statistics.fallback_threshold_counts[feature_idx] = 0;
        g_quantizer_statistics.refinement_counts[feature_idx] = 0;
        return 0;
    }

    selected = (double *)malloc((size_t)cut_count * sizeof(double));
    if (selected == NULL) {
        fprintf(stderr, "quantizer: failed to allocate ChiMerge threshold buffer.\n");
        return -1;
    }

    for (int i = 0; i < meaningful_count && i < cut_count; i++) {
        selected[i] = intervals[i].right_value +
                      0.5 * (intervals[i + 1].left_value - intervals[i].right_value);
    }

    double fill_value = 0.0;
    if (meaningful_count > 0) {
        fill_value = selected[meaningful_count - 1];
    } else if (sample_count > 0) {
        fill_value = sorted_samples[0].value;
    }

    for (int i = meaningful_count; i < cut_count; i++) {
        selected[i] = fill_value;
        fallback_count++;
    }

    for (int i = 1; i < cut_count; i++) {
        if (selected[i] <= selected[i - 1]) {
            selected[i] = nextafter(selected[i - 1], INFINITY);
            refinements++;
        }
    }

    for (int i = 0; i < cut_count; i++) {
        g_quantizer_state.boundaries[boundary_index(feature_idx, i)] = selected[i];
    }

    g_quantizer_statistics.fallback_threshold_counts[feature_idx] = fallback_count;
    g_quantizer_statistics.refinement_counts[feature_idx] = refinements;
    g_quantizer_statistics.total_fallback_thresholds += fallback_count;
    g_quantizer_statistics.total_refinements += refinements;

    free(selected);
    return 0;
}

static int fit_chimerge_feature(int feature_idx,
                                const feature_sample_t *sorted_samples,
                                int sample_count) {
    chimerge_interval_t *intervals = NULL;
    int interval_count = 0;

    if (g_quantizer_state.num_levels <= 1) {
        g_quantizer_statistics.initial_interval_counts[feature_idx] = 1;
        g_quantizer_statistics.fallback_threshold_counts[feature_idx] = 0;
        g_quantizer_statistics.refinement_counts[feature_idx] = 0;
        return 0;
    }

    intervals = (chimerge_interval_t *)calloc((size_t)sample_count, sizeof(chimerge_interval_t));
    if (intervals == NULL) {
        fprintf(stderr, "quantizer: failed to allocate ChiMerge intervals.\n");
        return -1;
    }

    for (int i = 0; i < sample_count; i++) {
        double value = sorted_samples[i].value;
        int label = sorted_samples[i].label;

        if (interval_count == 0 ||
            fabs(value - intervals[interval_count - 1].right_value) > CHIMERGE_THRESHOLD_EPS) {
            intervals[interval_count].left_value = value;
            intervals[interval_count].right_value = value;
            interval_count++;
        }

        chimerge_interval_t *interval = &intervals[interval_count - 1];
        interval->right_value = value;
        interval->counts[label] += 1;
        interval->sample_count += 1;
    }

    g_quantizer_statistics.initial_interval_counts[feature_idx] = interval_count;
    if (interval_count < g_quantizer_state.num_levels && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr,
                "quantizer: feature %d has only %d distinct value intervals for %d target bins; ChiMerge fallback will be used.\n",
                feature_idx,
                interval_count,
                g_quantizer_state.num_levels);
    }

    while (interval_count > g_quantizer_state.num_levels) {
        int best_pair_idx = 0;
        double best_score = compute_chimerge_score(&intervals[0], &intervals[1]);

        for (int pair_idx = 1; pair_idx < interval_count - 1; pair_idx++) {
            double score = compute_chimerge_score(&intervals[pair_idx], &intervals[pair_idx + 1]);
            if (score < best_score - CHIMERGE_THRESHOLD_EPS) {
                best_score = score;
                best_pair_idx = pair_idx;
            }
        }

        chimerge_interval_t *left = &intervals[best_pair_idx];
        chimerge_interval_t *right = &intervals[best_pair_idx + 1];
        left->right_value = right->right_value;
        left->sample_count += right->sample_count;
        for (int cls = 0; cls < NUM_CLASSES; cls++) {
            left->counts[cls] += right->counts[cls];
        }

        for (int i = best_pair_idx + 1; i < interval_count - 1; i++) {
            intervals[i] = intervals[i + 1];
        }
        interval_count--;
    }

    if (finalize_chimerge_boundaries(feature_idx, intervals, interval_count, sorted_samples, sample_count) != 0) {
        free(intervals);
        return -1;
    }

    free(intervals);
    return 0;
}
#endif

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING || BINNING_MODE == GA_REFINED_BINNING
static void compute_training_occupancy(double **training_data, int training_samples) {
    if (g_quantizer_statistics.training_occupancy == NULL || g_quantizer_state.num_features <= 0 || g_quantizer_state.num_levels <= 0) {
        return;
    }

    memset(g_quantizer_statistics.training_occupancy, 0, center_count_total_for(g_quantizer_state.num_features, g_quantizer_state.num_levels) * sizeof(int));
    memset(g_quantizer_statistics.empty_bin_counts, 0, (size_t)g_quantizer_state.num_features * sizeof(int));

    for (int sample = 0; sample < training_samples; sample++) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            int level = get_signal_level(feature, training_data[sample][feature]);
            g_quantizer_statistics.training_occupancy[occupancy_index(feature, level)] += 1;
        }
    }

    g_quantizer_statistics.total_empty_bins = 0;
    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        int empty_bins = 0;
        for (int level = 0; level < g_quantizer_state.num_levels; level++) {
            if (g_quantizer_statistics.training_occupancy[occupancy_index(feature, level)] == 0) {
                empty_bins++;
            }
        }
        g_quantizer_statistics.empty_bin_counts[feature] = empty_bins;
        g_quantizer_statistics.total_empty_bins += empty_bins;
    }
}
#endif

#if BINNING_MODE == KMEANS_1D_BINNING
static void print_kmeans_diagnostics(void) {
    if (output_mode >= OUTPUT_DETAILED) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            fprintf(stdout,
                    "quantizer: kmeans feature %d: iter=%d, empty_bins=%d, duplicate_centers=%d, zero_width_intervals=%d\n",
                    feature,
                    g_quantizer_statistics.iteration_counts[feature],
                    g_quantizer_statistics.empty_bin_counts[feature],
                    g_quantizer_statistics.duplicate_center_counts[feature],
                    g_quantizer_statistics.zero_width_interval_counts[feature]);
        }
    }

    if (output_mode >= OUTPUT_DEBUG) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            fprintf(stdout, "quantizer: feature %d centers:", feature);
            for (int i = 0; i < g_quantizer_state.num_levels; i++) {
                fprintf(stdout, " %.17g", g_quantizer_state.centers[center_index(feature, i)]);
            }
            fprintf(stdout, "\n");

            if (g_quantizer_state.num_levels > 1) {
                fprintf(stdout, "quantizer: feature %d boundaries:", feature);
                for (int i = 0; i < g_quantizer_state.num_levels - 1; i++) {
                    fprintf(stdout, " %.17g", g_quantizer_state.boundaries[boundary_index(feature, i)]);
                }
                fprintf(stdout, "\n");
            }

            fprintf(stdout, "quantizer: feature %d occupancy:", feature);
            for (int level = 0; level < g_quantizer_state.num_levels; level++) {
                fprintf(stdout, " %d", g_quantizer_statistics.training_occupancy[occupancy_index(feature, level)]);
            }
            fprintf(stdout, "\n");
        }
    }
}
#endif

#if BINNING_MODE == DECISION_TREE_1D_BINNING
static void print_decision_tree_diagnostics(void) {
    if (output_mode >= OUTPUT_DETAILED) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            fprintf(stdout,
                    "quantizer: tree feature %d: splits=%d, fallback_thresholds=%d, empty_bins=%d, refinements=%d\n",
                    feature,
                    g_quantizer_statistics.tree_split_counts[feature],
                    g_quantizer_statistics.fallback_threshold_counts[feature],
                    g_quantizer_statistics.empty_bin_counts[feature],
                    g_quantizer_statistics.refinement_counts[feature]);
        }
    }

    if (output_mode >= OUTPUT_DEBUG) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            if (g_quantizer_state.num_levels > 1) {
                fprintf(stdout, "quantizer: feature %d boundaries:", feature);
                for (int i = 0; i < g_quantizer_state.num_levels - 1; i++) {
                    fprintf(stdout, " %.17g", g_quantizer_state.boundaries[boundary_index(feature, i)]);
                }
                fprintf(stdout, "\n");
            }

            fprintf(stdout, "quantizer: feature %d occupancy:", feature);
            for (int level = 0; level < g_quantizer_state.num_levels; level++) {
                fprintf(stdout, " %d", g_quantizer_statistics.training_occupancy[occupancy_index(feature, level)]);
            }
            fprintf(stdout, "\n");
        }
    }
}
#endif

#if BINNING_MODE == CHIMERGE_BINNING
static void print_chimerge_diagnostics(void) {
    if (output_mode >= OUTPUT_DETAILED) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            fprintf(stdout,
                    "quantizer: chimerge feature %d: initial_intervals=%d, fallback_thresholds=%d, empty_bins=%d, refinements=%d\n",
                    feature,
                    g_quantizer_statistics.initial_interval_counts[feature],
                    g_quantizer_statistics.fallback_threshold_counts[feature],
                    g_quantizer_statistics.empty_bin_counts[feature],
                    g_quantizer_statistics.refinement_counts[feature]);
        }
    }

    if (output_mode >= OUTPUT_DEBUG) {
        for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
            if (g_quantizer_state.num_levels > 1) {
                fprintf(stdout, "quantizer: feature %d boundaries:", feature);
                for (int i = 0; i < g_quantizer_state.num_levels - 1; i++) {
                    fprintf(stdout, " %.17g", g_quantizer_state.boundaries[boundary_index(feature, i)]);
                }
                fprintf(stdout, "\n");
            }

            fprintf(stdout, "quantizer: feature %d occupancy:", feature);
            for (int level = 0; level < g_quantizer_state.num_levels; level++) {
                fprintf(stdout, " %d", g_quantizer_statistics.training_occupancy[occupancy_index(feature, level)]);
            }
            fprintf(stdout, "\n");
        }
    }
}
#endif

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING
static int prepare_sorted_feature_values(double **training_data,
                                         int training_samples,
                                         int feature_idx,
                                         double *sorted_values) {
    for (int sample = 0; sample < training_samples; sample++) {
        double value = training_data[sample][feature_idx];
        if (!isfinite(value)) {
            value = 0.0;
            g_quantizer_state.non_finite_replacements++;
        }
        sorted_values[sample] = value;
    }

    qsort(sorted_values, (size_t)training_samples, sizeof(double), compare_doubles);
    return 0;
}
#endif

#if BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING
static int prepare_sorted_feature_samples(double **training_data,
                                          const int *training_labels,
                                          int training_samples,
                                          int feature_idx,
                                          feature_sample_t *sorted_samples) {
    for (int sample = 0; sample < training_samples; sample++) {
        double value = training_data[sample][feature_idx];
        int label = training_labels[sample];
        if (label < 0 || label >= NUM_CLASSES) {
            fprintf(stderr,
                    "quantizer: label %d at sample %d is out of range [0,%d).\n",
                    label,
                    sample,
                    NUM_CLASSES);
            return -1;
        }
        if (!isfinite(value)) {
            value = 0.0;
            g_quantizer_state.non_finite_replacements++;
        }
        sorted_samples[sample].value = value;
        sorted_samples[sample].label = label;
    }

    qsort(sorted_samples, (size_t)training_samples, sizeof(feature_sample_t), compare_feature_samples);
    return 0;
}
#endif

static int finalize_quantizer_fit(double **training_data, int training_samples) {
    g_quantizer_state.fitted = 1;

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING || BINNING_MODE == GA_REFINED_BINNING
    compute_training_occupancy(training_data, training_samples);
#else
    (void)training_data;
    (void)training_samples;
#endif
    return 0;
}

static void print_quantizer_fit_summary(void) {
    if (g_quantizer_state.non_finite_replacements > 0 && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr,
                "quantizer: replaced %d non-finite training values with 0.0 during fit.\n",
                g_quantizer_state.non_finite_replacements);
    }

#if BINNING_MODE == QUANTILE_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (total refinements: %d).\n",
                quantizer_mode_name(),
                g_quantizer_state.num_features,
                g_quantizer_state.num_levels,
                g_quantizer_statistics.total_refinements);
    }
#elif BINNING_MODE == KMEANS_1D_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (duplicate centers: %d, zero-width intervals: %d, empty bins: %d).\n",
                quantizer_mode_name(),
                g_quantizer_state.num_features,
                g_quantizer_state.num_levels,
                g_quantizer_statistics.total_duplicate_centers,
                g_quantizer_statistics.total_zero_width_intervals,
                g_quantizer_statistics.total_empty_bins);
    }
    print_kmeans_diagnostics();
#elif BINNING_MODE == DECISION_TREE_1D_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (tree splits: %d, fallback thresholds: %d, refinements: %d, empty bins: %d).\n",
                quantizer_mode_name(),
                g_quantizer_state.num_features,
                g_quantizer_state.num_levels,
                g_quantizer_statistics.total_tree_splits,
                g_quantizer_statistics.total_fallback_thresholds,
                g_quantizer_statistics.total_refinements,
                g_quantizer_statistics.total_empty_bins);
    }
    print_decision_tree_diagnostics();
#elif BINNING_MODE == CHIMERGE_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (fallback thresholds: %d, refinements: %d, empty bins: %d).\n",
                quantizer_mode_name(),
                g_quantizer_state.num_features,
                g_quantizer_state.num_levels,
                g_quantizer_statistics.total_fallback_thresholds,
                g_quantizer_statistics.total_refinements,
                g_quantizer_statistics.total_empty_bins);
    }
    print_chimerge_diagnostics();
#elif BINNING_MODE == GA_REFINED_BINNING
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted %s boundaries for %d features, %d levels (refinements: %d, empty bins: %d, zero-width intervals: %d).\n",
                quantizer_mode_name(),
                g_quantizer_state.num_features,
                g_quantizer_state.num_levels,
                g_quantizer_statistics.total_refinements,
                g_quantizer_statistics.total_empty_bins,
                g_quantizer_statistics.total_zero_width_intervals);
    }
    print_ga_refined_diagnostics();
#endif
}

static int fit_uniform_quantizer(double **training_data,
                                 const int *training_labels,
                                 int training_samples) {
    (void)training_data;
    (void)training_labels;
    (void)training_samples;
    g_quantizer_state.fitted = 1;
    return 0;
}

#if BINNING_MODE == GA_REFINED_BINNING
static int fit_ga_refined_quantizer_init(double **training_data, int training_samples) {
    if (!training_data || training_samples <= 0) {
        fprintf(stderr, "quantizer: invalid fit input for GA-refined mode.\n");
        return -1;
    }

    g_quantizer_state.training_data_ref = training_data;
    g_quantizer_state.training_samples_ref = training_samples;
    g_quantizer_state.ga_refined_ready = 0;
    g_quantizer_state.fitted = 1;
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: initialized %s mode with temporary uniform lookup for %d features, %d levels.\n",
                quantizer_mode_name(),
                g_quantizer_state.num_features,
                g_quantizer_state.num_levels);
    }
    return 0;
}
#endif

#if BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING
static int fit_unsupervised_boundary_quantizer(double **training_data, int training_samples) {
    double *sorted_values = NULL;

    if (!training_data || training_samples <= 0) {
        fprintf(stderr, "quantizer: invalid fit input.\n");
        return -1;
    }

    sorted_values = (double *)malloc((size_t)training_samples * sizeof(double));
    if (sorted_values == NULL) {
        fprintf(stderr, "quantizer: failed to allocate training buffer.\n");
        return -1;
    }

    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        prepare_sorted_feature_values(training_data, training_samples, feature, sorted_values);
#if BINNING_MODE == QUANTILE_BINNING
        if (fit_quantile_feature(feature, sorted_values, training_samples) != 0) {
#else
        if (fit_kmeans_feature(feature, sorted_values, training_samples) != 0) {
#endif
            free(sorted_values);
            return -1;
        }
    }

    free(sorted_values);
    return 0;
}
#endif

#if BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING
static int fit_supervised_boundary_quantizer(double **training_data,
                                             const int *training_labels,
                                             int training_samples) {
    feature_sample_t *sorted_samples = NULL;

    if (!training_data || !training_labels || training_samples <= 0) {
#if BINNING_MODE == DECISION_TREE_1D_BINNING
        fprintf(stderr, "quantizer: invalid fit input for decision-tree mode.\n");
#else
        fprintf(stderr, "quantizer: invalid fit input for ChiMerge mode.\n");
#endif
        return -1;
    }

    sorted_samples = (feature_sample_t *)malloc((size_t)training_samples * sizeof(feature_sample_t));
    if (sorted_samples == NULL) {
#if BINNING_MODE == DECISION_TREE_1D_BINNING
        fprintf(stderr, "quantizer: failed to allocate decision-tree training buffers.\n");
#else
        fprintf(stderr, "quantizer: failed to allocate ChiMerge training buffers.\n");
#endif
        free(sorted_samples);
        return -1;
    }

    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        if (prepare_sorted_feature_samples(training_data,
                                           training_labels,
                                           training_samples,
                                           feature,
                                           sorted_samples) != 0) {
            free(sorted_samples);
            return -1;
        }
#if BINNING_MODE == DECISION_TREE_1D_BINNING
        if (fit_decision_tree_feature(feature, sorted_samples, training_samples) != 0) {
#else
        if (fit_chimerge_feature(feature, sorted_samples, training_samples) != 0) {
#endif
            free(sorted_samples);
            return -1;
        }
    }

    free(sorted_samples);
    return 0;
}
#endif

void quantizer_clear(void) {
    free(g_quantizer_state.boundaries);
    free(g_quantizer_state.centers);
    free(g_quantizer_statistics.refinement_counts);
    free(g_quantizer_statistics.duplicate_center_counts);
    free(g_quantizer_statistics.zero_width_interval_counts);
    free(g_quantizer_statistics.empty_bin_counts);
    free(g_quantizer_statistics.iteration_counts);
    free(g_quantizer_statistics.training_occupancy);
    free(g_quantizer_statistics.tree_split_counts);
    free(g_quantizer_statistics.fallback_threshold_counts);
    free(g_quantizer_statistics.initial_interval_counts);
#if BINNING_MODE == GA_REFINED_BINNING
    free(g_quantizer_state.ga_refined_flip_counts);
    free(g_quantizer_state.ga_refined_transition_weights);
#endif
    g_quantizer_state.boundaries = NULL;
    g_quantizer_state.centers = NULL;
    g_quantizer_statistics.refinement_counts = NULL;
    g_quantizer_statistics.duplicate_center_counts = NULL;
    g_quantizer_statistics.zero_width_interval_counts = NULL;
    g_quantizer_statistics.empty_bin_counts = NULL;
    g_quantizer_statistics.iteration_counts = NULL;
    g_quantizer_statistics.training_occupancy = NULL;
    g_quantizer_statistics.tree_split_counts = NULL;
    g_quantizer_statistics.fallback_threshold_counts = NULL;
    g_quantizer_statistics.initial_interval_counts = NULL;
#if BINNING_MODE == GA_REFINED_BINNING
    g_quantizer_state.ga_refined_flip_counts = NULL;
    g_quantizer_state.ga_refined_transition_weights = NULL;
    g_quantizer_state.training_data_ref = NULL;
    g_quantizer_state.training_samples_ref = 0;
    g_quantizer_state.ga_refined_ready = 0;
#endif
    g_quantizer_state.num_features = 0;
    g_quantizer_state.num_levels = 0;
    g_quantizer_state.fitted = 0;
    g_quantizer_state.non_finite_replacements = 0;
    g_quantizer_statistics.total_refinements = 0;
    g_quantizer_statistics.total_duplicate_centers = 0;
    g_quantizer_statistics.total_zero_width_intervals = 0;
    g_quantizer_statistics.total_empty_bins = 0;
    g_quantizer_statistics.total_tree_splits = 0;
    g_quantizer_statistics.total_fallback_thresholds = 0;
}

int quantizer_is_fitted(void) {
    return g_quantizer_state.fitted;
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

    g_quantizer_state.num_features = num_features;
    g_quantizer_state.num_levels = num_levels;

    if (allocate_quantizer_state(num_features, num_levels) != 0) {
        quantizer_clear();
        return -1;
    }

#if BINNING_MODE == UNIFORM_BINNING
    if (fit_uniform_quantizer(training_data, training_labels, training_samples) != 0) {
        quantizer_clear();
        return -1;
    }
    return 0;
#elif BINNING_MODE == GA_REFINED_BINNING
    (void)training_labels;
    if (fit_ga_refined_quantizer_init(training_data, training_samples) != 0) {
        quantizer_clear();
        return -1;
    }
    return 0;
#elif BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING
    (void)training_labels;
    if (fit_unsupervised_boundary_quantizer(training_data, training_samples) != 0) {
        quantizer_clear();
        return -1;
    }
#elif BINNING_MODE == DECISION_TREE_1D_BINNING
    if (fit_supervised_boundary_quantizer(training_data, training_labels, training_samples) != 0) {
        quantizer_clear();
        return -1;
    }
#elif BINNING_MODE == CHIMERGE_BINNING
    if (fit_supervised_boundary_quantizer(training_data, training_labels, training_samples) != 0) {
        quantizer_clear();
        return -1;
    }
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING, QUANTILE_BINNING, KMEANS_1D_BINNING, DECISION_TREE_1D_BINNING, CHIMERGE_BINNING, or GA_REFINED_BINNING."
#endif

    finalize_quantizer_fit(training_data, training_samples);
    print_quantizer_fit_summary();
    return 0;
}

#if BINNING_MODE == GA_REFINED_BINNING
int quantizer_refine_from_flip_counts(const uint16_t *flip_counts, int genome_length) {
    if (!g_quantizer_state.fitted) {
        fprintf(stderr, "quantizer: GA-refined thresholds requested before fit.\n");
        return -1;
    }

    int expected_length = g_quantizer_state.num_features * (g_quantizer_state.num_levels - 1);
    if (!flip_counts || genome_length != expected_length) {
        fprintf(stderr,
                "quantizer: invalid GA-refined flip-count input (got %d, expected %d).\n",
                genome_length,
                expected_length);
        return -1;
    }

    reset_ga_refined_feature_stats();
    if (g_quantizer_statistics.training_occupancy != NULL) {
        memset(g_quantizer_statistics.training_occupancy, 0, (size_t)g_quantizer_state.num_features * (size_t)g_quantizer_state.num_levels * sizeof(int));
    }

    int transitions = g_quantizer_state.num_levels - 1;
    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        if (fit_ga_refined_feature(feature, flip_counts + (size_t)feature * transitions) != 0) {
            return -1;
        }
    }

    g_quantizer_state.ga_refined_ready = 1;
    if (g_quantizer_state.training_data_ref && g_quantizer_state.training_samples_ref > 0) {
        compute_training_occupancy(g_quantizer_state.training_data_ref, g_quantizer_state.training_samples_ref);
    }

    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: installed GA-refined thresholds for %d features, %d levels.\n",
                g_quantizer_state.num_features,
                g_quantizer_state.num_levels);
    }
    print_ga_refined_diagnostics();
    return 0;
}
#endif

int get_signal_level(int feature_idx, double emg_value) {
#if BINNING_MODE == UNIFORM_BINNING
    (void)feature_idx;
    return get_signal_level_uniform(emg_value);
#elif BINNING_MODE == GA_REFINED_BINNING
    if (!g_quantizer_state.ga_refined_ready) {
        (void)feature_idx;
        return get_signal_level_uniform(emg_value);
    }
    return map_value_with_boundaries_checked(feature_idx, emg_value);
#elif BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING
    return map_value_with_boundaries_checked(feature_idx, emg_value);
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING, QUANTILE_BINNING, KMEANS_1D_BINNING, DECISION_TREE_1D_BINNING, CHIMERGE_BINNING, or GA_REFINED_BINNING."
#endif
}

const char *quantizer_get_mode_name(void) {
    return quantizer_mode_name();
}
#if BINNING_MODE == KMEANS_1D_BINNING
static int quantizer_export_centers_csv(const char *filepath) {
    if (!filepath || filepath[0] == '\0') {
        return -1;
    }
    if (!g_quantizer_state.fitted) {
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
            g_quantizer_state.num_features,
            g_quantizer_state.num_levels);
    fprintf(file, "feature");
    for (int k = 0; k < g_quantizer_state.num_levels; k++) {
        fprintf(file, ",center_%03d", k);
    }
    fprintf(file, "\n");

    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        fprintf(file, "%d", feature);
        for (int k = 0; k < g_quantizer_state.num_levels; k++) {
            fprintf(file, ",%.17g", g_quantizer_state.centers[center_index(feature, k)]);
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
    if (!g_quantizer_state.fitted) {
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
            g_quantizer_state.num_features,
            g_quantizer_state.num_levels);
    fprintf(file, "feature,refinement_count,tree_split_count,fallback_threshold_count,initial_interval_count\n");
    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        fprintf(file, "%d,0,0,0,0\n", feature);
    }
#elif BINNING_MODE == QUANTILE_BINNING || BINNING_MODE == KMEANS_1D_BINNING || BINNING_MODE == DECISION_TREE_1D_BINNING || BINNING_MODE == CHIMERGE_BINNING
    int cut_count = g_quantizer_state.num_levels - 1;
    fprintf(file,
            "#quantizer,mode=%s,num_features=%d,num_levels=%d,total_refinements=%d,non_finite_replacements=%d,total_tree_splits=%d,total_fallback_thresholds=%d\n",
            quantizer_mode_name(),
            g_quantizer_state.num_features,
            g_quantizer_state.num_levels,
            g_quantizer_statistics.total_refinements,
            g_quantizer_state.non_finite_replacements,
            g_quantizer_statistics.total_tree_splits,
            g_quantizer_statistics.total_fallback_thresholds);
    fprintf(file, "feature,refinement_count,tree_split_count,fallback_threshold_count,initial_interval_count");
    for (int k = 0; k < cut_count; k++) {
        fprintf(file, ",cut_%03d", k);
    }
    fprintf(file, "\n");

    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        fprintf(file,
                "%d,%d,%d,%d,%d",
                feature,
                g_quantizer_statistics.refinement_counts[feature],
                g_quantizer_statistics.tree_split_counts[feature],
                g_quantizer_statistics.fallback_threshold_counts[feature],
                g_quantizer_statistics.initial_interval_counts[feature]);
        for (int k = 0; k < cut_count; k++) {
            fprintf(file, ",%.17g", g_quantizer_state.boundaries[boundary_index(feature, k)]);
        }
        fprintf(file, "\n");
    }
#elif BINNING_MODE == GA_REFINED_BINNING
    int cut_count = g_quantizer_state.num_levels - 1;
    fprintf(file,
            "#quantizer,mode=%s,num_features=%d,num_levels=%d,total_refinements=%d,non_finite_replacements=%d,total_zero_width_intervals=%d,epsilon=%.17g,alpha=%.17g,refined_ready=%d\n",
            quantizer_mode_name(),
            g_quantizer_state.num_features,
            g_quantizer_state.num_levels,
            g_quantizer_statistics.total_refinements,
            g_quantizer_state.non_finite_replacements,
            g_quantizer_statistics.total_zero_width_intervals,
            (double)GA_BINNING_EPSILON,
            (double)GA_BINNING_ALPHA,
            g_quantizer_state.ga_refined_ready);
    fprintf(file, "feature,refinement_count,empty_bin_count,zero_width_interval_count");
    for (int k = 0; k < cut_count; k++) {
        fprintf(file, ",flip_%03d", k);
    }
    for (int k = 0; k < cut_count; k++) {
        fprintf(file, ",weight_%03d", k);
    }
    for (int k = 0; k < cut_count; k++) {
        fprintf(file, ",cut_%03d", k);
    }
    fprintf(file, "\n");

    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        fprintf(file,
                "%d,%d,%d,%d",
                feature,
                g_quantizer_statistics.refinement_counts[feature],
                g_quantizer_statistics.empty_bin_counts[feature],
                g_quantizer_statistics.zero_width_interval_counts[feature]);
        for (int k = 0; k < cut_count; k++) {
            fprintf(file, ",%u", (unsigned)g_quantizer_state.ga_refined_flip_counts[boundary_index(feature, k)]);
        }
        for (int k = 0; k < cut_count; k++) {
            fprintf(file, ",%.17g", g_quantizer_state.ga_refined_transition_weights[boundary_index(feature, k)]);
        }
        for (int k = 0; k < cut_count; k++) {
            fprintf(file, ",%.17g", g_quantizer_state.boundaries[boundary_index(feature, k)]);
        }
        fprintf(file, "\n");
    }
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING, QUANTILE_BINNING, KMEANS_1D_BINNING, DECISION_TREE_1D_BINNING, CHIMERGE_BINNING, or GA_REFINED_BINNING."
#endif

    fclose(file);
    return 0;
}
