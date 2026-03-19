#include "quantizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QUANTIZER_EXPORT_ENABLED 0
#define QUANTIZER_EXPORT_PATH_TEMPLATE "analysis/quantizer_cuts_dataset%02d.csv"

static double *g_cuts = NULL;
static int *g_refinement_counts = NULL;
static int g_num_features = 0;
static int g_num_levels = 0;
static int g_fitted = 0;
static int g_non_finite_replacements = 0;
static int g_total_refinements = 0;

#if BINNING_MODE == QUANTILE_BINNING
static int cut_index(int feature_idx, int cut_idx) {
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
static int map_quantile_value(int feature_idx, double x) {
    if (!g_fitted) {
        fprintf(stderr, "quantizer: map called before fit.\n");
        exit(EXIT_FAILURE);
    }
    if (feature_idx < 0 || feature_idx >= g_num_features) {
        fprintf(stderr, "quantizer: feature index %d out of range [0,%d).\n", feature_idx, g_num_features);
        exit(EXIT_FAILURE);
    }
    if (g_num_levels <= 1) {
        return 0;
    }

    int cut_count = g_num_levels - 1;
    const double *cuts = &g_cuts[feature_idx * cut_count];

    if (isnan(x) || x <= cuts[0]) {
        return 0;
    }
    if (x > cuts[cut_count - 1]) {
        return g_num_levels - 1;
    }

    int lo = 0;
    int hi = cut_count - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (x <= cuts[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
#endif

void quantizer_clear(void) {
    free(g_cuts);
    free(g_refinement_counts);
    g_cuts = NULL;
    g_refinement_counts = NULL;
    g_num_features = 0;
    g_num_levels = 0;
    g_fitted = 0;
    g_non_finite_replacements = 0;
    g_total_refinements = 0;
}

int quantizer_is_fitted(void) {
    return g_fitted;
}

int quantizer_fit_from_training(double **training_data,
                                int training_samples,
                                int num_features,
                                int num_levels) {
    quantizer_clear();

#if BINNING_MODE == UNIFORM_BINNING
    (void)training_data;
    (void)training_samples;
    if (num_features <= 0 || num_levels <= 0) {
        fprintf(stderr, "quantizer: invalid fit input.\n");
        return -1;
    }

    g_num_features = num_features;
    g_num_levels = num_levels;
    g_fitted = 1;
    return 0;
#elif BINNING_MODE == QUANTILE_BINNING
    if (!training_data || training_samples <= 0 || num_features <= 0 || num_levels <= 0) {
        fprintf(stderr, "quantizer: invalid fit input.\n");
        return -1;
    }
    if (num_levels <= 1) {
        g_num_features = num_features;
        g_num_levels = num_levels;
        g_fitted = 1;
        return 0;
    }

    g_num_features = num_features;
    g_num_levels = num_levels;

    int cut_count = num_levels - 1;
    size_t total_cuts = (size_t)num_features * (size_t)cut_count;
    g_cuts = (double *)malloc(total_cuts * sizeof(double));
    g_refinement_counts = (int *)calloc((size_t)num_features, sizeof(int));
    double *sorted_values = (double *)malloc((size_t)training_samples * sizeof(double));

    if (!g_cuts || !g_refinement_counts || !sorted_values) {
        fprintf(stderr, "quantizer: failed to allocate fit buffers.\n");
        free(sorted_values);
        quantizer_clear();
        return -1;
    }

    for (int feature = 0; feature < num_features; feature++) {
        for (int i = 0; i < training_samples; i++) {
            double v = training_data[i][feature];
            if (!isfinite(v)) {
                v = 0.0;
                g_non_finite_replacements++;
            }
            sorted_values[i] = v;
        }

        qsort(sorted_values, (size_t)training_samples, sizeof(double), compare_doubles);

        for (int k = 1; k < num_levels; k++) {
            double q = (double)k / (double)num_levels;
            double p = q * (double)(training_samples - 1);
            int left = (int)floor(p);
            int right = (int)ceil(p);
            double alpha = p - (double)left;
            double cut_value = sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha;
            g_cuts[cut_index(feature, k - 1)] = cut_value;
        }

        int refinements = 0;
        for (int k = 1; k < cut_count; k++) {
            int prev_idx = cut_index(feature, k - 1);
            int curr_idx = cut_index(feature, k);
            if (g_cuts[curr_idx] <= g_cuts[prev_idx]) {
                g_cuts[curr_idx] = nextafter(g_cuts[prev_idx], INFINITY);
                refinements++;
            }
        }
        g_refinement_counts[feature] = refinements;
        g_total_refinements += refinements;

        if (refinements > 0 && output_mode >= OUTPUT_BASIC) {
            fprintf(stderr,
                    "quantizer: feature %d required %d cut refinements to preserve strict bin order.\n",
                    feature,
                    refinements);
        }
    }

    free(sorted_values);
    g_fitted = 1;

    if (g_non_finite_replacements > 0 && output_mode >= OUTPUT_BASIC) {
        fprintf(stderr,
                "quantizer: replaced %d non-finite training values with 0.0 during fit.\n",
                g_non_finite_replacements);
    }
    if (output_mode >= OUTPUT_DETAILED) {
        fprintf(stdout,
                "quantizer: fitted quantile cuts for %d features, %d levels (total refinements: %d).\n",
                g_num_features,
                g_num_levels,
                g_total_refinements);
    }

    return 0;
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING or QUANTILE_BINNING."
#endif
}

int get_signal_level(int feature_idx, double emg_value) {
#if BINNING_MODE == QUANTILE_BINNING
    return map_quantile_value(feature_idx, emg_value);
#elif BINNING_MODE == UNIFORM_BINNING
    (void)feature_idx;
    return get_signal_level_uniform(emg_value);
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING or QUANTILE_BINNING."
#endif
}

int quantizer_export_cuts_csv_for_dataset(int dataset) {
    if (!QUANTIZER_EXPORT_ENABLED) {
        return 0;
    }

    char filepath[512];
    int written = snprintf(filepath,
                           sizeof(filepath),
                           QUANTIZER_EXPORT_PATH_TEMPLATE,
                           dataset);
    if (written < 0 || (size_t)written >= sizeof(filepath)) {
        fprintf(stderr, "quantizer: export path is too long for dataset %d.\n", dataset);
        return -1;
    }

    return quantizer_export_cuts_csv(filepath);
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

#if BINNING_MODE == QUANTILE_BINNING
    int cut_count = g_num_levels - 1;
    fprintf(file,
            "#quantizer,num_features=%d,num_levels=%d,total_refinements=%d,non_finite_replacements=%d\n",
            g_num_features,
            g_num_levels,
            g_total_refinements,
            g_non_finite_replacements);

    fprintf(file, "feature,refinement_count");
    for (int k = 0; k < cut_count; k++) {
        fprintf(file, ",cut_%03d", k);
    }
    fprintf(file, "\n");

    for (int feature = 0; feature < g_num_features; feature++) {
        fprintf(file, "%d,%d", feature, g_refinement_counts[feature]);
        for (int k = 0; k < cut_count; k++) {
            fprintf(file, ",%.17g", g_cuts[cut_index(feature, k)]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    return 0;
#elif BINNING_MODE == UNIFORM_BINNING
    fprintf(file,
            "#quantizer,mode=uniform,num_features=%d,num_levels=%d,total_refinements=0,non_finite_replacements=0\n",
            g_num_features,
            g_num_levels);
    fprintf(file, "feature,refinement_count\n");
    for (int feature = 0; feature < g_num_features; feature++) {
        fprintf(file, "%d,0\n", feature);
    }
    fclose(file);
    return 0;
#else
#error "Unsupported BINNING_MODE. Use UNIFORM_BINNING or QUANTILE_BINNING."
#endif
}
