#include "quantizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define QUANTIZER_EXPORT_ENABLED 0
#define QUANTIZER_EXPORT_CUTS_PATH_TEMPLATE "analysis/quantizer_cuts_dataset%02d.csv"

typedef struct {
    double *boundaries;
    int num_features;
    int num_levels;
    int fitted;
} quantizer_state_t;

static quantizer_state_t g_quantizer_state = {0};

static size_t boundary_count_total_for(int num_features, int num_levels) {
    if (num_features <= 0 || num_levels <= 1) {
        return 0u;
    }
    return (size_t)num_features * (size_t)(num_levels - 1);
}

static int boundary_index(int feature_idx, int cut_idx) {
    return feature_idx * (g_quantizer_state.num_levels - 1) + cut_idx;
}

static const char *quantizer_mode_name(void) {
    return "uniform";
}

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
        fprintf(stderr,
                "quantizer: feature index %d out of range [0,%d).\n",
                feature_idx,
                g_quantizer_state.num_features);
        exit(EXIT_FAILURE);
    }
    return map_value_with_boundaries_unchecked(feature_idx, x);
}

static int install_uniform_boundaries(void) {
    int cut_count = g_quantizer_state.num_levels - 1;
    if (cut_count <= 0 || g_quantizer_state.boundaries == NULL) {
        return 0;
    }

    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        for (int level = 0; level < cut_count; level++) {
            long long numerator = 20000LL * (long long)(level + 1) - 10000LL;
            long long threshold_scaled = (numerator + (long long)cut_count - 1LL) / (long long)cut_count;
            double boundary = ((double)(threshold_scaled - 1LL) - 10000.0) / 10000.0;
            g_quantizer_state.boundaries[boundary_index(feature, level)] = boundary;
        }
    }
    return 0;
}

void quantizer_clear(void) {
    free(g_quantizer_state.boundaries);
    g_quantizer_state.boundaries = NULL;
    g_quantizer_state.num_features = 0;
    g_quantizer_state.num_levels = 0;
    g_quantizer_state.fitted = 0;
}

int quantizer_is_fitted(void) {
    return g_quantizer_state.fitted;
}

int quantizer_fit_from_training(double **training_data,
                                const int *training_labels,
                                int training_samples,
                                int num_features,
                                int num_levels) {
    (void)training_data;
    (void)training_labels;
    (void)training_samples;

    quantizer_clear();

    if (num_features <= 0 || num_levels <= 0) {
        fprintf(stderr, "quantizer: invalid fit input.\n");
        return -1;
    }

    g_quantizer_state.num_features = num_features;
    g_quantizer_state.num_levels = num_levels;

    size_t boundary_count = boundary_count_total_for(num_features, num_levels);
    if (boundary_count > 0) {
        g_quantizer_state.boundaries = (double *)malloc(boundary_count * sizeof(double));
        if (g_quantizer_state.boundaries == NULL) {
            fprintf(stderr, "quantizer: failed to allocate state buffers.\n");
            quantizer_clear();
            return -1;
        }
    }

    if (install_uniform_boundaries() != 0) {
        quantizer_clear();
        return -1;
    }
    g_quantizer_state.fitted = 1;
    return 0;
}

int get_signal_level(int feature_idx, double emg_value) {
    return map_value_with_boundaries_checked(feature_idx, emg_value);
}

const char *quantizer_get_mode_name(void) {
    return quantizer_mode_name();
}

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

    return quantizer_export_cuts_csv(cuts_filepath);
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

    int cut_count = g_quantizer_state.num_levels - 1;
    fprintf(file,
            "#quantizer,mode=%s,num_features=%d,num_levels=%d,total_refinements=0,non_finite_replacements=0,total_tree_splits=0,total_fallback_thresholds=0\n",
            quantizer_mode_name(),
            g_quantizer_state.num_features,
            g_quantizer_state.num_levels);
    fprintf(file, "feature,refinement_count,tree_split_count,fallback_threshold_count,initial_interval_count");
    for (int k = 0; k < cut_count; k++) {
        fprintf(file, ",cut_%03d", k);
    }
    fprintf(file, "\n");
    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        fprintf(file, "%d,0,0,0,0", feature);
        for (int k = 0; k < cut_count; k++) {
            fprintf(file, ",%.17g", g_quantizer_state.boundaries[boundary_index(feature, k)]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    return 0;
}

int quantizer_export_systemc_text(const char *filepath) {
    if (!filepath || filepath[0] == '\0') {
        return -1;
    }
    if (!g_quantizer_state.fitted) {
        fprintf(stderr, "quantizer: SystemC export requested before fit.\n");
        return -1;
    }
    if (g_quantizer_state.num_features <= 0 || g_quantizer_state.num_levels <= 0) {
        fprintf(stderr, "quantizer: invalid quantizer dimensions for SystemC export.\n");
        return -1;
    }

    FILE *file = fopen(filepath, "w");
    if (!file) {
        perror("quantizer: failed to open SystemC export path");
        return -1;
    }

    fprintf(file,
            "#systemc_quantizer mode=%s num_features=%d num_levels=%d layout=feature_major_cut_minor\n",
            quantizer_mode_name(),
            g_quantizer_state.num_features,
            g_quantizer_state.num_levels);

    int cut_count = g_quantizer_state.num_levels - 1;
    for (int feature = 0; feature < g_quantizer_state.num_features; feature++) {
        fprintf(file, "%d", feature);
        for (int cut = 0; cut < cut_count; cut++) {
            fprintf(file, " %.17g", g_quantizer_state.boundaries[boundary_index(feature, cut)]);
        }
        fputc('\n', file);
    }

    fclose(file);
    return 0;
}
