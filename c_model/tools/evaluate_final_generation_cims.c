#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config.h"
#include "../include/data_reader.h"
#include "../include/assoc_mem.h"
#include "../include/encoder.h"
#include "../include/evaluator.h"
#include "../include/item_mem.h"
#include "../include/quantizer.h"
#include "../include/trainer.h"

int output_mode = OUTPUT_MODE;

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

struct cim_header_metrics {
    int generation;
    int candidate;
    int num_levels;
    int num_features;
    int dimension;
    double validation_accuracy;
    double similarity;
};

static int has_suffix(const char *s, const char *suffix) {
    size_t n = strlen(s);
    size_t m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *copy = (char *)malloc(n);
    if (copy) {
        memcpy(copy, s, n);
    }
    return copy;
}

static int parse_header_token_int(const char *header, const char *key, int *out) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "%s=", key);
    const char *p = strstr(header, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    *out = (int)strtol(p, NULL, 10);
    return 1;
}

static int parse_header_token_double(const char *header, const char *key, double *out) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "%s=", key);
    const char *p = strstr(header, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    *out = strtod(p, NULL);
    return 1;
}

static int read_cim_header(const char *path, struct cim_header_metrics *metrics) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }

    char line[2048];
    if (!fgets(line, sizeof(line), f)) {
        fprintf(stderr, "failed to read header from %s\n", path);
        fclose(f);
        return -1;
    }
    fclose(f);

    memset(metrics, 0, sizeof(*metrics));
    if (strncmp(line, "#ga_cim_export", 14) != 0) {
        fprintf(stderr, "missing ga_cim_export header in %s\n", path);
        return -1;
    }

    parse_header_token_int(line, "generation", &metrics->generation);
    parse_header_token_int(line, "candidate", &metrics->candidate);
    parse_header_token_int(line, "num_levels", &metrics->num_levels);
    parse_header_token_int(line, "num_features", &metrics->num_features);
    parse_header_token_int(line, "dimension", &metrics->dimension);
    parse_header_token_double(line, "accuracy", &metrics->validation_accuracy);
    parse_header_token_double(line, "similarity", &metrics->similarity);
    return 0;
}

static int compare_cim_names(const void *a, const void *b) {
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

static char **list_cim_files(const char *dir_path, int *count_out) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "failed to open directory %s: %s\n", dir_path, strerror(errno));
        return NULL;
    }

    int cap = 128;
    int count = 0;
    char **names = (char **)calloc((size_t)cap, sizeof(char *));
    if (!names) {
        closedir(dir);
        return NULL;
    }

    struct dirent *ent = NULL;
    while ((ent = readdir(dir)) != NULL) {
        if (strncmp(ent->d_name, "cim_", 4) != 0 || !has_suffix(ent->d_name, ".csv")) {
            continue;
        }
        if (count == cap) {
            cap *= 2;
            char **tmp = (char **)realloc(names, (size_t)cap * sizeof(char *));
            if (!tmp) {
                closedir(dir);
                for (int i = 0; i < count; i++) free(names[i]);
                free(names);
                return NULL;
            }
            names = tmp;
        }
        names[count] = xstrdup(ent->d_name);
        if (!names[count]) {
            closedir(dir);
            for (int i = 0; i < count; i++) free(names[i]);
            free(names);
            return NULL;
        }
        count++;
    }
    closedir(dir);

    qsort(names, (size_t)count, sizeof(char *), compare_cim_names);
    *count_out = count;
    return names;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [cim_generation_dir] [dataset]\n"
            "Default cim_generation_dir: analysis/generated_data/ccim_exports/ga_run_20260811_133846_01_dataset03_seed01_l40_d10000_pop128_gen64/generation_0064\n"
            "Default dataset: 3\n",
            prog);
}

int main(int argc, char **argv) {
    const char *cim_dir = "analysis/generated_data/ccim_exports/ga_run_20260811_133846_01_dataset03_seed01_l40_d10000_pop128_gen64/generation_0064";
    int dataset = 3;

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            usage(argv[0]);
            return EXIT_SUCCESS;
        }
        cim_dir = argv[1];
    }
    if (argc > 2) {
        dataset = atoi(argv[2]);
    }
    if (argc > 3) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    double **trainingData = NULL;
    double **validationData = NULL;
    double **testingData = NULL;
    int *trainingLabels = NULL;
    int *validationLabels = NULL;
    int *testingLabels = NULL;
    int trainingSamples = 0;
    int validationSamples = 0;
    int testingSamples = 0;

    getDataWithValSet(dataset,
                      &trainingData,
                      &validationData,
                      &testingData,
                      &trainingLabels,
                      &validationLabels,
                      &testingLabels,
                      &trainingSamples,
                      &validationSamples,
                      &testingSamples,
                      VALIDATION_RATIO);

    quantizer_clear();
    if (quantizer_fit_from_training(trainingData,
                                    trainingLabels,
                                    trainingSamples,
                                    NUM_FEATURES,
                                    NUM_LEVELS) != 0) {
        fprintf(stderr, "failed to fit quantizer for dataset %d\n", dataset);
        return EXIT_FAILURE;
    }

    int cim_count = 0;
    char **cim_names = list_cim_files(cim_dir, &cim_count);
    if (!cim_names || cim_count == 0) {
        fprintf(stderr, "no cim_*.csv files found in %s\n", cim_dir);
        return EXIT_FAILURE;
    }

    printf("candidate,file,header_validation_class_average_accuracy,header_similarity,recomputed_validation_overall_accuracy,recomputed_validation_class_average_accuracy,test_overall_accuracy,test_class_average_accuracy,validation_overall_minus_test_overall,abs_header_validation_class_average_delta,recomputed_similarity\n");

    for (int i = 0; i < cim_count; i++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", cim_dir, cim_names[i]);

        struct cim_header_metrics header;
        if (read_cim_header(path, &header) != 0) {
            continue;
        }

        struct item_memory itemMem;
        load_precomp_item_mem_from_csv(&itemMem, path, NUM_LEVELS, NUM_FEATURES);

        struct encoder enc;
        init_encoder(&enc, &itemMem);

        struct associative_memory assMem;
        init_assoc_mem(&assMem);
        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);

        struct timeseries_eval_result eval_val = evaluate_model_timeseries_direct(&enc, &assMem, validationData, validationLabels, validationSamples);
        struct timeseries_eval_result eval_test = evaluate_model_timeseries_direct(&enc, &assMem, testingData, testingLabels, testingSamples);

        double header_delta = eval_val.class_average_accuracy - header.validation_accuracy;
        if (header_delta < 0.0) header_delta = -header_delta;

        printf("%d,%s,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
               header.candidate,
               cim_names[i],
               header.validation_accuracy,
               header.similarity,
               eval_val.overall_accuracy,
               eval_val.class_average_accuracy,
               eval_test.overall_accuracy,
               eval_test.class_average_accuracy,
               eval_val.overall_accuracy - eval_test.overall_accuracy,
               header_delta,
               eval_val.class_vector_similarity);

        free_assoc_mem(&assMem);
        free_item_memory(&itemMem);
    }

    for (int i = 0; i < cim_count; i++) free(cim_names[i]);
    free(cim_names);

    freeData(trainingData, trainingSamples);
    freeData(validationData, validationSamples);
    freeData(testingData, testingSamples);
    freeCSVLabels(trainingLabels);
    freeCSVLabels(validationLabels);
    freeCSVLabels(testingLabels);

    return EXIT_SUCCESS;
}
