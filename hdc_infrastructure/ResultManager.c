#include "ResultManager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *result_file = NULL;
static int header_written = 0;

static void write_csv_header(FILE *file) {
    if (!file) {
        return;
    }
    fprintf(file,
            "num_levels,num_features,vector_dimension,bipolar_mode,precomputed_item_memory,"
            "use_genetic_item_memory,ga_selection_mode,n_gram_size,window,downsample,validation_ratio,"
            "overall_accuracy,class_average_accuracy,class_vector_similarity,correct,not_correct,transition_error,total,info\n");
}

static void write_csv_escaped(FILE *file, const char *value) {
    if (!file) {
        return;
    }
    if (!value) {
        fprintf(file, "\"");
        fprintf(file, "\"");
        return;
    }
    fputc('"', file);
    for (const char *p = value; *p; p++) {
        if (*p == '"') {
            fputc('"', file);
        }
        fputc(*p, file);
    }
    fputc('"', file);
}

void result_manager_init(void) {
    if (result_file) {
        return;
    }

#ifndef RESULT_CSV_PATH
#define RESULT_CSV_PATH "analysis/results.csv"
#endif

    result_file = fopen(RESULT_CSV_PATH, "a+");
    if (!result_file) {
        fprintf(stderr, "ResultManager: failed to open %s\n", RESULT_CSV_PATH);
        return;
    }

    if (!header_written) {
        fseek(result_file, 0, SEEK_END);
        long size = ftell(result_file);
        if (size <= 0) {
            write_csv_header(result_file);
        }
        header_written = 1;
    }
}

void result_manager_close(void) {
    if (result_file) {
        fclose(result_file);
        result_file = NULL;
        header_written = 0;
    }
}

void addResult(const struct timeseries_eval_result *result, const char *info) {
    if (!result) {
        return;
    }
    if (!result_file) {
        result_manager_init();
    }
    if (!result_file) {
        return;
    }

    fprintf(result_file,
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,",
            NUM_LEVELS,
            NUM_FEATURES,
            VECTOR_DIMENSION,
            BIPOLAR_MODE,
            PRECOMPUTED_ITEM_MEMORY,
            USE_GENETIC_ITEM_MEMORY,
#ifdef GA_SELECTION_MODE
            GA_SELECTION_MODE,
#else
            -1,
#endif
            N_GRAM_SIZE,
            WINDOW,
            DOWNSAMPLE,
            VALIDATION_RATIO);

    fprintf(result_file,
            "%.8f,%.8f,%.8f,%zu,%zu,%zu,%zu,",
            result->overall_accuracy,
            result->class_average_accuracy,
            result->class_vector_similarity,
            result->correct,
            result->not_correct,
            result->transition_error,
            result->total);

    write_csv_escaped(result_file, info);
    fputc('\n', result_file);
    fflush(result_file);
}
