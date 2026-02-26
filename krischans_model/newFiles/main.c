#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "hdc_types.h"
#include "hdc_memory.h"
#include "hdc_encode.h"
#include "hdc_train.h"
#include "hdc_classify.h"
#include "hdc_features.h"
#include "block_accumulator.h"
#include "hdc_utils.h"

// -------------------- Helpers --------------------

// popcount helper (portable)
static inline int popcount32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int c = 0;
    while (x) { x &= (x - 1); c++; }
    return c;
#endif
}

// Hamming distance between two hypervectors (assumes hv_t is uint32_t* chunks)
static int hv_hamming_local(const hv_t a, const hv_t b) {
    int dist = 0;
    const int n = chunks_per_vec();
    for (int i = 0; i < n; i++) {
        dist += popcount32(((const uint32_t*)a)[i] ^ ((const uint32_t*)b)[i]);
    }
    return dist;
}


// -------------------- Main --------------------

int main(int argc, char* argv[])
{
    // Usage:
    //   <D> <M> <mode> <train_fraction>
    if (argc < 5) {
        printf("Usage: %s <D> <M> <mode> <train_fraction>\n", argv[0]);
        printf("mode = 0: classic encoding\n");
        printf("mode = 1: rolling %d-block encoding\n", BLOCK_WINDOW);
        printf("train_fraction: e.g. 0.2 for 20%% (use dot, not comma)\n");
        return 1;
    }

    D = atoi(argv[1]);
    M = atoi(argv[2]);
    int mode = atoi(argv[3]);

    // parse train_fraction
    errno = 0;
    char* endp = NULL;
    float train_fraction = strtof(argv[4], &endp);
    if (errno != 0 || endp == argv[4] || *endp != '\0' || train_fraction <= 0.0f || train_fraction > 1.0f) {
        printf("ERROR: train_fraction must be a number in (0, 1], e.g. 0.2\n");
        return 1;
    }

    printf("=== CPU HDC Dynamic ===\n");
    printf("D = %d, M = %d, mode = %d, train_fraction = %.3f (%.1f%%)\n",
           D, M, mode, train_fraction, train_fraction * 100.0f);

    // Timing
    int64_t total_duration = 0; // testing
    struct timespec start_time, end_time;
    int64_t train_total_duration = 0;
    int train_measured_runs = 0;

    alloc_memory();
    if (mode == 1) block_init();

    load_im("memoryfiles/position-vectors.txt");
    load_cm("memoryfiles/value_vectors.txt");

    // -----------------------------
    // Load training data
    // -----------------------------
    float** X_train;
    int* y_train;
    int train_count;
    if (load_csv_features("datasets/training_emg.csv", &X_train, &train_count) != 0) {
        perror("load_csv_features training_emg.csv");
        return 1;
    }
    if (load_csv_labels("datasets/training_labels.csv", &y_train, &train_count) != 0) {
        perror("load_csv_labels training_labels.csv");
        return 1;
    }

    // -----------------------------
    // Calibrate quantization ranges (per feature) from training data
    // -----------------------------
    float minv[N];
    float maxv[N];
    for (int f = 0; f < N; f++) {
        minv[f] = 1e30f;
        maxv[f] = -1e30f;
    }

    for (int i = 0; i < train_count; i++) {
        for (int f = 0; f < N; f++) {
            float v = X_train[i][f];
            if (v < minv[f]) minv[f] = v;
            if (v > maxv[f]) maxv[f] = v;
        }
    }

    for (int f = 0; f < N; f++) {
        if (maxv[f] <= minv[f]) {
            maxv[f] = minv[f] + 1.0f;
        }
    }

    //set_quantization_ranges(minv, maxv);

    // -----------------------------
    // Load testing data
    // -----------------------------
    float** X_test;
    int* y_test;
    int test_count;
    if (load_csv_features("datasets/testing_emg.csv", &X_test, &test_count) != 0) {
        perror("load_csv_features testing_emg.csv");
        return 1;
    }
    if (load_csv_labels("datasets/testing_labels.csv", &y_test, &test_count) != 0) {
        perror("load_csv_labels testing_labels.csv");
        return 1;
    }

    // -----------------------------
    // Determine how many training HVs exist per class
    // -----------------------------
    int total_per_class[NUM_CLASSES]  = {0};
    int target_per_class[NUM_CLASSES] = {0};
    int seen_per_class[NUM_CLASSES]   = {0};

    for (int i = 0; i < train_count; i++) {
        int c = y_train[i];
        if (mode == 0) {
            total_per_class[c]++;
        } else {
            if (i >= BLOCK_WINDOW - 1) total_per_class[c]++;
        }
    }

    for (int c = 0; c < NUM_CLASSES; c++) {
        int k = (int)(total_per_class[c] * train_fraction);
        if (k < 1 && total_per_class[c] > 0) k = 1;
        if (total_per_class[c] == 0) k = 0;
        target_per_class[c] = k;
    }

    // -----------------------------
    // Allocate class vectors (ONLY target_per_class per label)
    // -----------------------------
    hv_t* class_vectors[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (target_per_class[c] > 0)
            class_vectors[c] = (hv_t*)malloc((size_t)target_per_class[c] * sizeof(hv_t));
        else
            class_vectors[c] = NULL;
    }

    int class_index[NUM_CLASSES] = {0};

    // For rolling
    hv_t rolling_acc = NULL;
    if (mode == 1) {
        rolling_acc = hv_alloc();
        block_reset();
    }

    srand(0);

    // -----------------------------
    // TRAINING
    // -----------------------------
    printf("[TRAINING]\n");

    hv_t hv_single = hv_alloc();

    for (int i = 0; i < train_count; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        int c = y_train[i];
        encode_sample(hv_single, X_train[i]);

        if (mode == 0) {
            if (target_per_class[c] > 0) {
                seen_per_class[c]++;
                if (class_index[c] < target_per_class[c]) {
                    class_vectors[c][class_index[c]] = hv_alloc();
                    hv_copy(class_vectors[c][class_index[c]], hv_single);
                    class_index[c]++;
                } else {
                    int j = rand() % seen_per_class[c];
                    if (j < target_per_class[c]) hv_copy(class_vectors[c][j], hv_single);
                }
            }
        } else {
            block_accumulate(rolling_acc, hv_single);
            if (i >= BLOCK_WINDOW - 1) {
                if (target_per_class[c] > 0) {
                    seen_per_class[c]++;
                    if (class_index[c] < target_per_class[c]) {
                        class_vectors[c][class_index[c]] = hv_alloc();
                        hv_copy(class_vectors[c][class_index[c]], rolling_acc);
                        class_index[c]++;
                    } else {
                        int j = rand() % seen_per_class[c];
                        if (j < target_per_class[c]) hv_copy(class_vectors[c][j], rolling_acc);
                    }
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &end_time);
        int64_t duration =
            (end_time.tv_sec - start_time.tv_sec) * 1000000LL +
            (end_time.tv_nsec - start_time.tv_nsec) / 1000LL;
        train_total_duration += duration;
        train_measured_runs++;
    }

    hv_free(hv_single);

    // Train AM
    for (int c = 0; c < NUM_CLASSES; c++) {
        printf("Training class %d with %d/%d vectors (%.0f%%)\n",
               c, class_index[c], total_per_class[c], train_fraction * 100.0f);

        if (class_index[c] > 0) train_class(AM[c], class_vectors[c], class_index[c]);
    }

    // Free stored training HVs
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int i = 0; i < class_index[c]; i++) hv_free(class_vectors[c][i]);
        free(class_vectors[c]);
    }

    // -----------------------------
    // Snapshot baseline AM (after offline training, before any online updates)
    // -----------------------------
    // -----------------------------
    // TESTING (no online learning)
    // -----------------------------
    printf("[TESTING]\n");

    hv_t hv_test_single = hv_alloc();
    hv_t hv_test_roll = NULL;
    if (mode == 1) {
        hv_test_roll = hv_alloc();
        memset(hv_test_roll, 0, (size_t)chunks_per_vec()*sizeof(uint32_t));
        block_reset();
    }
    
    FILE *fp_pred = fopen("predicted_labels.txt", "w");
    if (!fp_pred) { perror("fopen predicted_labels.txt"); return 1; }
    setvbuf(fp_pred, NULL, _IONBF, 0);
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) {
        printf("Writing predicted_labels.txt to: %s\n", cwd);
    }

    int correct_all = 0;
    int runs_all = 0;
    int measured_runs = 0;

    int64_t *run_durations = (int64_t*)malloc((size_t)test_count * sizeof(*run_durations));
    if (!run_durations) { perror("malloc run_durations"); return 1; }

    for (int i = 0; i < test_count; i++) {
        int pred_local = -1;
        int did_classify = 0;

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // Encode test sample
        encode_sample(hv_test_single, X_test[i]);

        if (mode == 0) {
            pred_local = classify(hv_test_single);
            did_classify = 1;
        } else {
            block_accumulate(hv_test_roll, hv_test_single);
            if (i >= BLOCK_WINDOW - 1) { pred_local = classify(hv_test_roll); did_classify = 1; }
        }

        // Write one line per test row
        if (fprintf(fp_pred, "%d,%d,%d\n", i, y_test[i], pred_local) < 0) {
            perror("fprintf predicted_labels.txt");
            fclose(fp_pred);
            return 1;
        }

        // timing only when classified
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        if (did_classify) {
            int64_t duration =
                (end_time.tv_sec - start_time.tv_sec) * 1000000LL +
                (end_time.tv_nsec - start_time.tv_nsec) / 1000LL;

            run_durations[measured_runs] = duration;
            total_duration += duration;
            measured_runs++;
        }

        // Accuracy only when classified AND label valid
        if (did_classify && y_test[i] != -1) {
            runs_all++;
            if (pred_local == y_test[i]) correct_all++;
        }
    }

    fclose(fp_pred);

    // Overall accuracy for the whole run
    if (runs_all > 0) {
        printf("Accuracy (online run / prequential): %.2f%% (%d/%d)\n",
               100.0 * (double)correct_all / (double)runs_all, correct_all, runs_all);
    } else {
        printf("Accuracy (online run): n/a\n");
    }

    if (train_measured_runs > 0) {
        printf("Avg train time: %.2f us\n", (double)train_total_duration / train_measured_runs);
    } else {
        printf("Avg train time: n/a\n");
    }

    if (measured_runs > 0) {
        printf("Avg test time: %.2f us\n", (double)total_duration / measured_runs);
    } else {
        printf("Avg test time: n/a\n");
    }

    free(run_durations);

    // -----------------------------
    // Cleanup
    // -----------------------------
    hv_free(hv_test_single);
    if (hv_test_roll) hv_free(hv_test_roll);
    if (rolling_acc) hv_free(rolling_acc);

    free_memory();
    return 0;
}
