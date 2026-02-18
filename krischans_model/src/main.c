#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "hdc_types.h"
#include "hdc_memory.h"
#include "hdc_encode.h"
#include "hdc_train.h"
#include "hdc_classify.h"
#include "hdc_features.h"
#include "block_accumulator.h"
#include "hdc_utils.h"

int main(int argc, char* argv[])
{
    if(argc < 4) {
        printf("Usage: %s <D> <M> <mode>\n", argv[0]);
        printf("mode = 0: classic encoding\n");
        printf("mode = 1: rolling 5-block encoding\n");
        return 1;
    }

    D = atoi(argv[1]);
    M = atoi(argv[2]);
    int mode = atoi(argv[3]);

    printf("=== CPU HDC Dynamic ===\n");
    printf("D = %d, M = %d, mode = %d\n", D, M, mode);

    alloc_memory();
    if(mode == 1) block_init();

    load_im("memoryfiles/position-vectors.txt");
    load_cm("memoryfiles/value_vectors.txt");

    // -----------------------------
    // Load training data
    // -----------------------------
    float** X_train;
    int* y_train;
    int train_count;

    load_csv_features("datasets/training_emg.csv", &X_train, &train_count);
    load_csv_labels("datasets/training_labels.csv", &y_train, &train_count);

    // -----------------------------
    // Load testing data
    // -----------------------------
    float** X_test;
    int* y_test;
    int test_count;

    load_csv_features("datasets/testing_emg.csv", &X_test, &test_count);
    load_csv_labels("datasets/testing_labels.csv", &y_test, &test_count);

    // -----------------------------
    // Allocate class vectors
    // -----------------------------
    hv_t* class_vectors[NUM_CLASSES];
    for(int c = 0; c < NUM_CLASSES; c++)
        class_vectors[c] = malloc(train_count * sizeof(hv_t));

    int class_index[NUM_CLASSES] = {0};

    // For rolling
    hv_t rolling_acc = NULL;
    if(mode == 1) {
        rolling_acc = hv_alloc();   // output des Rolling-Fensters
        block_reset();
    }

    // -----------------------------
    // TRAINING
    // -----------------------------
    printf("[TRAINING]\n");

    // temporärer HV für encode_sample()
    hv_t hv_single = hv_alloc();

    for(int i = 0; i < train_count; i++) 
    {
        int c = y_train[i];

        // Klassischer HV
        encode_sample(hv_single, X_train[i]);
        if(mode == 0) {
            // Klassischer Pfad — direkt trainieren
            class_vectors[c][class_index[c]] = hv_alloc();
            hv_copy(class_vectors[c][class_index[c]], hv_single);
            class_index[c]++;
        }
        else {
            // Rolling Pfad
            block_accumulate(rolling_acc, hv_single);

            if(i >= BLOCK_WINDOW - 1) {
                // Erst ab Sample 5 entsteht ein Rolling-HV
                class_vectors[c][class_index[c]] = hv_alloc();
                hv_copy(class_vectors[c][class_index[c]], rolling_acc);
                class_index[c]++;
            }
        }
    }

    hv_free(hv_single);

    // -----------------------------
    // Train AM über die gesammelten Hypervektoren
    // -----------------------------
    for(int c = 0; c < NUM_CLASSES; c++) {
        printf("Training class %d with %d vectors\n", c, class_index[c]);
        train_class(AM[c], class_vectors[c], class_index[c]);
    }

    // -----------------------------
    // TESTING
    // -----------------------------
    printf("[TESTING]\n");

    hv_t hv_test_single = hv_alloc();
    hv_t hv_test_roll = NULL;
    if(mode == 1) {
        hv_test_roll = hv_alloc();
        memset(hv_test_roll, 0, chunks_per_vec()*sizeof(uint32_t));
        block_reset();
    }

    int correct = 0;

    for(int i = 0; i < test_count; i++) 
    {
        encode_sample(hv_test_single, X_test[i]);

        int pred;
        if(mode == 0) {
            pred = classify(hv_test_single);
        } else {
            block_accumulate(hv_test_roll, hv_test_single);
            if(i < BLOCK_WINDOW - 1) 
                continue; // noch kein vollständiger Block
            pred = classify(hv_test_roll);
        }

        if(pred == y_test[i]) correct++;
    }

    printf("Accuracy: %.2f%%\n", 100.0f * correct / test_count);

    // -----------------------------
    // Cleanup
    // -----------------------------
    hv_free(hv_test_single);
    if(hv_test_roll) hv_free(hv_test_roll);

    if(rolling_acc) hv_free(rolling_acc);

    free_memory();

    return 0;
}
