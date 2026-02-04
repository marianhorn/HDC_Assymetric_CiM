//This serves as main file.
//Classifies Dietmars foot movements

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/item_mem.h"
#include "../hdc_infrastructure/asymItemMemory.h"
#include "../hdc_infrastructure/encoder.h"
#include "../hdc_infrastructure/operations.h"
#include "dataReaderFootEMG.h"
#include "configFoot.h"
#include "../hdc_infrastructure/evaluator.h"
#include "../hdc_infrastructure/vector.h"
#include "../hdc_infrastructure/trainer.h"

int main(){
    #if OUTPUT_MODE>=OUTPUT_BASIC
        printf("\nHDC-classification for EMG-signals:\n\n");
    #endif
    for(int dataset = 2; dataset<3;dataset++){

        #if OUTPUT_MODE>= OUTPUT_BASIC
            printf("\n\nModel for dataset #%d\n",dataset);
        #endif
        #if PRECOMPUTED_ITEM_MEMORY
        struct item_memory itemMem;
        init_precomp_item_memory(&itemMem,NUM_LEVELS,NUM_FEATURES);

        struct encoder enc;
        init_encoder(&enc,&itemMem);
        #else
        struct item_memory electrodes;
        struct item_memory intensityLevels;
        struct item_memory intensityLevelsBaseline;
        int has_baseline = 0;
        init_item_memory(&electrodes,NUM_FEATURES);
        init_continuous_item_memory(&intensityLevels,NUM_LEVELS);

        struct encoder enc;
        init_encoder(&enc,&electrodes,&intensityLevels);
        #endif

        double** trainingData;
        double** testingData;
        int* trainingLabels;
        int* testingLabels;
        int trainingSamples, testingSamples;

        struct associative_memory assMem;
        init_assoc_mem(&assMem);

        getData(dataset,&trainingData,&testingData,&trainingLabels,&testingLabels,&trainingSamples,&testingSamples);
        char naive_path[256];
        snprintf(naive_path, sizeof(naive_path), "./item_mem_naive_%d.csv", dataset);
        store_item_mem_to_csv(&intensityLevels, naive_path);

        #if USE_GENETIC_ITEM_MEMORY
                intensityLevelsBaseline.num_vectors = intensityLevels.num_vectors;
                intensityLevelsBaseline.base_vectors = (Vector **)malloc(intensityLevelsBaseline.num_vectors * sizeof(Vector *));
                for (int i = 0; i < intensityLevelsBaseline.num_vectors; i++) {
                    intensityLevelsBaseline.base_vectors[i] = create_uninitialized_vector();
                    memcpy(intensityLevelsBaseline.base_vectors[i]->data,
                           intensityLevels.base_vectors[i]->data,
                           VECTOR_DIMENSION * sizeof(vector_element));
                }
                has_baseline = 1;

                optimize_item_memory(&intensityLevels,
                                    &electrodes,
                                     trainingData,
                                     trainingLabels,
                                     trainingSamples,
                                     testingData,
                                     testingLabels,
                                     testingSamples);
        #endif
        char optimized_path[256];
        snprintf(optimized_path, sizeof(optimized_path), "./item_mem_optimized_%d.csv", dataset);
        store_item_mem_to_csv(&intensityLevels, optimized_path);

        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);
            
        evaluate_model_timeseries_direct(&enc,&assMem,testingData,testingLabels,testingSamples);

        #if USE_GENETIC_ITEM_MEMORY
            if (has_baseline) {
                struct encoder baseline_enc;
                init_encoder(&baseline_enc, &electrodes, &intensityLevelsBaseline);
                struct associative_memory baselineAssMem;
                init_assoc_mem(&baselineAssMem);
                train_model_timeseries(trainingData, trainingLabels, trainingSamples, &baselineAssMem, &baseline_enc);
                evaluate_model_timeseries_direct(&baseline_enc, &baselineAssMem, testingData, testingLabels, testingSamples);
                free_assoc_mem(&baselineAssMem);
            }
        #endif

        // Free allocated memory
        freeData(trainingData, trainingSamples);
        freeData(testingData, testingSamples);
        free(trainingLabels);
        free(testingLabels);
        free_assoc_mem(&assMem);

        #if PRECOMPUTED_ITEM_MEMORY
        free_item_memory(&itemMem);
        #else
        free_item_memory(&electrodes);
        free_item_memory(&intensityLevels);
        if (has_baseline) {
            free_item_memory(&intensityLevelsBaseline);
        }
        #endif
    }
    return 1;
}
