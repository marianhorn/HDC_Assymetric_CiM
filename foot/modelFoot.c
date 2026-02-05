//This serves as main file.
//Classifies Dietmars foot movements

#include <stdio.h>
#include <stdlib.h>
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

int output_mode = OUTPUT_MODE;

int main(){
    if (output_mode >= OUTPUT_BASIC) {
        printf("\nHDC-classification for EMG-signals:\n\n");
    }
    for(int dataset = 2; dataset<3;dataset++){

        if (output_mode >= OUTPUT_BASIC) {
            printf("\n\nModel for dataset #%d\n",dataset);
        }
        #if PRECOMPUTED_ITEM_MEMORY
        struct item_memory itemMem;
        init_precomp_item_memory(&itemMem,NUM_LEVELS,NUM_FEATURES);

        struct encoder enc;
        init_encoder(&enc,&itemMem);
        #else
        struct item_memory electrodes;
        struct item_memory intensityLevels;
        init_item_memory(&electrodes,NUM_FEATURES);
        init_continuous_item_memory(&intensityLevels,NUM_LEVELS);

        struct encoder enc;
        init_encoder(&enc,&electrodes,&intensityLevels);
        #endif

        double** trainingData;
        double** validationData;
        double** testingData;
        int* trainingLabels;
        int* validationLabels;
        int* testingLabels;
        int trainingSamples, validationSamples, testingSamples;

        struct associative_memory assMem;
        init_assoc_mem(&assMem);

        double validationRatio = 0.2;
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
                          validationRatio);

        #if USE_GENETIC_ITEM_MEMORY
        #if PRECOMPUTED_ITEM_MEMORY
        optimize_item_memory(&itemMem,
                             trainingData,
                             trainingLabels,
                             trainingSamples,
                             validationData,
                             validationLabels,
                             validationSamples);
        #else
        optimize_item_memory(&intensityLevels,
                             &electrodes,
                             trainingData,
                             trainingLabels,
                             trainingSamples,
                             validationData,
                             validationLabels,
                             validationSamples);
        #endif
        #endif

        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);
            
        (void)evaluate_model_timeseries_direct(&enc,&assMem,testingData,testingLabels,testingSamples);
        store_precomp_item_mem_to_csv(&itemMem,"./analysis/item_mem_naive.csv",NUM_LEVELS, NUM_FEATURES);
        // Free allocated memory
        freeData(trainingData, trainingSamples);
        if (validationData && validationSamples > 0) {
            freeData(validationData, validationSamples);
        }
        freeData(testingData, testingSamples);
        free(trainingLabels);
        free(validationLabels);
        free(testingLabels);
        free_assoc_mem(&assMem);

        #if PRECOMPUTED_ITEM_MEMORY
        free_item_memory(&itemMem);
        #else
        free_item_memory(&electrodes);
        free_item_memory(&intensityLevels);
        #endif
    }
    return 1;
}
