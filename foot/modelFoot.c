//This serves as main file.
//Classifies Dietmars foot movements

#include <stdio.h>
#include <stdlib.h>
#include "../hdc_infrastructure/assoc_mem.h"
#include "../hdc_infrastructure/item_mem.h"
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
    for(int dataset = 1; dataset<7;dataset++){

        #if OUTPUT_MODE>= OUTPUT_BASIC
            printf("\n\nModel for dataset #%d\n",dataset);
        #endif
        #if PRECOMPUTED_ITEM_MEMORY
        struct item_memory itemMem;
        init_binary_item_memory(&itemMem,NUM_LEVELS,NUM_FEATURES);

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
        double** testingData;
        int* trainingLabels;
        int* testingLabels;
        int trainingSamples, testingSamples;

        struct associative_memory assMem;
        init_assoc_mem(&assMem);

        getData(dataset,&trainingData,&testingData,&trainingLabels,&testingLabels,&trainingSamples,&testingSamples);

        train_model_timeseries(trainingData, trainingLabels, trainingSamples, &assMem, &enc);
            
        evaluate_model_timeseries_direct(&enc,&assMem,testingData,testingLabels,testingSamples);

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
        #endif
    }
    return 1;
}