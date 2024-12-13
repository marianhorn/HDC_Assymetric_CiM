#ifndef ONLINE_CLASSIFIER_H
#define ONLINE_CLASSIFIER_H
#endif

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include "vector.h"
#include "operations.h"
#include "assoc_mem.h"
#include "encoder.h"

/**
 * @brief Represents the online classifier for real-time predictions.
 *
 * This structure handles batch-based online classification using a pre-trained associative memory
 * and encoder.
 * 
 * Members:
 * - **assoc_mem**: Pointer to the associative memory used for classification.
 * - **enc**: Pointer to the encoder used for transforming input data into hypervectors.
 * - **batch_size**: The number of samples to process in each batch.
 */
struct onlineClassifier{
    struct associative_memory* assoc_mem;/**< Pointer to the associative memory. */
    struct encoder* enc;  /**< Pointer to the encoder. */
    int batch_size; /**< Number of samples in a batch. */
};

void init_online_classifier(struct onlineClassifier* classifier, struct associative_memory* assMem, struct encoder* enc, int batchSize);

int calculateUpdate(struct onlineClassifier* classifier,double** testing_data);
