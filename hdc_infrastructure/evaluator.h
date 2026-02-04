//evaluator.h
#ifndef EVALUATOR_H
#define EVALUATOR_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include "assoc_mem.h"
#include "encoder.h"
#include <stddef.h>

struct timeseries_eval_result {
    size_t correct;
    size_t not_correct;
    size_t transition_error;
    size_t total;
    double overall_accuracy;
    double class_average_accuracy;
    double class_vector_similarity;
    int confusion_matrix[NUM_CLASSES][NUM_CLASSES];
};

struct timeseries_eval_result evaluate_model_timeseries_with_window(struct encoder *enc,
                                                                    struct associative_memory *assMem,
                                                                    double **testingData,
                                                                    int *testingLabels,
                                                                    int testingSamples);
struct timeseries_eval_result evaluate_model_timeseries_direct(struct encoder *enc,
                                                               struct associative_memory *assMem,
                                                               double **testingData,
                                                               int *testingLabels,
                                                               int testingSamples);
struct timeseries_eval_result evaluate_model_general_direct(struct encoder *enc,
                                                            struct associative_memory *assoc_mem,
                                                            double **testing_data,
                                                            int *testing_labels,
                                                            int testing_samples);

#endif
