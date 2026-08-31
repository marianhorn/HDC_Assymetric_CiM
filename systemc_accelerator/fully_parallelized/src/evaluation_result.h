// SIMULATION / SOFTWARE ONLY: Evaluation metrics are not part of the HLS synthesis target.
#ifndef SYSTEMC_HDC_EVALUATION_RESULT_H
#define SYSTEMC_HDC_EVALUATION_RESULT_H

#include "config_systemc.h"

namespace hdc_systemc {

struct EvaluationResult {
    unsigned correct;
    unsigned not_correct;
    unsigned transition_error;
    unsigned total;
    double overall_accuracy;
    double non_transition_accuracy;
    unsigned confusion_matrix[NUM_CLASSES][NUM_CLASSES];
};

} // namespace hdc_systemc

#endif
