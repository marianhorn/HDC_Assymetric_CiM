#ifndef SYSTEMC_HDC_TRANSACTIONS_H
#define SYSTEMC_HDC_TRANSACTIONS_H

#include "systemc_types.h"

namespace hdc_systemc {

enum class AccelCommandKind {
    ResetTraining,
    ResetInference,
    TrainSample,
    InvalidTrainingStep,
    InferSample,
    Shutdown
};

struct QuantizedSample {
    level_t levels[NUM_FEATURES];
};

struct AccelCommand {
    AccelCommandKind kind;
    class_t class_id;
    QuantizedSample sample;
};

struct AccelResponse {
    bool valid_prediction;
    class_t predicted_class;
    distance_counter_t distances[NUM_CLASSES];
};

} // namespace hdc_systemc

#endif
