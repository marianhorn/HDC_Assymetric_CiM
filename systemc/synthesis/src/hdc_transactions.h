// ACCELERATOR BOUNDARY TYPES: Commands carry only command kind, class id, and quantized samples.
// Do not add raw EMG values, double fields, or floating-point quantizer data to this interface.
#ifndef SYSTEMC_HDC_TRANSACTIONS_H
#define SYSTEMC_HDC_TRANSACTIONS_H

#include "systemc_types.h"

enum AccelCommandKind {
    ResetTraining,
    ResetInference,
    TrainSample,
    InvalidTrainingStep,
    InferSample
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
    distance_counter_t distances[NUM_CLASSES];
};

#endif
