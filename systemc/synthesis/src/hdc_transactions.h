// ACCELERATOR BOUNDARY TYPES: Commands carry only command kind, class id, and quantized samples.
// Do not add raw EMG values, double fields, or floating-point quantizer data to this interface.
#ifndef SYSTEMC_HDC_TRANSACTIONS_H
#define SYSTEMC_HDC_TRANSACTIONS_H

#include <ostream>
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
    bool is_shutdown_ack;
    class_t predicted_class;
    distance_counter_t distances[NUM_CLASSES];
};

// Required by sc_fifo<T> print/dump instantiation for custom transaction types.
// Keep opaque: this is not accelerator debug output.
inline std::ostream &operator<<(std::ostream &os, const AccelCommand &) {
    return os << "AccelCommand";
}

inline std::ostream &operator<<(std::ostream &os, const AccelResponse &) {
    return os << "AccelResponse";
}

} // namespace hdc_systemc

#endif
