// ACCELERATOR BOUNDARY TYPES: Commands carry only command kind, class id, and quantized samples.
// Do not add raw EMG values, double fields, or floating-point quantizer data to this interface.
#ifndef SYSTEMC_HDC_TRANSACTIONS_H
#define SYSTEMC_HDC_TRANSACTIONS_H

#include <string>
#include "systemc_types.h"

namespace hdc_systemc {

enum class AccelCommandKind {
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
    class_t predicted_class;
    distance_counter_t distances[NUM_CLASSES];
};

inline bool operator==(const QuantizedSample &lhs, const QuantizedSample &rhs) {
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        if (lhs.levels[feature] != rhs.levels[feature]) {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const QuantizedSample &lhs, const QuantizedSample &rhs) {
    return !(lhs == rhs);
}

inline bool operator==(const AccelCommand &lhs, const AccelCommand &rhs) {
    return lhs.kind == rhs.kind &&
           lhs.class_id == rhs.class_id &&
           lhs.sample == rhs.sample;
}

inline bool operator!=(const AccelCommand &lhs, const AccelCommand &rhs) {
    return !(lhs == rhs);
}

inline bool operator==(const AccelResponse &lhs, const AccelResponse &rhs) {
    if (lhs.valid_prediction != rhs.valid_prediction ||
        lhs.predicted_class != rhs.predicted_class) {
        return false;
    }
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        if (lhs.distances[class_id] != rhs.distances[class_id]) {
            return false;
        }
    }
    return true;
}

inline bool operator!=(const AccelResponse &lhs, const AccelResponse &rhs) {
    return !(lhs == rhs);
}

inline void sc_trace(sc_core::sc_trace_file *, const QuantizedSample &, const std::string &) {}
inline void sc_trace(sc_core::sc_trace_file *, const AccelCommand &, const std::string &) {}
inline void sc_trace(sc_core::sc_trace_file *, const AccelResponse &, const std::string &) {}

} // namespace hdc_systemc

#endif
