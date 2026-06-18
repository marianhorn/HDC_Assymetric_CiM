// ACCELERATOR BOUNDARY TYPES: Commands carry only command kind, class id, and quantized samples.
// Do not add raw EMG values, double fields, or floating-point quantizer data to this interface.
#ifndef SYSTEMC_HDC_TRANSACTIONS_H
#define SYSTEMC_HDC_TRANSACTIONS_H

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

    bool operator==(const QuantizedSample &other) const {
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            if (levels[feature] != other.levels[feature]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const QuantizedSample &other) const {
        return !(*this == other);
    }
};

inline void clear_quantized_sample(QuantizedSample &sample) {
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        sample.levels[feature] = 0;
    }
}

inline void sc_trace(sc_core::sc_trace_file *tf, const QuantizedSample &sample, const std::string &name) {
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        sc_core::sc_trace(tf, sample.levels[feature], name + ".level" + std::to_string(feature));
    }
}

inline std::ostream &operator<<(std::ostream &os, const QuantizedSample &) {
    return os << "QuantizedSample";
}

struct AccelCommand {
    AccelCommandKind kind;
    class_t class_id;
    QuantizedSample sample;
};

struct AccelResponse {
    bool valid_prediction;
    distance_counter_t distances[NUM_CLASSES];
};

} // namespace hdc_systemc

#endif
