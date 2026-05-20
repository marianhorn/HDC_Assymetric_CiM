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

inline std::ostream &operator<<(std::ostream &os, AccelCommandKind kind) {
    switch (kind) {
    case AccelCommandKind::ResetTraining:
        return os << "ResetTraining";
    case AccelCommandKind::ResetInference:
        return os << "ResetInference";
    case AccelCommandKind::TrainSample:
        return os << "TrainSample";
    case AccelCommandKind::InvalidTrainingStep:
        return os << "InvalidTrainingStep";
    case AccelCommandKind::InferSample:
        return os << "InferSample";
    case AccelCommandKind::Shutdown:
        return os << "Shutdown";
    }
    return os << "Unknown";
}

inline std::ostream &operator<<(std::ostream &os, const QuantizedSample &sample) {
    os << "levels=[";
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        if (feature > 0) {
            os << ',';
        }
        os << sample.levels[feature].to_uint();
    }
    return os << ']';
}

inline std::ostream &operator<<(std::ostream &os, const AccelCommand &command) {
    return os << "AccelCommand{kind=" << command.kind
              << ", class_id=" << command.class_id.to_uint()
              << ", " << command.sample << '}';
}

inline std::ostream &operator<<(std::ostream &os, const AccelResponse &response) {
    os << "AccelResponse{valid_prediction=" << response.valid_prediction
       << ", is_shutdown_ack=" << response.is_shutdown_ack
       << ", predicted_class=" << response.predicted_class.to_uint()
       << ", distances=[";
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        if (class_id > 0) {
            os << ',';
        }
        os << response.distances[class_id].to_uint();
    }
    return os << "]}";
}

} // namespace hdc_systemc

#endif
