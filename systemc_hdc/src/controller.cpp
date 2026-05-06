#include "controller.h"

namespace hdc_systemc {

namespace {

} // namespace

Controller::Controller(sc_core::sc_module_name name)
    : sc_module(name), m_memory("hdc_memory"), m_accelerator("hdc_accelerator") {
    m_accelerator.bind_memory(&m_memory);
}

void Controller::load_cim(const hv_t *flat_cim) {
    m_memory.load_cim_flat(flat_cim);
}

void Controller::load_cim_file(const char *path) {
    m_memory.load_cim_text(path);
}

void Controller::load_quantizer_boundaries(const double *flat_boundaries) {
    m_memory.load_quantizer_boundaries(flat_boundaries);
}

void Controller::load_quantizer_file(const char *path) {
    m_memory.load_quantizer_text(path);
}

void Controller::reset_assoc_mem() {
    m_memory.clear_assoc_mem();
    m_accelerator.reset_training_state();
}

void Controller::quantize_window(const double *raw_window, level_t *quantized_window) const {
    if (raw_window == 0 || quantized_window == 0) {
        SC_REPORT_FATAL("Controller", "quantize_window inputs must not be null");
    }

    for (int i = 0; i < N_GRAM_SIZE; ++i) {
        m_memory.quantize_sample(&raw_window[i * NUM_FEATURES], &quantized_window[i * NUM_FEATURES]);
    }
}

void Controller::encode_window(const double *raw_window, hv_t &encoded) const {
    level_t quantized_window[N_GRAM_SIZE * NUM_FEATURES];
    quantize_window(raw_window, quantized_window);
    m_accelerator.encode(quantized_window, encoded);
}

void Controller::classify_window(const double *raw_window, distance_counter_t *distances) const {
    level_t quantized_window[N_GRAM_SIZE * NUM_FEATURES];
    quantize_window(raw_window, quantized_window);
    m_accelerator.classify(quantized_window, distances);
}

int Controller::predict_window(const double *raw_window) const {
    distance_counter_t distances[NUM_CLASSES];
    classify_window(raw_window, distances);

    int best_class = 0;
    distance_counter_t best_distance = distances[0];
    for (int class_id = 1; class_id < NUM_CLASSES; ++class_id) {
        if (distances[class_id] < best_distance) {
            best_distance = distances[class_id];
            best_class = class_id;
        }
    }
    return best_class;
}

bool Controller::is_window_stable(const int *labels) const {
    if (labels == 0) {
        SC_REPORT_FATAL("Controller", "labels must not be null");
    }
    return labels[0] == labels[N_GRAM_SIZE - 1];
}

void Controller::train_dataset(const double *raw_data, const int *labels, int num_samples) {
    if (raw_data == 0 || labels == 0) {
        SC_REPORT_FATAL("Controller", "training data and labels must not be null");
    }

    reset_assoc_mem();

    hv_t encoded_ngram;
    for (int j = 0; j < num_samples - N_GRAM_SIZE; ++j) {
        if (is_window_stable(&labels[j])) {
            const int class_id = labels[j];
            encode_window(&raw_data[j * NUM_FEATURES], encoded_ngram);
            m_accelerator.accumulate_class_vector(class_id, encoded_ngram);
        } else {
            j += (N_GRAM_SIZE - 1);
        }
    }

    m_accelerator.finalize_assoc_mem();
}

int Controller::mode_smallest_tie(const int *labels, int size) const {
    if (labels == 0 || size <= 0) {
        SC_REPORT_FATAL("Controller", "invalid labels input");
        return 0;
    }

    int max_value = 0;
    int max_count = 0;
    for (int i = 0; i < size; ++i) {
        int count = 0;
        for (int j = 0; j < size; ++j) {
            if (labels[j] == labels[i]) {
                ++count;
            }
        }
        if (count > max_count) {
            max_count = count;
            max_value = labels[i];
        } else if (count == max_count && labels[i] < max_value) {
            max_value = labels[i];
        }
    }

    return max_value;
}

EvaluationResult Controller::evaluate_dataset(const double *raw_data, const int *labels, int num_samples) const {
    if (raw_data == 0 || labels == 0) {
        SC_REPORT_FATAL("Controller", "evaluation data and labels must not be null");
    }

    EvaluationResult result;
    result.correct = 0;
    result.not_correct = 0;
    result.transition_error = 0;
    result.total = 0;
    result.overall_accuracy = 0.0;
    result.non_transition_accuracy = 0.0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            result.confusion_matrix[i][j] = 0;
        }
    }

    for (int j = 0; j < num_samples - N_GRAM_SIZE + 1; j += N_GRAM_SIZE) {
        const int actual = mode_smallest_tie(&labels[j], N_GRAM_SIZE);
        const int predicted = predict_window(&raw_data[j * NUM_FEATURES]);

        if (actual >= 0 && actual < NUM_CLASSES && predicted >= 0 && predicted < NUM_CLASSES) {
            result.confusion_matrix[actual][predicted] += 1;
        }

        if (predicted == actual) {
            ++result.correct;
        } else if (labels[j] != labels[j + N_GRAM_SIZE - 1]) {
            ++result.transition_error;
        } else {
            ++result.not_correct;
        }
    }

    result.total = result.correct + result.not_correct + result.transition_error;
    if (result.total > 0) {
        result.overall_accuracy = static_cast<double>(result.correct) / static_cast<double>(result.total);
    }
    if ((result.correct + result.not_correct) > 0) {
        result.non_transition_accuracy =
            static_cast<double>(result.correct) / static_cast<double>(result.correct + result.not_correct);
    }

    return result;
}

const hv_t &Controller::get_class_vector(unsigned class_id) const {
    return m_memory.read_assoc_class(class_id);
}

} // namespace hdc_systemc
