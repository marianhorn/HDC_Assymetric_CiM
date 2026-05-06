#include "controller.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

namespace hdc_systemc {

namespace {

bool is_comment_or_empty(const std::string &line) {
    for (std::string::size_type i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            continue;
        }
        return c == '#';
    }
    return true;
}

int parse_header_int_field(const std::string &line, const char *key) {
    const std::string pattern(key);
    const std::string::size_type pos = line.find(pattern);
    if (pos == std::string::npos) {
        SC_REPORT_FATAL("Controller", "missing required header field");
        return 0;
    }

    const std::string::size_type start = pos + pattern.size();
    std::string::size_type end = start;
    while (end < line.size() && line[end] != ' ' && line[end] != '\t' && line[end] != '\r' && line[end] != '\n') {
        ++end;
    }

    return std::atoi(line.substr(start, end - start).c_str());
}

} // namespace

Controller::Controller(sc_core::sc_module_name name)
    : sc_module(name), m_memory("hdc_memory"), m_accelerator("hdc_accelerator") {
    m_accelerator.bind_memory(&m_memory);
}

void Controller::load_cim(const char *path) {
    if (path == 0 || path[0] == '\0') {
        SC_REPORT_FATAL("Controller", "CiM path must not be null or empty");
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        SC_REPORT_FATAL("Controller", "failed to open CiM text file");
    }

    std::vector<hv_t> flat_cim(NUM_LEVELS * NUM_FEATURES);
    std::vector<bool> loaded_entries(NUM_LEVELS * NUM_FEATURES, false);
    std::string line;
    bool header_checked = false;
    int loaded_count = 0;

    while (std::getline(file, line)) {
        if (line.find("#systemc_precomp_cim") != std::string::npos) {
            const int header_levels = parse_header_int_field(line, "num_levels=");
            const int header_features = parse_header_int_field(line, "num_features=");
            const int header_dimension = parse_header_int_field(line, "dimension=");
            if (header_levels != NUM_LEVELS) {
                SC_REPORT_FATAL("Controller", "CiM header num_levels does not match config_systemc.h");
            }
            if (header_features != NUM_FEATURES) {
                SC_REPORT_FATAL("Controller", "CiM header num_features does not match config_systemc.h");
            }
            if (header_dimension != VECTOR_DIMENSION) {
                SC_REPORT_FATAL("Controller", "CiM header dimension does not match config_systemc.h");
            }
            header_checked = true;
            continue;
        }
        if (is_comment_or_empty(line)) {
            continue;
        }
        if (!header_checked) {
            SC_REPORT_FATAL("Controller", "CiM text file header missing or not checked before data");
        }

        std::istringstream iss(line);
        int level = -1;
        int feature = -1;
        std::string bits;
        if (!(iss >> level >> feature >> bits)) {
            SC_REPORT_FATAL("Controller", "invalid CiM text line");
        }
        if (level < 0 || level >= NUM_LEVELS) {
            SC_REPORT_FATAL("Controller", "CiM level out of range");
        }
        if (feature < 0 || feature >= NUM_FEATURES) {
            SC_REPORT_FATAL("Controller", "CiM feature out of range");
        }
        if (static_cast<int>(bits.size()) != VECTOR_DIMENSION) {
            SC_REPORT_FATAL("Controller", "CiM bitstring length mismatch");
        }

        const int index = level * NUM_FEATURES + feature;
        if (loaded_entries[static_cast<std::size_t>(index)]) {
            SC_REPORT_FATAL("Controller", "duplicate CiM entry in text file");
        }

        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            const char bit = bits[static_cast<std::string::size_type>(d)];
            if (bit == '0') {
                flat_cim[static_cast<std::size_t>(index)][d] = sc_dt::SC_LOGIC_0;
            } else if (bit == '1') {
                flat_cim[static_cast<std::size_t>(index)][d] = sc_dt::SC_LOGIC_1;
            } else {
                SC_REPORT_FATAL("Controller", "invalid CiM bit character");
            }
        }

        loaded_entries[static_cast<std::size_t>(index)] = true;
        ++loaded_count;
    }

    if (loaded_count != NUM_LEVELS * NUM_FEATURES) {
        SC_REPORT_FATAL("Controller", "CiM text file does not contain all entries");
    }

    m_memory.set_cim(flat_cim.data());
}

void Controller::load_quantizer(const char *path) {
    if (path == 0 || path[0] == '\0') {
        SC_REPORT_FATAL("Controller", "quantizer path must not be null or empty");
    }
    if (NUM_LEVELS <= 1) {
        m_memory.set_quantizer_boundaries(0);
        return;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        SC_REPORT_FATAL("Controller", "failed to open quantizer text file");
    }

    std::vector<double> flat_boundaries(NUM_FEATURES * (NUM_LEVELS - 1), 0.0);
    std::vector<bool> loaded_features(NUM_FEATURES, false);
    std::string line;
    bool header_checked = false;
    int loaded_count = 0;

    while (std::getline(file, line)) {
        if (line.find("#systemc_quantizer") != std::string::npos) {
            const int header_levels = parse_header_int_field(line, "num_levels=");
            const int header_features = parse_header_int_field(line, "num_features=");
            if (header_levels != NUM_LEVELS) {
                SC_REPORT_FATAL("Controller", "quantizer header num_levels does not match config_systemc.h");
            }
            if (header_features != NUM_FEATURES) {
                SC_REPORT_FATAL("Controller", "quantizer header num_features does not match config_systemc.h");
            }
            header_checked = true;
            continue;
        }
        if (is_comment_or_empty(line)) {
            continue;
        }
        if (!header_checked) {
            SC_REPORT_FATAL("Controller", "quantizer text file header missing or not checked before data");
        }

        std::istringstream iss(line);
        int feature = -1;
        if (!(iss >> feature)) {
            SC_REPORT_FATAL("Controller", "invalid quantizer text line");
        }
        if (feature < 0 || feature >= NUM_FEATURES) {
            SC_REPORT_FATAL("Controller", "quantizer feature out of range");
        }
        if (loaded_features[static_cast<std::size_t>(feature)]) {
            SC_REPORT_FATAL("Controller", "duplicate quantizer feature entry");
        }

        for (int cut = 0; cut < NUM_LEVELS - 1; ++cut) {
            double boundary = 0.0;
            if (!(iss >> boundary)) {
                SC_REPORT_FATAL("Controller", "missing quantizer boundary value");
            }
            flat_boundaries[static_cast<std::size_t>(feature * (NUM_LEVELS - 1) + cut)] = boundary;
        }

        loaded_features[static_cast<std::size_t>(feature)] = true;
        ++loaded_count;
    }

    if (loaded_count != NUM_FEATURES) {
        SC_REPORT_FATAL("Controller", "quantizer text file does not contain all features");
    }

    m_memory.set_quantizer_boundaries(flat_boundaries.data());
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
        quantize_sample(&raw_window[i * NUM_FEATURES], &quantized_window[i * NUM_FEATURES]);
    }
}

level_t Controller::quantize_value(unsigned feature, double value) const {
    if (feature >= static_cast<unsigned>(NUM_FEATURES)) {
        SC_REPORT_FATAL("Controller", "quantize_value feature index out of range");
        return 0;
    }
    if (NUM_LEVELS <= 1) {
        return 0;
    }

    const double *boundaries = m_memory.read_quantizer_row(feature);
    for (int cut = 0; cut < NUM_LEVELS - 1; ++cut) {
        if (value <= boundaries[cut]) {
            return static_cast<unsigned>(cut);
        }
    }
    return static_cast<unsigned>(NUM_LEVELS - 1);
}

void Controller::quantize_sample(const double *raw_sample, level_t *quantized_sample) const {
    if (raw_sample == 0 || quantized_sample == 0) {
        SC_REPORT_FATAL("Controller", "quantize_sample inputs must not be null");
    }

    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        quantized_sample[feature] = quantize_value(static_cast<unsigned>(feature), raw_sample[feature]);
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
