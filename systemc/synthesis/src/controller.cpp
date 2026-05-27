// SIMULATION / SOFTWARE ONLY: This file is not part of the HLS synthesis target.
// It contains file loading, floating-point quantization, dataset control, and metrics collection.
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

void clear_evaluation_result(EvaluationResult &result) {
    result.correct = 0;
    result.not_correct = 0;
    result.transition_error = 0;
    result.total = 0;
    result.overall_accuracy = 0.0;
    result.non_transition_accuracy = 0.0;
    for (int actual = 0; actual < NUM_CLASSES; ++actual) {
        for (int predicted = 0; predicted < NUM_CLASSES; ++predicted) {
            result.confusion_matrix[actual][predicted] = 0;
        }
    }
}

sample_levels_packed_t pack_sample_levels(const QuantizedSample &sample) {
    sample_levels_packed_t packed = 0;
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        const unsigned base = feature * LEVEL_BITS;
        for (unsigned bit = 0; bit < LEVEL_BITS; ++bit) {
            packed[base + bit] =
                sample.levels[feature][bit].to_bool() ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;
        }
    }
    return packed;
}

distance_counter_t unpack_distance(const distances_packed_t &packed, unsigned class_id) {
    distance_counter_t value = 0;
    const unsigned base = class_id * DISTANCE_BITS;
    for (unsigned bit = 0; bit < DISTANCE_BITS; ++bit) {
        value[bit] = packed[base + bit].to_bool();
    }
    return value;
}

} // namespace

Controller::Controller(sc_core::sc_module_name name)
    : sc_module(name),
      clk("clk"),
      rst("rst"),
      m_done(false),
      m_memory("hdc_memory"),
      m_cmd_valid("cmd_valid"),
      m_cmd_ready("cmd_ready"),
      m_rsp_valid("rsp_valid"),
      m_rsp_ready("rsp_ready"),
      m_rsp_valid_prediction("rsp_valid_prediction"),
      m_accelerator("hdc_accelerator") {
    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        m_dataset_configs[dataset].dataset_id = dataset;
        m_dataset_configs[dataset].cim_path = 0;
        m_dataset_configs[dataset].quantizer_path = 0;
        m_dataset_configs[dataset].dataset = 0;
        m_dataset_configs[dataset].configured = false;
        clear_evaluation_result(m_test_results[dataset]);
    }

    m_accelerator.clk(clk);
    m_accelerator.rst(rst);
    m_accelerator.cmd_valid(m_cmd_valid);
    m_accelerator.cmd_ready(m_cmd_ready);
    m_accelerator.cmd_kind(m_cmd_kind);
    m_accelerator.cmd_class_id(m_cmd_class_id);
    m_accelerator.cmd_sample_levels(m_cmd_sample_levels);
    m_accelerator.rsp_valid(m_rsp_valid);
    m_accelerator.rsp_ready(m_rsp_ready);
    m_accelerator.rsp_valid_prediction(m_rsp_valid_prediction);
    m_accelerator.rsp_distances(m_rsp_distances);
    SC_THREAD(main_thread);
}

void Controller::configure(int dataset_id,
                           const char *cim_path,
                           const char *quantizer_path,
                           const FootDataset *dataset) {
    if (dataset_id < 0 || dataset_id >= NUM_DATASETS) {
        SC_REPORT_FATAL("Controller", "dataset_id out of range");
    }
    if (cim_path == 0 || quantizer_path == 0 || dataset == 0) {
        SC_REPORT_FATAL("Controller", "invalid dataset configuration");
    }

    DatasetConfig &config = m_dataset_configs[dataset_id];
    config.dataset_id = dataset_id;
    config.cim_path = cim_path;
    config.quantizer_path = quantizer_path;
    config.dataset = dataset;
    config.configured = true;
}

bool Controller::done() const {
    return m_done;
}

const EvaluationResult &Controller::test_result(int dataset_id) const {
    if (dataset_id < 0 || dataset_id >= NUM_DATASETS) {
        SC_REPORT_FATAL("Controller", "test_result dataset_id out of range");
    }
    return m_test_results[dataset_id];
}

void Controller::main_thread() {
    m_cmd_valid.write(false);
    m_rsp_ready.write(false);
    do {
        wait(clk.posedge_event());
    } while (rst.read());

    for (int dataset = 0; dataset < NUM_DATASETS; ++dataset) {
        const DatasetConfig &config = m_dataset_configs[dataset];
        if (!config.configured) {
            SC_REPORT_FATAL("Controller", "dataset not configured before simulation start");
        }

        m_memory.clear_all();
        load_cim(config.cim_path);
        load_quantizer(config.quantizer_path);

        train_dataset(config.dataset->training.raw_data(),
                      config.dataset->training.raw_labels(),
                      config.dataset->training.samples);

        m_test_results[dataset] =
            evaluate_dataset(config.dataset->testing.raw_data(),
                             config.dataset->testing.raw_labels(),
                             config.dataset->testing.samples);
    }

    m_done = true;
    sc_core::sc_stop();
}

void Controller::copy_quantized_sample(const level_t *levels, QuantizedSample &sample) const {
    if (levels == 0) {
        SC_REPORT_FATAL("Controller", "quantized levels must not be null");
    }
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        sample.levels[feature] = levels[feature];
    }
}

void Controller::send_command(const AccelCommand &command) {
    m_cmd_kind.write(static_cast<unsigned>(command.kind));
    m_cmd_class_id.write(command.class_id);
    m_cmd_sample_levels.write(pack_sample_levels(command.sample));
    m_cmd_valid.write(true);
    while (true) {
        wait(clk.posedge_event());
        wait(sc_core::SC_ZERO_TIME);
        if (m_cmd_ready.read()) {
            break;
        }
    }
    m_cmd_valid.write(false);
}

AccelResponse Controller::read_response() {
    AccelResponse response = {};
    response.valid_prediction = m_rsp_valid_prediction.read();
    const distances_packed_t packed_distances = m_rsp_distances.read();
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        response.distances[class_id] = unpack_distance(packed_distances,
                                                       static_cast<unsigned>(class_id));
    }
    return response;
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

    for (int level = 0; level < NUM_LEVELS; ++level) {
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            m_accelerator.set_cim(static_cast<unsigned>(level),
                                  static_cast<unsigned>(feature),
                                  flat_cim[static_cast<std::size_t>(level * NUM_FEATURES + feature)]);
        }
    }
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

void Controller::train_dataset(const double *raw_data, const int *labels, int num_samples) {
    if (raw_data == 0 || labels == 0) {
        SC_REPORT_FATAL("Controller", "training data and labels must not be null");
    }

    hv_t empty_class_vector;
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        empty_class_vector = 0;
        m_accelerator.set_assoc_class(static_cast<unsigned>(class_id), empty_class_vector);
    }

    AccelCommand command = {};
    command.kind = AccelCommandKind::ResetTraining;
    command.class_id = 0;
    send_command(command);

    level_t quantized_sample[NUM_FEATURES];
    quantize_sample(raw_data, quantized_sample);
    command.kind = AccelCommandKind::TrainSample;
    command.class_id = static_cast<unsigned>(labels[0]);
    copy_quantized_sample(quantized_sample, command.sample);
    send_command(command);

    for (int j = 1; j < num_samples - 1; ++j) {
        if (labels[j] != labels[j - 1]) {
            command.kind = AccelCommandKind::InvalidTrainingStep;
            command.class_id = 0;
            send_command(command);
        }
        quantize_sample(&raw_data[j * NUM_FEATURES], quantized_sample);
        command.kind = AccelCommandKind::TrainSample;
        command.class_id = static_cast<unsigned>(labels[j]);
        copy_quantized_sample(quantized_sample, command.sample);
        send_command(command);
    }

    command.kind = AccelCommandKind::InvalidTrainingStep;
    command.class_id = 0;
    send_command(command);
}

int Controller::get_ngram_real_label(const int *labels, int size) const {
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

EvaluationResult Controller::evaluate_dataset(const double *raw_data, const int *labels, int num_samples) {
    if (raw_data == 0 || labels == 0) {
        SC_REPORT_FATAL("Controller", "evaluation data and labels must not be null");
    }

    EvaluationResult result;
    clear_evaluation_result(result);

    AccelCommand command = {};
    command.kind = AccelCommandKind::ResetInference;
    command.class_id = 0;
    send_command(command);

    level_t quantized_sample[NUM_FEATURES];
    int issued = 0;
    int received = 0;
    int outstanding = 0;
    bool command_pending = false;

    while (received < num_samples) {
        if (!command_pending && issued < num_samples && outstanding < MAX_SAMPLES_IN_PIPELINE) {
            quantize_sample(&raw_data[issued * NUM_FEATURES], quantized_sample);
            command.kind = AccelCommandKind::InferSample;
            command.class_id = 0;
            copy_quantized_sample(quantized_sample, command.sample);
            m_cmd_kind.write(static_cast<unsigned>(command.kind));
            m_cmd_class_id.write(command.class_id);
            m_cmd_sample_levels.write(pack_sample_levels(command.sample));
            m_cmd_valid.write(true);
            command_pending = true;
        }

        m_rsp_ready.write(outstanding > 0);
        wait(clk.posedge_event());
        wait(sc_core::SC_ZERO_TIME);

        if (command_pending && m_cmd_ready.read()) {
            m_cmd_valid.write(false);
            command_pending = false;
            ++issued;
            ++outstanding;
        }

        if (!(outstanding > 0 && m_rsp_valid.read() && m_rsp_ready.read())) {
            continue;
        }

        AccelResponse response = read_response();
        const int sample = received;
        ++received;
        --outstanding;

        if (!response.valid_prediction) {
            continue;
        }

        const int ngram_start = sample - N_GRAM_SIZE + 1;
        const int actual = get_ngram_real_label(&labels[ngram_start], N_GRAM_SIZE);
        int predicted = 0;
        distance_counter_t best_distance = response.distances[0];
        for (int class_id = 1; class_id < NUM_CLASSES; ++class_id) {
            if (response.distances[class_id] < best_distance) {
                best_distance = response.distances[class_id];
                predicted = class_id;
            }
        }

        if (actual >= 0 && actual < NUM_CLASSES && predicted >= 0 && predicted < NUM_CLASSES) {
            ++result.confusion_matrix[actual][predicted];
        }

        if (predicted == actual) {
            ++result.correct;
        } else if (labels[ngram_start] != labels[ngram_start + N_GRAM_SIZE - 1]) {
            ++result.transition_error;
        } else {
            ++result.not_correct;
        }
    }
    m_cmd_valid.write(false);
    m_rsp_ready.write(false);

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

} // namespace hdc_systemc
