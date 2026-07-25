// SIMULATION / SOFTWARE ONLY: Generates RTL stimulus and expected responses.
// The output is consumed by the Vivado XSim RTL testbench.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "foot_dataset_loader.h"
#include "hdc_accelerator.h"
#include "hdc_memory.h"

using namespace hdc_systemc;

namespace {

struct TracePaths {
    std::string out_dir;
    std::string commands;
    std::string responses;
    std::string stats;
};

struct TraceStats {
    unsigned long long cycles = 0;
    unsigned long long commands = 0;
    unsigned long long training_commands = 0;
    unsigned long long inference_commands = 0;
    unsigned long long responses = 0;
    unsigned train_samples = 0;
    unsigned test_samples = 0;
    unsigned correct = 0;
    unsigned not_correct = 0;
    unsigned transition_error = 0;
    unsigned total = 0;
};

struct TraceLimits {
    int train_samples = -1;
    int test_samples = -1;
};

static constexpr unsigned POST_COMMAND_HOLD_CYCLES = 5;

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
        SC_REPORT_FATAL("rtl_trace_export", "missing required header field");
        return 0;
    }

    const std::string::size_type start = pos + pattern.size();
    std::string::size_type end = start;
    while (end < line.size() && line[end] != ' ' && line[end] != '\t' &&
           line[end] != '\r' && line[end] != '\n') {
        ++end;
    }

    return std::atoi(line.substr(start, end - start).c_str());
}

void ensure_directory(const std::string &path) {
    if (path.empty()) {
        SC_REPORT_FATAL("rtl_trace_export", "empty output directory");
    }
    if (::mkdir(path.c_str(), 0775) != 0 && errno != EEXIST) {
        std::ostringstream msg;
        msg << "failed to create output directory '" << path << "': " << std::strerror(errno);
        SC_REPORT_FATAL("rtl_trace_export", msg.str().c_str());
    }
}

TracePaths make_trace_paths(const std::string &out_dir) {
    TracePaths paths;
    paths.out_dir = out_dir;
    paths.commands = out_dir + "/commands.txt";
    paths.responses = out_dir + "/expected_responses.txt";
    paths.stats = out_dir + "/stats.txt";
    return paths;
}

std::string dataset_file(const char *kind, int dataset_id) {
    char path[128];
    std::snprintf(path, sizeof(path), "import/%s_dataset%02d.txt", kind, dataset_id);
    return std::string(path);
}

void load_cim_into_accelerator(HDC_Accelerator &accelerator, const char *path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        SC_REPORT_FATAL("rtl_trace_export", "failed to open CiM text file");
    }

    std::vector<hv_t> flat_cim(NUM_LEVELS * NUM_FEATURES);
    for (std::size_t i = 0; i < flat_cim.size(); ++i) {
        hv_clear(flat_cim[i]);
    }
    std::vector<bool> loaded_entries(NUM_LEVELS * NUM_FEATURES, false);
    std::string line;
    bool header_checked = false;
    int loaded_count = 0;

    while (std::getline(file, line)) {
        if (line.find("#systemc_precomp_cim") != std::string::npos) {
            const int header_levels = parse_header_int_field(line, "num_levels=");
            const int header_features = parse_header_int_field(line, "num_features=");
            const int header_dimension = parse_header_int_field(line, "dimension=");
            if (header_levels != NUM_LEVELS || header_features != NUM_FEATURES ||
                header_dimension != VECTOR_DIMENSION) {
                SC_REPORT_FATAL("rtl_trace_export", "CiM header does not match config_systemc.h");
            }
            header_checked = true;
            continue;
        }
        if (is_comment_or_empty(line)) {
            continue;
        }
        if (!header_checked) {
            SC_REPORT_FATAL("rtl_trace_export", "CiM data found before checked header");
        }

        std::istringstream iss(line);
        int level = -1;
        int feature = -1;
        std::string bits;
        if (!(iss >> level >> feature >> bits)) {
            SC_REPORT_FATAL("rtl_trace_export", "invalid CiM text line");
        }
        if (level < 0 || level >= NUM_LEVELS || feature < 0 || feature >= NUM_FEATURES ||
            static_cast<int>(bits.size()) != VECTOR_DIMENSION) {
            SC_REPORT_FATAL("rtl_trace_export", "CiM entry out of range or wrong width");
        }

        const int index = level * NUM_FEATURES + feature;
        if (loaded_entries[static_cast<std::size_t>(index)]) {
            SC_REPORT_FATAL("rtl_trace_export", "duplicate CiM entry");
        }

        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            const char bit = bits[static_cast<std::string::size_type>(d)];
            if (bit == '0') {
                hv_set_bit(flat_cim[static_cast<std::size_t>(index)], static_cast<unsigned>(d), false);
            } else if (bit == '1') {
                hv_set_bit(flat_cim[static_cast<std::size_t>(index)], static_cast<unsigned>(d), true);
            } else {
                SC_REPORT_FATAL("rtl_trace_export", "invalid CiM bit character");
            }
        }

        loaded_entries[static_cast<std::size_t>(index)] = true;
        ++loaded_count;
    }

    if (loaded_count != NUM_LEVELS * NUM_FEATURES) {
        SC_REPORT_FATAL("rtl_trace_export", "CiM file does not contain all entries");
    }

    for (int level = 0; level < NUM_LEVELS; ++level) {
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            accelerator.set_cim(static_cast<unsigned>(level),
                                static_cast<unsigned>(feature),
                                flat_cim[static_cast<std::size_t>(level * NUM_FEATURES + feature)]);
        }
    }
}

void load_quantizer(HDC_Memory &memory, const char *path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        SC_REPORT_FATAL("rtl_trace_export", "failed to open quantizer text file");
    }

    if (NUM_LEVELS <= 1) {
        memory.set_quantizer_boundaries(0);
        return;
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
            if (header_levels != NUM_LEVELS || header_features != NUM_FEATURES) {
                SC_REPORT_FATAL("rtl_trace_export", "quantizer header does not match config_systemc.h");
            }
            header_checked = true;
            continue;
        }
        if (is_comment_or_empty(line)) {
            continue;
        }
        if (!header_checked) {
            SC_REPORT_FATAL("rtl_trace_export", "quantizer data found before checked header");
        }

        std::istringstream iss(line);
        int feature = -1;
        if (!(iss >> feature)) {
            SC_REPORT_FATAL("rtl_trace_export", "invalid quantizer text line");
        }
        if (feature < 0 || feature >= NUM_FEATURES) {
            SC_REPORT_FATAL("rtl_trace_export", "quantizer feature out of range");
        }
        if (loaded_features[static_cast<std::size_t>(feature)]) {
            SC_REPORT_FATAL("rtl_trace_export", "duplicate quantizer feature");
        }

        for (int cut = 0; cut < NUM_LEVELS - 1; ++cut) {
            double boundary = 0.0;
            if (!(iss >> boundary)) {
                SC_REPORT_FATAL("rtl_trace_export", "missing quantizer boundary");
            }
            flat_boundaries[static_cast<std::size_t>(feature * (NUM_LEVELS - 1) + cut)] = boundary;
        }

        loaded_features[static_cast<std::size_t>(feature)] = true;
        ++loaded_count;
    }

    if (loaded_count != NUM_FEATURES) {
        SC_REPORT_FATAL("rtl_trace_export", "quantizer file does not contain all features");
    }

    memory.set_quantizer_boundaries(flat_boundaries.data());
}

level_t quantize_value(const HDC_Memory &memory, unsigned feature, double value) {
    if (feature >= static_cast<unsigned>(NUM_FEATURES)) {
        SC_REPORT_FATAL("rtl_trace_export", "quantize feature index out of range");
        return 0;
    }
    if (NUM_LEVELS <= 1) {
        return 0;
    }

    const double *boundaries = memory.read_quantizer_row(feature);
    for (int cut = 0; cut < NUM_LEVELS - 1; ++cut) {
        if (value <= boundaries[cut]) {
            return static_cast<unsigned>(cut);
        }
    }
    return static_cast<unsigned>(NUM_LEVELS - 1);
}

void quantize_sample(const HDC_Memory &memory, const double *raw_sample, level_t *quantized_sample) {
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        quantized_sample[feature] = quantize_value(memory, static_cast<unsigned>(feature), raw_sample[feature]);
    }
}

void copy_quantized_sample(const level_t *levels, QuantizedSample &sample) {
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        sample.levels[feature] = levels[feature];
    }
}

int get_ngram_real_label(const int *labels, int size) {
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

int predicted_class_from_response(const AccelResponse &response) {
    int predicted = 0;
    distance_counter_t best_distance = response.distances[0];
    for (int class_id = 1; class_id < NUM_CLASSES; ++class_id) {
        if (response.distances[class_id] < best_distance) {
            best_distance = response.distances[class_id];
            predicted = class_id;
        }
    }
    return predicted;
}

class RtlTraceDriver {
public:
    RtlTraceDriver(HDC_Accelerator &accelerator,
                   HDC_Memory &memory,
                   const FootDataset &dataset,
                   const TraceLimits &limits,
                   std::ostream &commands,
                   std::ostream &responses,
                   TraceStats &stats)
        : m_accelerator(accelerator),
          m_memory(memory),
          m_dataset(dataset),
          m_limits(limits),
          m_commands(commands),
          m_responses(responses),
          m_stats(stats),
          clk("clk", sc_core::sc_time(10, sc_core::SC_NS)),
          rst("rst"),
          cmd_valid("cmd_valid"),
          cmd_ready("cmd_ready"),
          cmd_kind("cmd_kind"),
          cmd_class_id("cmd_class_id"),
          rsp_valid("rsp_valid"),
          rsp_ready("rsp_ready"),
          rsp_valid_prediction("rsp_valid_prediction") {
        m_accelerator.clk(clk);
        m_accelerator.rst(rst);
        m_accelerator.cmd_valid(cmd_valid);
        m_accelerator.cmd_ready(cmd_ready);
        m_accelerator.cmd_kind(cmd_kind);
        m_accelerator.cmd_class_id(cmd_class_id);
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            m_accelerator.cmd_sample_levels[feature](cmd_sample_levels[feature]);
        }
        m_accelerator.rsp_valid(rsp_valid);
        m_accelerator.rsp_ready(rsp_ready);
        m_accelerator.rsp_valid_prediction(rsp_valid_prediction);
        for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            m_accelerator.rsp_distances[class_id](rsp_distances[class_id]);
        }
    }

    void run() {
        reset();
        train_dataset(m_dataset.training.raw_data(),
                      m_dataset.training.raw_labels(),
                      limited_samples("training", m_dataset.training.samples, m_limits.train_samples, 2));
        evaluate_dataset(m_dataset.testing.raw_data(),
                         m_dataset.testing.raw_labels(),
                         limited_samples("testing", m_dataset.testing.samples, m_limits.test_samples, 1));
    }

private:
    int limited_samples(const char *name, int available, int limit, int minimum) const {
        int selected = available;
        if (limit >= 0 && limit < selected) {
            selected = limit;
        }
        if (selected < minimum) {
            std::ostringstream msg;
            msg << name << " sample limit too small; need at least " << minimum;
            SC_REPORT_FATAL("rtl_trace_export", msg.str().c_str());
        }
        return selected;
    }

    void tick() {
        sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_NS));
        sc_core::sc_start(sc_core::SC_ZERO_TIME);
        ++m_stats.cycles;
        if (m_stats.cycles > 100000000ULL) {
            SC_REPORT_FATAL("rtl_trace_export", "SystemC trace generation timed out");
        }
    }

    void reset() {
        cmd_valid.write(false);
        rsp_ready.write(false);
        rst.write(true);
        tick();
        tick();
        rst.write(false);
        tick();
    }

    void write_command_line(const AccelCommand &command) {
        m_commands << static_cast<unsigned>(command.kind) << ' '
                   << command.class_id.to_uint();
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            m_commands << ' ' << command.sample.levels[feature].to_uint();
        }
        m_commands << '\n';
    }

    void send_command(const AccelCommand &command) {
        cmd_kind.write(static_cast<unsigned>(command.kind));
        cmd_class_id.write(command.class_id);
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            cmd_sample_levels[feature].write(command.sample.levels[feature]);
        }
        cmd_valid.write(true);

        while (true) {
            rsp_ready.write(false);
            tick();
            if (cmd_ready.read()) {
                write_command_line(command);
                ++m_stats.commands;
                if (command.kind == AccelCommandKind::InferSample) {
                    ++m_stats.inference_commands;
                } else {
                    ++m_stats.training_commands;
                }
                break;
            }
        }
        cmd_valid.write(false);
        for (unsigned hold = 0; hold < POST_COMMAND_HOLD_CYCLES; ++hold) {
            rsp_ready.write(false);
            tick();
        }
    }

    AccelResponse read_response() const {
        AccelResponse response = {};
        response.valid_prediction = rsp_valid_prediction.read();
        for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            response.distances[class_id] = rsp_distances[class_id].read();
        }
        return response;
    }

    void write_response_line(const AccelResponse &response, int predicted, int actual) {
        m_responses << (response.valid_prediction ? 1 : 0);
        for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            m_responses << ' ' << response.distances[class_id].to_uint();
        }
        m_responses << ' ' << predicted << ' ' << actual << '\n';
    }

    void train_dataset(const double *raw_data, const int *labels, int num_samples) {
        if (raw_data == 0 || labels == 0 || num_samples <= 1) {
            SC_REPORT_FATAL("rtl_trace_export", "invalid training dataset");
        }
        m_stats.train_samples = static_cast<unsigned>(num_samples);

        hv_t empty_class_vector;
        hv_clear(empty_class_vector);
        for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            m_accelerator.set_assoc_class(static_cast<unsigned>(class_id), empty_class_vector);
        }

        AccelCommand command = {};
        command.kind = AccelCommandKind::ResetTraining;
        command.class_id = 0;
        send_command(command);

        level_t quantized_sample[NUM_FEATURES];
        quantize_sample(m_memory, raw_data, quantized_sample);
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
            quantize_sample(m_memory, &raw_data[j * NUM_FEATURES], quantized_sample);
            command.kind = AccelCommandKind::TrainSample;
            command.class_id = static_cast<unsigned>(labels[j]);
            copy_quantized_sample(quantized_sample, command.sample);
            send_command(command);
        }

        command.kind = AccelCommandKind::InvalidTrainingStep;
        command.class_id = 0;
        send_command(command);
    }

    void evaluate_dataset(const double *raw_data, const int *labels, int num_samples) {
        if (raw_data == 0 || labels == 0 || num_samples <= 0) {
            SC_REPORT_FATAL("rtl_trace_export", "invalid testing dataset");
        }
        m_stats.test_samples = static_cast<unsigned>(num_samples);

        AccelCommand command = {};
        command.kind = AccelCommandKind::ResetInference;
        command.class_id = 0;
        send_command(command);

        level_t quantized_sample[NUM_FEATURES];
        int issued = 0;
        int received = 0;
        int outstanding = 0;
        bool command_pending = false;
        unsigned command_hold_cycles = 0;

        while (received < num_samples) {
            if (!command_pending && command_hold_cycles == 0 &&
                issued < num_samples && outstanding < MAX_SAMPLES_IN_PIPELINE) {
                quantize_sample(m_memory, &raw_data[issued * NUM_FEATURES], quantized_sample);
                command.kind = AccelCommandKind::InferSample;
                command.class_id = 0;
                copy_quantized_sample(quantized_sample, command.sample);
                cmd_kind.write(static_cast<unsigned>(command.kind));
                cmd_class_id.write(command.class_id);
                for (int feature = 0; feature < NUM_FEATURES; ++feature) {
                    cmd_sample_levels[feature].write(command.sample.levels[feature]);
                }
                cmd_valid.write(true);
                command_pending = true;
            }

            rsp_ready.write(outstanding > 0);
            tick();

            if (!command_pending && command_hold_cycles > 0) {
                --command_hold_cycles;
            }

            if (command_pending && cmd_ready.read()) {
                write_command_line(command);
                cmd_valid.write(false);
                command_pending = false;
                command_hold_cycles = POST_COMMAND_HOLD_CYCLES;
                ++issued;
                ++outstanding;
                ++m_stats.commands;
                ++m_stats.inference_commands;
            }

            if (!(outstanding > 0 && rsp_valid.read() && rsp_ready.read())) {
                continue;
            }

            AccelResponse response = read_response();
            const int sample = received;
            ++received;
            --outstanding;
            ++m_stats.responses;

            int predicted = -1;
            int actual = -1;
            if (response.valid_prediction) {
                const int ngram_start = sample - N_GRAM_SIZE + 1;
                actual = get_ngram_real_label(&labels[ngram_start], N_GRAM_SIZE);
                predicted = predicted_class_from_response(response);
                if (predicted == actual) {
                    ++m_stats.correct;
                } else if (labels[ngram_start] != labels[ngram_start + N_GRAM_SIZE - 1]) {
                    ++m_stats.transition_error;
                } else {
                    ++m_stats.not_correct;
                }
            }

            write_response_line(response, predicted, actual);
        }

        cmd_valid.write(false);
        rsp_ready.write(false);
        m_stats.total = m_stats.correct + m_stats.not_correct + m_stats.transition_error;
    }

    HDC_Accelerator &m_accelerator;
    HDC_Memory &m_memory;
    const FootDataset &m_dataset;
    TraceLimits m_limits;
    std::ostream &m_commands;
    std::ostream &m_responses;
    TraceStats &m_stats;

    sc_core::sc_clock clk;
    sc_core::sc_signal<bool> rst;
    sc_core::sc_signal<bool> cmd_valid;
    sc_core::sc_signal<bool> cmd_ready;
    sc_core::sc_signal<command_kind_t> cmd_kind;
    sc_core::sc_signal<class_t> cmd_class_id;
    sc_core::sc_signal<level_t> cmd_sample_levels[NUM_FEATURES];
    sc_core::sc_signal<bool> rsp_valid;
    sc_core::sc_signal<bool> rsp_ready;
    sc_core::sc_signal<bool> rsp_valid_prediction;
    sc_core::sc_signal<distance_counter_t> rsp_distances[NUM_CLASSES];
};

void write_stats(const TracePaths &paths, int dataset_id, const TraceStats &stats) {
    std::ofstream out(paths.stats.c_str());
    if (!out.is_open()) {
        SC_REPORT_FATAL("rtl_trace_export", "failed to open stats.txt");
    }

    const double overall_accuracy =
        (stats.total == 0) ? 0.0 : static_cast<double>(stats.correct) / static_cast<double>(stats.total);
    const unsigned non_transition_total = stats.correct + stats.not_correct;
    const double non_transition_accuracy =
        (non_transition_total == 0)
            ? 0.0
            : static_cast<double>(stats.correct) / static_cast<double>(non_transition_total);

    out << "dataset=" << dataset_id << '\n';
    out << "cycles=" << stats.cycles << '\n';
    out << "commands=" << stats.commands << '\n';
    out << "train_samples=" << stats.train_samples << '\n';
    out << "test_samples=" << stats.test_samples << '\n';
    out << "training_commands=" << stats.training_commands << '\n';
    out << "inference_commands=" << stats.inference_commands << '\n';
    out << "expected_responses=" << stats.responses << '\n';
    out << "correct=" << stats.correct << '\n';
    out << "not_correct=" << stats.not_correct << '\n';
    out << "transition_error=" << stats.transition_error << '\n';
    out << "total=" << stats.total << '\n';
    out << std::fixed << std::setprecision(12);
    out << "overall_accuracy=" << overall_accuracy << '\n';
    out << "non_transition_accuracy=" << non_transition_accuracy << '\n';
}

void parse_args(int argc, char **argv, int &dataset_id, std::string &out_dir, TraceLimits &limits) {
    dataset_id = 0;
    out_dir = "build/rtl_trace_dataset00";
    limits.train_samples = -1;
    limits.test_samples = -1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--dataset" && i + 1 < argc) {
            dataset_id = std::atoi(argv[++i]);
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--sample-limit" && i + 1 < argc) {
            const int limit = std::atoi(argv[++i]);
            limits.train_samples = limit;
            limits.test_samples = limit;
        } else if (arg == "--train-limit" && i + 1 < argc) {
            limits.train_samples = std::atoi(argv[++i]);
        } else if (arg == "--test-limit" && i + 1 < argc) {
            limits.test_samples = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--dataset 0] [--out build/rtl_trace_dataset00]"
                      << " [--sample-limit N] [--train-limit N] [--test-limit N]\n";
            std::exit(EXIT_FAILURE);
        }
    }

    if (dataset_id < 0 || dataset_id >= NUM_DATASETS) {
        SC_REPORT_FATAL("rtl_trace_export", "dataset id out of range");
    }
    if (dataset_id != 0) {
        SC_REPORT_FATAL("rtl_trace_export", "RTL HLS build currently has only dataset-00 CiM compiled in");
    }
}

} // namespace

int sc_main(int argc, char **argv) {
    int dataset_id = 0;
    std::string out_dir;
    TraceLimits limits;
    parse_args(argc, argv, dataset_id, out_dir, limits);

    ensure_directory(out_dir);
    const TracePaths paths = make_trace_paths(out_dir);

    std::ofstream commands(paths.commands.c_str());
    std::ofstream responses(paths.responses.c_str());
    if (!commands.is_open() || !responses.is_open()) {
        SC_REPORT_FATAL("rtl_trace_export", "failed to open trace output files");
    }

    HDC_Accelerator accelerator("accelerator");
    HDC_Memory memory("memory");
    const std::string cim_path = dataset_file("cim", dataset_id);
    const std::string quantizer_path = dataset_file("quantizer", dataset_id);
    load_cim_into_accelerator(accelerator, cim_path.c_str());
    load_quantizer(memory, quantizer_path.c_str());

    FootDataset dataset = load_foot_dataset_by_id(dataset_id);
    TraceStats stats;
    RtlTraceDriver driver(accelerator, memory, dataset, limits, commands, responses, stats);
    driver.run();

    commands.close();
    responses.close();
    write_stats(paths, dataset_id, stats);

    std::cout << "Wrote RTL trace to " << out_dir << '\n';
    std::cout << "commands=" << stats.commands
              << " responses=" << stats.responses
              << " cycles=" << stats.cycles << '\n';
    return EXIT_SUCCESS;
}
