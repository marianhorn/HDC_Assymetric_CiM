// SIMULATION / SOFTWARE ONLY: Replays RTL trace files against the SystemC accelerator.
// This validates commands.txt and expected_responses.txt before debugging RTL/XSim.
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hdc_accelerator.h"

using namespace hdc_systemc;

namespace {

struct ReplayConfig {
    std::string trace_dir = "rtl_trace_dataset00_smoke20";
    int dataset_id = 0;
    unsigned long long timeout_cycles = 50000000ULL;
    unsigned long long progress_cycles = 100000ULL;
    unsigned reset_cycles = 32;
};

struct ReplayStats {
    unsigned long long cycles = 0;
    unsigned long long commands = 0;
    unsigned long long inference = 0;
    unsigned long long responses = 0;
    unsigned long long command_stall_cycles = 0;
    unsigned long long response_stall_cycles = 0;
    unsigned long long total_latency = 0;
    unsigned long long max_latency = 0;
    unsigned errors = 0;
};

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
        SC_REPORT_FATAL("rtl_trace_replay", "missing required header field");
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

std::string path_join(const std::string &dir, const char *file) {
    if (dir.empty()) {
        return std::string(file);
    }
    if (dir[dir.size() - 1] == '/' || dir[dir.size() - 1] == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

std::string dataset_file(const char *kind, int dataset_id) {
    char path[128];
    std::snprintf(path, sizeof(path), "import/%s_dataset%02d.txt", kind, dataset_id);
    return std::string(path);
}

void load_cim_into_accelerator(HDC_Accelerator &accelerator, const char *path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        SC_REPORT_FATAL("rtl_trace_replay", "failed to open CiM text file");
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
                SC_REPORT_FATAL("rtl_trace_replay", "CiM header does not match config_systemc.h");
            }
            header_checked = true;
            continue;
        }
        if (is_comment_or_empty(line)) {
            continue;
        }
        if (!header_checked) {
            SC_REPORT_FATAL("rtl_trace_replay", "CiM data found before checked header");
        }

        std::istringstream iss(line);
        int level = -1;
        int feature = -1;
        std::string bits;
        if (!(iss >> level >> feature >> bits)) {
            SC_REPORT_FATAL("rtl_trace_replay", "invalid CiM text line");
        }
        if (level < 0 || level >= NUM_LEVELS || feature < 0 || feature >= NUM_FEATURES ||
            static_cast<int>(bits.size()) != VECTOR_DIMENSION) {
            SC_REPORT_FATAL("rtl_trace_replay", "CiM entry out of range or wrong width");
        }

        const int index = level * NUM_FEATURES + feature;
        if (loaded_entries[static_cast<std::size_t>(index)]) {
            SC_REPORT_FATAL("rtl_trace_replay", "duplicate CiM entry");
        }

        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            const char bit = bits[static_cast<std::string::size_type>(d)];
            if (bit == '0') {
                hv_set_bit(flat_cim[static_cast<std::size_t>(index)], static_cast<unsigned>(d), false);
            } else if (bit == '1') {
                hv_set_bit(flat_cim[static_cast<std::size_t>(index)], static_cast<unsigned>(d), true);
            } else {
                SC_REPORT_FATAL("rtl_trace_replay", "invalid CiM bit character");
            }
        }

        loaded_entries[static_cast<std::size_t>(index)] = true;
        ++loaded_count;
    }

    if (loaded_count != NUM_LEVELS * NUM_FEATURES) {
        SC_REPORT_FATAL("rtl_trace_replay", "CiM file does not contain all entries");
    }

    for (int level = 0; level < NUM_LEVELS; ++level) {
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            accelerator.set_cim(static_cast<unsigned>(level),
                                static_cast<unsigned>(feature),
                                flat_cim[static_cast<std::size_t>(level * NUM_FEATURES + feature)]);
        }
    }
}

bool read_next_command(std::ifstream &commands, AccelCommand &command) {
    int kind = 0;
    int class_id = 0;
    if (!(commands >> kind >> class_id)) {
        return false;
    }
    command.kind = static_cast<AccelCommandKind>(kind);
    command.class_id = static_cast<unsigned>(class_id);
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        int level = 0;
        if (!(commands >> level)) {
            SC_REPORT_FATAL("rtl_trace_replay", "malformed commands.txt");
        }
        command.sample.levels[feature] = static_cast<unsigned>(level);
    }
    return true;
}

bool read_expected_response(std::ifstream &responses,
                            AccelResponse &response,
                            int &expected_predicted,
                            int &expected_actual) {
    int valid = 0;
    if (!(responses >> valid)) {
        return false;
    }
    response.valid_prediction = valid != 0;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        unsigned distance = 0;
        if (!(responses >> distance)) {
            SC_REPORT_FATAL("rtl_trace_replay", "malformed expected_responses.txt distance");
        }
        response.distances[class_id] = distance;
    }
    if (!(responses >> expected_predicted >> expected_actual)) {
        SC_REPORT_FATAL("rtl_trace_replay", "malformed expected_responses.txt predicted/actual");
    }
    return true;
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

class ReplayDriver {
public:
    ReplayDriver(HDC_Accelerator &accelerator,
                 std::ifstream &commands,
                 std::ifstream &responses,
                 const ReplayConfig &config,
                 ReplayStats &stats)
        : m_accelerator(accelerator),
          m_commands(commands),
          m_responses(responses),
          m_config(config),
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
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            m_accelerator.cmd_sample_levels[feature](cmd_sample_levels[feature]);
        }
        m_accelerator.rsp_valid(rsp_valid);
        m_accelerator.rsp_ready(rsp_ready);
        m_accelerator.rsp_valid_prediction(rsp_valid_prediction);
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            m_accelerator.rsp_distances[class_id](rsp_distances[class_id]);
        }
    }

    void run() {
        initialize_assoc_memory();
        reset();
        bool has_command = read_next_command(m_commands, m_current_command);
        bool command_pending = false;
        int outstanding = 0;
        std::vector<unsigned long long> issue_cycles;
        std::size_t issue_head = 0;

        while (true) {
            if (m_stats.cycles > m_config.timeout_cycles) {
                SC_REPORT_FATAL("rtl_trace_replay", "SystemC replay timed out");
            }

            if (!command_pending && has_command &&
                !(m_current_command.kind == AccelCommandKind::InferSample &&
                  outstanding >= MAX_SAMPLES_IN_PIPELINE)) {
                drive_command(m_current_command);
                command_pending = true;
            }
            rsp_ready.write(outstanding > 0);

            tick();
            if (m_config.progress_cycles > 0 &&
                (m_stats.cycles % m_config.progress_cycles) == 0) {
                std::cout << "progress cycles=" << m_stats.cycles
                          << " commands=" << m_stats.commands
                          << " inference=" << m_stats.inference
                          << " responses=" << m_stats.responses
                          << " outstanding=" << outstanding << std::endl;
                m_accelerator.print_training_debug_counters(std::cout);
            }

            if (command_pending && cmd_ready.read()) {
                ++m_stats.commands;
                if (m_current_command.kind == AccelCommandKind::InferSample) {
                    ++m_stats.inference;
                    ++outstanding;
                    issue_cycles.push_back(m_stats.cycles);
                }
                cmd_valid.write(false);
                command_pending = false;
                has_command = read_next_command(m_commands, m_current_command);
            } else if (command_pending && !cmd_ready.read()) {
                ++m_stats.command_stall_cycles;
            }

            if (rsp_valid.read() && !rsp_ready.read()) {
                ++m_stats.response_stall_cycles;
            }

            if (rsp_valid.read() && rsp_ready.read()) {
                compare_response();
                if (issue_head < issue_cycles.size()) {
                    const unsigned long long latency = m_stats.cycles - issue_cycles[issue_head];
                    ++issue_head;
                    m_stats.total_latency += latency;
                    if (latency > m_stats.max_latency) {
                        m_stats.max_latency = latency;
                    }
                }
                ++m_stats.responses;
                --outstanding;
            }

            if (!has_command && !command_pending && outstanding == 0) {
                AccelResponse extra;
                int extra_predicted = 0;
                int extra_actual = 0;
                if (read_expected_response(m_responses, extra, extra_predicted, extra_actual)) {
                    SC_REPORT_FATAL("rtl_trace_replay", "expected_responses.txt has extra responses");
                }
                break;
            }
        }

        cmd_valid.write(false);
        rsp_ready.write(false);
    }

private:
    void initialize_assoc_memory() {
        hv_t empty_class_vector;
        hv_clear(empty_class_vector);
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            m_accelerator.set_assoc_class(class_id, empty_class_vector);
        }
    }

    void reset() {
        cmd_valid.write(false);
        rsp_ready.write(false);
        rst.write(true);
        for (unsigned i = 0; i < m_config.reset_cycles; ++i) {
            tick();
        }
        rst.write(false);
        tick();
        tick();
    }

    void tick() {
        sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_NS));
        sc_core::sc_start(sc_core::SC_ZERO_TIME);
        ++m_stats.cycles;
    }

    void drive_command(const AccelCommand &command) {
        cmd_kind.write(static_cast<unsigned>(command.kind));
        cmd_class_id.write(command.class_id);
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            cmd_sample_levels[feature].write(command.sample.levels[feature]);
        }
        cmd_valid.write(true);
    }

    AccelResponse read_actual_response() const {
        AccelResponse response = {};
        response.valid_prediction = rsp_valid_prediction.read();
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            response.distances[class_id] = rsp_distances[class_id].read();
        }
        return response;
    }

    void compare_response() {
        AccelResponse expected;
        int expected_predicted = -1;
        int expected_actual = -1;
        if (!read_expected_response(m_responses, expected, expected_predicted, expected_actual)) {
            SC_REPORT_FATAL("rtl_trace_replay", "missing expected response");
        }

        const AccelResponse actual = read_actual_response();
        if (actual.valid_prediction != expected.valid_prediction) {
            ++m_stats.errors;
            std::cout << "valid_prediction mismatch response=" << m_stats.responses
                      << " actual=" << actual.valid_prediction
                      << " expected=" << expected.valid_prediction << std::endl;
        }

        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            if (actual.distances[class_id] != expected.distances[class_id]) {
                ++m_stats.errors;
                std::cout << "distance mismatch response=" << m_stats.responses
                          << " class=" << class_id
                          << " actual=" << actual.distances[class_id]
                          << " expected=" << expected.distances[class_id] << std::endl;
            }
        }

        if (expected.valid_prediction) {
            const int actual_predicted = predicted_class_from_response(actual);
            if (actual_predicted != expected_predicted) {
                ++m_stats.errors;
                std::cout << "predicted mismatch response=" << m_stats.responses
                          << " actual=" << actual_predicted
                          << " expected=" << expected_predicted
                          << " expected_actual_label=" << expected_actual << std::endl;
            }
        }
    }

    HDC_Accelerator &m_accelerator;
    std::ifstream &m_commands;
    std::ifstream &m_responses;
    const ReplayConfig &m_config;
    ReplayStats &m_stats;
    AccelCommand m_current_command;

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

void parse_args(int argc, char **argv, ReplayConfig &config) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--trace" && i + 1 < argc) {
            config.trace_dir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            config.dataset_id = std::atoi(argv[++i]);
        } else if (arg == "--timeout-cycles" && i + 1 < argc) {
            config.timeout_cycles = std::strtoull(argv[++i], 0, 10);
        } else if (arg == "--progress-cycles" && i + 1 < argc) {
            config.progress_cycles = std::strtoull(argv[++i], 0, 10);
        } else if (arg == "--reset-cycles" && i + 1 < argc) {
            config.reset_cycles = static_cast<unsigned>(std::strtoul(argv[++i], 0, 10));
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " --trace rtl_trace_dataset00_smoke20"
                      << " [--dataset 0] [--timeout-cycles N]"
                      << " [--progress-cycles N] [--reset-cycles N]\n";
            std::exit(EXIT_FAILURE);
        }
    }

    if (config.dataset_id != 0) {
        SC_REPORT_FATAL("rtl_trace_replay", "only dataset 00 is supported for the current RTL trace flow");
    }
}

} // namespace

int sc_main(int argc, char **argv) {
    ReplayConfig config;
    parse_args(argc, argv, config);

    const std::string commands_path = path_join(config.trace_dir, "commands.txt");
    const std::string responses_path = path_join(config.trace_dir, "expected_responses.txt");

    std::ifstream commands(commands_path.c_str());
    std::ifstream responses(responses_path.c_str());
    if (!commands.is_open()) {
        SC_REPORT_FATAL("rtl_trace_replay", "failed to open commands.txt");
    }
    if (!responses.is_open()) {
        SC_REPORT_FATAL("rtl_trace_replay", "failed to open expected_responses.txt");
    }

    HDC_Accelerator accelerator("accelerator");
    const std::string cim_path = dataset_file("cim", config.dataset_id);
    load_cim_into_accelerator(accelerator, cim_path.c_str());

    ReplayStats stats;
    ReplayDriver driver(accelerator, commands, responses, config, stats);
    driver.run();

    const double average_latency =
        (stats.responses == 0)
            ? 0.0
            : static_cast<double>(stats.total_latency) / static_cast<double>(stats.responses);

    std::cout << "SystemC trace replay complete" << std::endl;
    accelerator.print_training_debug_counters(std::cout);
    std::cout << "cycles=" << stats.cycles << std::endl;
    std::cout << "commands=" << stats.commands << std::endl;
    std::cout << "inference=" << stats.inference << std::endl;
    std::cout << "responses=" << stats.responses << std::endl;
    std::cout << "command_stall_cycles=" << stats.command_stall_cycles << std::endl;
    std::cout << "response_stall_cycles=" << stats.response_stall_cycles << std::endl;
    std::cout << "average_inference_latency=" << average_latency << std::endl;
    std::cout << "max_inference_latency=" << stats.max_latency << std::endl;
    std::cout << "errors=" << stats.errors << std::endl;

    return stats.errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
