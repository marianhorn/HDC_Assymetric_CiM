#include "hdc_memory.h"
#include <fstream>
#include <sstream>
#include <string>

namespace hdc_systemc {

namespace {

void clear_hv(hv_t &hv) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        hv[d] = sc_dt::SC_LOGIC_0;
    }
}

void copy_hv(const hv_t &src, hv_t &dst) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        dst[d] = src[d];
    }
}

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
        SC_REPORT_FATAL("HDC_Memory", "missing required header field");
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

HDC_Memory::HDC_Memory(sc_core::sc_module_name name)
    : sc_module(name), m_cim_loaded(false), m_quantizer_loaded(false) {
    clear_all();
}

void HDC_Memory::clear_all() {
    for (int i = 0; i < NUM_LEVELS * NUM_FEATURES; ++i) {
        clear_hv(m_cim[i]);
    }
    for (int i = 0; i < NUM_CLASSES; ++i) {
        clear_hv(m_assoc_mem[i]);
    }
    for (int i = 0; i < NUM_FEATURES * ((NUM_LEVELS > 1) ? (NUM_LEVELS - 1) : 1); ++i) {
        m_quantizer_boundaries[i] = 0.0;
    }
    m_cim_loaded = false;
    m_quantizer_loaded = false;
}

void HDC_Memory::load_cim_flat(const hv_t *flat_cim) {
    if (flat_cim == 0) {
        SC_REPORT_FATAL("HDC_Memory", "flat_cim must not be null");
    }

    for (int i = 0; i < NUM_LEVELS * NUM_FEATURES; ++i) {
        copy_hv(flat_cim[i], m_cim[i]);
    }
    m_cim_loaded = true;
}

void HDC_Memory::load_cim_text(const char *path) {
    if (path == 0 || path[0] == '\0') {
        SC_REPORT_FATAL("HDC_Memory", "CiM path must not be null or empty");
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        SC_REPORT_FATAL("HDC_Memory", "failed to open CiM text file");
    }

    bool loaded_entries[NUM_LEVELS * NUM_FEATURES];
    for (int i = 0; i < NUM_LEVELS * NUM_FEATURES; ++i) {
        loaded_entries[i] = false;
        clear_hv(m_cim[i]);
    }

    std::string line;
    bool header_checked = false;
    int loaded_count = 0;
    while (std::getline(file, line)) {
        if (line.find("#systemc_precomp_cim") != std::string::npos) {
            const int header_levels = parse_header_int_field(line, "num_levels=");
            const int header_features = parse_header_int_field(line, "num_features=");
            const int header_dimension = parse_header_int_field(line, "dimension=");
            if (header_levels != NUM_LEVELS) {
                SC_REPORT_FATAL("HDC_Memory", "CiM header num_levels does not match config_systemc.h");
            }
            if (header_features != NUM_FEATURES) {
                SC_REPORT_FATAL("HDC_Memory", "CiM header num_features does not match config_systemc.h");
            }
            if (header_dimension != VECTOR_DIMENSION) {
                SC_REPORT_FATAL("HDC_Memory", "CiM header dimension does not match config_systemc.h");
            }
            header_checked = true;
            continue;
        }
        if (is_comment_or_empty(line)) {
            continue;
        }
        if (!header_checked) {
            SC_REPORT_FATAL("HDC_Memory", "CiM text file header missing or not checked before data");
        }

        std::istringstream iss(line);
        int level = -1;
        int feature = -1;
        std::string bits;
        if (!(iss >> level >> feature >> bits)) {
            SC_REPORT_FATAL("HDC_Memory", "invalid CiM text line");
        }
        if (level < 0 || level >= NUM_LEVELS) {
            SC_REPORT_FATAL("HDC_Memory", "CiM level out of range");
        }
        if (feature < 0 || feature >= NUM_FEATURES) {
            SC_REPORT_FATAL("HDC_Memory", "CiM feature out of range");
        }
        if (static_cast<int>(bits.size()) != VECTOR_DIMENSION) {
            SC_REPORT_FATAL("HDC_Memory", "CiM bitstring length mismatch");
        }

        const int index = level * NUM_FEATURES + feature;
        if (loaded_entries[index]) {
            SC_REPORT_FATAL("HDC_Memory", "duplicate CiM entry in text file");
        }

        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            const char bit = bits[static_cast<std::string::size_type>(d)];
            if (bit == '0') {
                m_cim[index][d] = sc_dt::SC_LOGIC_0;
            } else if (bit == '1') {
                m_cim[index][d] = sc_dt::SC_LOGIC_1;
            } else {
                SC_REPORT_FATAL("HDC_Memory", "invalid CiM bit character");
            }
        }

        loaded_entries[index] = true;
        ++loaded_count;
    }

    if (loaded_count != NUM_LEVELS * NUM_FEATURES) {
        SC_REPORT_FATAL("HDC_Memory", "CiM text file does not contain all entries");
    }

    m_cim_loaded = true;
}

void HDC_Memory::load_quantizer_boundaries(const double *flat_boundaries) {
    if (NUM_LEVELS <= 1) {
        m_quantizer_loaded = true;
        return;
    }
    if (flat_boundaries == 0) {
        SC_REPORT_FATAL("HDC_Memory", "flat_boundaries must not be null");
    }

    const int count = NUM_FEATURES * (NUM_LEVELS - 1);
    for (int i = 0; i < count; ++i) {
        m_quantizer_boundaries[i] = flat_boundaries[i];
    }
    m_quantizer_loaded = true;
}

void HDC_Memory::load_quantizer_text(const char *path) {
    if (path == 0 || path[0] == '\0') {
        SC_REPORT_FATAL("HDC_Memory", "quantizer path must not be null or empty");
    }
    if (NUM_LEVELS <= 1) {
        m_quantizer_loaded = true;
        return;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        SC_REPORT_FATAL("HDC_Memory", "failed to open quantizer text file");
    }

    bool loaded_features[NUM_FEATURES];
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        loaded_features[feature] = false;
    }

    std::string line;
    bool header_checked = false;
    int loaded_count = 0;
    while (std::getline(file, line)) {
        if (line.find("#systemc_quantizer") != std::string::npos) {
            const int header_levels = parse_header_int_field(line, "num_levels=");
            const int header_features = parse_header_int_field(line, "num_features=");
            if (header_levels != NUM_LEVELS) {
                SC_REPORT_FATAL("HDC_Memory", "quantizer header num_levels does not match config_systemc.h");
            }
            if (header_features != NUM_FEATURES) {
                SC_REPORT_FATAL("HDC_Memory", "quantizer header num_features does not match config_systemc.h");
            }
            header_checked = true;
            continue;
        }
        if (is_comment_or_empty(line)) {
            continue;
        }
        if (!header_checked) {
            SC_REPORT_FATAL("HDC_Memory", "quantizer text file header missing or not checked before data");
        }

        std::istringstream iss(line);
        int feature = -1;
        if (!(iss >> feature)) {
            SC_REPORT_FATAL("HDC_Memory", "invalid quantizer text line");
        }
        if (feature < 0 || feature >= NUM_FEATURES) {
            SC_REPORT_FATAL("HDC_Memory", "quantizer feature out of range");
        }
        if (loaded_features[feature]) {
            SC_REPORT_FATAL("HDC_Memory", "duplicate quantizer feature entry");
        }

        for (int cut = 0; cut < NUM_LEVELS - 1; ++cut) {
            double boundary = 0.0;
            if (!(iss >> boundary)) {
                SC_REPORT_FATAL("HDC_Memory", "missing quantizer boundary value");
            }
            m_quantizer_boundaries[feature * (NUM_LEVELS - 1) + cut] = boundary;
        }

        loaded_features[feature] = true;
        ++loaded_count;
    }

    if (loaded_count != NUM_FEATURES) {
        SC_REPORT_FATAL("HDC_Memory", "quantizer text file does not contain all features");
    }

    m_quantizer_loaded = true;
}

const hv_t &HDC_Memory::read_cim(level_t level, unsigned feature) const {
    if (!m_cim_loaded) {
        SC_REPORT_FATAL("HDC_Memory", "CiM requested before load");
        return m_cim[0];
    }

    const unsigned level_index = level.to_uint();
    if (level_index >= NUM_LEVELS) {
        SC_REPORT_FATAL("HDC_Memory", "level index out of range");
        return m_cim[0];
    }
    if (feature >= static_cast<unsigned>(NUM_FEATURES)) {
        SC_REPORT_FATAL("HDC_Memory", "feature index out of range");
        return m_cim[0];
    }

    return m_cim[(level_index * NUM_FEATURES) + static_cast<int>(feature)];
}

level_t HDC_Memory::quantize_value(unsigned feature, double value) const {
    if (feature >= static_cast<unsigned>(NUM_FEATURES)) {
        SC_REPORT_FATAL("HDC_Memory", "feature index out of range");
        return 0;
    }
    if (NUM_LEVELS <= 1) {
        return 0;
    }
    if (!m_quantizer_loaded) {
        SC_REPORT_FATAL("HDC_Memory", "quantizer requested before load");
        return 0;
    }

    const double *boundaries = &m_quantizer_boundaries[feature * (NUM_LEVELS - 1)];
    if (value <= boundaries[0]) {
        return 0;
    }
    if (value > boundaries[NUM_LEVELS - 2]) {
        return NUM_LEVELS - 1;
    }

    int lo = 0;
    int hi = NUM_LEVELS - 2;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (value <= boundaries[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return static_cast<unsigned>(lo);
}

void HDC_Memory::quantize_sample(const double *raw_sample, level_t *quantized_sample) const {
    if (raw_sample == 0 || quantized_sample == 0) {
        SC_REPORT_FATAL("HDC_Memory", "quantize_sample inputs must not be null");
    }

    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        quantized_sample[feature] = quantize_value(static_cast<unsigned>(feature), raw_sample[feature]);
    }
}

void HDC_Memory::clear_assoc_mem() {
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        clear_hv(m_assoc_mem[class_id]);
    }
}

void HDC_Memory::write_assoc_class(unsigned class_id, const hv_t &class_hv) {
    if (class_id >= static_cast<unsigned>(NUM_CLASSES)) {
        SC_REPORT_FATAL("HDC_Memory", "class index out of range");
        return;
    }
    copy_hv(class_hv, m_assoc_mem[class_id]);
}

const hv_t &HDC_Memory::read_assoc_class(unsigned class_id) const {
    if (class_id >= static_cast<unsigned>(NUM_CLASSES)) {
        SC_REPORT_FATAL("HDC_Memory", "class index out of range");
        return m_assoc_mem[0];
    }
    return m_assoc_mem[class_id];
}

} // namespace hdc_systemc
