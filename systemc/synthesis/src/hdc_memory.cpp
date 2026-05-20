#include "hdc_memory.h"
namespace hdc_systemc {

namespace {

static constexpr std::uint64_t HV_BYTES = (VECTOR_DIMENSION + 7u) / 8u;
static constexpr std::uint64_t QUANTIZER_ROW_BYTES =
    static_cast<std::uint64_t>((NUM_LEVELS > 1) ? (NUM_LEVELS - 1) : 1) * sizeof(double);

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

} // namespace

HDC_Memory::HDC_Memory(sc_core::sc_module_name name)
    : sc_module(name), m_cim_loaded(false), m_quantizer_loaded(false) {
    reset_stats();
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

void HDC_Memory::reset_stats() {
    m_stats.quantizer_row_reads = 0;
    m_stats.quantizer_row_read_bytes = 0;
    m_stats.cim_reads = 0;
    m_stats.cim_read_bytes = 0;
    m_stats.assoc_reads = 0;
    m_stats.assoc_read_bytes = 0;
    m_stats.assoc_writes = 0;
    m_stats.assoc_write_bytes = 0;
}

const MemoryStats &HDC_Memory::stats() const {
    return m_stats;
}

void HDC_Memory::set_cim(const hv_t *flat_cim) {
    if (flat_cim == 0) {
        SC_REPORT_FATAL("HDC_Memory", "flat_cim must not be null");
    }

    for (int i = 0; i < NUM_LEVELS * NUM_FEATURES; ++i) {
        copy_hv(flat_cim[i], m_cim[i]);
    }
    m_cim_loaded = true;
}

void HDC_Memory::set_quantizer_boundaries(const double *flat_boundaries) {
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

const double *HDC_Memory::read_quantizer_row(unsigned feature) const {
    if (feature >= static_cast<unsigned>(NUM_FEATURES)) {
        SC_REPORT_FATAL("HDC_Memory", "quantizer feature index out of range");
        return 0;
    }
    if (!m_quantizer_loaded) {
        SC_REPORT_FATAL("HDC_Memory", "quantizer requested before load");
        return 0;
    }
    if (NUM_LEVELS <= 1) {
        ++m_stats.quantizer_row_reads;
        m_stats.quantizer_row_read_bytes += QUANTIZER_ROW_BYTES;
        return m_quantizer_boundaries;
    }

    ++m_stats.quantizer_row_reads;
    m_stats.quantizer_row_read_bytes += QUANTIZER_ROW_BYTES;
    return &m_quantizer_boundaries[feature * (NUM_LEVELS - 1)];
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

    ++m_stats.cim_reads;
    m_stats.cim_read_bytes += HV_BYTES;
    return m_cim[(level_index * NUM_FEATURES) + static_cast<int>(feature)];
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
    ++m_stats.assoc_writes;
    m_stats.assoc_write_bytes += HV_BYTES;
}

const hv_t &HDC_Memory::read_assoc_class(unsigned class_id) const {
    if (class_id >= static_cast<unsigned>(NUM_CLASSES)) {
        SC_REPORT_FATAL("HDC_Memory", "class index out of range");
        return m_assoc_mem[0];
    }
    ++m_stats.assoc_reads;
    m_stats.assoc_read_bytes += HV_BYTES;
    return m_assoc_mem[class_id];
}

} // namespace hdc_systemc
