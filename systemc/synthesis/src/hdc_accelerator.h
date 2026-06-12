// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// The synthesis boundary for the first hardware version is HDC_Accelerator only.
#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc.h>
#ifdef STRATUS_HLS
#include "stratus_hls.h"
#endif
#include "cynw_p2p.h"
#ifndef HLS_DEFINE_PROTOCOL
#define HLS_DEFINE_PROTOCOL(name) ((void)0)
#endif
#include "systemc_types.h"
#include "hdc_transactions.h"

using hdc_systemc::AccelCommandKind;
using hdc_systemc::QuantizedSample;
using hdc_systemc::class_t;
using hdc_systemc::command_kind_t;
using hdc_systemc::distance_counter_t;
using hdc_systemc::feature_score_t;
using hdc_systemc::hv_t;
using hdc_systemc::level_t;
using hdc_systemc::train_counter_t;
using hdc_systemc::train_score_t;

template <typename T>
using hdc_p2p = cynw_p2p<T>;

struct EncoderPacket {
    AccelCommandKind kind;
    class_t class_id;
    QuantizedSample sample;
    hv_t encoded;

    bool operator==(const EncoderPacket &other) const {
        return kind == other.kind &&
               class_id == other.class_id &&
               sample == other.sample &&
               encoded == other.encoded;
    }

    bool operator!=(const EncoderPacket &other) const {
        return !(*this == other);
    }
};

struct NGramPacket {
    AccelCommandKind kind;
    class_t class_id;
    hv_t ngram;
    bool valid_ngram;

    bool operator==(const NGramPacket &other) const {
        return kind == other.kind &&
               class_id == other.class_id &&
               ngram == other.ngram &&
               valid_ngram == other.valid_ngram;
    }

    bool operator!=(const NGramPacket &other) const {
        return !(*this == other);
    }
};

struct DistancePacket {
    bool valid_prediction;
    distance_counter_t distances[NUM_CLASSES];

    bool operator==(const DistancePacket &other) const {
        if (valid_prediction != other.valid_prediction) {
            return false;
        }
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            if (distances[class_id] != other.distances[class_id]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const DistancePacket &other) const {
        return !(*this == other);
    }
};

inline void sc_trace(sc_core::sc_trace_file *tf,
                     const EncoderPacket &packet,
                     const std::string &name) {
#ifndef STRATUS_HLS
    sc_core::sc_trace(tf, packet.class_id, name + ".class_id");
    sc_trace(tf, packet.sample, name + ".sample");
    sc_trace(tf, packet.encoded, name + ".encoded");
#else
    (void)tf;
    (void)packet;
    (void)name;
#endif
}

inline void sc_trace(sc_core::sc_trace_file *tf,
                     const NGramPacket &packet,
                     const std::string &name) {
#ifndef STRATUS_HLS
    sc_core::sc_trace(tf, packet.class_id, name + ".class_id");
    sc_trace(tf, packet.ngram, name + ".ngram");
    sc_core::sc_trace(tf, packet.valid_ngram, name + ".valid_ngram");
#else
    (void)tf;
    (void)packet;
    (void)name;
#endif
}

inline void sc_trace(sc_core::sc_trace_file *tf,
                     const DistancePacket &packet,
                     const std::string &name) {
#ifndef STRATUS_HLS
    sc_core::sc_trace(tf, packet.valid_prediction, name + ".valid_prediction");
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        std::ostringstream signal_name;
        signal_name << name << ".distance" << class_id;
        sc_core::sc_trace(tf, packet.distances[class_id], signal_name.str());
    }
#else
    (void)tf;
    (void)packet;
    (void)name;
#endif
}

#ifndef STRATUS_HLS
inline std::ostream &operator<<(std::ostream &os, const EncoderPacket &packet) {
    os << "EncoderPacket{kind=" << static_cast<unsigned>(packet.kind)
       << ",class=" << packet.class_id << '}';
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const NGramPacket &packet) {
    os << "NGramPacket{kind=" << static_cast<unsigned>(packet.kind)
       << ",class=" << packet.class_id
       << ",valid=" << packet.valid_ngram << '}';
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const DistancePacket &packet) {
    os << "DistancePacket{valid=" << packet.valid_prediction << '}';
    return os;
}
#endif

SC_MODULE(HDC_Accelerator) {
public:
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;
    sc_core::sc_in<bool> cmd_valid;
    sc_core::sc_out<bool> cmd_ready;
    sc_core::sc_in<command_kind_t> cmd_kind;
    sc_core::sc_in<class_t> cmd_class_id;
    sc_core::sc_in<level_t> cmd_sample_levels[NUM_FEATURES];
    sc_core::sc_out<bool> rsp_valid;
    sc_core::sc_in<bool> rsp_ready;
    sc_core::sc_out<bool> rsp_valid_prediction;
    sc_core::sc_out<distance_counter_t> rsp_distances[NUM_CLASSES];

    SC_CTOR(HDC_Accelerator);

    // Simulation/pre-synthesis preload helper only.
    // This is not a hardware runtime load interface; real deployment needs ROM
    // initialization, generated constants, or a dedicated preload path later.
    void set_cim(unsigned level, unsigned feature, const hv_t &value);
    // Simulation/pre-synthesis preload helper only.
    // This is not a hardware runtime load interface; real deployment needs ROM
    // initialization, generated constants, or a dedicated preload path later.
    void set_assoc_class(unsigned class_id, const hv_t &value);

private:
    static constexpr unsigned ENCODER_WORDS_PER_CYCLE = 1;
    static_assert(ENCODER_WORDS_PER_CYCLE == 1,
                  "multi-word-per-cycle encoder support is intentionally not implemented yet");
    static constexpr unsigned NGRAM_WORDS_PER_CYCLE = 1;
    static_assert(NGRAM_WORDS_PER_CYCLE == 1,
                  "multi-word-per-cycle n-gram support is intentionally not implemented yet");

    void command_thread();
    void encoder_thread();
    void ngram_thread();
    void train_thread();
    void distance_thread();
    void response_thread();

    // N-gram datapath.
    void push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample);

    // Training-side bundling.
    void add_ngram_to_bundling_buffer(const hv_t &encoded_ngram);
    void reset_response_ports();
    void reset_bundling_buffer_only();
    void finalize_current_class();
    void reset_ngram_buffer();

    hdc_p2p<EncoderPacket> encoder_in;
    hdc_p2p<EncoderPacket> encoder_out;
    hdc_p2p<NGramPacket> bundler_in;
    hdc_p2p<NGramPacket> distance_in;
    hdc_p2p<DistancePacket> distance_done;
    hdc_p2p<bool> control_done;

    hv_t m_ngram_buffer[N_GRAM_SIZE];
    unsigned m_ngram_buffer_write_pos;
    unsigned m_ngram_buffer_fill_count;

    // Signed bundling score for the currently trained class segment.
    train_score_t m_bundling_score[VECTOR_DIMENSION];
    train_counter_t m_current_class_count;
    class_t m_current_class_id;
    bool m_current_class_valid;

    hv_t m_cim[NUM_LEVELS][NUM_FEATURES];
    hv_t m_assoc_mem[NUM_CLASSES];
};

namespace hdc_systemc {

using ::HDC_Accelerator;

} // namespace hdc_systemc

#endif
