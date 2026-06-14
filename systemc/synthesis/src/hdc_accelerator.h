// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// The synthesis boundary for the first hardware version is HDC_Accelerator only.
#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc.h>
#ifdef STRATUS_HLS
#include "stratus_hls.h"
#endif
#ifndef HLS_DEFINE_PROTOCOL
#define HLS_DEFINE_PROTOCOL(name) if (true)
#endif
#ifndef HLS_UNROLL_LOOP
#define HLS_UNROLL_LOOP(mode, name) ((void)0)
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

inline void sc_trace(sc_core::sc_trace_file *tf, const EncoderPacket &packet, const std::string &name) {
    sc_core::sc_trace(tf, packet.class_id, name + ".class_id");
    sc_trace(tf, packet.sample, name + ".sample");
    sc_trace(tf, packet.encoded, name + ".encoded");
}

inline void sc_trace(sc_core::sc_trace_file *tf, const NGramPacket &packet, const std::string &name) {
    sc_core::sc_trace(tf, packet.class_id, name + ".class_id");
    sc_trace(tf, packet.ngram, name + ".ngram");
    sc_core::sc_trace(tf, packet.valid_ngram, name + ".valid_ngram");
}

inline void sc_trace(sc_core::sc_trace_file *tf, const DistancePacket &packet, const std::string &name) {
    sc_core::sc_trace(tf, packet.valid_prediction, name + ".valid_prediction");
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        sc_core::sc_trace(tf, packet.distances[class_id],
                          name + ".distance" + std::to_string(class_id));
    }
}

inline std::ostream &operator<<(std::ostream &os, const EncoderPacket &) {
    return os << "EncoderPacket";
}

inline std::ostream &operator<<(std::ostream &os, const NGramPacket &packet) {
    return os << "NGramPacket{valid=" << packet.valid_ngram << '}';
}

inline std::ostream &operator<<(std::ostream &os, const DistancePacket &packet) {
    return os << "DistancePacket{valid=" << packet.valid_prediction << '}';
}

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

    // Clocked pipeline stages.
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
    void reset_all_local_state();
    void reset_bundling_buffer_only();
    void finalize_current_class();
    void reset_ngram_buffer();

    sc_core::sc_signal<EncoderPacket> m_encoder_in_data;
    sc_core::sc_signal<bool> m_encoder_in_valid;
    sc_core::sc_signal<bool> m_encoder_in_ready;

    sc_core::sc_signal<EncoderPacket> m_encoder_out_data;
    sc_core::sc_signal<bool> m_encoder_out_valid;
    sc_core::sc_signal<bool> m_encoder_out_ready;

    sc_core::sc_signal<NGramPacket> m_bundler_in_data;
    sc_core::sc_signal<bool> m_bundler_in_valid;
    sc_core::sc_signal<bool> m_bundler_in_ready;

    sc_core::sc_signal<NGramPacket> m_distance_in_data;
    sc_core::sc_signal<bool> m_distance_in_valid;
    sc_core::sc_signal<bool> m_distance_in_ready;

    sc_core::sc_signal<DistancePacket> m_distance_done_data;
    sc_core::sc_signal<bool> m_distance_done_valid;
    sc_core::sc_signal<bool> m_distance_done_ready;

    sc_core::sc_signal<bool> m_ngram_control_done_valid;
    sc_core::sc_signal<bool> m_ngram_control_done_ready;
    sc_core::sc_signal<bool> m_train_control_done_valid;
    sc_core::sc_signal<bool> m_train_control_done_ready;

    hv_t m_ngram_buffer[N_GRAM_SIZE];
    unsigned m_ngram_buffer_write_pos;
    unsigned m_ngram_buffer_fill_count;
    bool m_ngram_bind_busy;
    unsigned m_ngram_bind_round;
    unsigned m_ngram_bind_word;
    unsigned m_ngram_oldest_slot;
    EncoderPacket m_ngram_work_packet;
    hv_t m_ngram_work;
    hv_t m_ngram_next;

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
