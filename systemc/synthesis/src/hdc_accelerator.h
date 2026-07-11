// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// The synthesis boundary for the first hardware version is HDC_Accelerator only.
#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc.h>
#ifdef STRATUS_HLS
#include "stratus_hls.h"
#include <cynw_p2p.h>
#endif
#if !defined(STRATUS_HLS) && !defined(HLS_DEFINE_PROTOCOL)
#define HLS_DEFINE_PROTOCOL(name) ((void)0)
#endif
#if !defined(STRATUS_HLS) && !defined(HLS_UNROLL_LOOP)
#define HLS_UNROLL_LOOP(mode, name) ((void)0)
#endif
#include "systemc_types.h"
#include "hdc_transactions.h"

using hdc_systemc::AccelCommandKind;
using hdc_systemc::CLASS_BITS;
using hdc_systemc::COMMAND_KIND_BITS;
using hdc_systemc::DISTANCE_BITS;
using hdc_systemc::HV_WORD_BITS;
using hdc_systemc::HV_WORDS;
using hdc_systemc::LEVEL_BITS;
using hdc_systemc::QuantizedSample;
using hdc_systemc::class_t;
using hdc_systemc::clear_quantized_sample;
using hdc_systemc::command_kind_t;
using hdc_systemc::distance_counter_t;
using hdc_systemc::feature_score_t;
using hdc_systemc::hv_clear;
using hdc_systemc::hv_t;
using hdc_systemc::level_t;
using hdc_systemc::train_counter_t;
using hdc_systemc::train_score_t;

struct EncoderPacket {
    AccelCommandKind kind = AccelCommandKind::ResetTraining;
    class_t class_id = 0;
    QuantizedSample sample = {};
    hv_t encoded = {};

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
    AccelCommandKind kind = AccelCommandKind::ResetTraining;
    class_t class_id = 0;
    hv_t ngram = {};
    bool valid_ngram = false;

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
    bool valid_prediction = false;
    distance_counter_t distances[NUM_CLASSES] = {};

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

typedef sc_dt::sc_biguint<NUM_FEATURES * LEVEL_BITS> sample_bits_t;
typedef sc_dt::sc_biguint<VECTOR_DIMENSION> hv_bits_t;
typedef sc_dt::sc_biguint<NUM_CLASSES * DISTANCE_BITS> distance_bits_t;
static constexpr unsigned COMMAND_CHANNEL_BITS =
    COMMAND_KIND_BITS + CLASS_BITS + (NUM_FEATURES * LEVEL_BITS);
static constexpr unsigned RESPONSE_CHANNEL_BITS =
    1u + (NUM_CLASSES * DISTANCE_BITS);
typedef sc_dt::sc_biguint<COMMAND_CHANNEL_BITS> command_channel_bits_t;
typedef sc_dt::sc_biguint<RESPONSE_CHANNEL_BITS> response_channel_bits_t;

#ifdef STRATUS_HLS
typedef cynw_p2p<command_channel_bits_t, CYN::PIN> HdcCommandChannel;
typedef cynw_p2p<response_channel_bits_t, CYN::PIN> HdcResponseChannel;
#endif

struct EncoderChannelPacket {
    command_kind_t kind = 0;
    class_t class_id = 0;
    sample_bits_t sample = 0;
    hv_bits_t encoded = 0;

    bool operator==(const EncoderChannelPacket &other) const {
        return kind == other.kind &&
               class_id == other.class_id &&
               sample == other.sample &&
               encoded == other.encoded;
    }

    bool operator!=(const EncoderChannelPacket &other) const {
        return !(*this == other);
    }
};

struct NGramChannelPacket {
    command_kind_t kind = 0;
    class_t class_id = 0;
    hv_bits_t ngram = 0;
    bool valid_ngram = false;

    bool operator==(const NGramChannelPacket &other) const {
        return kind == other.kind &&
               class_id == other.class_id &&
               ngram == other.ngram &&
               valid_ngram == other.valid_ngram;
    }

    bool operator!=(const NGramChannelPacket &other) const {
        return !(*this == other);
    }
};

struct DistanceChannelPacket {
    bool valid_prediction = false;
    distance_bits_t distances = 0;

    bool operator==(const DistanceChannelPacket &other) const {
        return valid_prediction == other.valid_prediction &&
               distances == other.distances;
    }

    bool operator!=(const DistanceChannelPacket &other) const {
        return !(*this == other);
    }
};

inline void clear_encoder_packet(EncoderPacket &packet) {
    packet.kind = AccelCommandKind::ResetTraining;
    packet.class_id = 0;
    clear_quantized_sample(packet.sample);
    hv_clear(packet.encoded);
}

inline void clear_ngram_packet(NGramPacket &packet) {
    packet.kind = AccelCommandKind::ResetTraining;
    packet.class_id = 0;
    hv_clear(packet.ngram);
    packet.valid_ngram = false;
}

inline void clear_distance_packet(DistancePacket &packet) {
    packet.valid_prediction = false;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        packet.distances[class_id] = 0;
    }
}

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

inline void sc_trace(sc_core::sc_trace_file *tf, const EncoderChannelPacket &packet,
                     const std::string &name) {
    sc_core::sc_trace(tf, packet.kind, name + ".kind");
    sc_core::sc_trace(tf, packet.class_id, name + ".class_id");
    sc_core::sc_trace(tf, packet.sample, name + ".sample");
    sc_core::sc_trace(tf, packet.encoded, name + ".encoded");
}

inline void sc_trace(sc_core::sc_trace_file *tf, const NGramChannelPacket &packet,
                     const std::string &name) {
    sc_core::sc_trace(tf, packet.kind, name + ".kind");
    sc_core::sc_trace(tf, packet.class_id, name + ".class_id");
    sc_core::sc_trace(tf, packet.ngram, name + ".ngram");
    sc_core::sc_trace(tf, packet.valid_ngram, name + ".valid_ngram");
}

inline void sc_trace(sc_core::sc_trace_file *tf, const DistanceChannelPacket &packet,
                     const std::string &name) {
    sc_core::sc_trace(tf, packet.valid_prediction, name + ".valid_prediction");
    sc_core::sc_trace(tf, packet.distances, name + ".distances");
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

inline std::ostream &operator<<(std::ostream &os, const EncoderChannelPacket &) {
    return os << "EncoderChannelPacket";
}

inline std::ostream &operator<<(std::ostream &os, const NGramChannelPacket &packet) {
    return os << "NGramChannelPacket{valid=" << packet.valid_ngram << '}';
}

inline std::ostream &operator<<(std::ostream &os, const DistanceChannelPacket &packet) {
    return os << "DistanceChannelPacket{valid=" << packet.valid_prediction << '}';
}

SC_MODULE(HDC_Accelerator) {
public:
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;
#ifdef STRATUS_HLS
    HdcCommandChannel::in cmd;
    HdcResponseChannel::out rsp;
#else
    sc_core::sc_in<bool> cmd_valid;
    sc_core::sc_out<bool> cmd_ready;
    sc_core::sc_in<command_kind_t> cmd_kind;
    sc_core::sc_in<class_t> cmd_class_id;
    sc_core::sc_in<level_t> cmd_sample_levels[NUM_FEATURES];
    sc_core::sc_out<bool> rsp_valid;
    sc_core::sc_in<bool> rsp_ready;
    sc_core::sc_out<bool> rsp_valid_prediction;
    sc_core::sc_out<distance_counter_t> rsp_distances[NUM_CLASSES];
#endif

    SC_CTOR(HDC_Accelerator);

    // Simulation/pre-synthesis preload helper only.
    // This is not a hardware runtime load interface; real deployment needs ROM
    // initialization, generated constants, or a dedicated preload path later.
    void set_cim(unsigned level, unsigned feature, const hv_t &value);
    // Simulation/pre-synthesis preload helper only.
    // This is not a hardware runtime load interface; real deployment needs ROM
    // initialization, generated constants, or a dedicated preload path later.
    void set_assoc_class(unsigned class_id, const hv_t &value);
#ifndef STRATUS_HLS
    void reset_training_debug_counters();
    void print_training_debug_counters(std::ostream &out) const;
    void dump_assoc_mem(std::ostream &out) const;
#endif

private:
    static constexpr unsigned ENCODER_WORDS_PER_CYCLE = 1;
    static_assert(ENCODER_WORDS_PER_CYCLE == 1,
                  "multi-word-per-cycle encoder support is intentionally not implemented yet");
    static constexpr unsigned NGRAM_WORDS_PER_CYCLE = 1;
    static_assert(NGRAM_WORDS_PER_CYCLE == 1,
                  "multi-word-per-cycle n-gram support is intentionally not implemented yet");
    static constexpr unsigned TRAIN_WORDS_PER_CYCLE = 1;
    static_assert(TRAIN_WORDS_PER_CYCLE == 1,
                  "multi-word-per-cycle train support is intentionally not implemented yet");

    // Clocked pipeline stages.
    void command_thread();
    void encoder_thread();
    void ngram_thread();
    void train_thread();
    void distance_thread();
    void response_thread();

    // N-gram datapath.
    void push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample);

    void reset_ngram_buffer();

#ifdef STRATUS_HLS
    cynw_p2p_direct<EncoderChannelPacket, CYN::PIN> m_encoder_in;

    cynw_p2p_direct<EncoderChannelPacket, CYN::PIN> m_encoder_out;

    cynw_p2p_direct<NGramChannelPacket, CYN::PIN> m_bundler_in;

    cynw_p2p_direct<NGramChannelPacket, CYN::PIN> m_distance_in;

    cynw_p2p_direct<DistanceChannelPacket, CYN::PIN> m_distance_done;

    cynw_p2p_direct<bool, CYN::PIN> m_ngram_control_done;

    cynw_p2p_direct<bool, CYN::PIN> m_train_control_done;
#else
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

    unsigned long long m_debug_encoder_out_train_tokens;
    unsigned long long m_debug_encoder_valid_popcount[NUM_CLASSES];
    unsigned long long m_debug_encoder_valid_weighted_sum[NUM_CLASSES];
    unsigned long long m_debug_encoder_valid_first_popcount[NUM_CLASSES];
    unsigned long long m_debug_encoder_valid_last_popcount[NUM_CLASSES];
    bool m_debug_encoder_valid_seen[NUM_CLASSES];
    unsigned long long m_debug_bundler_train_invalid_tokens;
    unsigned long long m_debug_bundler_train_valid_tokens;
    unsigned long long m_debug_bundler_invalid_training_step_tokens;
    unsigned long long m_debug_train_valid_ngram_tokens;
    unsigned long long m_debug_train_invalid_training_step_tokens;
    unsigned long long m_debug_bundler_valid_popcount[NUM_CLASSES];
    unsigned long long m_debug_bundler_valid_weighted_sum[NUM_CLASSES];
    unsigned long long m_debug_bundler_valid_first_popcount[NUM_CLASSES];
    unsigned long long m_debug_bundler_valid_last_popcount[NUM_CLASSES];
    bool m_debug_bundler_valid_seen[NUM_CLASSES];
    unsigned long long m_debug_train_valid_popcount[NUM_CLASSES];
    unsigned long long m_debug_train_valid_weighted_sum[NUM_CLASSES];
    unsigned long long m_debug_train_valid_first_popcount[NUM_CLASSES];
    unsigned long long m_debug_train_valid_last_popcount[NUM_CLASSES];
    bool m_debug_train_valid_seen[NUM_CLASSES];
#endif

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
    train_score_t m_bundling_score[HV_WORDS][HV_WORD_BITS];
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
