// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// The synthesis boundary for the first hardware version is HDC_Accelerator only.
#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc.h>
#include "systemc_types.h"
#include "hdc_transactions.h"

namespace hdc_systemc {

struct EncoderPacket {
    AccelCommandKind kind;
    class_t class_id;
    QuantizedSample sample;
    hv_t encoded;
};

struct NGramPacket {
    AccelCommandKind kind;
    class_t class_id;
    hv_t ngram;
    bool valid_ngram;
};

struct DistancePacket {
    bool valid_prediction;
    distance_counter_t distances[NUM_CLASSES];
};

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
    // Pipeline scheduler and stages.
    void pipeline_fsm();
    void command_stage();
    void encoder_stage();
    void ngram_stage();
    void train_stage();
    void distance_stage();
    void response_stage();

    // Encoder datapath.
    void encode_sample(const QuantizedSample &sample, hv_t &encoded_sample);

    // N-gram datapath.
    void push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample);
    void bind_ngram(hv_t &encoded_ngram);
    void permute_xor(const hv_t &input, const hv_t &rhs, hv_t &output);

    // Training-side bundling.
    void add_ngram_to_bundling_buffer(const hv_t &encoded_ngram);
    void reset_all_local_state();
    void reset_bundling_buffer_only();
    void finalize_current_class();
    void reset_ngram_buffer();

    // Distance datapath.
    void compute_hamming_distances(const hv_t &query, distance_counter_t *distances);

    EncoderPacket m_encoder_in_data;
    bool m_encoder_in_valid;

    EncoderPacket m_encoder_out_data;
    bool m_encoder_out_valid;

    NGramPacket m_bundler_in_data;
    bool m_bundler_in_valid;

    NGramPacket m_distance_in_data;
    bool m_distance_in_valid;

    DistancePacket m_distance_done_data;
    bool m_distance_done_valid;

    bool m_control_done_valid;
    bool m_control_busy;

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

} // namespace hdc_systemc

#endif
