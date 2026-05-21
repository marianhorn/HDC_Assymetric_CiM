// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// The synthesis boundary for the first hardware version is HDC_Accelerator only.
#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc>
#include "systemc_types.h"
#include "hdc_transactions.h"
#include "hdc_memory.h"

namespace hdc_systemc {

struct PipelineItem {
    AccelCommandKind kind;
    class_t class_id;
    QuantizedSample sample;
    hv_t encoded;
    hv_t ngram;
    bool valid_ngram;
};

struct DistanceResponse {
    bool valid_prediction;
    distance_counter_t distances[NUM_CLASSES];
};

SC_MODULE(HDC_Accelerator) {
public:
    sc_core::sc_fifo_in<AccelCommand> cmd_in;
    sc_core::sc_fifo_out<AccelResponse> rsp_out;

    SC_CTOR(HDC_Accelerator);

    void bind_memory(HDC_Memory *memory);

private:
    // Pipeline stage threads.
    void command_thread();
    void encoder_thread();
    void ngram_thread();
    void bundler_thread();
    void distance_thread();
    void forward_completed_distance_responses();

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

    PipelineItem m_encoder_in_data;
    bool m_encoder_in_valid;
    bool m_encoder_in_ready;

    PipelineItem m_encoder_out_data;
    bool m_encoder_out_valid;
    bool m_encoder_out_ready;

    PipelineItem m_bundler_in_data;
    bool m_bundler_in_valid;
    bool m_bundler_in_ready;

    PipelineItem m_distance_in_data;
    bool m_distance_in_valid;
    bool m_distance_in_ready;

    DistanceResponse m_distance_done_data;
    bool m_distance_done_valid;
    bool m_distance_done_ready;

    bool m_control_done_valid;

    hv_t m_ngram_buffer[N_GRAM_SIZE];
    int m_ngram_buffer_write_pos;
    int m_ngram_buffer_fill_count;

    // Signed bundling score for the currently trained class segment.
    train_score_t m_bundling_score[VECTOR_DIMENSION];
    train_counter_t m_current_class_count;
    int m_current_class_id;

    HDC_Memory *m_memory;
};

} // namespace hdc_systemc

#endif
