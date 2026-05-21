// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// The synthesis boundary for the first hardware version is HDC_Accelerator only.
#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <ostream>
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

inline std::ostream &operator<<(std::ostream &os, const PipelineItem &item) {
    return os << "PipelineItem{kind=" << item.kind
              << ", class_id=" << item.class_id.to_uint()
              << ", valid_ngram=" << item.valid_ngram << '}';
}

inline std::ostream &operator<<(std::ostream &os, const DistanceResponse &response) {
    os << "DistanceResponse{valid_prediction=" << response.valid_prediction
       << ", distances=[";
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        if (class_id > 0) {
            os << ',';
        }
        os << response.distances[class_id].to_uint();
    }
    return os << "]}";
}

SC_MODULE(HDC_Accelerator) {
public:
    sc_core::sc_fifo_in<AccelCommand> cmd_in;
    sc_core::sc_fifo_out<AccelResponse> rsp_out;

    SC_CTOR(HDC_Accelerator);

    void bind_memory(HDC_Memory *memory);
    void reset_stats();
    const AcceleratorStats &stats() const;

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

    // Internal pipeline FIFOs.
    sc_core::sc_fifo<PipelineItem> m_encoder_in_fifo;
    sc_core::sc_fifo<PipelineItem> m_encoder_out_fifo;
    sc_core::sc_fifo<PipelineItem> m_bundler_in_fifo;
    sc_core::sc_fifo<PipelineItem> m_distance_in_fifo;
    sc_core::sc_fifo<bool> m_control_done_fifo;
    sc_core::sc_fifo<DistanceResponse> m_distance_done_fifo;

    hv_t m_ngram_buffer[N_GRAM_SIZE];
    int m_ngram_buffer_write_pos;
    int m_ngram_buffer_fill_count;

    // Signed bundling score for the currently trained class segment.
    train_score_t m_bundling_score[VECTOR_DIMENSION];
    train_counter_t m_current_class_count;
    int m_current_class_id;

    AcceleratorStats m_stats;
    int m_infer_outstanding;
    HDC_Memory *m_memory;
};

} // namespace hdc_systemc

#endif
