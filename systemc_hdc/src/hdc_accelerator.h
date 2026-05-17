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

private:
    void command_thread();
    void encoder_thread();
    void ngram_thread();
    void bundler_thread();
    void distance_thread();
    void encode_sample_parallel(const QuantizedSample &sample, hv_t &encoded_sample);
    void encoder_pe_thread(unsigned pe_id);
    void push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample);
    void compute_hamming_distances_parallel(const hv_t &query, distance_counter_t *distances);
    void distance_class_pe_thread(unsigned class_id);
    void ngram_pe_thread(unsigned pe_id);
    void bind_ngram_parallel(hv_t &encoded_ngram);
    void permute_xor_parallel(const hv_t &input, const hv_t &rhs, hv_t &output);
    void add_ngram_to_bundling_buffer(const hv_t &encoded_ngram);
    void reset_training_state_local();
    void reset_bundling_buffer_only();
    void finalize_current_class();
    void reset_ngram_buffer();

    sc_core::sc_fifo<PipelineItem> m_encoder_in_fifo;
    sc_core::sc_fifo<PipelineItem> m_encoder_out_fifo;
    sc_core::sc_fifo<PipelineItem> m_bundler_in_fifo;
    sc_core::sc_fifo<PipelineItem> m_distance_in_fifo;
    sc_core::sc_fifo<bool> m_control_done_fifo;
    sc_core::sc_fifo<DistanceResponse> m_distance_done_fifo;
    sc_core::sc_event m_encode_start_event;
    sc_core::sc_event m_encode_done_event[ENCODER_PES];
    QuantizedSample m_encode_current_sample;
    hv_t m_encode_current_output;
    bool m_encode_done_flags[ENCODER_PES];
    sc_core::sc_event m_distance_start_event;
    sc_core::sc_event m_distance_done_event[NUM_CLASSES];
    hv_t m_distance_current_query;
    distance_counter_t m_distance_current_result[NUM_CLASSES];
    bool m_distance_done_flags[NUM_CLASSES];
    sc_core::sc_event m_ngram_start_event;
    sc_core::sc_event m_ngram_done_event[NGRAM_PES];
    hv_t m_ngram_current_output;
    const hv_t *m_ngram_work_input;
    const hv_t *m_ngram_work_rhs;
    hv_t *m_ngram_work_output;
    bool m_ngram_done_flags[NGRAM_PES];
    HDC_Memory *m_memory;
    hv_t m_ngram_buffer[N_GRAM_SIZE];
    int m_ngram_buffer_write_pos;
    int m_ngram_buffer_fill_count;
    train_counter_t m_bundling_buffer[VECTOR_DIMENSION];
    train_counter_t m_current_class_count;
    int m_current_class_id;
};

} // namespace hdc_systemc

#endif
