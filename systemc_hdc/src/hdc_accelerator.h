#ifndef SYSTEMC_HDC_HDC_ACCELERATOR_H
#define SYSTEMC_HDC_HDC_ACCELERATOR_H

#include <systemc>
#include "systemc_types.h"
#include "hdc_transactions.h"
#include "hdc_memory.h"

namespace hdc_systemc {

SC_MODULE(HDC_Accelerator) {
public:
    sc_core::sc_fifo_in<AccelCommand> cmd_in;
    sc_core::sc_fifo_out<AccelResponse> rsp_out;

    SC_CTOR(HDC_Accelerator);

    void bind_memory(HDC_Memory *memory);
    void reset_training_state();
    void push_training_sample(int class_id, const level_t *quantized_sample);
    void push_invalid_training_step();
    void reset_inference_state();
    bool push_inference_sample(const level_t *quantized_sample, distance_counter_t *distances);

private:
    void command_thread();
    void fill_inference_response(bool valid_prediction,
                                 const distance_counter_t *distances,
                                 AccelResponse &response) const;
    void encode_sample(const level_t *quantized_sample, hv_t &encoded_sample) const;
    void push_sample_to_ngram_buffer(const level_t *quantized_sample);
    void compute_hamming_distances(const hv_t &query, distance_counter_t *distances) const;
    void bind_ngram(hv_t &encoded) const;
    void add_ngram_to_bundling_buffer(const hv_t &encoded_ngram);
    void finalize_current_class();
    void reset_ngram_buffer();

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
