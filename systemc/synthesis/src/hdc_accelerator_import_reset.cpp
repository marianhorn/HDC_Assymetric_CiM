// SYNTHESIS IMPORT DIAGNOSTIC ONLY.
// Tests whether Stratus accepts the real reset helpers and state-clearing loops.
#include "hdc_accelerator.h"

namespace {

void clear_hv(hv_t &hv) {
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        hv[d] = sc_dt::SC_LOGIC_0;
    }
}

} // namespace

void HDC_Accelerator::set_cim(unsigned level, unsigned feature, const hv_t &value) {
    m_cim[level][feature] = value;
}

void HDC_Accelerator::set_assoc_class(unsigned class_id, const hv_t &value) {
    m_assoc_mem[class_id] = value;
}

void HDC_Accelerator::pipeline_fsm() {
    reset_all_local_state();
    wait();

    while (true) {
        cmd_ready.write(true);
        rsp_valid.write(false);
        rsp_valid_prediction.write(false);
        rsp_distances.write(0);
        wait();
    }
}

void HDC_Accelerator::reset_all_local_state() {
    m_encoder_in_valid = false;
    m_encoder_out_valid = false;
    m_bundler_in_valid = false;
    m_distance_in_valid = false;
    m_distance_done_valid = false;
    m_control_done_valid = false;
    m_control_busy = false;
    cmd_ready.write(false);
    rsp_valid.write(false);
    rsp_valid_prediction.write(false);
    rsp_distances.write(0);
    m_encoder_in_data = EncoderPacket();
    m_encoder_out_data = EncoderPacket();
    m_bundler_in_data = NGramPacket();
    m_distance_in_data = NGramPacket();
    m_distance_done_data = DistancePacket();
    reset_ngram_buffer();
    reset_bundling_buffer_only();
}

void HDC_Accelerator::reset_bundling_buffer_only() {
    m_current_class_count = 0;
    m_current_class_id = 0;
    m_current_class_valid = false;
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        m_bundling_score[d] = 0;
    }
}

void HDC_Accelerator::reset_ngram_buffer() {
    m_ngram_buffer_write_pos = 0;
    m_ngram_buffer_fill_count = 0;
    for (unsigned slot = 0; slot < N_GRAM_SIZE; ++slot) {
        clear_hv(m_ngram_buffer[slot]);
    }
}
