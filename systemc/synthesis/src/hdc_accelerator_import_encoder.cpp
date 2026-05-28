// SYNTHESIS IMPORT DIAGNOSTIC ONLY.
// Tests reset + command + encoder_stage() and encode_sample() with CiM reads.
#include "hdc_accelerator.h"

namespace {

bool get_bit(const hv_t &hv, int index) {
    return hv[index].to_bool();
}

void set_bit(hv_t &hv, int index, bool value) {
    hv[index] = value ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;
}

void clear_hv(hv_t &hv) {
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        hv[d] = sc_dt::SC_LOGIC_0;
    }
}

level_t get_packed_level(const sample_levels_packed_t &levels, unsigned feature) {
    level_t value = 0;
    const unsigned base = feature * LEVEL_BITS;
    for (unsigned bit = 0; bit < LEVEL_BITS; ++bit) {
        value[bit] = levels[base + bit].to_bool();
    }
    return value;
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
        encoder_stage();
        command_stage();
        wait();
    }
}

void HDC_Accelerator::command_stage() {
    if (m_control_busy) {
        if (m_control_done_valid) {
            m_control_done_valid = false;
            m_control_busy = false;
        }
        cmd_ready.write(false);
        return;
    }

    const bool can_accept_command = !m_encoder_in_valid;
    cmd_ready.write(can_accept_command);

    if (!(cmd_valid.read() && can_accept_command)) {
        return;
    }

    AccelCommand command = {};
    command.kind = static_cast<AccelCommandKind>(cmd_kind.read().to_uint());
    command.class_id = cmd_class_id.read();
    const sample_levels_packed_t packed_levels = cmd_sample_levels.read();
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        command.sample.levels[feature] = get_packed_level(packed_levels, feature);
    }
    EncoderPacket packet = {};

    switch (command.kind) {
    case ResetTraining:
        packet.kind = ResetTraining;
        m_control_busy = true;
        break;

    case ResetInference:
        packet.kind = ResetInference;
        m_control_busy = true;
        break;

    case TrainSample:
        packet.kind = TrainSample;
        packet.class_id = command.class_id;
        packet.sample = command.sample;
        break;

    case InvalidTrainingStep:
        packet.kind = InvalidTrainingStep;
        m_control_busy = true;
        break;

    case InferSample:
        packet.kind = InferSample;
        packet.class_id = 0;
        packet.sample = command.sample;
        break;
    }

    m_encoder_in_data = packet;
    m_encoder_in_valid = true;
}

void HDC_Accelerator::encoder_stage() {
    const bool can_encode = m_encoder_in_valid && !m_encoder_out_valid;
    if (can_encode) {
        EncoderPacket item = m_encoder_in_data;
        m_encoder_in_valid = false;
        if (item.kind == TrainSample || item.kind == InferSample) {
            encode_sample(item.sample, item.encoded);
        }
        m_encoder_out_data = item;
        m_encoder_out_valid = true;
    }
}

void HDC_Accelerator::encode_sample(const QuantizedSample &sample, hv_t &encoded_sample) {
    const feature_score_t signed_threshold =
        (NUM_FEATURES % 2 == 1) ? feature_score_t(-1) : feature_score_t(0);

    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        feature_score_t score = 0;
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            const hv_t &feature_hv = m_cim[sample.levels[feature].to_uint()][feature];
            if (get_bit(feature_hv, d)) {
                ++score;
            } else {
                --score;
            }
        }
        set_bit(encoded_sample, d, score >= signed_threshold);
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
