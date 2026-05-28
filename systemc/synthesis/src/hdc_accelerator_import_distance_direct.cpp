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
        distance_stage();
        train_stage();
        ngram_stage();
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

void HDC_Accelerator::ngram_stage() {
    if (!m_encoder_out_valid) {
        return;
    }

    EncoderPacket item = m_encoder_out_data;

    if (item.kind == ResetTraining) {
        m_encoder_out_valid = false;
        reset_ngram_buffer();
        reset_bundling_buffer_only();
        m_control_done_valid = true;
        return;
    }

    if (item.kind == ResetInference) {
        m_encoder_out_valid = false;
        reset_ngram_buffer();
        m_control_done_valid = true;
        return;
    }

    if (item.kind == InvalidTrainingStep) {
        const bool can_accept_bundler = !m_bundler_in_valid;
        if (!can_accept_bundler) {
            return;
        }
        m_encoder_out_valid = false;
        reset_ngram_buffer();
        NGramPacket packet = {};
        packet.kind = item.kind;
        packet.class_id = item.class_id;
        packet.valid_ngram = false;
        m_bundler_in_data = packet;
        m_bundler_in_valid = true;
        return;
    }

    if (item.kind == TrainSample || item.kind == InferSample) {
        const bool to_bundler = item.kind == TrainSample;
        const bool can_accept_bundler = !m_bundler_in_valid;
        const bool can_accept_distance = !m_distance_in_valid;
        if ((to_bundler && !can_accept_bundler) || (!to_bundler && !can_accept_distance)) {
            return;
        }

        m_encoder_out_valid = false;
        push_encoded_sample_to_ngram_buffer(item.encoded);
        NGramPacket packet = {};
        packet.kind = item.kind;
        packet.class_id = item.class_id;
        if (m_ngram_buffer_fill_count == N_GRAM_SIZE) {
            bind_ngram(packet.ngram);
            packet.valid_ngram = true;
        } else {
            packet.valid_ngram = false;
        }

        if (to_bundler) {
            m_bundler_in_data = packet;
            m_bundler_in_valid = true;
        } else {
            m_distance_in_data = packet;
            m_distance_in_valid = true;
        }
        return;
    }

    m_encoder_out_valid = false;
}

void HDC_Accelerator::push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample) {
    m_ngram_buffer[m_ngram_buffer_write_pos] = encoded_sample;
    ++m_ngram_buffer_write_pos;
    if (m_ngram_buffer_write_pos == N_GRAM_SIZE) {
        m_ngram_buffer_write_pos = 0;
    }
    if (m_ngram_buffer_fill_count < N_GRAM_SIZE) {
        ++m_ngram_buffer_fill_count;
    }
}

void HDC_Accelerator::bind_ngram(hv_t &encoded_ngram) {
    const unsigned oldest_slot = m_ngram_buffer_write_pos;
    encoded_ngram = m_ngram_buffer[oldest_slot];
    hv_t next_encoded;

    for (unsigned i = 1; i < N_GRAM_SIZE; ++i) {
        const unsigned slot = (oldest_slot + i) % N_GRAM_SIZE;
        permute_xor(encoded_ngram, m_ngram_buffer[slot], next_encoded);
        encoded_ngram = next_encoded;
    }
}

void HDC_Accelerator::permute_xor(const hv_t &input, const hv_t &rhs, hv_t &output) {
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        const unsigned source_index = (d == 0) ? (VECTOR_DIMENSION - 1u) : (d - 1u);
        const bool bit = get_bit(input, source_index) ^ get_bit(rhs, d);
        set_bit(output, d, bit);
    }
}

void HDC_Accelerator::train_stage() {
    if (!m_bundler_in_valid) {
        return;
    }

    const NGramPacket item = m_bundler_in_data;
    m_bundler_in_valid = false;

    if (item.kind == TrainSample) {
        if (item.valid_ngram) {
            if (!m_current_class_valid) {
                m_current_class_id = item.class_id;
                m_current_class_valid = true;
            }

            add_ngram_to_bundling_buffer(item.ngram);
        }
        return;
    }

    if (item.kind == InvalidTrainingStep) {
        finalize_current_class();
        reset_bundling_buffer_only();
        m_control_done_valid = true;
        return;
    }
}

void HDC_Accelerator::add_ngram_to_bundling_buffer(const hv_t &encoded_ngram) {
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        if (get_bit(encoded_ngram, d)) {
            ++m_bundling_score[d];
        } else {
            --m_bundling_score[d];
        }
    }
    ++m_current_class_count;
}

void HDC_Accelerator::finalize_current_class() {
    if (!m_current_class_valid) {
        m_current_class_count = 0;
        return;
    }

    hv_t class_vector;
    clear_hv(class_vector);
    const bool odd_count = (m_current_class_count.to_uint() & 1u) != 0u;
    const train_score_t signed_threshold = odd_count ? train_score_t(-1) : train_score_t(0);
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        set_bit(class_vector, d, m_bundling_score[d] >= signed_threshold);
        m_bundling_score[d] = 0;
    }
    m_assoc_mem[m_current_class_id.to_uint()] = class_vector;

    m_current_class_count = 0;
    m_current_class_id = 0;
    m_current_class_valid = false;
}

void HDC_Accelerator::distance_stage() {
    if (!m_distance_in_valid || m_distance_done_valid) {
        return;
    }

    const NGramPacket item = m_distance_in_data;
    m_distance_in_valid = false;

    if (!item.valid_ngram) {
        m_distance_done_data.valid_prediction = false;
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            m_distance_done_data.distances[class_id] = 0;
        }
        m_distance_done_valid = true;
        return;
    }

    m_distance_done_data.valid_prediction = true;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        const hv_t &class_vector = m_assoc_mem[class_id];
        distance_counter_t distance = 0;
        for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(item.ngram, d) != get_bit(class_vector, d)) {
                ++distance;
            }
        }
        m_distance_done_data.distances[class_id] = distance;
    }
    m_distance_done_valid = true;
}
