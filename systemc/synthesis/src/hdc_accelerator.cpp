// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// Keep dataset loading, floating-point quantization, and testbench code outside this file.
#include "hdc_accelerator.h"

namespace hdc_systemc {

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

} // namespace

HDC_Accelerator::HDC_Accelerator(sc_core::sc_module_name name)
    : sc_module(name),
      clk("clk"),
      rst("rst"),
      cmd_valid("cmd_valid"),
      cmd_ready("cmd_ready"),
      cmd_data("cmd_data"),
      rsp_valid("rsp_valid"),
      rsp_ready("rsp_ready"),
      rsp_data("rsp_data"),
      m_encoder_in_valid(false),
      m_encoder_out_valid(false),
      m_bundler_in_valid(false),
      m_distance_in_valid(false),
      m_distance_done_valid(false),
      m_control_done_valid(false),
      m_control_busy(false) {
    SC_CTHREAD(pipeline_fsm, clk.pos());
    reset_signal_is(rst, true);
}

void HDC_Accelerator::set_cim(unsigned level, unsigned feature, const hv_t &value) {
    m_cim[level][feature] = value;
}

void HDC_Accelerator::set_assoc_class(unsigned class_id, const hv_t &value) {
    m_assoc_mem[class_id] = value;
}

// Data commands are pipelined: TrainSample and InferSample are dispatched
// without waiting for completion. Control commands are blocking stream
// boundaries and wait until their token passes through the internal pipeline.
void HDC_Accelerator::pipeline_fsm() {
    reset_all_local_state();
    wait();

    while (true) {
        response_stage();
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

    AccelCommand command = cmd_data.read();
    EncoderPacket packet = {};

    switch (command.kind) {
    case AccelCommandKind::ResetTraining:
            packet.kind = AccelCommandKind::ResetTraining;
            m_control_busy = true;
            break;

    case AccelCommandKind::ResetInference:
            packet.kind = AccelCommandKind::ResetInference;
            m_control_busy = true;
            break;

    case AccelCommandKind::TrainSample:
            packet.kind = AccelCommandKind::TrainSample;
            packet.class_id = command.class_id;
            packet.sample = command.sample;
            break;

    case AccelCommandKind::InvalidTrainingStep:
            // InvalidTrainingStep is a flush token for the current training class segment.
            // Since it uses the same FIFO path as samples, previous samples are bundled first.
            packet.kind = AccelCommandKind::InvalidTrainingStep;
            m_control_busy = true;
            break;

    case AccelCommandKind::InferSample:
            packet.kind = AccelCommandKind::InferSample;
            packet.class_id = 0;
            packet.sample = command.sample;
            break;

    case AccelCommandKind::Shutdown:
            return;
    }

    m_encoder_in_data = packet;
    m_encoder_in_valid = true;
}

void HDC_Accelerator::response_stage() {
    const bool can_consume_response = rsp_ready.read();
    if (m_distance_done_valid) {
        AccelResponse response = {};
        response.valid_prediction = m_distance_done_data.valid_prediction;
        response.is_shutdown_ack = false;
        response.predicted_class = m_distance_done_data.predicted_class;
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            response.distances[class_id] = m_distance_done_data.distances[class_id];
        }

        rsp_valid.write(true);
        rsp_data.write(response);
        if (can_consume_response) {
            m_distance_done_valid = false;
        }
    } else {
        rsp_valid.write(false);
    }
}

void HDC_Accelerator::encoder_stage() {
    const bool can_encode = m_encoder_in_valid && !m_encoder_out_valid;
    if (can_encode) {
        EncoderPacket item = m_encoder_in_data;
        m_encoder_in_valid = false;
        if (item.kind == AccelCommandKind::TrainSample || item.kind == AccelCommandKind::InferSample) {
            encode_sample(item.sample, item.encoded);
        }
        m_encoder_out_data = item;
        m_encoder_out_valid = true;
    }
}

void HDC_Accelerator::ngram_stage() {
    if (!m_encoder_out_valid) {
        return;
    }

    EncoderPacket item = m_encoder_out_data;

    if (item.kind == AccelCommandKind::ResetTraining) {
            m_encoder_out_valid = false;
            reset_ngram_buffer();
            reset_bundling_buffer_only();
            m_control_done_valid = true;
            return;
    }

    if (item.kind == AccelCommandKind::ResetInference) {
            m_encoder_out_valid = false;
            reset_ngram_buffer();
            m_control_done_valid = true;
            return;
    }

    if (item.kind == AccelCommandKind::InvalidTrainingStep) {
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

    if (item.kind == AccelCommandKind::TrainSample || item.kind == AccelCommandKind::InferSample) {
            const bool to_bundler = item.kind == AccelCommandKind::TrainSample;
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

void HDC_Accelerator::train_stage() {
    if (!m_bundler_in_valid) {
        return;
    }

    const NGramPacket item = m_bundler_in_data;
    m_bundler_in_valid = false;

    if (item.kind == AccelCommandKind::TrainSample) {
            if (item.valid_ngram) {
                const unsigned class_id = item.class_id.to_uint();

                if (m_current_class_id < 0) {
                    m_current_class_id = static_cast<int>(class_id);
                }

                add_ngram_to_bundling_buffer(item.ngram);
            }
            return;
    }

    if (item.kind == AccelCommandKind::InvalidTrainingStep) {
            finalize_current_class();
            reset_bundling_buffer_only();
            m_control_done_valid = true;
            return;
    }
}

void HDC_Accelerator::distance_stage() {
    const bool can_accept_distance = !m_distance_done_valid;
    if (!m_distance_in_valid || !can_accept_distance) {
        return;
    }

    const NGramPacket item = m_distance_in_data;
    m_distance_in_valid = false;

    DistancePacket response = {};
    if (!item.valid_ngram) {
            response.valid_prediction = false;
            response.predicted_class = 0;
            for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
                response.distances[class_id] = 0;
            }
            m_distance_done_data = response;
            m_distance_done_valid = true;
            return;
    }

    response.valid_prediction = true;
    response.predicted_class = 0;
    compute_hamming_distances(item.ngram, response.distances);
    m_distance_done_data = response;
    m_distance_done_valid = true;
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
    m_current_class_id = -1;
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
    if (m_current_class_id < 0) {
        m_current_class_count = 0;
        return;
    }

    hv_t class_vector;
    clear_hv(class_vector);
    // Exact equivalent of the previous rule:
    //     ones >= floor(half of m_current_class_count)
    //
    // Signed score:
    //     score = ones - zeros = 2 * ones - m_current_class_count
    //
    // Therefore:
    //     even count: score >= 0
    //     odd count:  score >= -1
    //
    // This avoids division while preserving the old bundling result.
    const bool odd_count = (m_current_class_count.to_uint() & 1u) != 0u;
    const train_score_t signed_threshold = odd_count ? train_score_t(-1) : train_score_t(0);
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        set_bit(class_vector, d, m_bundling_score[d] >= signed_threshold);
        m_bundling_score[d] = 0;
    }
    m_assoc_mem[static_cast<unsigned>(m_current_class_id)] = class_vector;

    m_current_class_count = 0;
    m_current_class_id = -1;
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
        const unsigned source_index = (d + VECTOR_DIMENSION - 1u) % VECTOR_DIMENSION;
        const bool bit = get_bit(input, source_index) ^ get_bit(rhs, d);
        set_bit(output, d, bit);
    }
}

void HDC_Accelerator::push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample) {
    m_ngram_buffer[m_ngram_buffer_write_pos] = encoded_sample;
    m_ngram_buffer_write_pos = (m_ngram_buffer_write_pos + 1) % N_GRAM_SIZE;
    if (m_ngram_buffer_fill_count < N_GRAM_SIZE) {
        ++m_ngram_buffer_fill_count;
    }
}

void HDC_Accelerator::encode_sample(const QuantizedSample &sample, hv_t &encoded_sample) {
    // Feature bundling uses signed +1/-1 accumulation instead of counting ones.
    // This is exactly equivalent to:
    //     ones >= floor(half of NUM_FEATURES)
    //
    // For NUM_FEATURES even: score >= 0
    // For NUM_FEATURES odd:  score >= -1
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

void HDC_Accelerator::compute_hamming_distances(const hv_t &query, distance_counter_t *distances) {
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        const hv_t &class_vector = m_assoc_mem[class_id];
        distance_counter_t distance = 0;
        for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(query, d) != get_bit(class_vector, d)) {
                ++distance;
            }
        }
        distances[class_id] = distance;
    }
}

} // namespace hdc_systemc
