// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// Keep dataset loading, floating-point quantization, and testbench code outside this file.
#include "hdc_accelerator.h"

using namespace hdc_systemc;

namespace {

void clear_hv(hv_t &hv) {
    hv_clear(hv);
}

distance_counter_t hamming_distance_words(const hv_t &a, const hv_t &b) {
    distance_counter_t distance = 0;
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        const hv_word_t diff = a.words[word] ^ b.words[word];
        for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
            if (((diff >> bit) & hv_word_t(1)) != 0) {
                ++distance;
            }
        }
    }
    return distance;
}

hv_word_t encode_sample_word(const QuantizedSample &sample,
                             const hv_t cim[NUM_LEVELS][NUM_FEATURES],
                             unsigned word_index) {
    hv_word_t encoded_word = 0;
    const unsigned start_dim = word_index * HV_WORD_BITS;
    const feature_score_t signed_threshold =
        (NUM_FEATURES % 2 == 1) ? feature_score_t(-1) : feature_score_t(0);

    for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
        const unsigned d = start_dim + bit;
        feature_score_t score = 0;
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            const unsigned level = sample.levels[feature].to_uint();
            const hv_t &feature_hv = cim[level][feature];
            if (hv_get_bit(feature_hv, d)) {
                ++score;
            } else {
                --score;
            }
        }

        if (score >= signed_threshold) {
            encoded_word = encoded_word | (hv_word_t(1) << bit);
        }
    }

    return encoded_word;
}

hv_word_t permute_xor_word(const hv_t &input, const hv_t &rhs, unsigned word_index) {
    const unsigned prev_word = (word_index == 0) ? (HV_WORDS - 1u) : (word_index - 1u);
    hv_word_t rotated = input.words[word_index] << 1;
    const bool carry = ((input.words[prev_word] >> (HV_WORD_BITS - 1u)) & hv_word_t(1)) != 0;
    if (carry) {
        rotated = rotated | hv_word_t(1);
    } else {
        rotated = rotated & ~hv_word_t(1);
    }
    return rotated ^ rhs.words[word_index];
}

} // namespace

HDC_Accelerator::HDC_Accelerator(sc_core::sc_module_name name)
    : sc_module(name),
      clk("clk"),
      rst("rst"),
      cmd_valid("cmd_valid"),
      cmd_ready("cmd_ready"),
      cmd_kind("cmd_kind"),
      cmd_class_id("cmd_class_id"),
      rsp_valid("rsp_valid"),
      rsp_ready("rsp_ready"),
      rsp_valid_prediction("rsp_valid_prediction"),
      m_encoder_in_valid(false),
      m_encoder_out_valid(false),
      m_encoder_busy(false),
      m_encoder_word(0),
      m_bundler_in_valid(false),
      m_distance_in_valid(false),
      m_distance_done_valid(false),
      m_distance_busy(false),
      m_distance_class(0),
      m_control_done_valid(false),
      m_control_busy(false),
      m_ngram_bind_busy(false),
      m_ngram_bind_round(0),
      m_ngram_bind_word(0),
      m_ngram_oldest_slot(0),
      m_current_class_valid(false) {
    SC_CTHREAD(pipeline_fsm, clk.pos());
    reset_signal_is(rst, true);
}

// Simulation/pre-synthesis preload helper only.
// This is not a hardware runtime load interface; real deployment needs ROM
// initialization, generated constants, or a dedicated preload path later.
void HDC_Accelerator::set_cim(unsigned level, unsigned feature, const hv_t &value) {
    m_cim[level][feature] = value;
}

// Simulation/pre-synthesis preload helper only.
// This is not a hardware runtime load interface; real deployment needs ROM
// initialization, generated constants, or a dedicated preload path later.
void HDC_Accelerator::set_assoc_class(unsigned class_id, const hv_t &value) {
    m_assoc_mem[class_id] = value;
}

// Data commands are pipelined: TrainSample and InferSample are dispatched
// without waiting for completion. Control commands are blocking stream
// boundaries and wait until their token passes through the internal pipeline.
void HDC_Accelerator::pipeline_fsm() {
    {
        HLS_DEFINE_PROTOCOL("reset");

        reset_all_local_state();
        reset_output_ports();

        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("cycle");

            response_stage();
            distance_stage();
            train_stage();
            ngram_stage();
            encoder_stage();
            command_stage();

            wait();
        }
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

    const bool can_accept_command = !m_encoder_in_valid && !m_encoder_busy;
    cmd_ready.write(can_accept_command);

    if (!(cmd_valid.read() && can_accept_command)) {
        return;
    }

    AccelCommand command = {};
    command.kind = static_cast<AccelCommandKind>(cmd_kind.read().to_uint());
    command.class_id = cmd_class_id.read();
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        command.sample.levels[feature] = cmd_sample_levels[feature].read();
    }
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
    }

    m_encoder_in_data = packet;
    m_encoder_in_valid = true;
}

void HDC_Accelerator::response_stage() {
    const bool can_consume_response = rsp_ready.read();
    if (m_distance_done_valid) {
        rsp_valid.write(true);
        rsp_valid_prediction.write(m_distance_done_data.valid_prediction);
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            rsp_distances[class_id].write(m_distance_done_data.distances[class_id]);
        }
        if (can_consume_response) {
            m_distance_done_valid = false;
        }
    } else {
        rsp_valid.write(false);
        rsp_valid_prediction.write(false);
    }
}

void HDC_Accelerator::encoder_stage() {
    if (!m_encoder_busy) {
        if (!m_encoder_in_valid) {
            return;
        }
        if (m_encoder_out_valid) {
            return;
        }

        m_encoder_work = m_encoder_in_data;
        clear_hv(m_encoder_result);
        m_encoder_word = 0;
        m_encoder_busy = true;
        m_encoder_in_valid = false;
    }

    if (m_encoder_busy && m_encoder_out_valid) {
        return;
    }

    const bool should_encode =
        m_encoder_work.kind == AccelCommandKind::TrainSample ||
        m_encoder_work.kind == AccelCommandKind::InferSample;

    if (!should_encode) {
        EncoderPacket output = m_encoder_work;
        output.encoded = m_encoder_result;
        m_encoder_out_data = output;
        m_encoder_out_valid = true;
        m_encoder_busy = false;
        m_encoder_word = 0;
        return;
    }

    const unsigned word_index = m_encoder_word;
    const hv_word_t encoded_word =
        encode_sample_word(m_encoder_work.sample, m_cim, word_index);
    m_encoder_result.words[word_index] = encoded_word;

    if (word_index + 1u == HV_WORDS) {
        EncoderPacket output = m_encoder_work;
        output.encoded = m_encoder_result;
        output.encoded.words[word_index] = encoded_word;

        m_encoder_out_data = output;
        m_encoder_out_valid = true;

        m_encoder_busy = false;
        m_encoder_word = 0;
    } else {
        m_encoder_word = word_index + 1u;
    }
}

void HDC_Accelerator::ngram_stage() {
    if (!m_ngram_bind_busy) {
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
            const bool can_accept_distance = !m_distance_in_valid && !m_distance_busy;
            if ((to_bundler && !can_accept_bundler) || (!to_bundler && !can_accept_distance)) {
                return;
            }

            m_encoder_out_valid = false;
            push_encoded_sample_to_ngram_buffer(item.encoded);
            NGramPacket packet = {};
            packet.kind = item.kind;
            packet.class_id = item.class_id;
            if (m_ngram_buffer_fill_count != N_GRAM_SIZE) {
                packet.valid_ngram = false;
                if (to_bundler) {
                    m_bundler_in_data = packet;
                    m_bundler_in_valid = true;
                } else {
                    m_distance_in_data = packet;
                    m_distance_in_valid = true;
                }
                return;
            }

            m_ngram_oldest_slot = m_ngram_buffer_write_pos;
            m_ngram_work_packet = item;
            m_ngram_work = m_ngram_buffer[m_ngram_oldest_slot];
            clear_hv(m_ngram_next);
            m_ngram_bind_round = (N_GRAM_SIZE > 1) ? 1u : N_GRAM_SIZE;
            m_ngram_bind_word = 0;
            m_ngram_bind_busy = true;
        } else {
            m_encoder_out_valid = false;
            return;
        }
    }

    if (m_ngram_bind_round != N_GRAM_SIZE) {
        for (unsigned lane = 0; lane < NGRAM_WORDS_PER_CYCLE; ++lane) {
            const unsigned word = m_ngram_bind_word;
            const unsigned rhs_slot =
                (m_ngram_oldest_slot + m_ngram_bind_round) % N_GRAM_SIZE;

            m_ngram_next.words[word] =
                permute_xor_word(m_ngram_work, m_ngram_buffer[rhs_slot], word);

            if (word + 1u == HV_WORDS) {
                m_ngram_work = m_ngram_next;
                clear_hv(m_ngram_next);
                m_ngram_bind_word = 0;

                if (m_ngram_bind_round + 1u == N_GRAM_SIZE) {
                    m_ngram_bind_round = N_GRAM_SIZE;
                    break;
                } else {
                    ++m_ngram_bind_round;
                }
            } else {
                m_ngram_bind_word = word + 1u;
            }
        }
    }

    if (m_ngram_bind_round == N_GRAM_SIZE) {
        const bool to_bundler = m_ngram_work_packet.kind == AccelCommandKind::TrainSample;
        const bool can_emit_bundler = !m_bundler_in_valid;
        const bool can_emit_distance = !m_distance_in_valid && !m_distance_busy;
        if ((to_bundler && !can_emit_bundler) || (!to_bundler && !can_emit_distance)) {
            return;
        }

        NGramPacket packet = {};
        packet.kind = m_ngram_work_packet.kind;
        packet.class_id = m_ngram_work_packet.class_id;
        packet.ngram = m_ngram_work;
        packet.valid_ngram = true;

        if (to_bundler) {
            m_bundler_in_data = packet;
            m_bundler_in_valid = true;
        } else {
            m_distance_in_data = packet;
            m_distance_in_valid = true;
        }

        m_ngram_bind_busy = false;
        m_ngram_bind_round = 0;
        m_ngram_bind_word = 0;
    }
}

void HDC_Accelerator::train_stage() {
    if (!m_bundler_in_valid) {
        return;
    }

    const NGramPacket item = m_bundler_in_data;
    m_bundler_in_valid = false;

    if (item.kind == AccelCommandKind::TrainSample) {
            if (item.valid_ngram) {
                if (!m_current_class_valid) {
                    m_current_class_id = item.class_id;
                    m_current_class_valid = true;
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
    if (!m_distance_busy) {
        if (!m_distance_in_valid) {
            return;
        }
        if (m_distance_done_valid) {
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

        m_distance_work = item;
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            m_distance_acc[class_id] = 0;
        }
        m_distance_class = 0;
        m_distance_busy = true;
    }

    const unsigned class_id = m_distance_class;
    m_distance_acc[class_id] =
        hamming_distance_words(m_distance_work.ngram, m_assoc_mem[class_id]);

    if (class_id == (NUM_CLASSES - 1u)) {
        m_distance_done_data.valid_prediction = true;
        for (unsigned copy_class = 0; copy_class < NUM_CLASSES; ++copy_class) {
            m_distance_done_data.distances[copy_class] = m_distance_acc[copy_class];
        }
        m_distance_done_valid = true;
        m_distance_busy = false;
        m_distance_class = 0;
    } else {
        m_distance_class = class_id + 1u;
    }
}

void HDC_Accelerator::reset_output_ports() {
    cmd_ready.write(false);
    rsp_valid.write(false);
    rsp_valid_prediction.write(false);

    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        rsp_distances[class_id].write(0);
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
    m_encoder_in_data = EncoderPacket();
    m_encoder_out_data = EncoderPacket();
    m_encoder_busy = false;
    m_encoder_word = 0;
    m_encoder_work = EncoderPacket();
    clear_hv(m_encoder_result);
    m_bundler_in_data = NGramPacket();
    m_distance_in_data = NGramPacket();
    m_distance_done_data = DistancePacket();
    m_distance_busy = false;
    m_distance_class = 0;
    m_distance_work = NGramPacket();
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        m_distance_acc[class_id] = 0;
    }
    m_ngram_bind_busy = false;
    m_ngram_bind_round = 0;
    m_ngram_bind_word = 0;
    m_ngram_oldest_slot = 0;
    m_ngram_work_packet = EncoderPacket();
    clear_hv(m_ngram_work);
    clear_hv(m_ngram_next);
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

void HDC_Accelerator::add_ngram_to_bundling_buffer(const hv_t &encoded_ngram) {
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        if (hv_get_bit(encoded_ngram, d)) {
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
        hv_set_bit(class_vector, d, m_bundling_score[d] >= signed_threshold);
        m_bundling_score[d] = 0;
    }
    m_assoc_mem[m_current_class_id.to_uint()] = class_vector;

    m_current_class_count = 0;
    m_current_class_id = 0;
    m_current_class_valid = false;
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
