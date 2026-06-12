// SYNTHESIS TARGET: This module is intended to be refactored toward HLS/SystemC synthesis.
// Keep dataset loading, floating-point quantization, and testbench code outside this file.
#include "hdc_accelerator.h"

using namespace hdc_systemc;

namespace {

void clear_hv(hv_t &hv) {
    hv_clear(hv);
}

template <typename T>
T p2p_get(hdc_p2p<T> &channel) {
    return channel.get();
}

template <typename T>
void p2p_put(hdc_p2p<T> &channel, const T &value) {
    channel.put(value);
}

bool is_control_command(AccelCommandKind kind) {
    return kind == AccelCommandKind::ResetTraining ||
           kind == AccelCommandKind::ResetInference ||
           kind == AccelCommandKind::InvalidTrainingStep;
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
      encoder_in("encoder_in"),
      encoder_out("encoder_out"),
      bundler_in("bundler_in"),
      distance_in("distance_in"),
      distance_done("distance_done"),
      control_done("control_done"),
      m_current_class_valid(false) {
    encoder_in.clk_rst(clk, rst);
    encoder_out.clk_rst(clk, rst);
    bundler_in.clk_rst(clk, rst);
    distance_in.clk_rst(clk, rst);
    distance_done.clk_rst(clk, rst);
    control_done.clk_rst(clk, rst);

    SC_CTHREAD(command_thread, clk.pos());
    reset_signal_is(rst, true);

    SC_CTHREAD(encoder_thread, clk.pos());
    reset_signal_is(rst, true);

    SC_CTHREAD(ngram_thread, clk.pos());
    reset_signal_is(rst, true);

    SC_CTHREAD(train_thread, clk.pos());
    reset_signal_is(rst, true);

    SC_CTHREAD(distance_thread, clk.pos());
    reset_signal_is(rst, true);

    SC_CTHREAD(response_thread, clk.pos());
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

void HDC_Accelerator::command_thread() {
    EncoderPacket pending_packet = {};
    bool pending_command = false;
    bool pending_control = false;

    {
        HLS_DEFINE_PROTOCOL("command_reset");
        cmd_ready.write(false);
        wait();
    }

    while (true) {
        if (pending_command) {
            {
                HLS_DEFINE_PROTOCOL("command_send");
                cmd_ready.write(false);
            }

            p2p_put(encoder_in, pending_packet);
            pending_command = false;

            if (pending_control) {
                const bool done = p2p_get(control_done);
                (void)done;
                pending_control = false;
            }

            {
                HLS_DEFINE_PROTOCOL("command_after_send");
                wait();
            }
            continue;
        }

        {
            HLS_DEFINE_PROTOCOL("command_cycle");
            cmd_ready.write(true);

            if (cmd_valid.read()) {
                AccelCommand command = {};
                command.kind = static_cast<AccelCommandKind>(cmd_kind.read().to_uint());
                command.class_id = cmd_class_id.read();
                for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
                    command.sample.levels[feature] = cmd_sample_levels[feature].read();
                }

                EncoderPacket packet = {};
                packet.kind = command.kind;
                packet.class_id = command.class_id;
                packet.sample = command.sample;
                clear_hv(packet.encoded);

                if (command.kind == AccelCommandKind::InferSample) {
                    packet.class_id = 0;
                }

                pending_packet = packet;
                pending_command = true;
                pending_control = is_control_command(command.kind);
            }

            wait();
        }
    }
}

void HDC_Accelerator::encoder_thread() {
    {
        HLS_DEFINE_PROTOCOL("encoder_reset");
        wait();
    }

    while (true) {
        EncoderPacket input = p2p_get(encoder_in);
        EncoderPacket output = input;
        clear_hv(output.encoded);

        const bool should_encode =
            input.kind == AccelCommandKind::TrainSample ||
            input.kind == AccelCommandKind::InferSample;

        if (should_encode) {
            for (unsigned word = 0; word < HV_WORDS; ++word) {
                {
                    HLS_DEFINE_PROTOCOL("encoder_word");
                    const hv_word_t encoded_word = encode_sample_word(input.sample, m_cim, word);
                    output.encoded.words[word] = encoded_word;
                    wait();
                }
            }
        } else {
            {
                HLS_DEFINE_PROTOCOL("encoder_control");
                wait();
            }
        }

        p2p_put(encoder_out, output);
    }
}

void HDC_Accelerator::ngram_thread() {
    {
        HLS_DEFINE_PROTOCOL("ngram_reset");
        reset_ngram_buffer();
        wait();
    }

    while (true) {
        EncoderPacket item = p2p_get(encoder_out);

        if (item.kind == AccelCommandKind::ResetTraining) {
            NGramPacket packet = {};
            packet.kind = item.kind;
            packet.class_id = item.class_id;
            packet.valid_ngram = false;
            clear_hv(packet.ngram);

            {
                HLS_DEFINE_PROTOCOL("ngram_reset_training");
                reset_ngram_buffer();
                wait();
            }
            p2p_put(bundler_in, packet);
            continue;
        }

        if (item.kind == AccelCommandKind::ResetInference) {
            {
                HLS_DEFINE_PROTOCOL("ngram_reset_inference");
                reset_ngram_buffer();
                wait();
            }
            p2p_put(control_done, true);
            continue;
        }

        if (item.kind == AccelCommandKind::InvalidTrainingStep) {
            NGramPacket packet = {};
            packet.kind = item.kind;
            packet.class_id = item.class_id;
            packet.valid_ngram = false;
            clear_hv(packet.ngram);

            {
                HLS_DEFINE_PROTOCOL("ngram_invalid_training");
                reset_ngram_buffer();
                wait();
            }
            p2p_put(bundler_in, packet);
            continue;
        }

        if (item.kind != AccelCommandKind::TrainSample && item.kind != AccelCommandKind::InferSample) {
            {
                HLS_DEFINE_PROTOCOL("ngram_unknown");
                wait();
            }
            continue;
        }

        const bool to_bundler = item.kind == AccelCommandKind::TrainSample;

        {
            HLS_DEFINE_PROTOCOL("ngram_push_sample");
            push_encoded_sample_to_ngram_buffer(item.encoded);
            wait();
        }

        NGramPacket packet = {};
        packet.kind = item.kind;
        packet.class_id = item.class_id;
        packet.valid_ngram = false;
        clear_hv(packet.ngram);

        if (m_ngram_buffer_fill_count != N_GRAM_SIZE) {
            if (to_bundler) {
                p2p_put(bundler_in, packet);
            } else {
                p2p_put(distance_in, packet);
            }
            continue;
        }

        const unsigned oldest_slot = m_ngram_buffer_write_pos;
        hv_t work = m_ngram_buffer[oldest_slot];
        hv_t next;
        clear_hv(next);

        for (unsigned round = 1; round < N_GRAM_SIZE; ++round) {
            const unsigned rhs_slot = (oldest_slot + round) % N_GRAM_SIZE;
            for (unsigned word = 0; word < HV_WORDS; ++word) {
                {
                    HLS_DEFINE_PROTOCOL("ngram_bind_word");
                    next.words[word] = permute_xor_word(work, m_ngram_buffer[rhs_slot], word);
                    wait();
                }
            }
            work = next;
            clear_hv(next);
        }

        packet.ngram = work;
        packet.valid_ngram = true;

        if (to_bundler) {
            p2p_put(bundler_in, packet);
        } else {
            p2p_put(distance_in, packet);
        }
    }
}

void HDC_Accelerator::train_thread() {
    {
        HLS_DEFINE_PROTOCOL("train_reset");
        reset_bundling_buffer_only();
        wait();
    }

    while (true) {
        NGramPacket item = p2p_get(bundler_in);

        if (item.kind == AccelCommandKind::TrainSample) {
            {
                HLS_DEFINE_PROTOCOL("train_sample");
                if (item.valid_ngram) {
                    if (!m_current_class_valid) {
                        m_current_class_id = item.class_id;
                        m_current_class_valid = true;
                    }
                    add_ngram_to_bundling_buffer(item.ngram);
                }
                wait();
            }
            continue;
        }

        if (item.kind == AccelCommandKind::InvalidTrainingStep) {
            {
                HLS_DEFINE_PROTOCOL("train_finalize");
                finalize_current_class();
                reset_bundling_buffer_only();
                wait();
            }
            p2p_put(control_done, true);
            continue;
        }

        if (item.kind == AccelCommandKind::ResetTraining) {
            {
                HLS_DEFINE_PROTOCOL("train_reset_training");
                reset_bundling_buffer_only();
                wait();
            }
            p2p_put(control_done, true);
            continue;
        }

        {
            HLS_DEFINE_PROTOCOL("train_unknown");
            wait();
        }
    }
}

void HDC_Accelerator::distance_thread() {
    {
        HLS_DEFINE_PROTOCOL("distance_reset");
        wait();
    }

    while (true) {
        NGramPacket item = p2p_get(distance_in);
        DistancePacket result = {};

        if (!item.valid_ngram) {
            {
                HLS_DEFINE_PROTOCOL("distance_invalid");
                result.valid_prediction = false;
                for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
                    result.distances[class_id] = 0;
                }
                wait();
            }
            p2p_put(distance_done, result);
            continue;
        }

        result.valid_prediction = true;
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            {
                HLS_DEFINE_PROTOCOL("distance_class");
                result.distances[class_id] =
                    hamming_distance_words(item.ngram, m_assoc_mem[class_id]);
                wait();
            }
        }

        p2p_put(distance_done, result);
    }
}

void HDC_Accelerator::response_thread() {
    {
        HLS_DEFINE_PROTOCOL("response_reset");
        reset_response_ports();
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("response_idle");
            rsp_valid.write(false);
            rsp_valid_prediction.write(false);
            wait();
        }

        DistancePacket item = p2p_get(distance_done);
        bool consumed = false;
        while (!consumed) {
            {
                HLS_DEFINE_PROTOCOL("response_cycle");
                rsp_valid.write(true);
                rsp_valid_prediction.write(item.valid_prediction);
                for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
                    rsp_distances[class_id].write(item.distances[class_id]);
                }
                consumed = rsp_ready.read();
                wait();
            }
        }
    }
}

void HDC_Accelerator::reset_response_ports() {
    rsp_valid.write(false);
    rsp_valid_prediction.write(false);

    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        rsp_distances[class_id].write(0);
    }
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
