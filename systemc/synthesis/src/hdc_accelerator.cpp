// SYNTHESIS TARGET: This module is intended for HLS/SystemC synthesis.
// Keep dataset loading, floating-point quantization, and testbench code outside this file.
#include "hdc_accelerator.h"

using namespace hdc_systemc;

namespace {

static constexpr unsigned OUTPUT_NONE = 0;
static constexpr unsigned OUTPUT_BUNDLER = 1;
static constexpr unsigned OUTPUT_DISTANCE = 2;

void clear_hv(hv_t &hv) {
    hv_clear(hv);
}

bool is_train_control_command(AccelCommandKind kind) {
    return kind == AccelCommandKind::ResetTraining ||
           kind == AccelCommandKind::InvalidTrainingStep;
}

bool is_ngram_control_command(AccelCommandKind kind) {
    return kind == AccelCommandKind::ResetInference;
}

distance_counter_t hamming_distance_words(const hv_t &a, const hv_t &b) {
    distance_counter_t distance = 0;
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "hamming-words-loop");
        const hv_word_t diff = a.words[word] ^ b.words[word];
        for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
            HLS_UNROLL_LOOP(OFF, "hamming-bits-loop");
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
        HLS_UNROLL_LOOP(OFF, "encode-word-bits-loop");
        const unsigned d = start_dim + bit;
        feature_score_t score = 0;
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            HLS_UNROLL_LOOP(OFF, "encode-features-loop");
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
      m_encoder_in_data("encoder_in_data"),
      m_encoder_in_valid("encoder_in_valid"),
      m_encoder_in_ready("encoder_in_ready"),
      m_encoder_out_data("encoder_out_data"),
      m_encoder_out_valid("encoder_out_valid"),
      m_encoder_out_ready("encoder_out_ready"),
      m_bundler_in_data("bundler_in_data"),
      m_bundler_in_valid("bundler_in_valid"),
      m_bundler_in_ready("bundler_in_ready"),
      m_distance_in_data("distance_in_data"),
      m_distance_in_valid("distance_in_valid"),
      m_distance_in_ready("distance_in_ready"),
      m_distance_done_data("distance_done_data"),
      m_distance_done_valid("distance_done_valid"),
      m_distance_done_ready("distance_done_ready"),
      m_ngram_control_done_valid("ngram_control_done_valid"),
      m_ngram_control_done_ready("ngram_control_done_ready"),
      m_train_control_done_valid("train_control_done_valid"),
      m_train_control_done_ready("train_control_done_ready"),
      m_current_class_valid(false) {
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

// Data commands are pipelined: TrainSample and InferSample are dispatched
// without waiting for completion. Control commands are blocking stream
// boundaries and wait until their token passes through the internal pipeline.
void HDC_Accelerator::command_thread() {
    EncoderPacket output_packet = EncoderPacket();
    bool output_valid = false;
    bool output_presented = false;
    bool wait_ngram_control = false;
    bool wait_train_control = false;
    bool release_ngram_control = false;
    bool release_train_control = false;

    {
        HLS_DEFINE_PROTOCOL("command_reset");
        cmd_ready.write(false);
        m_encoder_in_data.write(EncoderPacket());
        m_encoder_in_valid.write(false);
        m_ngram_control_done_ready.write(false);
        m_train_control_done_ready.write(false);
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("command_cycle");

            if (output_valid && !output_presented) {
                output_presented = true;
            } else if (output_valid && m_encoder_in_ready.read()) {
                output_valid = false;
                output_presented = false;
            }

            if (release_ngram_control) {
                wait_ngram_control = false;
                release_ngram_control = false;
            } else if (wait_ngram_control && m_ngram_control_done_valid.read()) {
                release_ngram_control = true;
            }

            if (release_train_control) {
                wait_train_control = false;
                release_train_control = false;
            } else if (wait_train_control && m_train_control_done_valid.read()) {
                release_train_control = true;
            }

            const bool can_accept_command =
                !output_valid && !wait_ngram_control && !wait_train_control;
            cmd_ready.write(can_accept_command);
            m_ngram_control_done_ready.write(wait_ngram_control);
            m_train_control_done_ready.write(wait_train_control);

            if (cmd_valid.read() && can_accept_command) {
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

                output_packet = packet;
                output_valid = true;
                output_presented = false;

                if (is_ngram_control_command(command.kind)) {
                    wait_ngram_control = true;
                }
                if (is_train_control_command(command.kind)) {
                    wait_train_control = true;
                }
            }

            m_encoder_in_data.write(output_packet);
            m_encoder_in_valid.write(output_valid);
            wait();
        }
    }
}

void HDC_Accelerator::encoder_thread() {
    EncoderPacket work = EncoderPacket();
    EncoderPacket output_packet = EncoderPacket();
    hv_t encoder_result;
    bool busy = false;
    bool output_valid = false;
    bool output_presented = false;
    unsigned word_index = 0;

    {
        HLS_DEFINE_PROTOCOL("encoder_reset");
        clear_hv(encoder_result);
        clear_hv(output_packet.encoded);
        m_encoder_in_ready.write(false);
        m_encoder_out_data.write(EncoderPacket());
        m_encoder_out_valid.write(false);
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("encoder_cycle");

            if (output_valid && !output_presented) {
                output_presented = true;
            } else if (output_valid && m_encoder_out_ready.read()) {
                output_valid = false;
                output_presented = false;
            }

            const bool can_accept_input = !busy && !output_valid;
            m_encoder_in_ready.write(can_accept_input);

            if (can_accept_input && m_encoder_in_valid.read()) {
                work = m_encoder_in_data.read();
                clear_hv(encoder_result);
                word_index = 0;
                busy = true;
            }

            if (busy && !output_valid) {
                const bool should_encode =
                    work.kind == AccelCommandKind::TrainSample ||
                    work.kind == AccelCommandKind::InferSample;

                if (!should_encode) {
                    output_packet = work;
                    output_packet.encoded = encoder_result;
                    output_valid = true;
                    output_presented = false;
                    busy = false;
                    word_index = 0;
                } else {
                    const hv_word_t encoded_word =
                        encode_sample_word(work.sample, m_cim, word_index);
                    encoder_result.words[word_index] = encoded_word;

                    if (word_index + 1u == HV_WORDS) {
                        output_packet = work;
                        output_packet.encoded = encoder_result;
                        output_packet.encoded.words[word_index] = encoded_word;
                        output_valid = true;
                        output_presented = false;
                        busy = false;
                        word_index = 0;
                    } else {
                        word_index = word_index + 1u;
                    }
                }
            }

            m_encoder_out_data.write(output_packet);
            m_encoder_out_valid.write(output_valid);
            wait();
        }
    }
}

void HDC_Accelerator::ngram_thread() {
    NGramPacket output_packet = NGramPacket();
    EncoderPacket work_packet = EncoderPacket();
    hv_t work;
    hv_t next;
    bool bind_busy = false;
    bool output_pending = false;
    bool output_presented = false;
    unsigned output_target = OUTPUT_NONE;
    bool control_done_valid = false;
    bool control_done_presented = false;
    unsigned bind_round = 0;
    unsigned bind_word = 0;
    unsigned oldest_slot = 0;

    {
        HLS_DEFINE_PROTOCOL("ngram_reset");
        clear_hv(work);
        clear_hv(next);
        reset_ngram_buffer();
        m_encoder_out_ready.write(false);
        m_bundler_in_data.write(NGramPacket());
        m_bundler_in_valid.write(false);
        m_distance_in_data.write(NGramPacket());
        m_distance_in_valid.write(false);
        m_ngram_control_done_valid.write(false);
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("ngram_cycle");

            if (control_done_valid && !control_done_presented) {
                control_done_presented = true;
            } else if (control_done_valid && m_ngram_control_done_ready.read()) {
                control_done_valid = false;
                control_done_presented = false;
            }

            if (output_pending) {
                bool output_consumed = false;
                if (output_target == OUTPUT_BUNDLER) {
                    output_consumed = m_bundler_in_ready.read();
                } else if (output_target == OUTPUT_DISTANCE) {
                    output_consumed = m_distance_in_ready.read();
                }

                if (!output_presented) {
                    output_presented = true;
                } else if (output_consumed) {
                    output_pending = false;
                    output_presented = false;
                    output_target = OUTPUT_NONE;
                }
            }

            m_bundler_in_valid.write(output_pending && output_target == OUTPUT_BUNDLER);
            m_distance_in_valid.write(output_pending && output_target == OUTPUT_DISTANCE);
            m_ngram_control_done_valid.write(control_done_valid);
            m_bundler_in_data.write(output_packet);
            m_distance_in_data.write(output_packet);

            if (bind_busy) {
                m_encoder_out_ready.write(false);
                const unsigned rhs_slot = (oldest_slot + bind_round) % N_GRAM_SIZE;
                next.words[bind_word] = permute_xor_word(work, m_ngram_buffer[rhs_slot], bind_word);

                if (bind_word + 1u == HV_WORDS) {
                    work = next;
                    clear_hv(next);
                    bind_word = 0;

                    if (bind_round + 1u == N_GRAM_SIZE) {
                        output_packet = NGramPacket();
                        output_packet.kind = work_packet.kind;
                        output_packet.class_id = work_packet.class_id;
                        output_packet.ngram = work;
                        output_packet.valid_ngram = true;
                        output_target = (work_packet.kind == AccelCommandKind::TrainSample)
                                            ? OUTPUT_BUNDLER
                                            : OUTPUT_DISTANCE;
                        output_pending = true;
                        output_presented = false;
                        bind_busy = false;
                        bind_round = 0;
                    } else {
                        bind_round = bind_round + 1u;
                    }
                } else {
                    bind_word = bind_word + 1u;
                }

                wait();
                continue;
            }

            const bool can_accept_encoder = !output_pending && !control_done_valid;
            m_encoder_out_ready.write(can_accept_encoder);

            if (can_accept_encoder && m_encoder_out_valid.read()) {
                EncoderPacket item = m_encoder_out_data.read();

                if (item.kind == AccelCommandKind::ResetTraining) {
                    reset_ngram_buffer();
                    output_packet = NGramPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    clear_hv(output_packet.ngram);
                    output_target = OUTPUT_BUNDLER;
                    output_pending = true;
                    output_presented = false;
                } else if (item.kind == AccelCommandKind::ResetInference) {
                    reset_ngram_buffer();
                    control_done_valid = true;
                    control_done_presented = false;
                } else if (item.kind == AccelCommandKind::InvalidTrainingStep) {
                    reset_ngram_buffer();
                    output_packet = NGramPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    clear_hv(output_packet.ngram);
                    output_target = OUTPUT_BUNDLER;
                    output_pending = true;
                    output_presented = false;
                } else if (item.kind == AccelCommandKind::TrainSample ||
                           item.kind == AccelCommandKind::InferSample) {
                    push_encoded_sample_to_ngram_buffer(item.encoded);

                    output_packet = NGramPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    clear_hv(output_packet.ngram);

                    if (m_ngram_buffer_fill_count != N_GRAM_SIZE) {
                        output_target = (item.kind == AccelCommandKind::TrainSample)
                                            ? OUTPUT_BUNDLER
                                            : OUTPUT_DISTANCE;
                        output_pending = true;
                        output_presented = false;
                    } else {
                        oldest_slot = m_ngram_buffer_write_pos;
                        work_packet = item;
                        work = m_ngram_buffer[oldest_slot];
                        clear_hv(next);
                        bind_round = (N_GRAM_SIZE > 1) ? 1u : N_GRAM_SIZE;
                        bind_word = 0;
                        if (N_GRAM_SIZE > 1) {
                            bind_busy = true;
                        } else {
                            output_packet.kind = item.kind;
                            output_packet.class_id = item.class_id;
                            output_packet.ngram = work;
                            output_packet.valid_ngram = true;
                            output_target = (item.kind == AccelCommandKind::TrainSample)
                                                ? OUTPUT_BUNDLER
                                                : OUTPUT_DISTANCE;
                            output_pending = true;
                            output_presented = false;
                        }
                    }
                }
            }

            wait();
        }
    }
}

void HDC_Accelerator::train_thread() {
    bool done_valid = false;
    bool done_presented = false;

    {
        HLS_DEFINE_PROTOCOL("train_reset");
        reset_bundling_buffer_only();
        m_bundler_in_ready.write(false);
        m_train_control_done_valid.write(false);
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("train_cycle");

            if (done_valid && !done_presented) {
                done_presented = true;
            } else if (done_valid && m_train_control_done_ready.read()) {
                done_valid = false;
                done_presented = false;
            }

            if (!done_valid && m_bundler_in_valid.read()) {
                const NGramPacket item = m_bundler_in_data.read();

                if (item.kind == AccelCommandKind::TrainSample) {
                    if (item.valid_ngram) {
                        if (!m_current_class_valid) {
                            m_current_class_id = item.class_id;
                            m_current_class_valid = true;
                        }
                        add_ngram_to_bundling_buffer(item.ngram);
                    }
                } else if (item.kind == AccelCommandKind::InvalidTrainingStep) {
                    finalize_current_class();
                    reset_bundling_buffer_only();
                    done_valid = true;
                    done_presented = false;
                } else if (item.kind == AccelCommandKind::ResetTraining) {
                    reset_bundling_buffer_only();
                    done_valid = true;
                    done_presented = false;
                }
            }

            m_train_control_done_valid.write(done_valid);
            m_bundler_in_ready.write(!done_valid);
            wait();
        }
    }
}

void HDC_Accelerator::distance_thread() {
    NGramPacket work = NGramPacket();
    DistancePacket output_packet = DistancePacket();
    bool busy = false;
    bool output_valid = false;
    bool output_presented = false;
    unsigned class_id = 0;

    {
        HLS_DEFINE_PROTOCOL("distance_reset");
        m_distance_in_ready.write(false);
        m_distance_done_data.write(DistancePacket());
        m_distance_done_valid.write(false);
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("distance_cycle");

            if (output_valid && !output_presented) {
                output_presented = true;
            } else if (output_valid && m_distance_done_ready.read()) {
                output_valid = false;
                output_presented = false;
            }

            const bool can_accept_input = !busy && !output_valid;
            m_distance_in_ready.write(can_accept_input);

            if (can_accept_input && m_distance_in_valid.read()) {
                const NGramPacket item = m_distance_in_data.read();
                if (!item.valid_ngram) {
                    output_packet = DistancePacket();
                    output_packet.valid_prediction = false;
                    for (unsigned copy_class = 0; copy_class < NUM_CLASSES; ++copy_class) {
                        output_packet.distances[copy_class] = 0;
                    }
                    output_valid = true;
                    output_presented = false;
                } else {
                    work = item;
                    output_packet = DistancePacket();
                    output_packet.valid_prediction = true;
                    for (unsigned copy_class = 0; copy_class < NUM_CLASSES; ++copy_class) {
                        output_packet.distances[copy_class] = 0;
                    }
                    class_id = 0;
                    busy = true;
                }
            }

            if (busy && !output_valid) {
                output_packet.distances[class_id] =
                    hamming_distance_words(work.ngram, m_assoc_mem[class_id]);

                if (class_id == (NUM_CLASSES - 1u)) {
                    output_valid = true;
                    output_presented = false;
                    busy = false;
                    class_id = 0;
                } else {
                    class_id = class_id + 1u;
                }
            }

            m_distance_done_data.write(output_packet);
            m_distance_done_valid.write(output_valid);
            wait();
        }
    }
}

void HDC_Accelerator::response_thread() {
    DistancePacket work = DistancePacket();
    bool holding_response = false;
    bool distance_token_consumed = false;

    {
        HLS_DEFINE_PROTOCOL("response_reset");
        rsp_valid.write(false);
        rsp_valid_prediction.write(false);
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            rsp_distances[class_id].write(0);
        }
        m_distance_done_ready.write(false);
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("response_cycle");

            if (holding_response) {
                rsp_valid.write(true);
                rsp_valid_prediction.write(work.valid_prediction);
                for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
                    rsp_distances[class_id].write(work.distances[class_id]);
                }
                m_distance_done_ready.write(false);

                if (rsp_ready.read()) {
                    holding_response = false;
                }
            } else {
                rsp_valid.write(false);
                rsp_valid_prediction.write(false);

                if (distance_token_consumed) {
                    m_distance_done_ready.write(false);
                    if (!m_distance_done_valid.read()) {
                        distance_token_consumed = false;
                    }
                } else {
                    m_distance_done_ready.write(true);
                }

                if (!distance_token_consumed && m_distance_done_valid.read()) {
                    work = m_distance_done_data.read();
                    holding_response = true;
                    distance_token_consumed = true;
                }
            }

            wait();
        }
    }
}

void HDC_Accelerator::reset_all_local_state() {
    reset_ngram_buffer();
    reset_bundling_buffer_only();
}

void HDC_Accelerator::reset_bundling_buffer_only() {
    m_current_class_count = 0;
    m_current_class_id = 0;
    m_current_class_valid = false;
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        HLS_UNROLL_LOOP(OFF, "reset-bundling-loop");
        m_bundling_score[d] = 0;
    }
}

void HDC_Accelerator::reset_ngram_buffer() {
    m_ngram_buffer_write_pos = 0;
    m_ngram_buffer_fill_count = 0;
    for (unsigned slot = 0; slot < N_GRAM_SIZE; ++slot) {
        HLS_UNROLL_LOOP(OFF, "reset-ngram-loop");
        clear_hv(m_ngram_buffer[slot]);
    }
}

void HDC_Accelerator::add_ngram_to_bundling_buffer(const hv_t &encoded_ngram) {
    for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
        HLS_UNROLL_LOOP(OFF, "bundling-loop");
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
        HLS_UNROLL_LOOP(OFF, "finalize-class-loop");
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
