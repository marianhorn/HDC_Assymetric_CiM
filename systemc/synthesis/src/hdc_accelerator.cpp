// SYNTHESIS TARGET: This module is intended for HLS/SystemC synthesis.
// Keep dataset loading, floating-point quantization, and testbench code outside this file.
#include "hdc_accelerator.h"
#ifdef STRATUS_HLS
#include "generated_cim_rom_dataset00.h"
#endif

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

AccelCommandKind decode_kind(command_kind_t kind) {
    return static_cast<AccelCommandKind>(kind.to_uint());
}

command_kind_t encode_kind(AccelCommandKind kind) {
    return command_kind_t(static_cast<unsigned>(kind));
}

level_t get_sample_level(const sample_bits_t &sample, unsigned feature) {
    const unsigned low = feature * LEVEL_BITS;
    const unsigned high = low + LEVEL_BITS - 1u;
    level_t level = 0;
    level = sample.range(high, low);
    return level;
}

void set_sample_level(sample_bits_t &sample, unsigned feature, level_t level) {
    const unsigned low = feature * LEVEL_BITS;
    const unsigned high = low + LEVEL_BITS - 1u;
    sample.range(high, low) = level;
}

hv_word_t get_hv_word(const hv_bits_t &hv, unsigned word) {
    const unsigned low = word * HV_WORD_BITS;
    const unsigned high = low + HV_WORD_BITS - 1u;
    hv_word_t value = 0;
    value = hv.range(high, low);
    return value;
}

void set_hv_word(hv_bits_t &hv, unsigned word, hv_word_t value) {
    const unsigned low = word * HV_WORD_BITS;
    const unsigned high = low + HV_WORD_BITS - 1u;
    hv.range(high, low) = value;
}

hv_bits_t pack_hv(const hv_t &hv) {
    hv_bits_t packed = 0;
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "pack-hv-words-loop");
        set_hv_word(packed, word, hv.words[word]);
    }
    return packed;
}

void unpack_hv(hv_t &hv, const hv_bits_t &packed) {
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "unpack-hv-words-loop");
        hv.words[word] = get_hv_word(packed, word);
    }
}

distance_counter_t get_distance_word(const distance_bits_t &distances, unsigned class_id) {
    const unsigned low = class_id * DISTANCE_BITS;
    const unsigned high = low + DISTANCE_BITS - 1u;
    distance_counter_t value = 0;
    value = distances.range(high, low);
    return value;
}

void set_distance_word(distance_bits_t &distances,
                       unsigned class_id,
                       distance_counter_t value) {
    const unsigned low = class_id * DISTANCE_BITS;
    const unsigned high = low + DISTANCE_BITS - 1u;
    distances.range(high, low) = value;
}

distance_bits_t pack_distances(const DistancePacket &packet) {
    distance_bits_t packed = 0;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        HLS_UNROLL_LOOP(OFF, "pack-distance-loop");
        set_distance_word(packed, class_id, packet.distances[class_id]);
    }
    return packed;
}

void unpack_distances(DistancePacket &packet, const DistanceChannelPacket &channel) {
    packet.valid_prediction = channel.valid_prediction;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        HLS_UNROLL_LOOP(OFF, "unpack-distance-loop");
        packet.distances[class_id] = get_distance_word(channel.distances, class_id);
    }
}

#ifdef STRATUS_HLS
distance_counter_t popcount_word(hv_word_t value) {
    distance_counter_t count = 0;
    for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
        HLS_UNROLL_LOOP(OFF, "popcount-word-bits-loop");
        if (((value >> bit) & hv_word_t(1)) != 0) {
            ++count;
        }
    }
    return count;
}
#endif

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

#ifdef STRATUS_HLS
hv_word_t encode_sample_word(const sample_bits_t &sample, unsigned word_index) {
    hv_word_t encoded_word = 0;
    const feature_score_t signed_threshold =
        (NUM_FEATURES % 2 == 1) ? feature_score_t(-1) : feature_score_t(0);

    for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
        HLS_UNROLL_LOOP(OFF, "encode-word-bits-loop");
        feature_score_t score = 0;
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            HLS_UNROLL_LOOP(OFF, "encode-features-loop");
            const unsigned level = get_sample_level(sample, feature).to_uint();
            const hv_word_t feature_word = HDC_CIM_ROM_DATASET00[level][feature][word_index];
            if (((feature_word >> bit) & hv_word_t(1)) != 0) {
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
#else
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
#endif

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
#ifdef STRATUS_HLS
      m_encoder_in("m_encoder_in"),
      m_encoder_out("m_encoder_out"),
      m_bundler_in("m_bundler_in"),
      m_distance_in("m_distance_in"),
      m_distance_done("m_distance_done"),
      m_ngram_control_done("m_ngram_control_done"),
      m_train_control_done("m_train_control_done"),
#endif
      m_current_class_valid(false) {
#ifdef STRATUS_HLS
    m_encoder_in.clk_rst(clk, rst, true);
    m_encoder_out.clk_rst(clk, rst, true);
    m_bundler_in.clk_rst(clk, rst, true);
    m_distance_in.clk_rst(clk, rst, true);
    m_distance_done.clk_rst(clk, rst, true);
    m_ngram_control_done.clk_rst(clk, rst, true);
    m_train_control_done.clk_rst(clk, rst, true);
#endif

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
#ifdef STRATUS_HLS
    enum CommandState {
        CMD_IDLE,
        CMD_DEASSERT_READY,
        CMD_SEND_PENDING
    };

    EncoderChannelPacket send_packet;
    CommandState state = CMD_IDLE;
    bool wait_ngram_control = false;
    bool wait_train_control = false;
    bool release_ngram_control = false;
    bool release_train_control = false;

    {
        HLS_DEFINE_PROTOCOL("command_reset");
        cmd_ready.write(false);
        state = CMD_IDLE;
        wait_ngram_control = false;
        wait_train_control = false;
        release_ngram_control = false;
        release_train_control = false;
        m_encoder_in.input.reset();
        m_ngram_control_done.output.reset();
        m_train_control_done.output.reset();
        wait();
    }

    while (true) {
        {
            if (state == CMD_SEND_PENDING) {
                cmd_ready.write(false);
                m_encoder_in.input.put(send_packet);
                state = CMD_IDLE;
            } else if (state == CMD_DEASSERT_READY) {
                cmd_ready.write(false);
                state = CMD_SEND_PENDING;
            } else {
                if (release_ngram_control) {
                    wait_ngram_control = false;
                    release_ngram_control = false;
                } else if (wait_ngram_control) {
                    bool done_token = false;
                    if (m_ngram_control_done.output.nb_get(done_token)) {
                        release_ngram_control = true;
                    }
                }

                if (release_train_control) {
                    wait_train_control = false;
                    release_train_control = false;
                } else if (wait_train_control) {
                    bool done_token = false;
                    if (m_train_control_done.output.nb_get(done_token)) {
                        release_train_control = true;
                    }
                }

                const bool can_accept_command = !wait_ngram_control && !wait_train_control;
                cmd_ready.write(can_accept_command);

                if (cmd_valid.read() && can_accept_command) {
                    const AccelCommandKind command_kind =
                        static_cast<AccelCommandKind>(cmd_kind.read().to_uint());

                    EncoderChannelPacket packet;
                    packet.kind = encode_kind(command_kind);
                    packet.class_id = cmd_class_id.read();
                    packet.sample = 0;
                    packet.encoded = 0;

                    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
                        HLS_UNROLL_LOOP(OFF, "command-pack-sample-loop");
                        set_sample_level(packet.sample,
                                         feature,
                                         cmd_sample_levels[feature].read());
                    }

                    if (command_kind == AccelCommandKind::InferSample) {
                        packet.class_id = 0;
                    }

                    if (is_ngram_control_command(command_kind)) {
                        wait_ngram_control = true;
                    }
                    if (is_train_control_command(command_kind)) {
                        wait_train_control = true;
                    }

                    send_packet = packet;
                    state = CMD_DEASSERT_READY;
                }
            }

            {
                HLS_DEFINE_PROTOCOL("command_wait");
                wait();
            }
        }
    }
#else
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
#endif
}

void HDC_Accelerator::encoder_thread() {
#ifdef STRATUS_HLS
    enum EncoderState {
        ENC_WAIT_INPUT,
        ENC_COMPUTE,
        ENC_SEND
    };

    EncoderChannelPacket work;
    EncoderChannelPacket output_packet;
    EncoderChannelPacket send_packet;
    hv_bits_t encoder_result;
    EncoderState state = ENC_WAIT_INPUT;
    unsigned word_index = 0;

    {
        HLS_DEFINE_PROTOCOL("encoder_reset");
        state = ENC_WAIT_INPUT;
        word_index = 0;
        encoder_result = 0;
        m_encoder_in.output.reset();
        m_encoder_out.input.reset();
        wait();
    }

    while (true) {
        {
            if (state == ENC_SEND) {
                m_encoder_out.input.put(send_packet);
                state = ENC_WAIT_INPUT;
            } else if (state == ENC_WAIT_INPUT) {
                work = m_encoder_in.output.get();
                encoder_result = 0;
                word_index = 0;
                state = ENC_COMPUTE;
            } else {
                const AccelCommandKind work_kind = decode_kind(work.kind);
                const bool should_encode =
                    work_kind == AccelCommandKind::TrainSample ||
                    work_kind == AccelCommandKind::InferSample;

                if (!should_encode) {
                    output_packet = work;
                    output_packet.encoded = encoder_result;
                    send_packet = output_packet;
                    state = ENC_SEND;
                } else {
                    const hv_word_t encoded_word =
                        encode_sample_word(work.sample, word_index);
                    set_hv_word(encoder_result, word_index, encoded_word);

                    if (word_index + 1u == HV_WORDS) {
                        output_packet = work;
                        output_packet.encoded = encoder_result;
                        set_hv_word(output_packet.encoded, word_index, encoded_word);
                        send_packet = output_packet;
                        state = ENC_SEND;
                    } else {
                        word_index = word_index + 1u;
                    }
                }
            }

            {
                HLS_DEFINE_PROTOCOL("encoder_wait");
                wait();
            }
        }
    }
#else
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
#endif
}

void HDC_Accelerator::ngram_thread() {
#ifdef STRATUS_HLS
    enum NGramState {
        NGRAM_INIT_CLEAR,
        NGRAM_WAIT_INPUT,
        NGRAM_BIND,
        NGRAM_SEND,
        NGRAM_SEND_CONTROL
    };

    NGramChannelPacket output_packet;
    EncoderChannelPacket work_packet;
    hv_t work;
    hv_t next;
    NGramChannelPacket send_packet;
    NGramState state = NGRAM_INIT_CLEAR;
    unsigned send_target = OUTPUT_NONE;
    unsigned bind_round = 0;
    unsigned bind_word = 0;
    unsigned oldest_slot = 0;
    unsigned reset_slot = 0;

    {
        HLS_DEFINE_PROTOCOL("ngram_reset");
        m_ngram_buffer_write_pos = 0;
        m_ngram_buffer_fill_count = 0;
        state = NGRAM_INIT_CLEAR;
        send_target = OUTPUT_NONE;
        bind_round = 0;
        bind_word = 0;
        oldest_slot = 0;
        reset_slot = 0;
        m_encoder_out.output.reset();
        m_bundler_in.input.reset();
        m_distance_in.input.reset();
        m_ngram_control_done.input.reset();
        wait();
    }

    while (true) {
        {
            if (state == NGRAM_INIT_CLEAR) {
                clear_hv(m_ngram_buffer[reset_slot]);
                if (reset_slot + 1u == N_GRAM_SIZE) {
                    reset_slot = 0;
                    state = NGRAM_WAIT_INPUT;
                } else {
                    reset_slot = reset_slot + 1u;
                }
            } else if (state == NGRAM_SEND_CONTROL) {
                const bool done = true;
                m_ngram_control_done.input.put(done);
                state = NGRAM_WAIT_INPUT;
            } else if (state == NGRAM_SEND) {
                if (send_target == OUTPUT_BUNDLER) {
                    m_bundler_in.input.put(send_packet);
                } else if (send_target == OUTPUT_DISTANCE) {
                    m_distance_in.input.put(send_packet);
                }
                send_target = OUTPUT_NONE;
                state = NGRAM_WAIT_INPUT;
            } else if (state == NGRAM_BIND) {
                const unsigned rhs_slot = (oldest_slot + bind_round) % N_GRAM_SIZE;
                next.words[bind_word] = permute_xor_word(work, m_ngram_buffer[rhs_slot], bind_word);

                if (bind_word + 1u == HV_WORDS) {
                    work = next;
                    clear_hv(next);
                    bind_word = 0;

                    if (bind_round + 1u == N_GRAM_SIZE) {
                        output_packet = NGramChannelPacket();
                        output_packet.kind = work_packet.kind;
                        output_packet.class_id = work_packet.class_id;
                        output_packet.ngram = pack_hv(work);
                        output_packet.valid_ngram = true;
                        send_packet = output_packet;
                        send_target = (decode_kind(work_packet.kind) == AccelCommandKind::TrainSample)
                                          ? OUTPUT_BUNDLER
                                          : OUTPUT_DISTANCE;
                        bind_round = 0;
                        state = NGRAM_SEND;
                    } else {
                        bind_round = bind_round + 1u;
                    }
                } else {
                    bind_word = bind_word + 1u;
                }
            } else {
                EncoderChannelPacket item = m_encoder_out.output.get();
                const AccelCommandKind item_kind = decode_kind(item.kind);

                if (item_kind == AccelCommandKind::ResetTraining) {
                    reset_ngram_buffer();
                    output_packet = NGramChannelPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    send_packet = output_packet;
                    send_target = OUTPUT_BUNDLER;
                    state = NGRAM_SEND;
                } else if (item_kind == AccelCommandKind::ResetInference) {
                    reset_ngram_buffer();
                    state = NGRAM_SEND_CONTROL;
                } else if (item_kind == AccelCommandKind::InvalidTrainingStep) {
                    reset_ngram_buffer();
                    output_packet = NGramChannelPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    send_packet = output_packet;
                    send_target = OUTPUT_BUNDLER;
                    state = NGRAM_SEND;
                } else if (item_kind == AccelCommandKind::TrainSample ||
                           item_kind == AccelCommandKind::InferSample) {
                    unpack_hv(work, item.encoded);
                    push_encoded_sample_to_ngram_buffer(work);

                    output_packet = NGramChannelPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;

                    if (m_ngram_buffer_fill_count != N_GRAM_SIZE) {
                        send_packet = output_packet;
                        send_target = (item_kind == AccelCommandKind::TrainSample)
                                          ? OUTPUT_BUNDLER
                                          : OUTPUT_DISTANCE;
                        state = NGRAM_SEND;
                    } else {
                        oldest_slot = m_ngram_buffer_write_pos;
                        work_packet = item;
                        work = m_ngram_buffer[oldest_slot];
                        clear_hv(next);
                        bind_round = (N_GRAM_SIZE > 1) ? 1u : N_GRAM_SIZE;
                        bind_word = 0;

                        if (N_GRAM_SIZE > 1) {
                            state = NGRAM_BIND;
                        } else {
                            output_packet.kind = item.kind;
                            output_packet.class_id = item.class_id;
                            output_packet.ngram = pack_hv(work);
                            output_packet.valid_ngram = true;
                            send_packet = output_packet;
                            send_target = (item_kind == AccelCommandKind::TrainSample)
                                              ? OUTPUT_BUNDLER
                                              : OUTPUT_DISTANCE;
                            state = NGRAM_SEND;
                        }
                    }
                }
            }

            {
                HLS_DEFINE_PROTOCOL("ngram_wait");
                wait();
            }
        }
    }
#else
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
#endif
}

void HDC_Accelerator::train_thread() {
    enum TrainState {
        TRAIN_INIT_RESET_SCORES,
        TRAIN_INIT_RESET_ASSOC,
        TRAIN_IDLE,
        TRAIN_ADD_NGRAM,
        TRAIN_FINALIZE_CLASS,
        TRAIN_RESET_TRAINING,
        TRAIN_SEND_DONE
    };

    TrainState state = TRAIN_INIT_RESET_SCORES;
#ifdef STRATUS_HLS
    NGramChannelPacket work;
    unsigned word_index = 0;
    unsigned assoc_class = 0;

    {
        HLS_DEFINE_PROTOCOL("train_reset");
        state = TRAIN_INIT_RESET_SCORES;
        word_index = 0;
        assoc_class = 0;
        m_current_class_count = 0;
        m_current_class_id = 0;
        m_current_class_valid = false;
        m_bundler_in.output.reset();
        m_train_control_done.input.reset();
        wait();
    }

    while (true) {
        {
            if (state == TRAIN_SEND_DONE) {
                const bool done = true;
                m_train_control_done.input.put(done);
                state = TRAIN_IDLE;
            } else if (state == TRAIN_INIT_RESET_SCORES) {
                const unsigned base_dim = word_index * HV_WORD_BITS;
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-init-reset-score-word-loop");
                    m_bundling_score[base_dim + bit] = 0;
                }

                if (word_index + 1u == HV_WORDS) {
                    word_index = 0;
                    assoc_class = 0;
                    state = TRAIN_INIT_RESET_ASSOC;
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_INIT_RESET_ASSOC) {
                {
                    HLS_DEFINE_PROTOCOL("train_assoc_clear_word");
                    m_assoc_mem[assoc_class].words[word_index] = 0;
                }

                if (word_index + 1u == HV_WORDS) {
                    word_index = 0;
                    if (assoc_class + 1u == NUM_CLASSES) {
                        assoc_class = 0;
                        state = TRAIN_IDLE;
                    } else {
                        assoc_class = assoc_class + 1u;
                    }
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_ADD_NGRAM) {
                const unsigned base_dim = word_index * HV_WORD_BITS;
                const hv_word_t word = get_hv_word(work.ngram, word_index);
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-add-ngram-word-loop");
                    if (((word >> bit) & hv_word_t(1)) != 0) {
                        ++m_bundling_score[base_dim + bit];
                    } else {
                        --m_bundling_score[base_dim + bit];
                    }
                }

                if (word_index + 1u == HV_WORDS) {
                    ++m_current_class_count;
                    word_index = 0;
                    state = TRAIN_IDLE;
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_FINALIZE_CLASS) {
                const bool odd_count = (m_current_class_count.to_uint() & 1u) != 0u;
                const train_score_t signed_threshold = odd_count ? train_score_t(-1) : train_score_t(0);
                const unsigned base_dim = word_index * HV_WORD_BITS;
                hv_word_t class_word = 0;
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-finalize-word-loop");
                    const unsigned dim = base_dim + bit;
                    if (m_bundling_score[dim] >= signed_threshold) {
                        class_word = class_word | (hv_word_t(1) << bit);
                    }
                    m_bundling_score[dim] = 0;
                }
                {
                    HLS_DEFINE_PROTOCOL("train_assoc_write_word");
                    m_assoc_mem[m_current_class_id.to_uint()].words[word_index] = class_word;
                }

                if (word_index + 1u == HV_WORDS) {
                    m_current_class_count = 0;
                    m_current_class_id = 0;
                    m_current_class_valid = false;
                    word_index = 0;
                    state = TRAIN_SEND_DONE;
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_RESET_TRAINING) {
                const unsigned base_dim = word_index * HV_WORD_BITS;
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-reset-score-word-loop");
                    m_bundling_score[base_dim + bit] = 0;
                }

                if (word_index + 1u == HV_WORDS) {
                    m_current_class_count = 0;
                    m_current_class_id = 0;
                    m_current_class_valid = false;
                    word_index = 0;
                    state = TRAIN_SEND_DONE;
                } else {
                    word_index = word_index + 1u;
                }
            } else {
                const NGramChannelPacket item = m_bundler_in.output.get();
                const AccelCommandKind item_kind = decode_kind(item.kind);

                if (item_kind == AccelCommandKind::TrainSample) {
                    if (item.valid_ngram) {
                        work = item;
                        if (!m_current_class_valid) {
                            m_current_class_id = item.class_id;
                            m_current_class_valid = true;
                        }
                        word_index = 0;
                        state = TRAIN_ADD_NGRAM;
                    }
                } else if (item_kind == AccelCommandKind::InvalidTrainingStep) {
                    word_index = 0;
                    if (m_current_class_valid) {
                        state = TRAIN_FINALIZE_CLASS;
                    } else {
                        state = TRAIN_RESET_TRAINING;
                    }
                } else if (item_kind == AccelCommandKind::ResetTraining) {
                    word_index = 0;
                    state = TRAIN_RESET_TRAINING;
                }
            }

            {
                HLS_DEFINE_PROTOCOL("train_wait");
                wait();
            }
        }
    }
#else
    NGramPacket work;
    bool done_valid = false;
    bool done_presented = false;
    unsigned word_index = 0;
    unsigned assoc_class = 0;

    {
        HLS_DEFINE_PROTOCOL("train_reset");
        for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            HLS_UNROLL_LOOP(OFF, "reset-assoc-mem-loop");
            for (unsigned word = 0; word < HV_WORDS; ++word) {
                HLS_UNROLL_LOOP(OFF, "reset-assoc-mem-word-loop");
                m_assoc_mem[class_id].words[word] = 0;
            }
        }
        m_current_class_count = 0;
        m_current_class_id = 0;
        m_current_class_valid = false;
        m_bundler_in_ready.write(false);
        m_train_control_done_valid.write(false);
        wait();
    }

    while (true) {
        {

            if (done_valid && !done_presented) {
                done_presented = true;
            } else if (done_valid && m_train_control_done_ready.read()) {
                done_valid = false;
                done_presented = false;
            }

            if (state == TRAIN_INIT_RESET_SCORES) {
                const unsigned base_dim = word_index * HV_WORD_BITS;
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-init-reset-score-word-loop");
                    m_bundling_score[base_dim + bit] = 0;
                }

                if (word_index + 1u == HV_WORDS) {
                    word_index = 0;
                    assoc_class = 0;
                    state = TRAIN_INIT_RESET_ASSOC;
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_INIT_RESET_ASSOC) {
                m_assoc_mem[assoc_class].words[word_index] = 0;

                if (word_index + 1u == HV_WORDS) {
                    word_index = 0;
                    if (assoc_class + 1u == NUM_CLASSES) {
                        assoc_class = 0;
                        state = TRAIN_IDLE;
                    } else {
                        assoc_class = assoc_class + 1u;
                    }
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_ADD_NGRAM) {
                const unsigned base_dim = word_index * HV_WORD_BITS;
                const hv_word_t word = work.ngram.words[word_index];
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-add-ngram-word-loop");
                    if (((word >> bit) & hv_word_t(1)) != 0) {
                        ++m_bundling_score[base_dim + bit];
                    } else {
                        --m_bundling_score[base_dim + bit];
                    }
                }

                if (word_index + 1u == HV_WORDS) {
                    ++m_current_class_count;
                    word_index = 0;
                    state = TRAIN_IDLE;
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_FINALIZE_CLASS) {
                const bool odd_count = (m_current_class_count.to_uint() & 1u) != 0u;
                const train_score_t signed_threshold = odd_count ? train_score_t(-1) : train_score_t(0);
                const unsigned base_dim = word_index * HV_WORD_BITS;
                hv_word_t class_word = 0;
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-finalize-word-loop");
                    const unsigned dim = base_dim + bit;
                    if (m_bundling_score[dim] >= signed_threshold) {
                        class_word = class_word | (hv_word_t(1) << bit);
                    }
                    m_bundling_score[dim] = 0;
                }
                m_assoc_mem[m_current_class_id.to_uint()].words[word_index] = class_word;

                if (word_index + 1u == HV_WORDS) {
                    m_current_class_count = 0;
                    m_current_class_id = 0;
                    m_current_class_valid = false;
                    word_index = 0;
                    done_valid = true;
                    done_presented = false;
                    state = TRAIN_IDLE;
                } else {
                    word_index = word_index + 1u;
                }
            } else if (state == TRAIN_RESET_TRAINING) {
                const unsigned base_dim = word_index * HV_WORD_BITS;
                for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
                    HLS_UNROLL_LOOP(OFF, "train-reset-score-word-loop");
                    m_bundling_score[base_dim + bit] = 0;
                }

                if (word_index + 1u == HV_WORDS) {
                    m_current_class_count = 0;
                    m_current_class_id = 0;
                    m_current_class_valid = false;
                    word_index = 0;
                    done_valid = true;
                    done_presented = false;
                    state = TRAIN_IDLE;
                } else {
                    word_index = word_index + 1u;
                }
            } else if (!done_valid && m_bundler_in_valid.read()) {
                const NGramPacket item = m_bundler_in_data.read();

                if (item.kind == AccelCommandKind::TrainSample) {
                    if (item.valid_ngram) {
                        work = item;
                        if (!m_current_class_valid) {
                            m_current_class_id = item.class_id;
                            m_current_class_valid = true;
                        }
                        word_index = 0;
                        state = TRAIN_ADD_NGRAM;
                    }
                } else if (item.kind == AccelCommandKind::InvalidTrainingStep) {
                    word_index = 0;
                    if (m_current_class_valid) {
                        state = TRAIN_FINALIZE_CLASS;
                    } else {
                        state = TRAIN_RESET_TRAINING;
                    }
                } else if (item.kind == AccelCommandKind::ResetTraining) {
                    word_index = 0;
                    state = TRAIN_RESET_TRAINING;
                }
            }

            m_train_control_done_valid.write(done_valid);
            m_bundler_in_ready.write(state == TRAIN_IDLE && !done_valid);
            wait();
        }
    }
#endif
}

void HDC_Accelerator::distance_thread() {
#ifdef STRATUS_HLS
    enum DistanceState {
        DIST_WAIT_INPUT,
        DIST_COMPUTE,
        DIST_SEND
    };

    NGramChannelPacket work;
    DistancePacket output_packet;
    DistanceChannelPacket send_packet;
    DistanceState state = DIST_WAIT_INPUT;
    unsigned class_id = 0;
    unsigned word_index = 0;
    distance_counter_t distance_acc = 0;

    {
        HLS_DEFINE_PROTOCOL("distance_reset");
        state = DIST_WAIT_INPUT;
        class_id = 0;
        word_index = 0;
        distance_acc = 0;
        m_distance_in.output.reset();
        m_distance_done.input.reset();
        wait();
    }

    while (true) {
        {
            if (state == DIST_SEND) {
                m_distance_done.input.put(send_packet);
                state = DIST_WAIT_INPUT;
            } else if (state == DIST_WAIT_INPUT) {
                const NGramChannelPacket item = m_distance_in.output.get();
                if (!item.valid_ngram) {
                    clear_distance_packet(output_packet);
                    output_packet.valid_prediction = false;
                    send_packet.valid_prediction = false;
                    send_packet.distances = pack_distances(output_packet);
                    state = DIST_SEND;
                } else {
                    work = item;
                    clear_distance_packet(output_packet);
                    output_packet.valid_prediction = true;
                    class_id = 0;
                    word_index = 0;
                    distance_acc = 0;
                    state = DIST_COMPUTE;
                }
            } else {
                hv_word_t assoc_word = 0;
                {
                    HLS_DEFINE_PROTOCOL("distance_assoc_read_word");
                    assoc_word = m_assoc_mem[class_id].words[word_index];
                }

                const hv_word_t diff = get_hv_word(work.ngram, word_index) ^ assoc_word;
                const distance_counter_t next_distance =
                    distance_acc + popcount_word(diff);

                if (word_index == (HV_WORDS - 1u)) {
                    output_packet.distances[class_id] = next_distance;
                    word_index = 0;
                    distance_acc = 0;

                    if (class_id == (NUM_CLASSES - 1u)) {
                        class_id = 0;
                        send_packet.valid_prediction = output_packet.valid_prediction;
                        send_packet.distances = pack_distances(output_packet);
                        state = DIST_SEND;
                    } else {
                        class_id = class_id + 1u;
                    }
                } else {
                    word_index = word_index + 1u;
                    distance_acc = next_distance;
                }
            }

            {
                HLS_DEFINE_PROTOCOL("distance_wait");
                wait();
            }
        }
    }
#else
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
#endif
}

void HDC_Accelerator::response_thread() {
    DistancePacket work;

#ifdef STRATUS_HLS
    enum ResponseState {
        RSP_WAIT_PACKET,
        RSP_PRESENT
    };

    ResponseState state = RSP_WAIT_PACKET;
    bool response_valid_prediction = false;
    distance_counter_t response_distance0 = 0;
    distance_counter_t response_distance1 = 0;
    distance_counter_t response_distance2 = 0;
    distance_counter_t response_distance3 = 0;
    distance_counter_t response_distance4 = 0;

    {
        HLS_DEFINE_PROTOCOL("response_reset");
        rsp_valid.write(false);
        rsp_valid_prediction.write(false);
#if NUM_CLASSES > 0
        rsp_distances[0].write(0);
#endif
#if NUM_CLASSES > 1
        rsp_distances[1].write(0);
#endif
#if NUM_CLASSES > 2
        rsp_distances[2].write(0);
#endif
#if NUM_CLASSES > 3
        rsp_distances[3].write(0);
#endif
#if NUM_CLASSES > 4
        rsp_distances[4].write(0);
#endif
        state = RSP_WAIT_PACKET;
        response_valid_prediction = false;
        response_distance0 = 0;
        response_distance1 = 0;
        response_distance2 = 0;
        response_distance3 = 0;
        response_distance4 = 0;
        m_distance_done.output.reset();
        wait();
    }

    while (true) {
        {
            if (state == RSP_WAIT_PACKET) {
                {
                    HLS_DEFINE_PROTOCOL("response_idle_outputs");
                    rsp_valid.write(false);
                    rsp_valid_prediction.write(false);
                }
                const DistanceChannelPacket channel_packet = m_distance_done.output.get();
                unpack_distances(work, channel_packet);
                response_valid_prediction = work.valid_prediction;
#if NUM_CLASSES > 0
                response_distance0 = work.distances[0];
#endif
#if NUM_CLASSES > 1
                response_distance1 = work.distances[1];
#endif
#if NUM_CLASSES > 2
                response_distance2 = work.distances[2];
#endif
#if NUM_CLASSES > 3
                response_distance3 = work.distances[3];
#endif
#if NUM_CLASSES > 4
                response_distance4 = work.distances[4];
#endif
                state = RSP_PRESENT;
            } else {
                {
                    HLS_DEFINE_PROTOCOL("response_outputs");
                    rsp_valid.write(true);
                    rsp_valid_prediction.write(response_valid_prediction);
#if NUM_CLASSES > 0
                    rsp_distances[0].write(response_distance0);
#endif
#if NUM_CLASSES > 1
                    rsp_distances[1].write(response_distance1);
#endif
#if NUM_CLASSES > 2
                    rsp_distances[2].write(response_distance2);
#endif
#if NUM_CLASSES > 3
                    rsp_distances[3].write(response_distance3);
#endif
#if NUM_CLASSES > 4
                    rsp_distances[4].write(response_distance4);
#endif
                }

                bool rsp_ready_snapshot = false;
                {
                    HLS_DEFINE_PROTOCOL("response_ready");
                    rsp_ready_snapshot = rsp_ready.read();
                }
                if (rsp_ready_snapshot) {
                    state = RSP_WAIT_PACKET;
                }
            }

            {
                HLS_DEFINE_PROTOCOL("response_wait");
                wait();
            }
        }
    }
#else
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
#endif
}

void HDC_Accelerator::reset_ngram_buffer() {
    m_ngram_buffer_write_pos = 0;
    m_ngram_buffer_fill_count = 0;
    for (unsigned slot = 0; slot < N_GRAM_SIZE; ++slot) {
        HLS_UNROLL_LOOP(OFF, "reset-ngram-loop");
        clear_hv(m_ngram_buffer[slot]);
    }
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
