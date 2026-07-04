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

command_kind_t command_kind_value(AccelCommandKind kind) {
    return static_cast<unsigned>(kind);
}

bool command_kind_is(command_kind_t actual, AccelCommandKind expected) {
    return actual == command_kind_value(expected);
}

bool is_train_control_command(command_kind_t kind) {
    return command_kind_is(kind, AccelCommandKind::ResetTraining) ||
           command_kind_is(kind, AccelCommandKind::InvalidTrainingStep);
}

bool is_ngram_control_command(command_kind_t kind) {
    return command_kind_is(kind, AccelCommandKind::ResetInference);
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
hv_word_t encode_sample_word(const QuantizedSample &sample, unsigned word_index) {
    hv_word_t encoded_word = 0;
    const feature_score_t signed_threshold =
        (NUM_FEATURES % 2 == 1) ? feature_score_t(-1) : feature_score_t(0);

    for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
        HLS_UNROLL_LOOP(OFF, "encode-word-bits-loop");
        feature_score_t score = 0;
        for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
            HLS_UNROLL_LOOP(OFF, "encode-features-loop");
            const unsigned level = sample.levels[feature].to_uint();
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

void HDC_Accelerator::write_encoder_in_packet(const EncoderPacket &packet) {
    m_encoder_in_kind.write(packet.kind);
    m_encoder_in_class_id.write(packet.class_id);
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        HLS_UNROLL_LOOP(OFF, "write-encoder-in-sample-loop");
        m_encoder_in_sample_levels[feature].write(packet.sample.levels[feature]);
    }
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "write-encoder-in-encoded-loop");
        m_encoder_in_encoded_words[word].write(packet.encoded.words[word]);
    }
}

EncoderPacket HDC_Accelerator::read_encoder_in_packet() const {
    EncoderPacket packet = EncoderPacket();
    packet.kind = m_encoder_in_kind.read();
    packet.class_id = m_encoder_in_class_id.read();
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        HLS_UNROLL_LOOP(OFF, "read-encoder-in-sample-loop");
        packet.sample.levels[feature] = m_encoder_in_sample_levels[feature].read();
    }
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "read-encoder-in-encoded-loop");
        packet.encoded.words[word] = m_encoder_in_encoded_words[word].read();
    }
    return packet;
}

void HDC_Accelerator::write_encoder_out_packet(const EncoderPacket &packet) {
    m_encoder_out_kind.write(packet.kind);
    m_encoder_out_class_id.write(packet.class_id);
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        HLS_UNROLL_LOOP(OFF, "write-encoder-out-sample-loop");
        m_encoder_out_sample_levels[feature].write(packet.sample.levels[feature]);
    }
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "write-encoder-out-encoded-loop");
        m_encoder_out_encoded_words[word].write(packet.encoded.words[word]);
    }
}

EncoderPacket HDC_Accelerator::read_encoder_out_packet() const {
    EncoderPacket packet = EncoderPacket();
    packet.kind = m_encoder_out_kind.read();
    packet.class_id = m_encoder_out_class_id.read();
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        HLS_UNROLL_LOOP(OFF, "read-encoder-out-sample-loop");
        packet.sample.levels[feature] = m_encoder_out_sample_levels[feature].read();
    }
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "read-encoder-out-encoded-loop");
        packet.encoded.words[word] = m_encoder_out_encoded_words[word].read();
    }
    return packet;
}

void HDC_Accelerator::write_bundler_in_packet(const NGramPacket &packet) {
    m_bundler_in_kind.write(packet.kind);
    m_bundler_in_class_id.write(packet.class_id);
    m_bundler_in_valid_ngram.write(packet.valid_ngram);
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "write-bundler-in-ngram-loop");
        m_bundler_in_ngram_words[word].write(packet.ngram.words[word]);
    }
}

NGramPacket HDC_Accelerator::read_bundler_in_packet() const {
    NGramPacket packet = NGramPacket();
    packet.kind = m_bundler_in_kind.read();
    packet.class_id = m_bundler_in_class_id.read();
    packet.valid_ngram = m_bundler_in_valid_ngram.read();
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "read-bundler-in-ngram-loop");
        packet.ngram.words[word] = m_bundler_in_ngram_words[word].read();
    }
    return packet;
}

void HDC_Accelerator::write_distance_in_packet(const NGramPacket &packet) {
    m_distance_in_kind.write(packet.kind);
    m_distance_in_class_id.write(packet.class_id);
    m_distance_in_valid_ngram.write(packet.valid_ngram);
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "write-distance-in-ngram-loop");
        m_distance_in_ngram_words[word].write(packet.ngram.words[word]);
    }
}

NGramPacket HDC_Accelerator::read_distance_in_packet() const {
    NGramPacket packet = NGramPacket();
    packet.kind = m_distance_in_kind.read();
    packet.class_id = m_distance_in_class_id.read();
    packet.valid_ngram = m_distance_in_valid_ngram.read();
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        HLS_UNROLL_LOOP(OFF, "read-distance-in-ngram-loop");
        packet.ngram.words[word] = m_distance_in_ngram_words[word].read();
    }
    return packet;
}

void HDC_Accelerator::write_distance_done_packet(const DistancePacket &packet) {
    m_distance_done_valid_prediction.write(packet.valid_prediction);
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        HLS_UNROLL_LOOP(OFF, "write-distance-done-loop");
        m_distance_done_distances[class_id].write(packet.distances[class_id]);
    }
}

DistancePacket HDC_Accelerator::read_distance_done_packet() const {
    DistancePacket packet = DistancePacket();
    packet.valid_prediction = m_distance_done_valid_prediction.read();
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        HLS_UNROLL_LOOP(OFF, "read-distance-done-loop");
        packet.distances[class_id] = m_distance_done_distances[class_id].read();
    }
    return packet;
}

// Data commands are pipelined: TrainSample and InferSample are dispatched
// without waiting for completion. Control commands are blocking stream
// boundaries and wait until their token passes through the internal pipeline.
void HDC_Accelerator::command_thread() {
    bool output_valid = false;
    bool output_payload_presented = false;
    bool output_presented = false;
    bool wait_ngram_control = false;
    bool wait_train_control = false;
    bool ngram_control_done_seen = false;
    bool train_control_done_seen = false;

    {
        cmd_ready.write(false);
        write_encoder_in_packet(EncoderPacket());
        m_encoder_in_valid.write(false);
        m_ngram_control_done_ready.write(false);
        m_train_control_done_ready.write(false);
        wait();
    }

    while (true) {
        {
            const bool cmd_valid_snapshot = cmd_valid.read();
            const command_kind_t cmd_kind_snapshot = cmd_kind.read();
            const class_t cmd_class_id_snapshot = cmd_class_id.read();
            QuantizedSample cmd_sample_snapshot = {};
            for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
                HLS_UNROLL_LOOP(OFF, "command-sample-snapshot-loop");
                cmd_sample_snapshot.levels[feature] = cmd_sample_levels[feature].read();
            }

            if (output_valid && !output_payload_presented) {
                output_payload_presented = true;
            } else if (output_valid && !output_presented) {
                output_presented = true;
            } else if (output_valid && m_encoder_in_ready.read()) {
                output_valid = false;
                output_payload_presented = false;
                output_presented = false;
            }

            if (wait_ngram_control) {
                if (!ngram_control_done_seen && m_ngram_control_done_valid.read()) {
                    ngram_control_done_seen = true;
                } else if (ngram_control_done_seen && !m_ngram_control_done_valid.read()) {
                    wait_ngram_control = false;
                    ngram_control_done_seen = false;
                }
            }

            if (wait_train_control) {
                if (!train_control_done_seen && m_train_control_done_valid.read()) {
                    train_control_done_seen = true;
                } else if (train_control_done_seen && !m_train_control_done_valid.read()) {
                    wait_train_control = false;
                    train_control_done_seen = false;
                }
            }

            const bool can_accept_command =
                !output_valid && !wait_ngram_control && !wait_train_control;
            cmd_ready.write(can_accept_command);
            m_ngram_control_done_ready.write(wait_ngram_control ||
                                             m_ngram_control_done_valid.read());
            m_train_control_done_ready.write(wait_train_control ||
                                             m_train_control_done_valid.read());

            if (cmd_valid_snapshot && can_accept_command) {
                EncoderPacket packet = {};
                packet.kind = cmd_kind_snapshot;
                packet.class_id = cmd_class_id_snapshot;
                packet.sample = cmd_sample_snapshot;
                clear_hv(packet.encoded);

                if (command_kind_is(packet.kind, AccelCommandKind::InferSample)) {
                    packet.class_id = 0;
                }

                write_encoder_in_packet(packet);
                output_valid = true;
                output_payload_presented = false;
                output_presented = false;

                if (is_ngram_control_command(packet.kind)) {
                    wait_ngram_control = true;
                    ngram_control_done_seen = false;
                }
                if (is_train_control_command(packet.kind)) {
                    wait_train_control = true;
                    train_control_done_seen = false;
                }
            }

            m_encoder_in_valid.write(output_valid && output_payload_presented);
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
        clear_hv(encoder_result);
        clear_hv(output_packet.encoded);
        m_encoder_in_ready.write(false);
        write_encoder_out_packet(EncoderPacket());
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
                work = EncoderPacket();
                work.kind = m_encoder_in_kind.read();
                work.class_id = m_encoder_in_class_id.read();
                for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
                    HLS_UNROLL_LOOP(OFF, "encoder-read-input-sample-loop");
                    work.sample.levels[feature] = m_encoder_in_sample_levels[feature].read();
                }
                for (unsigned word = 0; word < HV_WORDS; ++word) {
                    HLS_UNROLL_LOOP(OFF, "encoder-read-input-encoded-loop");
                    work.encoded.words[word] = m_encoder_in_encoded_words[word].read();
                }
                clear_hv(encoder_result);
                word_index = 0;
                busy = true;
            }

            if (busy && !output_valid) {
                const bool should_encode =
                    command_kind_is(work.kind, AccelCommandKind::TrainSample) ||
                    command_kind_is(work.kind, AccelCommandKind::InferSample);

                if (!should_encode) {
                    output_packet = work;
                    output_packet.encoded = encoder_result;
                    output_valid = true;
                    output_presented = false;
                    busy = false;
                    word_index = 0;
                } else {
#ifdef STRATUS_HLS
                    const hv_word_t encoded_word =
                        encode_sample_word(work.sample, word_index);
#else
                    const hv_word_t encoded_word =
                        encode_sample_word(work.sample, m_cim, word_index);
#endif
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

            write_encoder_out_packet(output_packet);
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
        clear_hv(work);
        clear_hv(next);
        reset_ngram_buffer();
        m_encoder_out_ready.write(false);
        write_bundler_in_packet(NGramPacket());
        m_bundler_in_valid.write(false);
        write_distance_in_packet(NGramPacket());
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
            write_bundler_in_packet(output_packet);
            write_distance_in_packet(output_packet);

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
                        output_target = command_kind_is(work_packet.kind, AccelCommandKind::TrainSample)
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
                EncoderPacket item = read_encoder_out_packet();

                if (command_kind_is(item.kind, AccelCommandKind::ResetTraining)) {
                    reset_ngram_buffer();
                    output_packet = NGramPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    clear_hv(output_packet.ngram);
                    output_target = OUTPUT_BUNDLER;
                    output_pending = true;
                    output_presented = false;
                } else if (command_kind_is(item.kind, AccelCommandKind::ResetInference)) {
                    reset_ngram_buffer();
                    control_done_valid = true;
                    control_done_presented = false;
                } else if (command_kind_is(item.kind, AccelCommandKind::InvalidTrainingStep)) {
                    reset_ngram_buffer();
                    output_packet = NGramPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    clear_hv(output_packet.ngram);
                    output_target = OUTPUT_BUNDLER;
                    output_pending = true;
                    output_presented = false;
                } else if (command_kind_is(item.kind, AccelCommandKind::TrainSample) ||
                           command_kind_is(item.kind, AccelCommandKind::InferSample)) {
                    push_encoded_sample_to_ngram_buffer(item.encoded);

                    output_packet = NGramPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    clear_hv(output_packet.ngram);

                    if (m_ngram_buffer_fill_count != N_GRAM_SIZE) {
                        output_target = command_kind_is(item.kind, AccelCommandKind::TrainSample)
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
                            output_target = command_kind_is(item.kind, AccelCommandKind::TrainSample)
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
    NGramPacket work;
    bool done_valid = false;
    bool done_presented = false;
    bool init_reset_done_pending = false;
    bool init_reset_token_drain = false;
    unsigned word_index = 0;
    unsigned assoc_class = 0;

    {
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
            const bool bundler_valid = m_bundler_in_valid.read();
            NGramPacket bundler_item = NGramPacket();
            if (bundler_valid) {
                bundler_item = read_bundler_in_packet();
            }

            if (init_reset_token_drain && !bundler_valid) {
                init_reset_token_drain = false;
            }

            const bool startup_clear_active =
                state == TRAIN_INIT_RESET_SCORES ||
                state == TRAIN_INIT_RESET_ASSOC;
            if (startup_clear_active && bundler_valid &&
                command_kind_is(bundler_item.kind, AccelCommandKind::ResetTraining)) {
                init_reset_done_pending = true;
                init_reset_token_drain = true;
            }

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
                        if (init_reset_done_pending) {
                            done_valid = true;
                            done_presented = false;
                            init_reset_done_pending = false;
                        }
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
            } else if (!done_valid && !init_reset_token_drain && bundler_valid) {
                const NGramPacket item = bundler_item;

                if (command_kind_is(item.kind, AccelCommandKind::TrainSample)) {
                    if (item.valid_ngram) {
                        work = item;
                        if (!m_current_class_valid) {
                            m_current_class_id = item.class_id;
                            m_current_class_valid = true;
                        }
                        word_index = 0;
                        state = TRAIN_ADD_NGRAM;
                    }
                } else if (command_kind_is(item.kind, AccelCommandKind::InvalidTrainingStep)) {
                    word_index = 0;
                    if (m_current_class_valid) {
                        state = TRAIN_FINALIZE_CLASS;
                    } else {
                        state = TRAIN_RESET_TRAINING;
                    }
                } else if (command_kind_is(item.kind, AccelCommandKind::ResetTraining)) {
                    word_index = 0;
                    state = TRAIN_RESET_TRAINING;
                }
            }

            m_train_control_done_valid.write(done_valid);
            m_bundler_in_ready.write((state == TRAIN_IDLE && !done_valid) ||
                                     init_reset_token_drain);
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
    bool input_token_consumed = false;
    unsigned class_id = 0;

    {
        m_distance_in_ready.write(false);
        write_distance_done_packet(DistancePacket());
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

            if (!m_distance_in_valid.read()) {
                input_token_consumed = false;
            }

            const bool can_accept_input = !busy && !output_valid;
            m_distance_in_ready.write(can_accept_input ||
                                      (input_token_consumed && m_distance_in_valid.read()));

            if (can_accept_input && m_distance_in_valid.read() && !input_token_consumed) {
                input_token_consumed = true;
                const NGramPacket item = read_distance_in_packet();
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

            write_distance_done_packet(output_packet);
            m_distance_done_valid.write(output_valid);
            wait();
        }
    }
}

void HDC_Accelerator::response_thread() {
    DistancePacket work;

    bool holding_response = false;
    bool distance_token_consumed = false;

    {
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
                    work = read_distance_done_packet();
                    holding_response = true;
                    distance_token_consumed = true;
                }
            }

            wait();
        }
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
