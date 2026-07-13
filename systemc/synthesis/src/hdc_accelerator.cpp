// SYNTHESIS TARGET: This module is intended for HLS/SystemC synthesis.
// Keep dataset loading, floating-point quantization, and testbench code outside this file.
#include "hdc_accelerator.h"
#ifndef STRATUS_HLS
#include <iomanip>
#endif
#ifdef STRATUS_HLS
#include "generated_cim_rom_dataset00_banked.h"
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

#ifdef STRATUS_HLS
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

EncoderChannelPacket unpack_command_channel(command_channel_bits_t word) {
    EncoderChannelPacket packet;
    unsigned low = 0;
    unsigned high = COMMAND_KIND_BITS - 1u;
    packet.kind = word.range(high, low);

    low = high + 1u;
    high = low + CLASS_BITS - 1u;
    packet.class_id = word.range(high, low);

    packet.sample = 0;
    for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
        HLS_UNROLL_LOOP(OFF, "unpack-command-sample-loop");
        low = COMMAND_KIND_BITS + CLASS_BITS + (feature * LEVEL_BITS);
        high = low + LEVEL_BITS - 1u;
        level_t level = 0;
        level = word.range(high, low);
        set_sample_level(packet.sample, feature, level);
    }

    packet.encoded = 0;
    return packet;
}

response_channel_bits_t pack_response_channel(const DistanceChannelPacket &packet) {
    response_channel_bits_t word = 0;
    word.range(0, 0) = packet.valid_prediction ? 1u : 0u;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        HLS_UNROLL_LOOP(OFF, "pack-response-distance-loop");
        const unsigned low = 1u + (class_id * DISTANCE_BITS);
        const unsigned high = low + DISTANCE_BITS - 1u;
        const unsigned distance_low = class_id * DISTANCE_BITS;
        const unsigned distance_high = distance_low + DISTANCE_BITS - 1u;
        word.range(high, low) = packet.distances.range(distance_high, distance_low);
    }
    return word;
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

distance_counter_t popcount_word(hv_word_t value) {
    hv_word_t x = value;
    x = x - ((x >> 1) & hv_word_t(0x5555555555555555ULL));
    x = (x & hv_word_t(0x3333333333333333ULL)) +
        ((x >> 2) & hv_word_t(0x3333333333333333ULL));
    x = (x + (x >> 4)) & hv_word_t(0x0F0F0F0F0F0F0F0FULL);
    x = x + (x >> 8);
    x = x + (x >> 16);
    x = x + (x >> 32);
    return distance_counter_t(x.range(6, 0));
}

hv_bits_t rotate_left_one(const hv_bits_t &input) {
    hv_bits_t rotated = input << 1;
    rotated[0] = input[VECTOR_DIMENSION - 1u];
    return rotated;
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

#ifndef STRATUS_HLS
unsigned long long hv_popcount_debug(const hv_t &hv) {
    unsigned long long count = 0;
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
            if (((hv.words[word] >> bit) & hv_word_t(1)) != 0) {
                ++count;
            }
        }
    }
    return count;
}

unsigned long long hv_weighted_sum_debug(const hv_t &hv) {
    unsigned long long sum = 0;
    for (unsigned word = 0; word < HV_WORDS; ++word) {
        const unsigned base_dim = word * HV_WORD_BITS;
        for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
            if (((hv.words[word] >> bit) & hv_word_t(1)) != 0) {
                sum += static_cast<unsigned long long>(base_dim + bit);
            }
        }
    }
    return sum;
}
#endif

#ifdef STRATUS_HLS
inline hv_word_t get_cim_feature_word(unsigned feature, unsigned level, unsigned word_index) {
    switch (feature) {
    case 0: return HDC_CIM_ROM_DATASET00_F00[level][word_index];
    case 1: return HDC_CIM_ROM_DATASET00_F01[level][word_index];
    case 2: return HDC_CIM_ROM_DATASET00_F02[level][word_index];
    case 3: return HDC_CIM_ROM_DATASET00_F03[level][word_index];
    case 4: return HDC_CIM_ROM_DATASET00_F04[level][word_index];
    case 5: return HDC_CIM_ROM_DATASET00_F05[level][word_index];
    case 6: return HDC_CIM_ROM_DATASET00_F06[level][word_index];
    case 7: return HDC_CIM_ROM_DATASET00_F07[level][word_index];
    case 8: return HDC_CIM_ROM_DATASET00_F08[level][word_index];
    case 9: return HDC_CIM_ROM_DATASET00_F09[level][word_index];
    case 10: return HDC_CIM_ROM_DATASET00_F10[level][word_index];
    case 11: return HDC_CIM_ROM_DATASET00_F11[level][word_index];
    case 12: return HDC_CIM_ROM_DATASET00_F12[level][word_index];
    case 13: return HDC_CIM_ROM_DATASET00_F13[level][word_index];
    case 14: return HDC_CIM_ROM_DATASET00_F14[level][word_index];
    case 15: return HDC_CIM_ROM_DATASET00_F15[level][word_index];
    case 16: return HDC_CIM_ROM_DATASET00_F16[level][word_index];
    case 17: return HDC_CIM_ROM_DATASET00_F17[level][word_index];
    case 18: return HDC_CIM_ROM_DATASET00_F18[level][word_index];
    case 19: return HDC_CIM_ROM_DATASET00_F19[level][word_index];
    case 20: return HDC_CIM_ROM_DATASET00_F20[level][word_index];
    case 21: return HDC_CIM_ROM_DATASET00_F21[level][word_index];
    case 22: return HDC_CIM_ROM_DATASET00_F22[level][word_index];
    case 23: return HDC_CIM_ROM_DATASET00_F23[level][word_index];
    case 24: return HDC_CIM_ROM_DATASET00_F24[level][word_index];
    case 25: return HDC_CIM_ROM_DATASET00_F25[level][word_index];
    case 26: return HDC_CIM_ROM_DATASET00_F26[level][word_index];
    case 27: return HDC_CIM_ROM_DATASET00_F27[level][word_index];
    case 28: return HDC_CIM_ROM_DATASET00_F28[level][word_index];
    case 29: return HDC_CIM_ROM_DATASET00_F29[level][word_index];
    case 30: return HDC_CIM_ROM_DATASET00_F30[level][word_index];
    default: return HDC_CIM_ROM_DATASET00_F31[level][word_index];
    }
}

hv_word_t encode_sample_word(const sample_bits_t &sample, unsigned word_index) {
    hv_word_t encoded_word = 0;
    const feature_score_t signed_threshold =
        (NUM_FEATURES % 2 == 1) ? feature_score_t(-1) : feature_score_t(0);

#define HDC_LOAD_FEATURE_WORD(index) \
    const hv_word_t feature_word_##index = \
        get_cim_feature_word(index, get_sample_level(sample, index).to_uint(), word_index)
    HDC_LOAD_FEATURE_WORD(0);
    HDC_LOAD_FEATURE_WORD(1);
    HDC_LOAD_FEATURE_WORD(2);
    HDC_LOAD_FEATURE_WORD(3);
    HDC_LOAD_FEATURE_WORD(4);
    HDC_LOAD_FEATURE_WORD(5);
    HDC_LOAD_FEATURE_WORD(6);
    HDC_LOAD_FEATURE_WORD(7);
    HDC_LOAD_FEATURE_WORD(8);
    HDC_LOAD_FEATURE_WORD(9);
    HDC_LOAD_FEATURE_WORD(10);
    HDC_LOAD_FEATURE_WORD(11);
    HDC_LOAD_FEATURE_WORD(12);
    HDC_LOAD_FEATURE_WORD(13);
    HDC_LOAD_FEATURE_WORD(14);
    HDC_LOAD_FEATURE_WORD(15);
    HDC_LOAD_FEATURE_WORD(16);
    HDC_LOAD_FEATURE_WORD(17);
    HDC_LOAD_FEATURE_WORD(18);
    HDC_LOAD_FEATURE_WORD(19);
    HDC_LOAD_FEATURE_WORD(20);
    HDC_LOAD_FEATURE_WORD(21);
    HDC_LOAD_FEATURE_WORD(22);
    HDC_LOAD_FEATURE_WORD(23);
    HDC_LOAD_FEATURE_WORD(24);
    HDC_LOAD_FEATURE_WORD(25);
    HDC_LOAD_FEATURE_WORD(26);
    HDC_LOAD_FEATURE_WORD(27);
    HDC_LOAD_FEATURE_WORD(28);
    HDC_LOAD_FEATURE_WORD(29);
    HDC_LOAD_FEATURE_WORD(30);
    HDC_LOAD_FEATURE_WORD(31);
#undef HDC_LOAD_FEATURE_WORD

    for (unsigned bit = 0; bit < HV_WORD_BITS; ++bit) {
        HLS_UNROLL_LOOP(ON, "encode-word-bits-loop");
        feature_score_t score = 0;
#define HDC_ACCUM_FEATURE_WORD(index) \
        if (((feature_word_##index >> bit) & hv_word_t(1)) != 0) { \
            ++score; \
        } else { \
            --score; \
        }
        HDC_ACCUM_FEATURE_WORD(0);
        HDC_ACCUM_FEATURE_WORD(1);
        HDC_ACCUM_FEATURE_WORD(2);
        HDC_ACCUM_FEATURE_WORD(3);
        HDC_ACCUM_FEATURE_WORD(4);
        HDC_ACCUM_FEATURE_WORD(5);
        HDC_ACCUM_FEATURE_WORD(6);
        HDC_ACCUM_FEATURE_WORD(7);
        HDC_ACCUM_FEATURE_WORD(8);
        HDC_ACCUM_FEATURE_WORD(9);
        HDC_ACCUM_FEATURE_WORD(10);
        HDC_ACCUM_FEATURE_WORD(11);
        HDC_ACCUM_FEATURE_WORD(12);
        HDC_ACCUM_FEATURE_WORD(13);
        HDC_ACCUM_FEATURE_WORD(14);
        HDC_ACCUM_FEATURE_WORD(15);
        HDC_ACCUM_FEATURE_WORD(16);
        HDC_ACCUM_FEATURE_WORD(17);
        HDC_ACCUM_FEATURE_WORD(18);
        HDC_ACCUM_FEATURE_WORD(19);
        HDC_ACCUM_FEATURE_WORD(20);
        HDC_ACCUM_FEATURE_WORD(21);
        HDC_ACCUM_FEATURE_WORD(22);
        HDC_ACCUM_FEATURE_WORD(23);
        HDC_ACCUM_FEATURE_WORD(24);
        HDC_ACCUM_FEATURE_WORD(25);
        HDC_ACCUM_FEATURE_WORD(26);
        HDC_ACCUM_FEATURE_WORD(27);
        HDC_ACCUM_FEATURE_WORD(28);
        HDC_ACCUM_FEATURE_WORD(29);
        HDC_ACCUM_FEATURE_WORD(30);
        HDC_ACCUM_FEATURE_WORD(31);
#undef HDC_ACCUM_FEATURE_WORD

        encoded_word[bit] = (score >= signed_threshold) ? 1 : 0;
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
#ifdef STRATUS_HLS
      cmd("cmd"),
      rsp("rsp"),
#else
      cmd_valid("cmd_valid"),
      cmd_ready("cmd_ready"),
      cmd_kind("cmd_kind"),
      cmd_class_id("cmd_class_id"),
      rsp_valid("rsp_valid"),
      rsp_ready("rsp_ready"),
      rsp_valid_prediction("rsp_valid_prediction"),
#endif
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
    cmd.clk_rst(clk, rst, true);
    rsp.clk_rst(clk, rst, true);
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

#ifndef STRATUS_HLS
    reset_training_debug_counters();
#endif
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

#ifndef STRATUS_HLS
void HDC_Accelerator::reset_training_debug_counters() {
    m_debug_encoder_out_train_tokens = 0;
    m_debug_bundler_train_invalid_tokens = 0;
    m_debug_bundler_train_valid_tokens = 0;
    m_debug_bundler_invalid_training_step_tokens = 0;
    m_debug_train_valid_ngram_tokens = 0;
    m_debug_train_invalid_training_step_tokens = 0;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        m_debug_encoder_valid_popcount[class_id] = 0;
        m_debug_encoder_valid_weighted_sum[class_id] = 0;
        m_debug_encoder_valid_first_popcount[class_id] = 0;
        m_debug_encoder_valid_last_popcount[class_id] = 0;
        m_debug_encoder_valid_seen[class_id] = false;
        m_debug_bundler_valid_popcount[class_id] = 0;
        m_debug_bundler_valid_weighted_sum[class_id] = 0;
        m_debug_bundler_valid_first_popcount[class_id] = 0;
        m_debug_bundler_valid_last_popcount[class_id] = 0;
        m_debug_bundler_valid_seen[class_id] = false;
        m_debug_train_valid_popcount[class_id] = 0;
        m_debug_train_valid_weighted_sum[class_id] = 0;
        m_debug_train_valid_first_popcount[class_id] = 0;
        m_debug_train_valid_last_popcount[class_id] = 0;
        m_debug_train_valid_seen[class_id] = false;
    }
}

void HDC_Accelerator::print_training_debug_counters(std::ostream &out) const {
    out << "debug training_path_counters"
        << " encoder_out_train=" << m_debug_encoder_out_train_tokens
        << " bundler_train_invalid=" << m_debug_bundler_train_invalid_tokens
        << " bundler_train_valid=" << m_debug_bundler_train_valid_tokens
        << " bundler_invalid_training_step=" << m_debug_bundler_invalid_training_step_tokens
        << " train_valid_ngram=" << m_debug_train_valid_ngram_tokens
        << " train_invalid_training_step=" << m_debug_train_invalid_training_step_tokens
        << std::endl;
    out << "debug encoder_payload_popcount";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "=" << m_debug_encoder_valid_popcount[class_id];
    }
    out << std::endl;
    out << "debug encoder_payload_weighted_sum";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "=" << m_debug_encoder_valid_weighted_sum[class_id];
    }
    out << std::endl;
    out << "debug encoder_payload_first_last";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "_first=" << m_debug_encoder_valid_first_popcount[class_id]
            << " c" << class_id << "_last=" << m_debug_encoder_valid_last_popcount[class_id];
    }
    out << std::endl;
    out << "debug bundler_payload_popcount";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "=" << m_debug_bundler_valid_popcount[class_id];
    }
    out << std::endl;
    out << "debug bundler_payload_weighted_sum";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "=" << m_debug_bundler_valid_weighted_sum[class_id];
    }
    out << std::endl;
    out << "debug bundler_payload_first_last";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "_first=" << m_debug_bundler_valid_first_popcount[class_id]
            << " c" << class_id << "_last=" << m_debug_bundler_valid_last_popcount[class_id];
    }
    out << std::endl;
    out << "debug train_payload_popcount";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "=" << m_debug_train_valid_popcount[class_id];
    }
    out << std::endl;
    out << "debug train_payload_weighted_sum";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "=" << m_debug_train_valid_weighted_sum[class_id];
    }
    out << std::endl;
    out << "debug train_payload_first_last";
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        out << " c" << class_id << "_first=" << m_debug_train_valid_first_popcount[class_id]
            << " c" << class_id << "_last=" << m_debug_train_valid_last_popcount[class_id];
    }
    out << std::endl;
}

void HDC_Accelerator::dump_assoc_mem(std::ostream &out) const {
    out << "# SystemC valid/ready associative-memory dump after replay" << std::endl;
    out << "# scalarized assoc words count=" << (NUM_CLASSES * HV_WORDS)
        << " words_per_class=" << HV_WORDS << std::endl;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        for (unsigned word = 0; word < HV_WORDS; ++word) {
            const unsigned flat = class_id * HV_WORDS + word;
            out << "assoc_word flat=" << flat
                << " class=" << class_id
                << " word=" << word
                << " value=" << std::hex << std::uppercase << std::setw(16)
                << std::setfill('0') << m_assoc_mem[class_id].words[word].to_uint64()
                << std::dec << std::nouppercase << std::setfill(' ') << std::endl;
        }
    }
}
#endif

// Data commands are pipelined: TrainSample and InferSample are dispatched
// without waiting for completion. Control commands are blocking stream
// boundaries and wait until their token passes through the internal pipeline.
void HDC_Accelerator::command_thread() {
#ifdef STRATUS_HLS
    enum CommandState {
        CMD_GET_COMMAND,
        CMD_SEND_PENDING,
        CMD_WAIT_NGRAM_DONE,
        CMD_WAIT_TRAIN_DONE
    };

    EncoderChannelPacket send_packet;
    CommandState state = CMD_GET_COMMAND;
    bool pending_ngram_control = false;
    bool pending_train_control = false;

    {
        HLS_DEFINE_PROTOCOL("command_reset");
        state = CMD_GET_COMMAND;
        pending_ngram_control = false;
        pending_train_control = false;
        cmd.reset();
        m_encoder_in.input.reset();
        m_ngram_control_done.output.reset();
        m_train_control_done.output.reset();
        wait();
    }

    while (true) {
        {
            if (state == CMD_GET_COMMAND) {
                const command_channel_bits_t command_word = cmd.get();
                EncoderChannelPacket packet = unpack_command_channel(command_word);
                const AccelCommandKind command_kind = decode_kind(packet.kind);

                if (command_kind == AccelCommandKind::InferSample) {
                    packet.class_id = 0;
                }

                pending_ngram_control = is_ngram_control_command(command_kind);
                pending_train_control = is_train_control_command(command_kind);
                send_packet = packet;
                state = CMD_SEND_PENDING;
            } else if (state == CMD_SEND_PENDING) {
                m_encoder_in.input.put(send_packet);
                if (pending_ngram_control) {
                    state = CMD_WAIT_NGRAM_DONE;
                } else if (pending_train_control) {
                    state = CMD_WAIT_TRAIN_DONE;
                } else {
                    state = CMD_GET_COMMAND;
                }
            } else if (state == CMD_WAIT_NGRAM_DONE) {
                (void)m_ngram_control_done.output.get();
                pending_ngram_control = false;
                state = CMD_GET_COMMAND;
            } else {
                (void)m_train_control_done.output.get();
                pending_train_control = false;
                state = CMD_GET_COMMAND;
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
        reset_training_debug_counters();
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
                        if (output_packet.kind == AccelCommandKind::TrainSample) {
                            ++m_debug_encoder_out_train_tokens;
                            const unsigned class_id = output_packet.class_id.to_uint();
                            const unsigned long long popcount =
                                hv_popcount_debug(output_packet.encoded);
                            m_debug_encoder_valid_popcount[class_id] += popcount;
                            m_debug_encoder_valid_weighted_sum[class_id] +=
                                hv_weighted_sum_debug(output_packet.encoded);
                            if (!m_debug_encoder_valid_seen[class_id]) {
                                m_debug_encoder_valid_first_popcount[class_id] = popcount;
                                m_debug_encoder_valid_seen[class_id] = true;
                            }
                            m_debug_encoder_valid_last_popcount[class_id] = popcount;
                        }
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
        NGRAM_ROTATE,
        NGRAM_XOR,
        NGRAM_SEND,
        NGRAM_SEND_CONTROL
    };

    NGramChannelPacket output_packet;
    EncoderChannelPacket work_packet;
    hv_bits_t work_bits = 0;
    hv_bits_t rotated_bits = 0;
    hv_bits_t rhs_bits = 0;
    NGramChannelPacket send_packet;
    NGramState state = NGRAM_INIT_CLEAR;
    unsigned send_target = OUTPUT_NONE;
    unsigned bind_round = 0;
    unsigned oldest_slot = 0;
    unsigned reset_slot = 0;

    {
        HLS_DEFINE_PROTOCOL("ngram_reset");
        m_ngram_buffer_write_pos = 0;
        m_ngram_buffer_fill_count = 0;
        state = NGRAM_INIT_CLEAR;
        send_target = OUTPUT_NONE;
        bind_round = 0;
        oldest_slot = 0;
        reset_slot = 0;
        work_bits = 0;
        rotated_bits = 0;
        rhs_bits = 0;
        m_encoder_out.output.reset();
        m_bundler_in.input.reset();
        m_distance_in.input.reset();
        m_ngram_control_done.input.reset();
        wait();
    }

    while (true) {
        {
            if (state == NGRAM_INIT_CLEAR) {
                m_ngram_buffer_bits[reset_slot] = 0;
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
            } else if (state == NGRAM_ROTATE) {
                const unsigned rhs_slot = (oldest_slot + bind_round) % N_GRAM_SIZE;
                rotated_bits = rotate_left_one(work_bits);
                rhs_bits = m_ngram_buffer_bits[rhs_slot];
                state = NGRAM_XOR;
            } else if (state == NGRAM_XOR) {
                const hv_bits_t next_bits = rotated_bits ^ rhs_bits;
                work_bits = next_bits;

                if (bind_round + 1u == N_GRAM_SIZE) {
                    output_packet = NGramChannelPacket();
                    output_packet.kind = work_packet.kind;
                    output_packet.class_id = work_packet.class_id;
                    output_packet.ngram = next_bits;
                    output_packet.valid_ngram = true;
                    send_packet = output_packet;
                    send_target = (decode_kind(work_packet.kind) == AccelCommandKind::TrainSample)
                                      ? OUTPUT_BUNDLER
                                      : OUTPUT_DISTANCE;
                    bind_round = 0;
                    state = NGRAM_SEND;
                } else {
                    bind_round = bind_round + 1u;
                    state = NGRAM_ROTATE;
                }
            } else {
                EncoderChannelPacket item = m_encoder_out.output.get();
                const AccelCommandKind item_kind = decode_kind(item.kind);

                if (item_kind == AccelCommandKind::ResetTraining) {
                    m_ngram_buffer_write_pos = 0;
                    m_ngram_buffer_fill_count = 0;
                    output_packet = NGramChannelPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    send_packet = output_packet;
                    send_target = OUTPUT_BUNDLER;
                    state = NGRAM_SEND;
                } else if (item_kind == AccelCommandKind::ResetInference) {
                    m_ngram_buffer_write_pos = 0;
                    m_ngram_buffer_fill_count = 0;
                    state = NGRAM_SEND_CONTROL;
                } else if (item_kind == AccelCommandKind::InvalidTrainingStep) {
                    m_ngram_buffer_write_pos = 0;
                    m_ngram_buffer_fill_count = 0;
                    output_packet = NGramChannelPacket();
                    output_packet.kind = item.kind;
                    output_packet.class_id = item.class_id;
                    output_packet.valid_ngram = false;
                    send_packet = output_packet;
                    send_target = OUTPUT_BUNDLER;
                    state = NGRAM_SEND;
                } else if (item_kind == AccelCommandKind::TrainSample ||
                           item_kind == AccelCommandKind::InferSample) {
                    m_ngram_buffer_bits[m_ngram_buffer_write_pos] = item.encoded;
                    ++m_ngram_buffer_write_pos;
                    if (m_ngram_buffer_write_pos == N_GRAM_SIZE) {
                        m_ngram_buffer_write_pos = 0;
                    }
                    if (m_ngram_buffer_fill_count < N_GRAM_SIZE) {
                        ++m_ngram_buffer_fill_count;
                    }

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
                        work_bits = m_ngram_buffer_bits[oldest_slot];
                        bind_round = (N_GRAM_SIZE > 1) ? 1u : N_GRAM_SIZE;

                        if (N_GRAM_SIZE > 1) {
                            state = NGRAM_ROTATE;
                        } else {
                            output_packet.kind = item.kind;
                            output_packet.class_id = item.class_id;
                            output_packet.ngram = work_bits;
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
                        if (output_target == OUTPUT_BUNDLER) {
                            ++m_debug_bundler_train_valid_tokens;
                            const unsigned class_id = output_packet.class_id.to_uint();
                            const unsigned long long popcount = hv_popcount_debug(output_packet.ngram);
                            m_debug_bundler_valid_popcount[class_id] += popcount;
                            m_debug_bundler_valid_weighted_sum[class_id] +=
                                hv_weighted_sum_debug(output_packet.ngram);
                            if (!m_debug_bundler_valid_seen[class_id]) {
                                m_debug_bundler_valid_first_popcount[class_id] = popcount;
                                m_debug_bundler_valid_seen[class_id] = true;
                            }
                            m_debug_bundler_valid_last_popcount[class_id] = popcount;
                        }
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
                    ++m_debug_bundler_invalid_training_step_tokens;
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
                        if (output_target == OUTPUT_BUNDLER) {
                            ++m_debug_bundler_train_invalid_tokens;
                        }
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
                            if (output_target == OUTPUT_BUNDLER) {
                                ++m_debug_bundler_train_valid_tokens;
                                const unsigned class_id = output_packet.class_id.to_uint();
                                const unsigned long long popcount = hv_popcount_debug(output_packet.ngram);
                                m_debug_bundler_valid_popcount[class_id] += popcount;
                                m_debug_bundler_valid_weighted_sum[class_id] +=
                                    hv_weighted_sum_debug(output_packet.ngram);
                                if (!m_debug_bundler_valid_seen[class_id]) {
                                    m_debug_bundler_valid_first_popcount[class_id] = popcount;
                                    m_debug_bundler_valid_seen[class_id] = true;
                                }
                                m_debug_bundler_valid_last_popcount[class_id] = popcount;
                            }
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

#define HDC_APPLY_TO_HV_BIT_BANKS(M) \
    M(0)                             \
    M(1)                             \
    M(2)                             \
    M(3)                             \
    M(4)                             \
    M(5)                             \
    M(6)                             \
    M(7)                             \
    M(8)                             \
    M(9)                             \
    M(10)                            \
    M(11)                            \
    M(12)                            \
    M(13)                            \
    M(14)                            \
    M(15)                            \
    M(16)                            \
    M(17)                            \
    M(18)                            \
    M(19)                            \
    M(20)                            \
    M(21)                            \
    M(22)                            \
    M(23)                            \
    M(24)                            \
    M(25)                            \
    M(26)                            \
    M(27)                            \
    M(28)                            \
    M(29)                            \
    M(30)                            \
    M(31)                            \
    M(32)                            \
    M(33)                            \
    M(34)                            \
    M(35)                            \
    M(36)                            \
    M(37)                            \
    M(38)                            \
    M(39)                            \
    M(40)                            \
    M(41)                            \
    M(42)                            \
    M(43)                            \
    M(44)                            \
    M(45)                            \
    M(46)                            \
    M(47)                            \
    M(48)                            \
    M(49)                            \
    M(50)                            \
    M(51)                            \
    M(52)                            \
    M(53)                            \
    M(54)                            \
    M(55)                            \
    M(56)                            \
    M(57)                            \
    M(58)                            \
    M(59)                            \
    M(60)                            \
    M(61)                            \
    M(62)                            \
    M(63)

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
#define HDC_CLEAR_SCORE_BANK(index) m_bundling_score_##index[word_index] = 0;
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_CLEAR_SCORE_BANK)
#undef HDC_CLEAR_SCORE_BANK

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
                const hv_word_t word = get_hv_word(work.ngram, word_index);
#define HDC_ADD_SCORE_BANK(index)                         \
                if (((word >> index) & hv_word_t(1)) != 0) { \
                    ++m_bundling_score_##index[word_index];  \
                } else {                                      \
                    --m_bundling_score_##index[word_index];  \
                }
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_ADD_SCORE_BANK)
#undef HDC_ADD_SCORE_BANK

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
                hv_word_t class_word = 0;
#define HDC_FINALIZE_SCORE_BANK(index)                         \
                if (m_bundling_score_##index[word_index] >= signed_threshold) { \
                    class_word[index] = 1;                         \
                }                                                  \
                m_bundling_score_##index[word_index] = 0;
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_FINALIZE_SCORE_BANK)
#undef HDC_FINALIZE_SCORE_BANK
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
#define HDC_CLEAR_SCORE_BANK(index) m_bundling_score_##index[word_index] = 0;
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_CLEAR_SCORE_BANK)
#undef HDC_CLEAR_SCORE_BANK

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
#define HDC_CLEAR_SCORE_BANK(index) m_bundling_score_##index[word_index] = 0;
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_CLEAR_SCORE_BANK)
#undef HDC_CLEAR_SCORE_BANK

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
                const hv_word_t word = work.ngram.words[word_index];
#define HDC_ADD_SCORE_BANK(index)                         \
                if (((word >> index) & hv_word_t(1)) != 0) { \
                    ++m_bundling_score_##index[word_index];  \
                } else {                                      \
                    --m_bundling_score_##index[word_index];  \
                }
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_ADD_SCORE_BANK)
#undef HDC_ADD_SCORE_BANK

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
                hv_word_t class_word = 0;
#define HDC_FINALIZE_SCORE_BANK(index)                         \
                if (m_bundling_score_##index[word_index] >= signed_threshold) { \
                    class_word[index] = 1;                         \
                }                                                  \
                m_bundling_score_##index[word_index] = 0;
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_FINALIZE_SCORE_BANK)
#undef HDC_FINALIZE_SCORE_BANK
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
#define HDC_CLEAR_SCORE_BANK(index) m_bundling_score_##index[word_index] = 0;
                HDC_APPLY_TO_HV_BIT_BANKS(HDC_CLEAR_SCORE_BANK)
#undef HDC_CLEAR_SCORE_BANK

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
                        ++m_debug_train_valid_ngram_tokens;
                        const unsigned class_id = item.class_id.to_uint();
                        const unsigned long long popcount = hv_popcount_debug(item.ngram);
                        m_debug_train_valid_popcount[class_id] += popcount;
                        m_debug_train_valid_weighted_sum[class_id] +=
                            hv_weighted_sum_debug(item.ngram);
                        if (!m_debug_train_valid_seen[class_id]) {
                            m_debug_train_valid_first_popcount[class_id] = popcount;
                            m_debug_train_valid_seen[class_id] = true;
                        }
                        m_debug_train_valid_last_popcount[class_id] = popcount;
                        work = item;
                        if (!m_current_class_valid) {
                            m_current_class_id = item.class_id;
                            m_current_class_valid = true;
                        }
                        word_index = 0;
                        state = TRAIN_ADD_NGRAM;
                    }
                } else if (item.kind == AccelCommandKind::InvalidTrainingStep) {
                    ++m_debug_train_invalid_training_step_tokens;
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

#undef HDC_APPLY_TO_HV_BIT_BANKS

void HDC_Accelerator::distance_thread() {
#ifdef STRATUS_HLS
    static_assert(NUM_CLASSES == 5,
                  "class-parallel distance path assumes five classes");
    enum DistanceState {
        DIST_WAIT_INPUT,
        DIST_COMPUTE,
        DIST_SEND
    };

    NGramChannelPacket work;
    DistanceChannelPacket send_packet;
    DistanceState state = DIST_WAIT_INPUT;
    unsigned word_index = 0;
    distance_counter_t distance_acc0 = 0;
    distance_counter_t distance_acc1 = 0;
    distance_counter_t distance_acc2 = 0;
    distance_counter_t distance_acc3 = 0;
    distance_counter_t distance_acc4 = 0;

    {
        HLS_DEFINE_PROTOCOL("distance_reset");
        state = DIST_WAIT_INPUT;
        word_index = 0;
        distance_acc0 = 0;
        distance_acc1 = 0;
        distance_acc2 = 0;
        distance_acc3 = 0;
        distance_acc4 = 0;
        send_packet.valid_prediction = false;
        send_packet.distances = 0;
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
                    send_packet.valid_prediction = false;
                    send_packet.distances = 0;
                    state = DIST_SEND;
                } else {
                    work = item;
                    word_index = 0;
                    distance_acc0 = 0;
                    distance_acc1 = 0;
                    distance_acc2 = 0;
                    distance_acc3 = 0;
                    distance_acc4 = 0;
                    state = DIST_COMPUTE;
                }
            } else {
                hv_word_t assoc_word0 = 0;
                hv_word_t assoc_word1 = 0;
                hv_word_t assoc_word2 = 0;
                hv_word_t assoc_word3 = 0;
                hv_word_t assoc_word4 = 0;
                {
                    HLS_DEFINE_PROTOCOL("distance_assoc_read_word");
                    assoc_word0 = m_assoc_mem[0].words[word_index];
                    assoc_word1 = m_assoc_mem[1].words[word_index];
                    assoc_word2 = m_assoc_mem[2].words[word_index];
                    assoc_word3 = m_assoc_mem[3].words[word_index];
                    assoc_word4 = m_assoc_mem[4].words[word_index];
                }

                const hv_word_t ngram_word = get_hv_word(work.ngram, word_index);
                const distance_counter_t next_distance0 =
                    distance_acc0 + popcount_word(ngram_word ^ assoc_word0);
                const distance_counter_t next_distance1 =
                    distance_acc1 + popcount_word(ngram_word ^ assoc_word1);
                const distance_counter_t next_distance2 =
                    distance_acc2 + popcount_word(ngram_word ^ assoc_word2);
                const distance_counter_t next_distance3 =
                    distance_acc3 + popcount_word(ngram_word ^ assoc_word3);
                const distance_counter_t next_distance4 =
                    distance_acc4 + popcount_word(ngram_word ^ assoc_word4);

                if (word_index == (HV_WORDS - 1u)) {
                    distance_bits_t final_distances = 0;
                    set_distance_word(final_distances, 0, next_distance0);
                    set_distance_word(final_distances, 1, next_distance1);
                    set_distance_word(final_distances, 2, next_distance2);
                    set_distance_word(final_distances, 3, next_distance3);
                    set_distance_word(final_distances, 4, next_distance4);
                    word_index = 0;
                    distance_acc0 = 0;
                    distance_acc1 = 0;
                    distance_acc2 = 0;
                    distance_acc3 = 0;
                    distance_acc4 = 0;
                    send_packet.valid_prediction = true;
                    send_packet.distances = final_distances;
                    state = DIST_SEND;
                } else {
                    word_index = word_index + 1u;
                    distance_acc0 = next_distance0;
                    distance_acc1 = next_distance1;
                    distance_acc2 = next_distance2;
                    distance_acc3 = next_distance3;
                    distance_acc4 = next_distance4;
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
#ifdef STRATUS_HLS
    enum ResponseState {
        RSP_WAIT_PACKET,
        RSP_SEND_PACKET
    };

    ResponseState state = RSP_WAIT_PACKET;
    response_channel_bits_t response_word = 0;

    {
        HLS_DEFINE_PROTOCOL("response_reset");
        state = RSP_WAIT_PACKET;
        response_word = 0;
        m_distance_done.output.reset();
        rsp.reset();
        wait();
    }

    while (true) {
        {
            if (state == RSP_WAIT_PACKET) {
                const DistanceChannelPacket channel_packet = m_distance_done.output.get();
                response_word = pack_response_channel(channel_packet);
                state = RSP_SEND_PACKET;
            } else {
                rsp.put(response_word);
                state = RSP_WAIT_PACKET;
            }

            {
                HLS_DEFINE_PROTOCOL("response_wait");
                wait();
            }
        }
    }

#else
    DistancePacket work;
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

#ifndef STRATUS_HLS
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
#endif
