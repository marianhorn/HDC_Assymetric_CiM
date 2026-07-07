#include "p2p_pipeline.h"

namespace {

void fill_token(P2PToken &token,
                const sc_dt::sc_uint<3> &kind,
                const sc_dt::sc_uint<8> &value) {
    token.kind = kind;
    token.class_id = kind;
    token.value = value;
#if P2P_HAS_SAMPLE_PAYLOAD
    for (unsigned index = 0; index < P2P_SAMPLE_LEVELS; ++index) {
        token.sample_levels[index] =
            (value + kind + static_cast<unsigned>(3u * index)) & 0xffu;
    }
#endif
#if P2P_HAS_ENCODED_PAYLOAD
    for (unsigned index = 0; index < P2P_HV_WORDS; ++index) {
        sc_dt::sc_uint<64> word = 0;
        word.range(7, 0) = value;
        word.range(15, 8) = static_cast<unsigned>(index);
        word.range(18, 16) = kind;
        word.range(31, 19) = static_cast<unsigned>(0x123u + index);
        word.range(63, 32) = static_cast<unsigned>(0xabc00000u + (index << 4));
        token.encoded_words[index] = word;
    }
#endif
}

sc_dt::sc_uint<16> sample_checksum(const P2PToken &token) {
    sc_dt::sc_uint<16> checksum = 0;
#if P2P_HAS_SAMPLE_PAYLOAD
    for (unsigned index = 0; index < P2P_SAMPLE_LEVELS; ++index) {
        checksum = checksum + token.sample_levels[index];
    }
#else
    (void)token;
#endif
    return checksum;
}

sc_dt::sc_uint<16> encoded_checksum(const P2PToken &token) {
    sc_dt::sc_uint<16> checksum = 0;
#if P2P_HAS_ENCODED_PAYLOAD
    for (unsigned index = 0; index < P2P_HV_WORDS; ++index) {
        checksum = checksum ^ token.encoded_words[index].range(15, 0);
        checksum = checksum ^ token.encoded_words[index].range(31, 16);
        checksum = checksum ^ token.encoded_words[index].range(47, 32);
        checksum = checksum ^ token.encoded_words[index].range(63, 48);
    }
#else
    (void)token;
#endif
    return checksum;
}

P2PToken transform_token(const P2PToken &input) {
    P2PToken output = input;
    output.value = input.value + 1u;
    return output;
}

}  // namespace

void P2PPipeline::source_thread() {
    enum SourceState {
        SOURCE_WAIT_INPUT,
        SOURCE_PUT
    };

    SourceState state = SOURCE_WAIT_INPUT;
    P2PToken pending;
    sc_dt::sc_uint<16> count = 0;

    {
        HLS_DEFINE_PROTOCOL("source_reset");
        in_ready.write(false);
        source_count.write(0);
        state = SOURCE_WAIT_INPUT;
        m_source_to_stage.input.reset();
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("source_cycle");

            if (state == SOURCE_WAIT_INPUT) {
                in_ready.write(true);
                if (in_valid.read()) {
                    fill_token(pending, in_kind.read(), in_value.read());
                    state = SOURCE_PUT;
                }
            } else {
                in_ready.write(false);
#ifdef P2P_EXPERIMENT_NB
                if (m_source_to_stage.input.nb_can_put()) {
                    m_source_to_stage.input.nb_put(pending);
                    state = SOURCE_WAIT_INPUT;
                    count = count + 1u;
                }
#else
                m_source_to_stage.input.put(pending);
                state = SOURCE_WAIT_INPUT;
                count = count + 1u;
#endif
            }

            source_count.write(count);
            wait();
        }
    }
}

void P2PPipeline::stage_thread() {
    enum StageState {
        STAGE_GET,
        STAGE_PUT
    };

    StageState state = STAGE_GET;
    P2PToken work;
    sc_dt::sc_uint<16> count = 0;

    {
        HLS_DEFINE_PROTOCOL("stage_reset");
        stage_count.write(0);
        m_source_to_stage.output.reset();
        m_stage_to_sink.input.reset();
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("stage_cycle");

            if (state == STAGE_GET) {
#ifdef P2P_EXPERIMENT_NB
                P2PToken input;
                if (m_source_to_stage.output.nb_can_get()) {
                    m_source_to_stage.output.nb_get(input);
                    work = transform_token(input);
                    state = STAGE_PUT;
                }
#else
                const P2PToken input = m_source_to_stage.output.get();
                work = transform_token(input);
                state = STAGE_PUT;
#endif
            } else {
#ifdef P2P_EXPERIMENT_NB
                if (m_stage_to_sink.input.nb_can_put()) {
                    m_stage_to_sink.input.nb_put(work);
                    count = count + 1u;
                    state = STAGE_GET;
                }
#else
                m_stage_to_sink.input.put(work);
                count = count + 1u;
                state = STAGE_GET;
#endif
            }

            stage_count.write(count);
            wait();
        }
    }
}

void P2PPipeline::sink_thread() {
    enum SinkState {
        SINK_GET,
        SINK_HOLD
    };

    SinkState state = SINK_GET;
    P2PToken work;
    sc_dt::sc_uint<16> count = 0;

    {
        HLS_DEFINE_PROTOCOL("sink_reset");
        out_valid.write(false);
        out_kind.write(0);
        out_value.write(0);
        out_sample_checksum.write(0);
        out_encoded_checksum.write(0);
        sink_count.write(0);
        m_stage_to_sink.output.reset();
        wait();
    }

    while (true) {
        {
            HLS_DEFINE_PROTOCOL("sink_cycle");

            if (state == SINK_GET) {
                out_valid.write(false);
#ifdef P2P_EXPERIMENT_NB
                P2PToken input;
                if (m_stage_to_sink.output.nb_can_get()) {
                    m_stage_to_sink.output.nb_get(input);
                    work = input;
                    state = SINK_HOLD;
                }
#else
                work = m_stage_to_sink.output.get();
                state = SINK_HOLD;
#endif
            } else {
                out_valid.write(true);
                out_kind.write(work.kind);
                out_value.write(work.value);
                out_sample_checksum.write(sample_checksum(work));
                out_encoded_checksum.write(encoded_checksum(work));

                if (out_ready.read()) {
                    count = count + 1u;
                    state = SINK_GET;
                }
            }

            sink_count.write(count);
            wait();
        }
    }
}
