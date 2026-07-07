#ifndef P2P_EXPERIMENTS_P2P_PIPELINE_H
#define P2P_EXPERIMENTS_P2P_PIPELINE_H

#include <systemc.h>
#include <cynw_p2p.h>

#if !defined(STRATUS_HLS) && !defined(HLS_DEFINE_PROTOCOL)
#define HLS_DEFINE_PROTOCOL(name) ((void)0)
#endif

#if !defined(STRATUS_HLS) && !defined(HLS_UNROLL_LOOP)
#define HLS_UNROLL_LOOP(mode, name) ((void)0)
#endif

#if defined(P2P_INTERNAL_SOURCE)
#if !defined(P2P_ENCODER_SCALAR_MIMIC)
#define P2P_ENCODER_SCALAR_MIMIC
#endif
#endif

#if defined(P2P_ENCODER_SCALAR_MIMIC)
#if !defined(P2P_ENCODER_MIMIC)
#define P2P_ENCODER_MIMIC
#endif
#endif

static const unsigned P2P_INTERNAL_SOURCE_TOKENS = 16;

#if defined(P2P_ENCODER_MIMIC_NB)
#if !defined(P2P_ENCODER_MIMIC)
#define P2P_ENCODER_MIMIC
#endif
#if !defined(P2P_EXPERIMENT_NB)
#define P2P_EXPERIMENT_NB
#endif
#endif

static const unsigned P2P_SAMPLE_LEVELS = 32;
static const unsigned P2P_HV_WORDS = 16;

#if defined(P2P_ENCODER_SCALAR_MIMIC)
#define P2P_HAS_SAMPLE_PAYLOAD 0
#define P2P_HAS_ENCODED_PAYLOAD 0
#elif defined(P2P_ENCODER_MIMIC)
#define P2P_HAS_SAMPLE_PAYLOAD 1
#define P2P_HAS_ENCODED_PAYLOAD 1
#elif defined(P2P_PAYLOAD_FULL)
#define P2P_HAS_SAMPLE_PAYLOAD 1
#define P2P_HAS_ENCODED_PAYLOAD 1
#elif defined(P2P_PAYLOAD_SAMPLE)
#define P2P_HAS_SAMPLE_PAYLOAD 1
#define P2P_HAS_ENCODED_PAYLOAD 0
#elif defined(P2P_PAYLOAD_ENCODED)
#define P2P_HAS_SAMPLE_PAYLOAD 0
#define P2P_HAS_ENCODED_PAYLOAD 1
#else
#define P2P_HAS_SAMPLE_PAYLOAD 0
#define P2P_HAS_ENCODED_PAYLOAD 0
#endif

struct P2PToken {
    sc_dt::sc_uint<3> kind;
    sc_dt::sc_uint<3> class_id;
    sc_dt::sc_uint<8> value;
#if defined(P2P_ENCODER_SCALAR_MIMIC)
    sc_dt::sc_uint<16> sample_checksum;
    sc_dt::sc_uint<16> encoded_checksum;
#endif
#if P2P_HAS_SAMPLE_PAYLOAD
    sc_dt::sc_uint<8> sample_levels[P2P_SAMPLE_LEVELS];
#endif
#if P2P_HAS_ENCODED_PAYLOAD
    sc_dt::sc_uint<64> encoded_words[P2P_HV_WORDS];
#endif

    P2PToken() : kind(0), class_id(0), value(0)
#if defined(P2P_ENCODER_SCALAR_MIMIC)
                 ,
                 sample_checksum(0),
                 encoded_checksum(0)
#endif
    {}
};

inline bool operator==(const P2PToken &lhs, const P2PToken &rhs) {
    return lhs.kind == rhs.kind && lhs.class_id == rhs.class_id && lhs.value == rhs.value;
}

inline std::ostream &operator<<(std::ostream &os, const P2PToken &token) {
    os << "{kind=" << token.kind << ", class_id=" << token.class_id
       << ", value=" << token.value << "}";
    return os;
}

inline void sc_trace(sc_core::sc_trace_file *tf, const P2PToken &token,
                     const std::string &name) {
    sc_core::sc_trace(tf, token.kind, name + ".kind");
    sc_core::sc_trace(tf, token.class_id, name + ".class_id");
    sc_core::sc_trace(tf, token.value, name + ".value");
}

SC_MODULE(P2PPipeline) {
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    sc_core::sc_in<bool> in_valid;
    sc_core::sc_out<bool> in_ready;
    sc_core::sc_in<sc_dt::sc_uint<3> > in_kind;
    sc_core::sc_in<sc_dt::sc_uint<8> > in_value;

    sc_core::sc_out<bool> out_valid;
    sc_core::sc_in<bool> out_ready;
    sc_core::sc_out<sc_dt::sc_uint<3> > out_kind;
    sc_core::sc_out<sc_dt::sc_uint<8> > out_value;
    sc_core::sc_out<sc_dt::sc_uint<16> > out_sample_checksum;
    sc_core::sc_out<sc_dt::sc_uint<16> > out_encoded_checksum;

    sc_core::sc_out<sc_dt::sc_uint<16> > source_count;
    sc_core::sc_out<sc_dt::sc_uint<16> > stage_count;
    sc_core::sc_out<sc_dt::sc_uint<16> > sink_count;

    SC_CTOR(P2PPipeline)
        : clk("clk"),
          rst("rst"),
          in_valid("in_valid"),
          in_ready("in_ready"),
          in_kind("in_kind"),
          in_value("in_value"),
          out_valid("out_valid"),
          out_ready("out_ready"),
          out_kind("out_kind"),
          out_value("out_value"),
          out_sample_checksum("out_sample_checksum"),
          out_encoded_checksum("out_encoded_checksum"),
          source_count("source_count"),
          stage_count("stage_count"),
          sink_count("sink_count"),
          m_source_to_stage("m_source_to_stage"),
          m_stage_to_sink("m_stage_to_sink") {
        m_source_to_stage.clk_rst(clk, rst, true);
        m_stage_to_sink.clk_rst(clk, rst, true);

        SC_CTHREAD(source_thread, clk.pos());
        reset_signal_is(rst, true);

        SC_CTHREAD(stage_thread, clk.pos());
        reset_signal_is(rst, true);

        SC_CTHREAD(sink_thread, clk.pos());
        reset_signal_is(rst, true);
    }

    void source_thread();
    void stage_thread();
    void sink_thread();

  private:
    cynw_p2p_direct<P2PToken, CYN::PIN> m_source_to_stage;
    cynw_p2p_direct<P2PToken, CYN::PIN> m_stage_to_sink;
};

#endif
