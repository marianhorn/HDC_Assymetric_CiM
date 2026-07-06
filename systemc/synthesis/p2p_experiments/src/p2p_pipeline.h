#ifndef P2P_EXPERIMENTS_P2P_PIPELINE_H
#define P2P_EXPERIMENTS_P2P_PIPELINE_H

#include <systemc.h>
#include <cynw_p2p.h>

#if !defined(STRATUS_HLS) && !defined(HLS_DEFINE_PROTOCOL)
#define HLS_DEFINE_PROTOCOL(name) ((void)0)
#endif

struct P2PToken {
    sc_dt::sc_uint<3> kind;
    sc_dt::sc_uint<8> value;

    P2PToken() : kind(0), value(0) {}
};

inline bool operator==(const P2PToken &lhs, const P2PToken &rhs) {
    return lhs.kind == rhs.kind && lhs.value == rhs.value;
}

inline std::ostream &operator<<(std::ostream &os, const P2PToken &token) {
    os << "{kind=" << token.kind << ", value=" << token.value << "}";
    return os;
}

inline void sc_trace(sc_core::sc_trace_file *tf, const P2PToken &token,
                     const std::string &name) {
    sc_core::sc_trace(tf, token.kind, name + ".kind");
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
          source_count("source_count"),
          stage_count("stage_count"),
          sink_count("sink_count"),
          m_source_to_stage("m_source_to_stage"),
          m_stage_to_sink("m_stage_to_sink") {
        m_source_to_stage.clk_rst(clk, rst);
        m_stage_to_sink.clk_rst(clk, rst);

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

