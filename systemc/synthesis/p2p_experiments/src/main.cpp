#include "p2p_pipeline.h"

#include <cstdlib>
#include <iostream>

namespace {

unsigned expected_sample_checksum(unsigned kind, unsigned value) {
    unsigned checksum = 0;
#if defined(P2P_ENCODER_SCALAR_MIMIC) || P2P_HAS_SAMPLE_PAYLOAD
    for (unsigned index = 0; index < P2P_SAMPLE_LEVELS; ++index) {
        checksum = (checksum + ((value + kind + 3u * index) & 0xffu)) & 0xffffu;
    }
#else
    (void)kind;
    (void)value;
#endif
    return checksum;
}

unsigned expected_encoded_checksum(unsigned kind, unsigned value) {
    unsigned checksum = 0;
#if defined(P2P_ENCODER_MIMIC)
    if (kind != 2u && kind != 4u) {
        return 0;
    }

    const unsigned scalar_sample_checksum = expected_sample_checksum(kind, value);
    for (unsigned word_index = 0; word_index < P2P_HV_WORDS; ++word_index) {
        unsigned acc0 = 0;
        unsigned acc1 = 0;
        unsigned acc2 = 0;
        unsigned acc3 = 0;

        for (unsigned feature = 0; feature < P2P_SAMPLE_LEVELS; ++feature) {
#if defined(P2P_ENCODER_SCALAR_MIMIC)
            const unsigned level = (scalar_sample_checksum + 3u * feature + word_index) & 0xffu;
#else
            const unsigned level = (value + kind + 3u * feature) & 0xffu;
#endif
            const unsigned term = level + kind + kind + feature + word_index;

            acc0 = (acc0 + term) & 0xffffu;
            acc1 = (acc1 ^ ((term << (feature & 3u)) & 0xffffu)) & 0xffffu;
            acc2 = (acc2 + term * (feature + 1u)) & 0xffffu;
            acc3 = (acc3 ^ ((term + word_index * 17u) & 0xffffu)) & 0xffffu;
        }

        checksum ^= acc0;
        checksum ^= acc1;
        checksum ^= acc2;
        checksum ^= acc3;
    }
#elif P2P_HAS_ENCODED_PAYLOAD
    for (unsigned index = 0; index < P2P_HV_WORDS; ++index) {
        const unsigned low = value | (index << 8) | (kind << 16) |
                             ((0x123u + index) << 19);
        const unsigned high = 0xabc00000u + (index << 4);
        checksum ^= low & 0xffffu;
        checksum ^= (low >> 16) & 0xffffu;
        checksum ^= high & 0xffffu;
        checksum ^= (high >> 16) & 0xffffu;
    }
#else
    (void)kind;
    (void)value;
#endif
    return checksum & 0xffffu;
}

}  // namespace

int sc_main(int, char **) {
    static const unsigned NUM_TOKENS = 16;
    static const unsigned TIMEOUT_CYCLES = 1000;

    sc_core::sc_clock clk("clk", 10, sc_core::SC_NS);
    sc_core::sc_signal<bool> rst;
    sc_core::sc_signal<bool> in_valid;
    sc_core::sc_signal<bool> in_ready;
    sc_core::sc_signal<sc_dt::sc_uint<3> > in_kind;
    sc_core::sc_signal<sc_dt::sc_uint<8> > in_value;
    sc_core::sc_signal<bool> out_valid;
    sc_core::sc_signal<bool> out_ready;
    sc_core::sc_signal<sc_dt::sc_uint<3> > out_kind;
    sc_core::sc_signal<sc_dt::sc_uint<8> > out_value;
    sc_core::sc_signal<sc_dt::sc_uint<16> > out_sample_checksum;
    sc_core::sc_signal<sc_dt::sc_uint<16> > out_encoded_checksum;
    sc_core::sc_signal<sc_dt::sc_uint<16> > source_count;
    sc_core::sc_signal<sc_dt::sc_uint<16> > stage_count;
    sc_core::sc_signal<sc_dt::sc_uint<16> > sink_count;

    P2PPipeline dut("dut");
    dut.clk(clk);
    dut.rst(rst);
    dut.in_valid(in_valid);
    dut.in_ready(in_ready);
    dut.in_kind(in_kind);
    dut.in_value(in_value);
    dut.out_valid(out_valid);
    dut.out_ready(out_ready);
    dut.out_kind(out_kind);
    dut.out_value(out_value);
    dut.out_sample_checksum(out_sample_checksum);
    dut.out_encoded_checksum(out_encoded_checksum);
    dut.source_count(source_count);
    dut.stage_count(stage_count);
    dut.sink_count(sink_count);

    rst.write(true);
    in_valid.write(false);
    in_kind.write(0);
    in_value.write(0);
    out_ready.write(false);

    sc_core::sc_start(5, sc_core::SC_NS);
    for (unsigned i = 0; i < 5; ++i) {
        sc_core::sc_start(10, sc_core::SC_NS);
    }
    rst.write(false);
    out_ready.write(true);

    unsigned sent = 0;
    unsigned received = 0;
    unsigned errors = 0;
#if defined(P2P_INTERNAL_SOURCE)
    sent = NUM_TOKENS;
#endif

    for (unsigned cycle = 0; cycle < TIMEOUT_CYCLES && received < NUM_TOKENS; ++cycle) {
#if !defined(P2P_INTERNAL_SOURCE)
        if (!in_valid.read() && sent < NUM_TOKENS) {
            in_kind.write(sent % 5u);
            in_value.write(10u + sent);
            in_valid.write(true);
        }
#endif

        sc_core::sc_start(10, sc_core::SC_NS);

#if !defined(P2P_INTERNAL_SOURCE)
        if (in_valid.read() && in_ready.read()) {
            ++sent;
            in_valid.write(false);
        }
#endif

        if (out_valid.read() && out_ready.read()) {
            const unsigned expected_index = received;
            const unsigned expected_kind = expected_index % 5u;
            const unsigned input_value = 10u + expected_index;
#if defined(P2P_ENCODER_MIMIC)
            const unsigned expected_value = input_value;
#else
            const unsigned expected_value = input_value + 1u;
#endif
            const unsigned expected_sample = expected_sample_checksum(expected_kind, input_value);
            const unsigned expected_encoded = expected_encoded_checksum(expected_kind, input_value);
            const unsigned got_kind = out_kind.read().to_uint();
            const unsigned got_value = out_value.read().to_uint();
            const unsigned got_sample = out_sample_checksum.read().to_uint();
            const unsigned got_encoded = out_encoded_checksum.read().to_uint();

            if (got_kind != expected_kind || got_value != expected_value ||
                got_sample != expected_sample || got_encoded != expected_encoded) {
                std::cerr << "Mismatch at token " << received
                          << ": got kind=" << got_kind << " value=" << got_value
                          << " sample_checksum=" << got_sample
                          << " encoded_checksum=" << got_encoded
                          << ", expected kind=" << expected_kind
                          << " value=" << expected_value
                          << " sample_checksum=" << expected_sample
                          << " encoded_checksum=" << expected_encoded << '\n';
                ++errors;
            }
            ++received;
        }
    }

    std::cout << "p2p experiment complete\n"
              << "sent=" << sent << '\n'
              << "received=" << received << '\n'
              << "source_count=" << source_count.read() << '\n'
              << "stage_count=" << stage_count.read() << '\n'
              << "sink_count=" << sink_count.read() << '\n'
              << "errors=" << errors << '\n';

    if (sent != NUM_TOKENS || received != NUM_TOKENS || errors != 0) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
