#!/usr/bin/env python3
"""Generate a 32-bit AXI4-Stream wrapper around the Stratus P2P RTL top."""

import argparse
import pathlib

from generate_xsim_tb import (
    canonical,
    find_exact,
    find_p2p_group,
    parse_ports,
    port_connection,
    width_to_int,
)


def connections(ports, mapping):
    return ",\n        ".join(
        port_connection(port, mapping[canonical(port.name)])
        for port in ports.values()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    ports = parse_ports(args.top, "HDC_Accelerator")
    clk = find_exact(ports, "clk")
    rst = find_exact(ports, "rst")
    cmd_data, cmd_vld, cmd_busy = find_p2p_group(ports, "cmd")
    rsp_data, rsp_vld, rsp_busy = find_p2p_group(ports, "rsp")
    command_width = width_to_int(cmd_data.width)
    response_width = width_to_int(rsp_data.width)
    command_beats = (command_width + 31) // 32
    response_beats = (response_width + 31) // 32

    core_mapping = {
        canonical(clk.name): "aclk",
        canonical(rst.name): "areset",
        canonical(cmd_data.name): "core_cmd_data",
        canonical(cmd_vld.name): "core_cmd_vld",
        canonical(cmd_busy.name): "core_cmd_busy",
        canonical(rsp_data.name): "core_rsp_data",
        canonical(rsp_vld.name): "core_rsp_vld",
        canonical(rsp_busy.name): "core_rsp_busy",
    }
    if set(core_mapping) != {canonical(port.name) for port in ports.values()}:
        raise SystemExit("HLS top has unsupported ports in addition to clk/rst/cmd/rsp")

    text = f"""`timescale 1ns/1ps

module HDC_AcceleratorAxis (
    input  wire        aclk,
    input  wire        areset,
    input  wire [31:0] s_axis_cmd_tdata,
    input  wire        s_axis_cmd_tvalid,
    output wire        s_axis_cmd_tready,
    input  wire        s_axis_cmd_tlast,
    output wire [31:0] m_axis_rsp_tdata,
    output wire        m_axis_rsp_tvalid,
    input  wire        m_axis_rsp_tready,
    output wire        m_axis_rsp_tlast
);
    localparam integer COMMAND_WIDTH = {command_width};
    localparam integer RESPONSE_WIDTH = {response_width};
    localparam integer COMMAND_BEATS = {command_beats};
    localparam integer RESPONSE_BEATS = {response_beats};
    localparam integer COMMAND_PAD_BITS = COMMAND_BEATS * 32;
    localparam integer RESPONSE_PAD_BITS = RESPONSE_BEATS * 32;

    reg [COMMAND_PAD_BITS-1:0] command_buffer;
    reg [RESPONSE_PAD_BITS-1:0] response_buffer;
    integer command_beat;
    integer response_beat;
    reg command_pending;
    reg response_pending;

    wire [COMMAND_WIDTH-1:0] core_cmd_data =
        command_buffer[COMMAND_WIDTH-1:0];
    wire core_cmd_vld = command_pending;
    wire core_cmd_busy;
    wire [RESPONSE_WIDTH-1:0] core_rsp_data;
    wire core_rsp_vld;
    wire core_rsp_busy = response_pending;

    assign s_axis_cmd_tready = !command_pending;
    assign m_axis_rsp_tvalid = response_pending;
    assign m_axis_rsp_tdata = response_buffer >> (response_beat * 32);
    assign m_axis_rsp_tlast = response_pending &&
                              (response_beat == RESPONSE_BEATS - 1);

    // Pack least-significant AXI beats first. TLAST marks packet boundaries;
    // fixed packet widths determine the expected final beat.
    always @(posedge aclk) begin
        if (areset) begin
            command_buffer <= {{COMMAND_WIDTH{{1'b0}}}};
            command_beat <= 0;
            command_pending <= 1'b0;
        end else begin
            if (command_pending && !core_cmd_busy)
                command_pending <= 1'b0;

            if (s_axis_cmd_tvalid && s_axis_cmd_tready) begin
                command_buffer[command_beat * 32 +: 32] <= s_axis_cmd_tdata;
                if (command_beat == COMMAND_BEATS - 1) begin
                    command_beat <= 0;
                    if (s_axis_cmd_tlast)
                        command_pending <= 1'b1;
                    else
                        command_buffer <= {{COMMAND_WIDTH{{1'b0}}}};
                end else if (s_axis_cmd_tlast) begin
                    // Discard malformed short packets without forwarding them.
                    command_buffer <= {{COMMAND_WIDTH{{1'b0}}}};
                    command_beat <= 0;
                end else begin
                    command_beat <= command_beat + 1;
                end
            end
        end
    end

    always @(posedge aclk) begin
        if (areset) begin
            response_buffer <= {{RESPONSE_WIDTH{{1'b0}}}};
            response_beat <= 0;
            response_pending <= 1'b0;
        end else begin
            if (!response_pending && core_rsp_vld) begin
                response_buffer <= {{{{(RESPONSE_PAD_BITS-RESPONSE_WIDTH){{1'b0}}}},
                                     core_rsp_data}};
                response_beat <= 0;
                response_pending <= 1'b1;
            end else if (response_pending &&
                         m_axis_rsp_tvalid && m_axis_rsp_tready) begin
                if (response_beat == RESPONSE_BEATS - 1) begin
                    response_beat <= 0;
                    response_pending <= 1'b0;
                end else begin
                    response_beat <= response_beat + 1;
                end
            end
        end
    end

    HDC_Accelerator core (
        {connections(ports, core_mapping)}
    );
endmodule

// Simulation-only compatibility shim. It lets the established packed-P2P
// trace testbench exercise the AXI wrapper without changing its checker.
module HDC_AcceleratorAxisTestShim (
    input  wire                      clk,
    input  wire                      rst,
    input  wire [{command_width - 1}:0]  cmd_data,
    input  wire                      cmd_vld,
    output wire                      cmd_busy,
    output wire [{response_width - 1}:0] rsp_data,
    output wire                      rsp_vld,
    input  wire                      rsp_busy
);
    localparam integer COMMAND_WIDTH = {command_width};
    localparam integer RESPONSE_WIDTH = {response_width};
    localparam integer COMMAND_BEATS = {command_beats};
    localparam integer RESPONSE_BEATS = {response_beats};

    reg [COMMAND_WIDTH-1:0] command_shift;
    reg [RESPONSE_WIDTH-1:0] response_collect;
    integer command_beat;
    integer response_beat;
    reg command_active;
    reg response_valid;

    wire [31:0] s_axis_cmd_tdata =
        command_shift >> (command_beat * 32);
    wire s_axis_cmd_tvalid = command_active;
    wire s_axis_cmd_tready;
    wire s_axis_cmd_tlast = command_active &&
                            (command_beat == COMMAND_BEATS - 1);
    wire [31:0] m_axis_rsp_tdata;
    wire m_axis_rsp_tvalid;
    wire m_axis_rsp_tready = !response_valid;
    wire m_axis_rsp_tlast;

    assign cmd_busy = command_active;
    assign rsp_data = response_collect;
    assign rsp_vld = response_valid;
    wire axis_command_start = command_active &&
                              (command_beat == 0) &&
                              s_axis_cmd_tvalid &&
                              s_axis_cmd_tready;
    wire axis_command_final = command_active &&
                              (command_beat == COMMAND_BEATS - 1) &&
                              s_axis_cmd_tvalid &&
                              s_axis_cmd_tready;
    wire axis_core_command_fire = axis_dut.core_cmd_vld &&
                                  !axis_dut.core_cmd_busy;
    wire [2:0] axis_command_start_kind = command_shift[2:0];
    wire [2:0] axis_command_final_kind = command_shift[2:0];
    wire [2:0] axis_core_command_kind = axis_dut.core_cmd_data[2:0];

    always @(posedge clk) begin
        if (rst) begin
            command_shift <= {{COMMAND_WIDTH{{1'b0}}}};
            command_beat <= 0;
            command_active <= 1'b0;
            response_collect <= {{RESPONSE_WIDTH{{1'b0}}}};
            response_beat <= 0;
            response_valid <= 1'b0;
        end else begin
            if (!command_active && cmd_vld) begin
                command_shift <= cmd_data;
                command_beat <= 0;
                command_active <= 1'b1;
            end else if (command_active &&
                         s_axis_cmd_tvalid && s_axis_cmd_tready) begin
                if (command_beat == COMMAND_BEATS - 1) begin
                    command_beat <= 0;
                    command_active <= 1'b0;
                end else begin
                    command_beat <= command_beat + 1;
                end
            end

            if (response_valid && !rsp_busy)
                response_valid <= 1'b0;

            if (m_axis_rsp_tvalid && m_axis_rsp_tready) begin
                response_collect[response_beat * 32 +: 32] <=
                    m_axis_rsp_tdata;
                if (m_axis_rsp_tlast ||
                    response_beat == RESPONSE_BEATS - 1) begin
                    response_beat <= 0;
                    response_valid <= 1'b1;
                end else begin
                    response_beat <= response_beat + 1;
                end
            end
        end
    end

    HDC_AcceleratorAxis axis_dut (
        .aclk(clk),
        .areset(rst),
        .s_axis_cmd_tdata(s_axis_cmd_tdata),
        .s_axis_cmd_tvalid(s_axis_cmd_tvalid),
        .s_axis_cmd_tready(s_axis_cmd_tready),
        .s_axis_cmd_tlast(s_axis_cmd_tlast),
        .m_axis_rsp_tdata(m_axis_rsp_tdata),
        .m_axis_rsp_tvalid(m_axis_rsp_tvalid),
        .m_axis_rsp_tready(m_axis_rsp_tready),
        .m_axis_rsp_tlast(m_axis_rsp_tlast)
    );
endmodule
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)


if __name__ == "__main__":
    main()
