#!/usr/bin/env python
import re
import sys


NUM_TOKENS = 16
TIMEOUT_CYCLES = 5000


def normalize(name):
    if name.startswith("\\"):
        name = name[1:]
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()


def parse_ports(rtl_path):
    ports = []
    decl_re = re.compile(r"^\s*(input|output)\s+(?:wire\s+|reg\s+)?(?:\[(\d+):(\d+)\]\s+)?(.+?);")
    with open(rtl_path, "r") as handle:
        for line in handle:
            match = decl_re.match(line)
            if not match:
                continue
            direction = match.group(1)
            left = match.group(2)
            right = match.group(3)
            width = 1
            if left is not None and right is not None:
                width = abs(int(left) - int(right)) + 1
            for raw_name in match.group(4).split(","):
                name = raw_name.strip()
                if name:
                    ports.append({
                        "name": name,
                        "norm": normalize(name),
                        "direction": direction,
                        "width": width,
                    })
    return ports


def find_one(ports, predicate, description):
    matches = [port for port in ports if predicate(port)]
    if len(matches) != 1:
        sys.stderr.write("ERROR: expected one {}, found {}\n".format(description, len(matches)))
        for port in matches:
            sys.stderr.write("  {} {} width={}\n".format(port["direction"], port["name"], port["width"]))
        sys.exit(1)
    return matches[0]


def find_p2p_group(ports, prefix):
    group = [port for port in ports if prefix in port["norm"]]
    if not group:
        sys.stderr.write("ERROR: no ports found for {}\n".format(prefix))
        sys.exit(1)

    vld = find_one(group, lambda port: "vld" in port["norm"] or "valid" in port["norm"],
                   "{} valid".format(prefix))
    busy = find_one(group, lambda port: "busy" in port["norm"],
                    "{} busy".format(prefix))
    data_candidates = [
        port for port in group
        if port is not vld and port is not busy and port["width"] > 1
    ]
    if not data_candidates:
        sys.stderr.write("ERROR: no data port found for {}\n".format(prefix))
        for port in group:
            sys.stderr.write("  {} {} width={}\n".format(port["direction"], port["name"], port["width"]))
        sys.exit(1)
    data = sorted(data_candidates, key=lambda port: port["width"], reverse=True)[0]
    return data, vld, busy


def sv_port_ref(port):
    name = port["name"]
    if name.startswith("\\"):
        return ".{} ".format(name)
    return ".{}".format(name)


def declare_signal(name, port, driven_by_tb):
    kind = "logic" if driven_by_tb else "wire"
    if port["width"] == 1:
        return "    {} {};".format(kind, name)
    return "    {} [{}:0] {};".format(kind, port["width"] - 1, name)


def generate(rtl_path, out_path):
    ports = parse_ports(rtl_path)
    clk = find_one(ports, lambda port: port["norm"] == "clk", "clk")
    rst = find_one(ports, lambda port: port["norm"] == "rst", "rst")
    in_data, in_vld, in_busy = find_p2p_group(ports, "in_p2p")
    out_data, out_vld, out_busy = find_p2p_group(ports, "out_p2p")
    source_count = find_one(ports, lambda port: port["norm"] == "source_count", "source_count")
    stage_count = find_one(ports, lambda port: port["norm"] == "stage_count", "stage_count")
    sink_count = find_one(ports, lambda port: port["norm"] == "sink_count", "sink_count")

    signal_for_port = {
        clk["name"]: "clk",
        rst["name"]: "rst",
        in_data["name"]: "in_p2p_data",
        in_vld["name"]: "in_p2p_vld",
        in_busy["name"]: "in_p2p_busy",
        out_data["name"]: "out_p2p_data",
        out_vld["name"]: "out_p2p_vld",
        out_busy["name"]: "out_p2p_busy",
        source_count["name"]: "source_count",
        stage_count["name"]: "stage_count",
        sink_count["name"]: "sink_count",
    }

    missing = [port for port in ports if port["name"] not in signal_for_port]
    if missing:
        sys.stderr.write("ERROR: unconnected generated ports:\n")
        for port in missing:
            sys.stderr.write("  {} {} width={}\n".format(port["direction"], port["name"], port["width"]))
        sys.exit(1)

    with open(out_path, "w") as out:
        out.write("`timescale 1ns/1ps\n\n")
        out.write("module p2p_pipeline_rtl_tb;\n")
        out.write("    localparam integer NUM_TOKENS = {};\n".format(NUM_TOKENS))
        out.write("    localparam integer TIMEOUT_CYCLES = {};\n\n".format(TIMEOUT_CYCLES))
        out.write("    logic clk;\n")
        out.write("    logic rst;\n")
        out.write(declare_signal("in_p2p_data", in_data, True) + "\n")
        out.write(declare_signal("in_p2p_vld", in_vld, True) + "\n")
        out.write(declare_signal("in_p2p_busy", in_busy, False) + "\n")
        out.write(declare_signal("out_p2p_data", out_data, False) + "\n")
        out.write(declare_signal("out_p2p_vld", out_vld, False) + "\n")
        out.write(declare_signal("out_p2p_busy", out_busy, True) + "\n")
        out.write(declare_signal("source_count", source_count, False) + "\n")
        out.write(declare_signal("stage_count", stage_count, False) + "\n")
        out.write(declare_signal("sink_count", sink_count, False) + "\n\n")
        out.write("    integer cycle_count;\n")
        out.write("    integer sent;\n")
        out.write("    integer received;\n")
        out.write("    integer errors;\n\n")
        out.write("    function automatic [63:0] make_word(input integer index);\n")
        out.write("        integer kind;\n")
        out.write("        integer value;\n")
        out.write("        integer sample;\n")
        out.write("        integer i;\n")
        out.write("        begin\n")
        out.write("            kind = index % 5;\n")
        out.write("            value = 10 + index;\n")
        out.write("            sample = 0;\n")
        out.write("            for (i = 0; i < 32; i = i + 1) begin\n")
        out.write("                sample = (sample + ((value + kind + 3 * i) & 8'hff)) & 16'hffff;\n")
        out.write("            end\n")
        out.write("            make_word = 64'd0;\n")
        out.write("            make_word[2:0] = kind[2:0];\n")
        out.write("            make_word[5:3] = kind[2:0];\n")
        out.write("            make_word[13:6] = value[7:0];\n")
        out.write("            make_word[29:14] = sample[15:0];\n")
        out.write("        end\n")
        out.write("    endfunction\n\n")
        out.write("    function automatic integer expected_encoded_checksum(input integer kind, input integer value);\n")
        out.write("        integer checksum;\n")
        out.write("        integer scalar_sample_checksum;\n")
        out.write("        integer word_index;\n")
        out.write("        integer feature;\n")
        out.write("        integer level;\n")
        out.write("        integer term;\n")
        out.write("        integer acc0;\n")
        out.write("        integer acc1;\n")
        out.write("        integer acc2;\n")
        out.write("        integer acc3;\n")
        out.write("        begin\n")
        out.write("            checksum = 0;\n")
        out.write("            if (kind == 2 || kind == 4) begin\n")
        out.write("                scalar_sample_checksum = make_word(value - 10)[29:14];\n")
        out.write("                for (word_index = 0; word_index < 16; word_index = word_index + 1) begin\n")
        out.write("                    acc0 = 0; acc1 = 0; acc2 = 0; acc3 = 0;\n")
        out.write("                    for (feature = 0; feature < 32; feature = feature + 1) begin\n")
        out.write("                        level = (scalar_sample_checksum + 3 * feature + word_index) & 8'hff;\n")
        out.write("                        term = level + kind + kind + feature + word_index;\n")
        out.write("                        acc0 = (acc0 + term) & 16'hffff;\n")
        out.write("                        acc1 = (acc1 ^ ((term << (feature & 3)) & 16'hffff)) & 16'hffff;\n")
        out.write("                        acc2 = (acc2 + term * (feature + 1)) & 16'hffff;\n")
        out.write("                        acc3 = (acc3 ^ ((term + word_index * 17) & 16'hffff)) & 16'hffff;\n")
        out.write("                    end\n")
        out.write("                    checksum = checksum ^ acc0 ^ acc1 ^ acc2 ^ acc3;\n")
        out.write("                end\n")
        out.write("            end\n")
        out.write("            expected_encoded_checksum = checksum & 16'hffff;\n")
        out.write("        end\n")
        out.write("    endfunction\n\n")
        out.write("    P2PPipeline dut (\n")
        connections = []
        for port in ports:
            connections.append("        {}({})".format(sv_port_ref(port), signal_for_port[port["name"]]))
        out.write(",\n".join(connections))
        out.write("\n    );\n\n")
        out.write("    always #5 clk = ~clk;\n\n")
        out.write("    initial begin\n")
        out.write("        clk = 1'b0;\n")
        out.write("        rst = 1'b1;\n")
        out.write("        in_p2p_vld = 1'b0;\n")
        out.write("        in_p2p_data = '0;\n")
        out.write("        out_p2p_busy = 1'b0;\n")
        out.write("        cycle_count = 0;\n")
        out.write("        sent = 0;\n")
        out.write("        received = 0;\n")
        out.write("        errors = 0;\n")
        out.write("        repeat (5) @(posedge clk);\n")
        out.write("        rst = 1'b0;\n")
        out.write("    end\n\n")
        out.write("    always @(posedge clk) begin\n")
        out.write("        if (rst) begin\n")
        out.write("            in_p2p_vld <= 1'b0;\n")
        out.write("            in_p2p_data <= '0;\n")
        out.write("            sent <= 0;\n")
        out.write("        end else begin\n")
        out.write("            if (!in_p2p_vld && sent < NUM_TOKENS) begin\n")
        out.write("                in_p2p_data <= make_word(sent);\n")
        out.write("                in_p2p_vld <= 1'b1;\n")
        out.write("            end else if (in_p2p_vld && !in_p2p_busy) begin\n")
        out.write("                sent <= sent + 1;\n")
        out.write("                in_p2p_vld <= 1'b0;\n")
        out.write("            end\n")
        out.write("        end\n")
        out.write("    end\n\n")
        out.write("    always @(posedge clk) begin\n")
        out.write("        if (!rst) begin\n")
        out.write("            cycle_count <= cycle_count + 1;\n")
        out.write("            if (out_p2p_vld && !out_p2p_busy) begin\n")
        out.write("                integer expected_kind;\n")
        out.write("                integer expected_value;\n")
        out.write("                integer expected_sample;\n")
        out.write("                integer expected_encoded;\n")
        out.write("                expected_kind = received % 5;\n")
        out.write("                expected_value = 10 + received;\n")
        out.write("                expected_sample = make_word(received)[29:14];\n")
        out.write("                expected_encoded = expected_encoded_checksum(expected_kind, expected_value);\n")
        out.write("                if (out_p2p_data[2:0] !== expected_kind[2:0] ||\n")
        out.write("                    out_p2p_data[13:6] !== expected_value[7:0] ||\n")
        out.write("                    out_p2p_data[29:14] !== expected_sample[15:0] ||\n")
        out.write("                    out_p2p_data[45:30] !== expected_encoded[15:0]) begin\n")
        out.write("                    $display(\"Mismatch token %0d got kind=%0d value=%0d sample=%0d encoded=%0d expected kind=%0d value=%0d sample=%0d encoded=%0d\",\n")
        out.write("                             received, out_p2p_data[2:0], out_p2p_data[13:6], out_p2p_data[29:14], out_p2p_data[45:30],\n")
        out.write("                             expected_kind, expected_value, expected_sample, expected_encoded);\n")
        out.write("                    errors <= errors + 1;\n")
        out.write("                end\n")
        out.write("                received <= received + 1;\n")
        out.write("            end\n")
        out.write("            if ((cycle_count % 1000) == 0) begin\n")
        out.write("                $display(\"progress cycles=%0d sent=%0d received=%0d source=%0d stage=%0d sink=%0d in_vld=%0b in_busy=%0b out_vld=%0b out_busy=%0b\",\n")
        out.write("                         cycle_count, sent, received, source_count, stage_count, sink_count,\n")
        out.write("                         in_p2p_vld, in_p2p_busy, out_p2p_vld, out_p2p_busy);\n")
        out.write("            end\n")
        out.write("            if (received == NUM_TOKENS) begin\n")
        out.write("                $display(\"P2P external RTL simulation complete\");\n")
        out.write("                $display(\"cycles=%0d\", cycle_count);\n")
        out.write("                $display(\"sent=%0d\", sent);\n")
        out.write("                $display(\"received=%0d\", received);\n")
        out.write("                $display(\"source_count=%0d\", source_count);\n")
        out.write("                $display(\"stage_count=%0d\", stage_count);\n")
        out.write("                $display(\"sink_count=%0d\", sink_count);\n")
        out.write("                $display(\"errors=%0d\", errors);\n")
        out.write("                if (errors != 0) $fatal(1, \"P2P external RTL mismatches\");\n")
        out.write("                $finish;\n")
        out.write("            end\n")
        out.write("            if (cycle_count > TIMEOUT_CYCLES) begin\n")
        out.write("                $fatal(1, \"Timeout sent=%0d received=%0d source=%0d stage=%0d sink=%0d\", sent, received, source_count, stage_count, sink_count);\n")
        out.write("            end\n")
        out.write("        end\n")
        out.write("    end\n")
        out.write("endmodule\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("usage: generate_external_p2p_tb.py <top_rtl.v> <out_tb.sv>\n")
        sys.exit(1)
    generate(sys.argv[1], sys.argv[2])
