#!/usr/bin/env python3
"""Generate an XSim SystemVerilog testbench for the Stratus RTL top module."""

import argparse
import json
import pathlib
import re
from typing import Dict, Iterable, List, Optional


class Port:
    def __init__(self, direction: str, name: str, width: str) -> None:
        self.direction = direction
        self.name = name
        self.width = width


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def module_text(verilog: str, module_name: str) -> str:
    match = re.search(rf"\bmodule\s+{re.escape(module_name)}\b", verilog)
    if not match:
        raise SystemExit(f"module {module_name!r} not found")
    end = re.search(r"\bendmodule\b", verilog[match.start() :])
    if not end:
        raise SystemExit(f"endmodule for {module_name!r} not found")
    return verilog[match.start() : match.start() + end.end()]


def read_module_body(verilog_path: pathlib.Path, module_name: str) -> str:
    text = strip_comments(verilog_path.read_text())
    return module_text(text, module_name)


def parse_ports(verilog_path: pathlib.Path, module_name: str) -> Dict[str, Port]:
    body = read_module_body(verilog_path, module_name)
    decl_re = re.compile(
        r"\b(input|output|inout)\b\s+"
        r"(?:(?:wire|reg|logic|signed|unsigned)\s+)*"
        r"(\[[^\]]+\])?\s*"
        r"([^;]+);",
        re.S,
    )

    ports: Dict[str, Port] = {}
    for direction, width, names in decl_re.findall(body):
        width = width.strip() if width else ""
        for raw_name in names.split(","):
            name = raw_name.strip()
            if not name:
                continue
            if "=" in name:
                name = name.split("=", 1)[0].strip()
            name = re.sub(r"\s+", " ", name)
            ports[name] = Port(direction=direction, name=name, width=width)

    if not ports:
        raise SystemExit(f"no ports parsed from {verilog_path}")
    return ports


def canonical(name: str) -> str:
    name = name.strip()
    if name.startswith("\\"):
        name = name[1:]
    return name.strip()


def find_exact(ports: Dict[str, Port], expected: str) -> Port:
    for port in ports.values():
        if canonical(port.name) == expected:
            return port
    available = ", ".join(sorted(canonical(p.name) for p in ports.values()))
    raise SystemExit(f"missing required port {expected!r}; available ports: {available}")


def extract_index(name: str, prefix: str) -> Optional[int]:
    name = canonical(name)
    if prefix not in name:
        return None
    bracket = re.search(r"\[(\d+)\]", name)
    if bracket:
        return int(bracket.group(1))
    suffix = re.search(r"(?:_|_r_|_s_|_z)?(\d+)$", name)
    if suffix:
        return int(suffix.group(1))
    return None


def find_indexed(ports: Dict[str, Port], prefix: str, expected_count: int) -> List[Port]:
    indexed = []
    for port in ports.values():
        idx = extract_index(port.name, prefix)
        if idx is not None:
            indexed.append((idx, port))
    indexed.sort(key=lambda item: item[0])
    if len(indexed) != expected_count:
        found = ", ".join(f"{idx}:{canonical(port.name)}" for idx, port in indexed)
        raise SystemExit(
            f"expected {expected_count} ports matching {prefix!r}, found {len(indexed)}: {found}"
        )
    for expected_index, (idx, _) in enumerate(indexed):
        if idx != expected_index:
            raise SystemExit(f"non-contiguous {prefix!r} port index: expected {expected_index}, got {idx}")
    return [port for _, port in indexed]


def port_connection(port: Port, signal: str) -> str:
    if port.name.startswith("\\"):
        return f".{port.name} ({signal})"
    return f".{port.name}({signal})"


def width_or_scalar(width: str) -> str:
    return f" {width}" if width else ""


def join_connections(connections: Iterable[str]) -> str:
    return ",\n        ".join(connections)


def has_internal_signal(module_body: str, signal: str) -> bool:
    return re.search(r"\b" + re.escape(signal) + r"\b", module_body) is not None


def make_debug_display(label: str, fields: List[tuple], module_body: str, value_format: str) -> str:
    present = [(name, signal) for name, signal in fields if has_internal_signal(module_body, signal)]
    if not present:
        return ""

    format_fields = " ".join(f"{name}={value_format}" for name, _ in present)
    args = ", ".join(f"dut.{signal}" for _, signal in present)
    return (
        f'            $display("debug {label} {format_fields}",\n'
        f"                     {args});\n"
    )


def internal_handshake_channels(module_body: str) -> List[tuple]:
    candidates = [
        ("enc_in", "m_encoder_in_valid", "m_encoder_in_ready"),
        ("enc_out", "m_encoder_out_valid", "m_encoder_out_ready"),
        ("bundler_in", "m_bundler_in_valid", "m_bundler_in_ready"),
        ("distance_in", "m_distance_in_valid", "m_distance_in_ready"),
        ("distance_done", "m_distance_done_valid", "m_distance_done_ready"),
        ("ngram_control_done", "m_ngram_control_done_valid", "m_ngram_control_done_ready"),
        ("train_control_done", "m_train_control_done_valid", "m_train_control_done_ready"),
    ]
    return [
        item for item in candidates
        if has_internal_signal(module_body, item[1]) and has_internal_signal(module_body, item[2])
    ]


def first_present_signal(module_body: str, candidates: List[str]) -> Optional[str]:
    for signal in candidates:
        if has_internal_signal(module_body, signal):
            return signal
    return None


def internal_payload_kind_channels(module_body: str) -> List[tuple]:
    candidates = [
        (
            "enc_in",
            "m_encoder_in_valid",
            "m_encoder_in_ready",
            [
                "m_encoder_in_kind",
                "m_encoder_in_data_kind",
                "m_encoder_in_m_chan_data_kind",
                "m_encoder_in_m_chan_data_kind_slice",
            ],
        ),
        (
            "enc_out",
            "m_encoder_out_valid",
            "m_encoder_out_ready",
            [
                "m_encoder_out_kind",
                "m_encoder_out_data_kind",
                "m_encoder_out_m_chan_data_kind",
            ],
        ),
        (
            "bundler_in",
            "m_bundler_in_valid",
            "m_bundler_in_ready",
            [
                "m_bundler_in_kind",
                "m_bundler_in_data_kind",
                "m_bundler_in_m_chan_data_kind",
            ],
        ),
        (
            "distance_in",
            "m_distance_in_valid",
            "m_distance_in_ready",
            [
                "m_distance_in_kind",
                "m_distance_in_data_kind",
                "m_distance_in_m_chan_data_kind",
            ],
        ),
    ]

    channels = []
    for label, valid, ready, kind_candidates in candidates:
        kind_signal = first_present_signal(module_body, kind_candidates)
        if (
            has_internal_signal(module_body, valid)
            and has_internal_signal(module_body, ready)
            and kind_signal is not None
        ):
            channels.append((label, valid, ready, kind_signal))
    return channels


def internal_payload_bool_channels(module_body: str) -> List[tuple]:
    candidates = [
        (
            "bundler_in_valid_ngram",
            "m_bundler_in_valid",
            "m_bundler_in_ready",
            "m_bundler_in_valid_ngram",
        ),
        (
            "distance_in_valid_ngram",
            "m_distance_in_valid",
            "m_distance_in_ready",
            "m_distance_in_valid_ngram",
        ),
        (
            "distance_done_valid_prediction",
            "m_distance_done_valid",
            "m_distance_done_ready",
            "m_distance_done_valid_prediction",
        ),
    ]
    return [
        item for item in candidates
        if has_internal_signal(module_body, item[1])
        and has_internal_signal(module_body, item[2])
        and has_internal_signal(module_body, item[3])
    ]


def generate_internal_counter_declarations(module_body: str) -> str:
    channels = internal_handshake_channels(module_body)
    kind_channels = internal_payload_kind_channels(module_body)
    bool_channels = internal_payload_bool_channels(module_body)
    declarations = "".join(f"    integer internal_{label}_fire;\n" for label, _, _ in channels)
    for label, _, _, _ in kind_channels:
        for kind in range(5):
            declarations += f"    integer internal_{label}_kind{kind}_fire;\n"
    for label, _, _, _ in bool_channels:
        declarations += f"    integer internal_{label}_false_fire;\n"
        declarations += f"    integer internal_{label}_true_fire;\n"
    return declarations


def generate_internal_counter_reset(module_body: str) -> str:
    channels = internal_handshake_channels(module_body)
    kind_channels = internal_payload_kind_channels(module_body)
    bool_channels = internal_payload_bool_channels(module_body)
    resets = "".join(f"        internal_{label}_fire = 0;\n" for label, _, _ in channels)
    for label, _, _, _ in kind_channels:
        for kind in range(5):
            resets += f"        internal_{label}_kind{kind}_fire = 0;\n"
    for label, _, _, _ in bool_channels:
        resets += f"        internal_{label}_false_fire = 0;\n"
        resets += f"        internal_{label}_true_fire = 0;\n"
    return resets


def generate_internal_counter_update(module_body: str) -> str:
    channels = internal_handshake_channels(module_body)
    lines = []
    for label, valid, ready in channels:
        lines.append(
            f"            if (dut.{valid} && dut.{ready}) begin\n"
            f"                internal_{label}_fire = internal_{label}_fire + 1;\n"
            f"            end\n"
        )
    for label, valid, ready, kind_signal in internal_payload_kind_channels(module_body):
        lines.append(
            f"            if (dut.{valid} && dut.{ready}) begin\n"
            f"                case (dut.{kind_signal})\n"
        )
        for kind in range(5):
            lines.append(
                f"                    {kind}: internal_{label}_kind{kind}_fire = "
                f"internal_{label}_kind{kind}_fire + 1;\n"
            )
        lines.append("                    default: begin end\n")
        lines.append("                endcase\n")
        lines.append("            end\n")
    for label, valid, ready, bool_signal in internal_payload_bool_channels(module_body):
        lines.append(
            f"            if (dut.{valid} && dut.{ready}) begin\n"
            f"                if (dut.{bool_signal}) begin\n"
            f"                    internal_{label}_true_fire = internal_{label}_true_fire + 1;\n"
            f"                end else begin\n"
            f"                    internal_{label}_false_fire = internal_{label}_false_fire + 1;\n"
            f"                end\n"
            f"            end\n"
        )
    return "".join(lines)


def generate_internal_counter_display(module_body: str) -> str:
    channels = internal_handshake_channels(module_body)
    display = ""
    if channels:
        fields = " ".join(f"{label}=%0d" for label, _, _ in channels)
        args = ", ".join(f"internal_{label}_fire" for label, _, _ in channels)
        display += (
            f'                $display("debug internal_fires {fields}",\n'
            f"                         {args});\n"
        )
    for label, _, _, _ in internal_payload_kind_channels(module_body):
        display += (
            f'                $display("debug {label}_kind_fires '
            f'k0=%0d k1=%0d k2=%0d k3=%0d k4=%0d",\n'
            f"                         internal_{label}_kind0_fire,\n"
            f"                         internal_{label}_kind1_fire,\n"
            f"                         internal_{label}_kind2_fire,\n"
            f"                         internal_{label}_kind3_fire,\n"
            f"                         internal_{label}_kind4_fire);\n"
        )
    for label, _, _, _ in internal_payload_bool_channels(module_body):
        display += (
            f'                $display("debug {label}_fires false=%0d true=%0d",\n'
            f"                         internal_{label}_false_fire,\n"
            f"                         internal_{label}_true_fire);\n"
        )
    return display


def generate_final_internal_counter_display(module_body: str) -> str:
    return generate_internal_counter_display(module_body).replace(
        '                $display("debug ', '                $display("final debug '
    )


def generate_debug_task(module_body: str) -> str:
    state_fields = [
        ("cmd", "global_state5"),
        ("enc", "global_state4"),
        ("ngram", "global_state3"),
        ("train", "global_state2"),
        ("dist", "global_state1"),
        ("rsp", "global_state"),
    ]

    channel_groups = [
        (
            "enc_in",
            [
                ("valid", "m_encoder_in_valid"),
                ("ready", "m_encoder_in_ready"),
                ("vld", "m_encoder_in_m_chan_vld"),
                ("busy", "m_encoder_in_m_chan_busy"),
                ("out_unval", "m_encoder_in_output_m_unvalidated_req"),
                ("in_unack", "m_encoder_in_input_m_unacked_req"),
            ],
        ),
        (
            "enc_out",
            [
                ("valid", "m_encoder_out_valid"),
                ("ready", "m_encoder_out_ready"),
                ("vld", "m_encoder_out_m_chan_vld"),
                ("busy", "m_encoder_out_m_chan_busy"),
                ("out_unval", "m_encoder_out_output_m_unvalidated_req"),
                ("in_unack", "m_encoder_out_input_m_unacked_req"),
            ],
        ),
        (
            "bundler_in",
            [
                ("valid", "m_bundler_in_valid"),
                ("ready", "m_bundler_in_ready"),
                ("vld", "m_bundler_in_m_chan_vld"),
                ("busy", "m_bundler_in_m_chan_busy"),
                ("out_unval", "m_bundler_in_output_m_unvalidated_req"),
                ("in_unack", "m_bundler_in_input_m_unacked_req"),
            ],
        ),
        (
            "distance_in",
            [
                ("valid", "m_distance_in_valid"),
                ("ready", "m_distance_in_ready"),
                ("vld", "m_distance_in_m_chan_vld"),
                ("busy", "m_distance_in_m_chan_busy"),
                ("out_unval", "m_distance_in_output_m_unvalidated_req"),
                ("in_unack", "m_distance_in_input_m_unacked_req"),
            ],
        ),
        (
            "distance_done",
            [
                ("valid", "m_distance_done_valid"),
                ("ready", "m_distance_done_ready"),
                ("vld", "m_distance_done_m_chan_vld"),
                ("busy", "m_distance_done_m_chan_busy"),
                ("out_unval", "m_distance_done_output_m_unvalidated_req"),
                ("in_unack", "m_distance_done_input_m_unacked_req"),
            ],
        ),
        (
            "ngram_control_done",
            [
                ("valid", "m_ngram_control_done_valid"),
                ("ready", "m_ngram_control_done_ready"),
                ("vld", "m_ngram_control_done_m_chan_vld"),
                ("busy", "m_ngram_control_done_m_chan_busy"),
                ("in_unack", "m_ngram_control_done_input_m_unacked_req"),
            ],
        ),
        (
            "train_control_done",
            [
                ("valid", "m_train_control_done_valid"),
                ("ready", "m_train_control_done_ready"),
                ("vld", "m_train_control_done_m_chan_vld"),
                ("busy", "m_train_control_done_m_chan_busy"),
                ("in_unack", "m_train_control_done_input_m_unacked_req"),
            ],
        ),
    ]

    payload_groups = [
        (
            "enc_in_data",
            [
                ("kind", "m_encoder_in_kind"),
                ("class", "m_encoder_in_class_id"),
                ("kind", "m_encoder_in_data_kind"),
                ("class", "m_encoder_in_data_class_id"),
                ("kind", "m_encoder_in_m_chan_data_kind"),
                ("kind_slice", "m_encoder_in_m_chan_data_kind_slice"),
                ("class", "m_encoder_in_m_chan_data_class_id"),
            ],
        ),
        (
            "enc_out_data",
            [
                ("kind", "m_encoder_out_kind"),
                ("class", "m_encoder_out_class_id"),
                ("kind", "m_encoder_out_data_kind"),
                ("class", "m_encoder_out_data_class_id"),
                ("kind", "m_encoder_out_m_chan_data_kind"),
                ("class", "m_encoder_out_m_chan_data_class_id"),
            ],
        ),
        (
            "bundler_in_data",
            [
                ("kind", "m_bundler_in_kind"),
                ("class", "m_bundler_in_class_id"),
                ("valid_ngram", "m_bundler_in_valid_ngram"),
                ("kind", "m_bundler_in_data_kind"),
                ("class", "m_bundler_in_data_class_id"),
                ("valid_ngram", "m_bundler_in_data_valid_ngram"),
                ("kind", "m_bundler_in_m_chan_data_kind"),
                ("class", "m_bundler_in_m_chan_data_class_id"),
                ("valid_ngram", "m_bundler_in_m_chan_data_valid_ngram"),
            ],
        ),
        (
            "distance_in_data",
            [
                ("kind", "m_distance_in_kind"),
                ("class", "m_distance_in_class_id"),
                ("valid_ngram", "m_distance_in_valid_ngram"),
                ("kind", "m_distance_in_data_kind"),
                ("class", "m_distance_in_data_class_id"),
                ("valid_ngram", "m_distance_in_data_valid_ngram"),
                ("kind", "m_distance_in_m_chan_data_kind"),
                ("class", "m_distance_in_m_chan_data_class_id"),
                ("valid_ngram", "m_distance_in_m_chan_data_valid_ngram"),
            ],
        ),
        (
            "distance_done_data",
            [
                ("valid_prediction", "m_distance_done_valid_prediction"),
                ("d0", "m_distance_done_distances_0"),
                ("d1", "m_distance_done_distances_1"),
                ("d2", "m_distance_done_distances_2"),
                ("d3", "m_distance_done_distances_3"),
                ("d4", "m_distance_done_distances_4"),
                ("valid_prediction", "m_distance_done_data_valid_prediction"),
                ("d0", "m_distance_done_data_distances_0"),
                ("d1", "m_distance_done_data_distances_1"),
                ("d2", "m_distance_done_data_distances_2"),
                ("d3", "m_distance_done_data_distances_3"),
                ("d4", "m_distance_done_data_distances_4"),
                ("valid_prediction", "m_distance_done_m_chan_data_valid_prediction"),
                ("d0", "m_distance_done_m_chan_data_distances_0"),
                ("d1", "m_distance_done_m_chan_data_distances_1"),
                ("d2", "m_distance_done_m_chan_data_distances_2"),
                ("d3", "m_distance_done_m_chan_data_distances_3"),
                ("d4", "m_distance_done_m_chan_data_distances_4"),
            ],
        ),
    ]

    displays = make_debug_display("states", state_fields, module_body, "%0d")
    for label, fields in channel_groups:
        displays += make_debug_display(label, fields, module_body, "%0b")
    for label, fields in payload_groups:
        displays += make_debug_display(label, fields, module_body, "%0d")

    return f"""
    task print_dut_debug;
        begin
            $display("debug top cmd_v=%0b cmd_r=%0b rsp_v=%0b rsp_r=%0b outstanding=%0d",
                     cmd_valid, cmd_ready, rsp_valid, rsp_ready, outstanding);
{displays.rstrip()}
        end
    endtask
"""


def generate_tb(args: argparse.Namespace) -> str:
    module_body = read_module_body(args.top, "HDC_Accelerator")
    ports = parse_ports(args.top, "HDC_Accelerator")
    clk = find_exact(ports, "clk")
    rst = find_exact(ports, "rst")
    cmd_valid = find_exact(ports, "cmd_valid")
    cmd_ready = find_exact(ports, "cmd_ready")
    cmd_kind = find_exact(ports, "cmd_kind")
    cmd_class_id = find_exact(ports, "cmd_class_id")
    rsp_valid = find_exact(ports, "rsp_valid")
    rsp_ready = find_exact(ports, "rsp_ready")
    rsp_valid_prediction = find_exact(ports, "rsp_valid_prediction")
    sample_ports = find_indexed(ports, "cmd_sample_levels", args.num_features)
    distance_ports = find_indexed(ports, "rsp_distances", args.num_classes)

    command_path = json.dumps(str(args.trace_dir / "commands.txt"))
    response_path = json.dumps(str(args.trace_dir / "expected_responses.txt"))

    connections = [
        port_connection(clk, "clk"),
        port_connection(rst, "rst"),
        port_connection(cmd_valid, "cmd_valid"),
        port_connection(cmd_ready, "cmd_ready"),
        port_connection(cmd_kind, "cmd_kind"),
        port_connection(cmd_class_id, "cmd_class_id"),
        port_connection(rsp_valid, "rsp_valid"),
        port_connection(rsp_ready, "rsp_ready"),
        port_connection(rsp_valid_prediction, "rsp_valid_prediction"),
    ]
    for idx, port in enumerate(sample_ports):
        connections.append(port_connection(port, f"cmd_sample_levels[{idx}]"))
    for idx, port in enumerate(distance_ports):
        connections.append(port_connection(port, f"rsp_distances[{idx}]"))

    level_width = width_or_scalar(sample_ports[0].width)
    dist_width = width_or_scalar(distance_ports[0].width)
    cmd_kind_width = width_or_scalar(cmd_kind.width)
    class_width = width_or_scalar(cmd_class_id.width)

    return f"""`timescale 1ns/1ps

module hdc_accelerator_rtl_tb;
    localparam integer NUM_FEATURES = {args.num_features};
    localparam integer NUM_CLASSES = {args.num_classes};
    localparam integer MAX_OUTSTANDING = {args.max_outstanding};
    localparam integer TIMEOUT_CYCLES = {args.timeout_cycles};
    localparam integer PROGRESS_CYCLES = {args.progress_cycles};
    localparam integer RESET_CYCLES = {args.reset_cycles};
    localparam integer POST_COMMAND_HOLD_CYCLES = {args.post_command_hold_cycles};
    localparam string COMMAND_PATH = {command_path};
    localparam string RESPONSE_PATH = {response_path};

    logic clk;
    logic rst;
    logic cmd_valid;
    wire cmd_ready;
    logic{cmd_kind_width} cmd_kind;
    logic{class_width} cmd_class_id;
    logic{level_width} cmd_sample_levels [0:NUM_FEATURES-1];
    wire rsp_valid;
    logic rsp_ready;
    wire rsp_valid_prediction;
    wire{dist_width} rsp_distances [0:NUM_CLASSES-1];

    HDC_Accelerator dut (
        {join_connections(connections)}
    );

    integer command_fd;
    integer response_fd;
    integer cycle_count;
    integer commands_sent;
    integer inference_sent;
    integer responses_received;
    integer command_stall_cycles;
    integer response_stall_cycles;
    integer outstanding;
    integer next_kind;
    integer next_class_id;
    integer next_levels [0:NUM_FEATURES-1];
    integer has_command;
    integer issue_cycles [0:1048575];
    integer issue_head;
    integer issue_tail;
    integer command_hold_cycles;
    integer total_latency;
    integer max_latency;
    integer error_count;
{generate_internal_counter_declarations(module_body)}

    always #5 clk = ~clk;

    task read_next_command(output integer ok);
        integer rc;
        integer feature;
        begin
            rc = $fscanf(command_fd, "%d %d", next_kind, next_class_id);
            if (rc == 2) begin
                ok = 1;
                for (feature = 0; feature < NUM_FEATURES; feature = feature + 1) begin
                    rc = $fscanf(command_fd, "%d", next_levels[feature]);
                    if (rc != 1) begin
                        $fatal(1, "Malformed command file while reading feature %0d", feature);
                    end
                end
            end else begin
                ok = 0;
            end
        end
    endtask

    function integer can_drive_next_command;
        begin
            if (!has_command) begin
                can_drive_next_command = 0;
            end else if (next_kind == 4 && outstanding >= MAX_OUTSTANDING) begin
                can_drive_next_command = 0;
            end else begin
                can_drive_next_command = 1;
            end
        end
    endfunction

    task drive_next_command;
        integer feature;
        begin
            cmd_kind = next_kind;
            cmd_class_id = next_class_id;
            for (feature = 0; feature < NUM_FEATURES; feature = feature + 1) begin
                cmd_sample_levels[feature] = next_levels[feature];
            end
            cmd_valid = 1'b1;
        end
    endtask

    task check_response;
        integer rc;
        integer expected_valid;
        integer expected_distance [0:NUM_CLASSES-1];
        integer expected_predicted;
        integer expected_actual;
        integer class_id;
        integer predicted;
        integer best_distance;
        integer latency;
        string expected_predicted_text;
        string expected_actual_text;
        begin
            rc = $fscanf(response_fd, "%d %d %d %d %d %d %s %s",
                         expected_valid,
                         expected_distance[0],
                         expected_distance[1],
                         expected_distance[2],
                         expected_distance[3],
                         expected_distance[4],
                         expected_predicted_text,
                         expected_actual_text);
            if (rc != 8) begin
                $fatal(1, "Malformed expected response %0d: parsed %0d fields",
                       responses_received, rc);
            end

            expected_predicted = expected_predicted_text.atoi();
            expected_actual = expected_actual_text.atoi();

            if (expected_predicted < 0) begin
                expected_predicted = 0;
                if (expected_valid != 0) begin
                    best_distance = expected_distance[0];
                    for (class_id = 1; class_id < NUM_CLASSES; class_id = class_id + 1) begin
                        if (expected_distance[class_id] < best_distance) begin
                            best_distance = expected_distance[class_id];
                            expected_predicted = class_id;
                        end
                    end
                end
            end

            if (rsp_valid_prediction !== (expected_valid != 0)) begin
                $error("valid_prediction mismatch at response %0d: got %0d expected %0d",
                       responses_received, rsp_valid_prediction, expected_valid);
                error_count = error_count + 1;
            end

            predicted = 0;
            best_distance = rsp_distances[0];
            for (class_id = 0; class_id < NUM_CLASSES; class_id = class_id + 1) begin
                if (rsp_distances[class_id] != expected_distance[class_id]) begin
                    $error("distance[%0d] mismatch at response %0d: got %0d expected %0d",
                           class_id, responses_received, rsp_distances[class_id], expected_distance[class_id]);
                    error_count = error_count + 1;
                end
                if (class_id > 0 && rsp_distances[class_id] < best_distance) begin
                    best_distance = rsp_distances[class_id];
                    predicted = class_id;
                end
            end

            if (expected_valid != 0 && predicted != expected_predicted) begin
                $error("predicted mismatch at response %0d: got %0d expected %0d actual %0d",
                       responses_received, predicted, expected_predicted, expected_actual);
                error_count = error_count + 1;
            end

            latency = cycle_count - issue_cycles[issue_head];
            issue_head = issue_head + 1;
            total_latency = total_latency + latency;
            if (latency > max_latency) begin
                max_latency = latency;
            end
        end
    endtask

    task finish_if_complete;
        integer extra;
        integer rc;
        real average_latency;
        begin
            if (!has_command && !cmd_valid && outstanding == 0) begin
                rc = $fscanf(response_fd, "%d", extra);
                if (rc == 1) begin
                    $fatal(1, "Expected response file contains extra response data");
                end

                average_latency = (responses_received == 0)
                    ? 0.0
                    : (1.0 * total_latency) / responses_received;
                $display("RTL simulation complete");
                $display("cycles=%0d", cycle_count);
                $display("commands_sent=%0d", commands_sent);
                $display("inference_sent=%0d", inference_sent);
                $display("responses_received=%0d", responses_received);
                $display("command_stall_cycles=%0d", command_stall_cycles);
                $display("response_stall_cycles=%0d", response_stall_cycles);
                $display("average_inference_latency=%0f", average_latency);
                $display("max_inference_latency=%0d", max_latency);
                $display("errors=%0d", error_count);
{generate_final_internal_counter_display(module_body)}

                $fclose(command_fd);
                $fclose(response_fd);
                if (error_count != 0) begin
                    $fatal(1, "RTL response comparison failed");
                end
                $finish;
            end
        end
    endtask

{generate_debug_task(module_body)}

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        cmd_valid = 1'b0;
        cmd_kind = '0;
        cmd_class_id = '0;
        rsp_ready = 1'b0;
        cycle_count = 0;
        commands_sent = 0;
        inference_sent = 0;
        responses_received = 0;
        command_stall_cycles = 0;
        response_stall_cycles = 0;
        outstanding = 0;
        has_command = 0;
        issue_head = 0;
        issue_tail = 0;
        command_hold_cycles = 0;
        total_latency = 0;
        max_latency = 0;
        error_count = 0;
{generate_internal_counter_reset(module_body)}

        command_fd = $fopen(COMMAND_PATH, "r");
        if (command_fd == 0) begin
            $fatal(1, "Failed to open command file: %s", COMMAND_PATH);
        end
        response_fd = $fopen(RESPONSE_PATH, "r");
        if (response_fd == 0) begin
            $fatal(1, "Failed to open expected response file: %s", RESPONSE_PATH);
        end
        read_next_command(has_command);

        repeat (RESET_CYCLES) @(posedge clk);
        #1;
        rst = 1'b0;
        repeat (2) @(posedge clk);
        #1;

        forever begin
            if (cycle_count > TIMEOUT_CYCLES) begin
                $fatal(1, "RTL simulation timeout after %0d cycles", cycle_count);
            end

            if (!cmd_valid && command_hold_cycles == 0 && can_drive_next_command()) begin
                drive_next_command();
            end
            rsp_ready = (outstanding > 0);

            @(posedge clk);
            #1;
            cycle_count = cycle_count + 1;
{generate_internal_counter_update(module_body)}
            if (PROGRESS_CYCLES > 0 && (cycle_count % PROGRESS_CYCLES) == 0) begin
                $display("progress cycles=%0d commands=%0d inference=%0d responses=%0d outstanding=%0d",
                         cycle_count, commands_sent, inference_sent, responses_received, outstanding);
{generate_internal_counter_display(module_body)}
                print_dut_debug();
            end

            if (cmd_valid && cmd_ready) begin
                commands_sent = commands_sent + 1;
                if (cmd_kind == 4) begin
                    inference_sent = inference_sent + 1;
                    outstanding = outstanding + 1;
                    issue_cycles[issue_tail] = cycle_count;
                    issue_tail = issue_tail + 1;
                end
                cmd_valid = 1'b0;
                read_next_command(has_command);
                command_hold_cycles = POST_COMMAND_HOLD_CYCLES;
            end else if (cmd_valid && !cmd_ready) begin
                command_stall_cycles = command_stall_cycles + 1;
            end

            if (!cmd_valid && command_hold_cycles > 0) begin
                command_hold_cycles = command_hold_cycles - 1;
            end

            if (rsp_valid && !rsp_ready) begin
                response_stall_cycles = response_stall_cycles + 1;
            end

            if (rsp_valid && rsp_ready) begin
                check_response();
                responses_received = responses_received + 1;
                outstanding = outstanding - 1;
            end

            finish_if_complete();
        end
    end
endmodule
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", required=True, type=pathlib.Path)
    parser.add_argument("--trace-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--num-features", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--max-outstanding", type=int, default=32)
    parser.add_argument("--timeout-cycles", type=int, default=50000000)
    parser.add_argument("--progress-cycles", type=int, default=100000)
    parser.add_argument("--reset-cycles", type=int, default=32)
    parser.add_argument("--post-command-hold-cycles", type=int, default=8)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generate_tb(args))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
