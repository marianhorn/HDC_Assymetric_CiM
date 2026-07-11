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


def find_optional_exact(ports: Dict[str, Port], expected: str) -> Optional[Port]:
    for port in ports.values():
        if canonical(port.name) == expected:
            return port
    return None


def width_to_int(width: str) -> int:
    if not width:
        return 1
    match = re.match(r"\[(\d+)\s*:\s*(\d+)\]", width)
    if not match:
        raise SystemExit(f"unsupported port width syntax: {width!r}")
    return abs(int(match.group(1)) - int(match.group(2))) + 1


def find_p2p_group(ports: Dict[str, Port], prefix: str) -> tuple:
    group = [port for port in ports.values() if prefix in canonical(port.name).lower()]
    if not group:
        available = ", ".join(sorted(canonical(p.name) for p in ports.values()))
        raise SystemExit(f"missing P2P group {prefix!r}; available ports: {available}")

    valid = [port for port in group if "vld" in canonical(port.name).lower()
             or "valid" in canonical(port.name).lower()]
    busy = [port for port in group if "busy" in canonical(port.name).lower()]
    data = [port for port in group
            if port not in valid and port not in busy and width_to_int(port.width) > 1]
    if len(valid) != 1 or len(busy) != 1 or not data:
        found = ", ".join(f"{p.direction}:{canonical(p.name)}{p.width}" for p in group)
        raise SystemExit(
            f"could not identify P2P {prefix!r} data/valid/busy ports: {found}"
        )
    data.sort(key=lambda port: width_to_int(port.width), reverse=True)
    return data[0], valid[0], busy[0]


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


def generate_p2p_counter_logic(module_body: str) -> Dict[str, str]:
    channels = [
        ("enc_in", "m_encoder_in"),
        ("enc_out", "m_encoder_out"),
        ("bundler_in", "m_bundler_in"),
        ("distance_in", "m_distance_in"),
        ("distance_done", "m_distance_done"),
        ("ngram_control_done", "m_ngram_control_done"),
        ("train_control_done", "m_train_control_done"),
    ]

    decls: List[str] = []
    inits: List[str] = []
    updates: List[str] = []
    prev_updates: List[str] = []
    displays: List[str] = []

    present_channels = []
    for label, prefix in channels:
        vld = f"{prefix}_m_chan_vld"
        if not has_internal_signal(module_body, vld):
            continue

        busy = f"{prefix}_m_chan_busy"
        has_busy = has_internal_signal(module_body, busy)
        present_channels.append((label, vld, busy if has_busy else None))

        decls.extend(
            [
                f"    integer {label}_vld_cycles;",
                f"    integer {label}_vld_edges;",
                f"    integer {label}_busy_cycles;",
                f"    integer {label}_prev_vld;",
                f"    integer {label}_first_vld_edge_cycle;",
                f"    integer {label}_last_vld_edge_cycle;",
                f"    integer {label}_max_vld_edge_gap;",
                f"    integer {label}_fire_cycles;",
                f"    integer {label}_first_fire_cycle;",
                f"    integer {label}_last_fire_cycle;",
                f"    integer {label}_max_fire_gap;",
            ]
        )
        inits.extend(
            [
                f"        {label}_vld_cycles = 0;",
                f"        {label}_vld_edges = 0;",
                f"        {label}_busy_cycles = 0;",
                f"        {label}_prev_vld = 0;",
                f"        {label}_first_vld_edge_cycle = -1;",
                f"        {label}_last_vld_edge_cycle = -1;",
                f"        {label}_max_vld_edge_gap = 0;",
                f"        {label}_fire_cycles = 0;",
                f"        {label}_first_fire_cycle = -1;",
                f"        {label}_last_fire_cycle = -1;",
                f"        {label}_max_fire_gap = 0;",
            ]
        )
        updates.append(
            f"""            if (dut.{vld}) begin
                {label}_vld_cycles = {label}_vld_cycles + 1;
                if (!{label}_prev_vld) begin
                    if ({label}_first_vld_edge_cycle < 0) begin
                        {label}_first_vld_edge_cycle = cycle_count;
                    end
                    if ({label}_last_vld_edge_cycle >= 0 &&
                        (cycle_count - {label}_last_vld_edge_cycle) > {label}_max_vld_edge_gap) begin
                        {label}_max_vld_edge_gap = cycle_count - {label}_last_vld_edge_cycle;
                    end
                    {label}_last_vld_edge_cycle = cycle_count;
                    {label}_vld_edges = {label}_vld_edges + 1;
                end
            end"""
        )
        prev_updates.append(f"            {label}_prev_vld = dut.{vld};")
        if has_busy:
            updates.append(
                f"""            if (dut.{busy}) begin
                {label}_busy_cycles = {label}_busy_cycles + 1;
            end"""
            )
        fire_condition = f"dut.{vld}"
        if has_busy:
            fire_condition = f"dut.{vld} && !dut.{busy}"
        updates.append(
            f"""            if ({fire_condition}) begin
                if ({label}_first_fire_cycle < 0) begin
                    {label}_first_fire_cycle = cycle_count;
                end
                if ({label}_last_fire_cycle >= 0 &&
                    (cycle_count - {label}_last_fire_cycle) > {label}_max_fire_gap) begin
                    {label}_max_fire_gap = cycle_count - {label}_last_fire_cycle;
                end
                {label}_last_fire_cycle = cycle_count;
                {label}_fire_cycles = {label}_fire_cycles + 1;
            end"""
        )

    if present_channels:
        format_fields = " ".join(
            f"{label}_vld_cycles=%0d {label}_vld_edges=%0d {label}_busy_cycles=%0d"
            for label, _, _ in present_channels
        )
        args = ", ".join(
            f"{label}_vld_cycles, {label}_vld_edges, {label}_busy_cycles"
            for label, _, _ in present_channels
        )
        displays.append(
            f'            $display("debug p2p_counters {format_fields}",\n'
            f"                     {args});"
        )
        for label, _, _ in present_channels:
            displays.append(
                f"""            $display("debug p2p_timing {label} first_edge=%0d last_edge=%0d max_edge_gap=%0d first_fire=%0d last_fire=%0d max_fire_gap=%0d fire_cycles=%0d",
                     {label}_first_vld_edge_cycle, {label}_last_vld_edge_cycle,
                     {label}_max_vld_edge_gap, {label}_first_fire_cycle,
                     {label}_last_fire_cycle, {label}_max_fire_gap,
                     {label}_fire_cycles);"""
            )

    enc_kind_signal = None
    for candidate in (
        "m_encoder_in_m_chan_data_kind",
        "m_encoder_in_m_chan_data_kind_slice",
    ):
        if has_internal_signal(module_body, candidate):
            enc_kind_signal = candidate
            break

    if enc_kind_signal and has_internal_signal(module_body, "m_encoder_in_m_chan_vld"):
        decls.extend(
            [
                "    integer enc_in_kind_vld_cycles [0:4];",
                "    integer enc_in_kind_vld_edges [0:4];",
            ]
        )
        inits.append(
            """        for (counter_index = 0; counter_index < 5; counter_index = counter_index + 1) begin
            enc_in_kind_vld_cycles[counter_index] = 0;
            enc_in_kind_vld_edges[counter_index] = 0;
        end"""
        )
        updates.append(
            f"""            if (dut.m_encoder_in_m_chan_vld && dut.{enc_kind_signal} >= 0 && dut.{enc_kind_signal} <= 4) begin
                enc_in_kind_vld_cycles[dut.{enc_kind_signal}] =
                    enc_in_kind_vld_cycles[dut.{enc_kind_signal}] + 1;
                if (!enc_in_prev_vld) begin
                    enc_in_kind_vld_edges[dut.{enc_kind_signal}] =
                        enc_in_kind_vld_edges[dut.{enc_kind_signal}] + 1;
                end
            end"""
        )
        displays.append(
            """            $display("debug enc_in_kind_vld_cycles k0=%0d k1=%0d k2=%0d k3=%0d k4=%0d",
                     enc_in_kind_vld_cycles[0], enc_in_kind_vld_cycles[1],
                     enc_in_kind_vld_cycles[2], enc_in_kind_vld_cycles[3],
                     enc_in_kind_vld_cycles[4]);
            $display("debug enc_in_kind_vld_edges k0=%0d k1=%0d k2=%0d k3=%0d k4=%0d",
                     enc_in_kind_vld_edges[0], enc_in_kind_vld_edges[1],
                     enc_in_kind_vld_edges[2], enc_in_kind_vld_edges[3],
                     enc_in_kind_vld_edges[4]);"""
        )

    def p2p_fire_condition(prefix: str) -> str:
        vld = f"{prefix}_m_chan_vld"
        busy = f"{prefix}_m_chan_busy"
        if has_internal_signal(module_body, busy):
            return f"dut.{vld} && !dut.{busy}"
        return f"dut.{vld}"

    enc_out_kind = "m_encoder_out_m_chan_data_kind"
    enc_out_class = "m_encoder_out_m_chan_data_class_id"
    enc_out_encoded = "m_encoder_out_m_chan_data_encoded"
    if (
        has_internal_signal(module_body, "m_encoder_out_m_chan_vld")
        and has_internal_signal(module_body, enc_out_kind)
    ):
        decls.extend(
            [
                "    integer training_enc_out_train_vld_edges;",
                "    integer training_enc_out_train_fire;",
            ]
        )
        inits.extend(
            [
                "        training_enc_out_train_vld_edges = 0;",
                "        training_enc_out_train_fire = 0;",
            ]
        )
        updates.append(
            f"""            if (dut.m_encoder_out_m_chan_vld && !enc_out_prev_vld &&
                dut.{enc_out_kind} == 2) begin
                training_enc_out_train_vld_edges = training_enc_out_train_vld_edges + 1;
            end
            if ({p2p_fire_condition("m_encoder_out")} && dut.{enc_out_kind} == 2) begin
                training_enc_out_train_fire = training_enc_out_train_fire + 1;
            end"""
        )

    if (
        has_internal_signal(module_body, "m_encoder_out_m_chan_vld")
        and has_internal_signal(module_body, enc_out_kind)
        and has_internal_signal(module_body, enc_out_class)
        and has_internal_signal(module_body, enc_out_encoded)
    ):
        decls.extend(
            [
                "    integer training_encoder_payload_popcount [0:4];",
                "    integer training_encoder_payload_weighted_sum [0:4];",
                "    integer training_encoder_payload_first_popcount [0:4];",
                "    integer training_encoder_payload_last_popcount [0:4];",
                "    integer training_encoder_payload_seen [0:4];",
                """    function integer training_encoder_weighted_sum;
        input logic [VECTOR_DIMENSION-1:0] value;
        integer bit_index;
        begin
            training_encoder_weighted_sum = 0;
            for (bit_index = 0; bit_index < VECTOR_DIMENSION; bit_index = bit_index + 1) begin
                if (value[bit_index]) begin
                    training_encoder_weighted_sum = training_encoder_weighted_sum + bit_index;
                end
            end
        end
    endfunction""",
            ]
        )
        inits.append(
            """        for (counter_index = 0; counter_index < 5; counter_index = counter_index + 1) begin
            training_encoder_payload_popcount[counter_index] = 0;
            training_encoder_payload_weighted_sum[counter_index] = 0;
            training_encoder_payload_first_popcount[counter_index] = 0;
            training_encoder_payload_last_popcount[counter_index] = 0;
            training_encoder_payload_seen[counter_index] = 0;
        end"""
        )
        enc_out_fire = p2p_fire_condition("m_encoder_out")
        updates.append(
            f"""            if ({enc_out_fire} && dut.{enc_out_kind} == 2 &&
                dut.{enc_out_class} >= 0 && dut.{enc_out_class} <= 4) begin
                if (!training_encoder_payload_seen[dut.{enc_out_class}]) begin
                    training_encoder_payload_first_popcount[dut.{enc_out_class}] =
                        $countones(dut.{enc_out_encoded});
                    training_encoder_payload_seen[dut.{enc_out_class}] = 1;
                end
                training_encoder_payload_popcount[dut.{enc_out_class}] =
                    training_encoder_payload_popcount[dut.{enc_out_class}] +
                    $countones(dut.{enc_out_encoded});
                training_encoder_payload_weighted_sum[dut.{enc_out_class}] =
                    training_encoder_payload_weighted_sum[dut.{enc_out_class}] +
                    training_encoder_weighted_sum(dut.{enc_out_encoded});
                training_encoder_payload_last_popcount[dut.{enc_out_class}] =
                    $countones(dut.{enc_out_encoded});
            end"""
        )

    bundler_kind = "m_bundler_in_m_chan_data_kind"
    bundler_class = "m_bundler_in_m_chan_data_class_id"
    bundler_valid_ngram = "m_bundler_in_m_chan_data_valid_ngram"
    bundler_ngram = "m_bundler_in_m_chan_data_ngram"
    if (
        has_internal_signal(module_body, "m_bundler_in_m_chan_vld")
        and has_internal_signal(module_body, bundler_kind)
        and has_internal_signal(module_body, bundler_valid_ngram)
    ):
        decls.extend(
            [
                "    integer training_bundler_train_invalid_vld_edges;",
                "    integer training_bundler_train_valid_vld_edges;",
                "    integer training_bundler_invalid_step_vld_edges;",
                "    integer training_bundler_train_invalid_fire;",
                "    integer training_bundler_train_valid_fire;",
                "    integer training_bundler_invalid_step_fire;",
            ]
        )
        inits.extend(
            [
                "        training_bundler_train_invalid_vld_edges = 0;",
                "        training_bundler_train_valid_vld_edges = 0;",
                "        training_bundler_invalid_step_vld_edges = 0;",
                "        training_bundler_train_invalid_fire = 0;",
                "        training_bundler_train_valid_fire = 0;",
                "        training_bundler_invalid_step_fire = 0;",
            ]
        )
        bundler_fire = p2p_fire_condition("m_bundler_in")
        updates.append(
            f"""            if (dut.m_bundler_in_m_chan_vld && !bundler_in_prev_vld) begin
                if (dut.{bundler_kind} == 2 && !dut.{bundler_valid_ngram}) begin
                    training_bundler_train_invalid_vld_edges =
                        training_bundler_train_invalid_vld_edges + 1;
                end
                if (dut.{bundler_kind} == 2 && dut.{bundler_valid_ngram}) begin
                    training_bundler_train_valid_vld_edges =
                        training_bundler_train_valid_vld_edges + 1;
                end
                if (dut.{bundler_kind} == 3) begin
                    training_bundler_invalid_step_vld_edges =
                        training_bundler_invalid_step_vld_edges + 1;
                end
            end
            if ({bundler_fire}) begin
                if (dut.{bundler_kind} == 2 && !dut.{bundler_valid_ngram}) begin
                    training_bundler_train_invalid_fire =
                        training_bundler_train_invalid_fire + 1;
                end
                if (dut.{bundler_kind} == 2 && dut.{bundler_valid_ngram}) begin
                    training_bundler_train_valid_fire =
                        training_bundler_train_valid_fire + 1;
                end
                if (dut.{bundler_kind} == 3) begin
                    training_bundler_invalid_step_fire =
                        training_bundler_invalid_step_fire + 1;
                end
            end"""
        )

    if (
        has_internal_signal(module_body, "m_bundler_in_m_chan_vld")
        and has_internal_signal(module_body, bundler_kind)
        and has_internal_signal(module_body, bundler_class)
        and has_internal_signal(module_body, bundler_valid_ngram)
        and has_internal_signal(module_body, bundler_ngram)
    ):
        decls.extend(
            [
                "    integer training_bundler_payload_popcount [0:4];",
                "    integer training_bundler_payload_weighted_sum [0:4];",
                "    integer training_bundler_payload_first_popcount [0:4];",
                "    integer training_bundler_payload_last_popcount [0:4];",
                "    integer training_bundler_payload_seen [0:4];",
                """    function integer training_weighted_sum;
        input logic [VECTOR_DIMENSION-1:0] value;
        integer bit_index;
        begin
            training_weighted_sum = 0;
            for (bit_index = 0; bit_index < VECTOR_DIMENSION; bit_index = bit_index + 1) begin
                if (value[bit_index]) begin
                    training_weighted_sum = training_weighted_sum + bit_index;
                end
            end
        end
    endfunction""",
            ]
        )
        inits.append(
            """        for (counter_index = 0; counter_index < 5; counter_index = counter_index + 1) begin
            training_bundler_payload_popcount[counter_index] = 0;
            training_bundler_payload_weighted_sum[counter_index] = 0;
            training_bundler_payload_first_popcount[counter_index] = 0;
            training_bundler_payload_last_popcount[counter_index] = 0;
            training_bundler_payload_seen[counter_index] = 0;
        end"""
        )
        bundler_fire = p2p_fire_condition("m_bundler_in")
        updates.append(
            f"""            if ({bundler_fire} && dut.{bundler_kind} == 2 &&
                dut.{bundler_valid_ngram} && dut.{bundler_class} >= 0 &&
                dut.{bundler_class} <= 4) begin
                if (!training_bundler_payload_seen[dut.{bundler_class}]) begin
                    training_bundler_payload_first_popcount[dut.{bundler_class}] =
                        $countones(dut.{bundler_ngram});
                    training_bundler_payload_seen[dut.{bundler_class}] = 1;
                end
                training_bundler_payload_popcount[dut.{bundler_class}] =
                    training_bundler_payload_popcount[dut.{bundler_class}] +
                    $countones(dut.{bundler_ngram});
                training_bundler_payload_weighted_sum[dut.{bundler_class}] =
                    training_bundler_payload_weighted_sum[dut.{bundler_class}] +
                    training_weighted_sum(dut.{bundler_ngram});
                training_bundler_payload_last_popcount[dut.{bundler_class}] =
                    $countones(dut.{bundler_ngram});
            end"""
        )

    distance_kind = "m_distance_in_m_chan_data_kind"
    distance_valid_ngram = "m_distance_in_m_chan_data_valid_ngram"
    if (
        has_internal_signal(module_body, "m_encoder_out_m_chan_vld")
        and has_internal_signal(module_body, enc_out_kind)
        and has_internal_signal(module_body, "m_bundler_in_m_chan_vld")
        and has_internal_signal(module_body, bundler_kind)
        and has_internal_signal(module_body, bundler_valid_ngram)
        and has_internal_signal(module_body, "m_distance_in_m_chan_vld")
        and has_internal_signal(module_body, distance_kind)
        and has_internal_signal(module_body, distance_valid_ngram)
    ):
        decls.extend(
            [
                "    integer ngram_train_issue_cycles [0:1048575];",
                "    integer ngram_infer_issue_cycles [0:1048575];",
                "    integer ngram_train_head;",
                "    integer ngram_train_tail;",
                "    integer ngram_infer_head;",
                "    integer ngram_infer_tail;",
                "    integer ngram_train_latency_count;",
                "    integer ngram_train_latency_total;",
                "    integer ngram_train_latency_max;",
                "    integer ngram_infer_latency_count;",
                "    integer ngram_infer_latency_total;",
                "    integer ngram_infer_latency_max;",
                "    integer ngram_bundler_train_invalid_fire;",
                "    integer ngram_bundler_train_valid_fire;",
                "    integer ngram_distance_infer_invalid_fire;",
                "    integer ngram_distance_infer_valid_fire;",
                "    integer ngram_latency_value;",
            ]
        )
        inits.extend(
            [
                "        ngram_train_head = 0;",
                "        ngram_train_tail = 0;",
                "        ngram_infer_head = 0;",
                "        ngram_infer_tail = 0;",
                "        ngram_train_latency_count = 0;",
                "        ngram_train_latency_total = 0;",
                "        ngram_train_latency_max = 0;",
                "        ngram_infer_latency_count = 0;",
                "        ngram_infer_latency_total = 0;",
                "        ngram_infer_latency_max = 0;",
                "        ngram_bundler_train_invalid_fire = 0;",
                "        ngram_bundler_train_valid_fire = 0;",
                "        ngram_distance_infer_invalid_fire = 0;",
                "        ngram_distance_infer_valid_fire = 0;",
                "        ngram_latency_value = 0;",
            ]
        )
        enc_out_fire = p2p_fire_condition("m_encoder_out")
        bundler_fire = p2p_fire_condition("m_bundler_in")
        distance_fire = p2p_fire_condition("m_distance_in")
        updates.append(
            f"""            if ({enc_out_fire} && dut.{enc_out_kind} == 2) begin
                ngram_train_issue_cycles[ngram_train_tail] = cycle_count;
                ngram_train_tail = ngram_train_tail + 1;
            end
            if ({enc_out_fire} && dut.{enc_out_kind} == 4) begin
                ngram_infer_issue_cycles[ngram_infer_tail] = cycle_count;
                ngram_infer_tail = ngram_infer_tail + 1;
            end
            if ({bundler_fire} && dut.{bundler_kind} == 2) begin
                if (dut.{bundler_valid_ngram}) begin
                    ngram_bundler_train_valid_fire = ngram_bundler_train_valid_fire + 1;
                end else begin
                    ngram_bundler_train_invalid_fire = ngram_bundler_train_invalid_fire + 1;
                end
                if (ngram_train_head >= ngram_train_tail) begin
                    $error("ngram train output without queued encoder input at cycle %0d", cycle_count);
                    error_count = error_count + 1;
                end else begin
                    ngram_latency_value = cycle_count - ngram_train_issue_cycles[ngram_train_head];
                    ngram_train_head = ngram_train_head + 1;
                    ngram_train_latency_count = ngram_train_latency_count + 1;
                    ngram_train_latency_total = ngram_train_latency_total + ngram_latency_value;
                    if (ngram_latency_value > ngram_train_latency_max) begin
                        ngram_train_latency_max = ngram_latency_value;
                    end
                end
            end
            if ({distance_fire} && dut.{distance_kind} == 4) begin
                if (dut.{distance_valid_ngram}) begin
                    ngram_distance_infer_valid_fire = ngram_distance_infer_valid_fire + 1;
                end else begin
                    ngram_distance_infer_invalid_fire = ngram_distance_infer_invalid_fire + 1;
                end
                if (ngram_infer_head >= ngram_infer_tail) begin
                    $error("ngram infer output without queued encoder input at cycle %0d", cycle_count);
                    error_count = error_count + 1;
                end else begin
                    ngram_latency_value = cycle_count - ngram_infer_issue_cycles[ngram_infer_head];
                    ngram_infer_head = ngram_infer_head + 1;
                    ngram_infer_latency_count = ngram_infer_latency_count + 1;
                    ngram_infer_latency_total = ngram_infer_latency_total + ngram_latency_value;
                    if (ngram_latency_value > ngram_infer_latency_max) begin
                        ngram_infer_latency_max = ngram_latency_value;
                    end
                end
            end"""
        )
        displays.append(
            """            $display("debug ngram_latency train_count=%0d train_avg=%0d train_max=%0d train_pending=%0d infer_count=%0d infer_avg=%0d infer_max=%0d infer_pending=%0d",
                     ngram_train_latency_count,
                     (ngram_train_latency_count == 0) ? 0 :
                         (ngram_train_latency_total / ngram_train_latency_count),
                     ngram_train_latency_max,
                     ngram_train_tail - ngram_train_head,
                     ngram_infer_latency_count,
                     (ngram_infer_latency_count == 0) ? 0 :
                         (ngram_infer_latency_total / ngram_infer_latency_count),
                     ngram_infer_latency_max,
                     ngram_infer_tail - ngram_infer_head);
            $display("debug ngram_output_counts train_invalid=%0d train_valid=%0d infer_invalid=%0d infer_valid=%0d",
                     ngram_bundler_train_invalid_fire,
                     ngram_bundler_train_valid_fire,
                     ngram_distance_infer_invalid_fire,
                     ngram_distance_infer_valid_fire);"""
        )

    training_display_fields = []
    if (
        has_internal_signal(module_body, "m_encoder_out_m_chan_vld")
        and has_internal_signal(module_body, enc_out_kind)
    ):
        training_display_fields.extend(
            [
                ("enc_out_train_edges", "training_enc_out_train_vld_edges"),
                ("enc_out_train_fire", "training_enc_out_train_fire"),
            ]
        )
    if (
        has_internal_signal(module_body, "m_bundler_in_m_chan_vld")
        and has_internal_signal(module_body, bundler_kind)
        and has_internal_signal(module_body, bundler_valid_ngram)
    ):
        training_display_fields.extend(
            [
                ("bundler_train_invalid_edges", "training_bundler_train_invalid_vld_edges"),
                ("bundler_train_valid_edges", "training_bundler_train_valid_vld_edges"),
                ("bundler_invalid_step_edges", "training_bundler_invalid_step_vld_edges"),
                ("bundler_train_invalid_fire", "training_bundler_train_invalid_fire"),
                ("bundler_train_valid_fire", "training_bundler_train_valid_fire"),
                ("bundler_invalid_step_fire", "training_bundler_invalid_step_fire"),
            ]
        )
    if training_display_fields:
        training_format = " ".join(f"{name}=%0d" for name, _ in training_display_fields)
        training_args = ", ".join(signal for _, signal in training_display_fields)
        displays.append(
            f'            $display("debug training_path_counters_rtl {training_format}",\n'
            f"                     {training_args});"
        )
    if (
        has_internal_signal(module_body, "m_encoder_out_m_chan_vld")
        and has_internal_signal(module_body, enc_out_kind)
        and has_internal_signal(module_body, enc_out_class)
        and has_internal_signal(module_body, enc_out_encoded)
    ):
        displays.append(
            """            $display("debug rtl_encoder_payload_popcount c0=%0d c1=%0d c2=%0d c3=%0d c4=%0d",
                     training_encoder_payload_popcount[0],
                     training_encoder_payload_popcount[1],
                     training_encoder_payload_popcount[2],
                     training_encoder_payload_popcount[3],
                     training_encoder_payload_popcount[4]);
            $display("debug rtl_encoder_payload_weighted_sum c0=%0d c1=%0d c2=%0d c3=%0d c4=%0d",
                     training_encoder_payload_weighted_sum[0],
                     training_encoder_payload_weighted_sum[1],
                     training_encoder_payload_weighted_sum[2],
                     training_encoder_payload_weighted_sum[3],
                     training_encoder_payload_weighted_sum[4]);
            $display("debug rtl_encoder_payload_first_last c0_first=%0d c0_last=%0d c1_first=%0d c1_last=%0d c2_first=%0d c2_last=%0d c3_first=%0d c3_last=%0d c4_first=%0d c4_last=%0d",
                     training_encoder_payload_first_popcount[0],
                     training_encoder_payload_last_popcount[0],
                     training_encoder_payload_first_popcount[1],
                     training_encoder_payload_last_popcount[1],
                     training_encoder_payload_first_popcount[2],
                     training_encoder_payload_last_popcount[2],
                     training_encoder_payload_first_popcount[3],
                     training_encoder_payload_last_popcount[3],
                     training_encoder_payload_first_popcount[4],
                     training_encoder_payload_last_popcount[4]);"""
        )
    if (
        has_internal_signal(module_body, "m_bundler_in_m_chan_vld")
        and has_internal_signal(module_body, bundler_kind)
        and has_internal_signal(module_body, bundler_class)
        and has_internal_signal(module_body, bundler_valid_ngram)
        and has_internal_signal(module_body, bundler_ngram)
    ):
        displays.append(
            """            $display("debug rtl_bundler_payload_popcount c0=%0d c1=%0d c2=%0d c3=%0d c4=%0d",
                     training_bundler_payload_popcount[0],
                     training_bundler_payload_popcount[1],
                     training_bundler_payload_popcount[2],
                     training_bundler_payload_popcount[3],
                     training_bundler_payload_popcount[4]);
            $display("debug rtl_bundler_payload_weighted_sum c0=%0d c1=%0d c2=%0d c3=%0d c4=%0d",
                     training_bundler_payload_weighted_sum[0],
                     training_bundler_payload_weighted_sum[1],
                     training_bundler_payload_weighted_sum[2],
                     training_bundler_payload_weighted_sum[3],
                     training_bundler_payload_weighted_sum[4]);
            $display("debug rtl_bundler_payload_first_last c0_first=%0d c0_last=%0d c1_first=%0d c1_last=%0d c2_first=%0d c2_last=%0d c3_first=%0d c3_last=%0d c4_first=%0d c4_last=%0d",
                     training_bundler_payload_first_popcount[0],
                     training_bundler_payload_last_popcount[0],
                     training_bundler_payload_first_popcount[1],
                     training_bundler_payload_last_popcount[1],
                     training_bundler_payload_first_popcount[2],
                     training_bundler_payload_last_popcount[2],
                     training_bundler_payload_first_popcount[3],
                     training_bundler_payload_last_popcount[3],
                     training_bundler_payload_first_popcount[4],
                     training_bundler_payload_last_popcount[4]);"""
        )

    return {
        "decls": "\n".join(decls),
        "inits": "\n".join(inits),
        "updates": "\n".join(updates + prev_updates),
        "displays": "\n".join(displays),
    }


def generate_debug_task(
    module_body: str,
    cmd_valid_signal: str = "cmd_valid",
    cmd_ready_signal: str = "cmd_ready",
    rsp_valid_signal: str = "rsp_valid",
    rsp_ready_signal: str = "rsp_ready",
    top_cmd_extra_format: str = " hold=%0d",
    top_cmd_extra_args: str = ", command_hold_cycles",
) -> str:
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
                ("vld", "m_encoder_in_m_chan_vld"),
                ("busy", "m_encoder_in_m_chan_busy"),
                ("out_unval", "m_encoder_in_output_m_unvalidated_req"),
                ("in_unack", "m_encoder_in_input_m_unacked_req"),
            ],
        ),
        (
            "enc_out",
            [
                ("vld", "m_encoder_out_m_chan_vld"),
                ("busy", "m_encoder_out_m_chan_busy"),
                ("out_unval", "m_encoder_out_output_m_unvalidated_req"),
                ("in_unack", "m_encoder_out_input_m_unacked_req"),
            ],
        ),
        (
            "bundler_in",
            [
                ("vld", "m_bundler_in_m_chan_vld"),
                ("busy", "m_bundler_in_m_chan_busy"),
                ("out_unval", "m_bundler_in_output_m_unvalidated_req"),
                ("in_unack", "m_bundler_in_input_m_unacked_req"),
            ],
        ),
        (
            "distance_in",
            [
                ("vld", "m_distance_in_m_chan_vld"),
                ("busy", "m_distance_in_m_chan_busy"),
                ("out_unval", "m_distance_in_output_m_unvalidated_req"),
                ("in_unack", "m_distance_in_input_m_unacked_req"),
            ],
        ),
        (
            "distance_done",
            [
                ("vld", "m_distance_done_m_chan_vld"),
                ("busy", "m_distance_done_m_chan_busy"),
                ("out_unval", "m_distance_done_output_m_unvalidated_req"),
                ("in_unack", "m_distance_done_input_m_unacked_req"),
            ],
        ),
        (
            "ngram_control_done",
            [
                ("vld", "m_ngram_control_done_m_chan_vld"),
                ("busy", "m_ngram_control_done_m_chan_busy"),
                ("in_unack", "m_ngram_control_done_input_m_unacked_req"),
            ],
        ),
        (
            "train_control_done",
            [
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
                ("kind", "m_encoder_in_m_chan_data_kind"),
                ("kind_slice", "m_encoder_in_m_chan_data_kind_slice"),
                ("class", "m_encoder_in_m_chan_data_class_id"),
            ],
        ),
        (
            "enc_out_data",
            [
                ("kind", "m_encoder_out_m_chan_data_kind"),
                ("class", "m_encoder_out_m_chan_data_class_id"),
            ],
        ),
        (
            "bundler_in_data",
            [
                ("kind", "m_bundler_in_m_chan_data_kind"),
                ("class", "m_bundler_in_m_chan_data_class_id"),
                ("valid_ngram", "m_bundler_in_m_chan_data_valid_ngram"),
            ],
        ),
        (
            "distance_in_data",
            [
                ("kind", "m_distance_in_m_chan_data_kind"),
                ("class", "m_distance_in_m_chan_data_class_id"),
                ("valid_ngram", "m_distance_in_m_chan_data_valid_ngram"),
            ],
        ),
        (
            "distance_done_data",
            [
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

    p2p_counter_logic = generate_p2p_counter_logic(module_body)

    return f"""
    task print_dut_debug;
        begin
            $display("debug top_cmd_kind_fires k0=%0d k1=%0d k2=%0d k3=%0d k4=%0d{top_cmd_extra_format}",
                     top_cmd_kind_fires[0], top_cmd_kind_fires[1],
                     top_cmd_kind_fires[2], top_cmd_kind_fires[3],
                     top_cmd_kind_fires[4]{top_cmd_extra_args});
{p2p_counter_logic["displays"]}
            $display("debug top cmd_v=%0b cmd_r=%0b rsp_v=%0b rsp_r=%0b outstanding=%0d",
                     {cmd_valid_signal}, {cmd_ready_signal}, {rsp_valid_signal}, {rsp_ready_signal}, outstanding);
{displays.rstrip()}
        end
    endtask
"""


def generate_p2p_tb(args: argparse.Namespace,
                    module_body: str,
                    ports: Dict[str, Port]) -> str:
    p2p_counter_logic = generate_p2p_counter_logic(module_body)
    clk = find_exact(ports, "clk")
    rst = find_exact(ports, "rst")
    cmd_data, cmd_vld, cmd_busy = find_p2p_group(ports, "cmd")
    rsp_data, rsp_vld, rsp_busy = find_p2p_group(ports, "rsp")

    command_width = width_to_int(cmd_data.width)
    response_width = width_to_int(rsp_data.width)
    command_kind_bits = 3
    class_bits = 3
    level_bits = (command_width - command_kind_bits - class_bits) // args.num_features
    distance_bits = (response_width - 1) // args.num_classes
    if command_kind_bits + class_bits + (level_bits * args.num_features) != command_width:
        raise SystemExit(f"cannot infer command P2P layout from width {command_width}")
    if 1 + (distance_bits * args.num_classes) != response_width:
        raise SystemExit(f"cannot infer response P2P layout from width {response_width}")

    command_path = json.dumps(str(args.trace_dir / "commands.txt"))
    response_path = json.dumps(str(args.trace_dir / "expected_responses.txt"))

    signal_for_port = {
        clk.name: "clk",
        rst.name: "rst",
        cmd_data.name: "cmd_data",
        cmd_vld.name: "cmd_vld",
        cmd_busy.name: "cmd_busy",
        rsp_data.name: "rsp_data",
        rsp_vld.name: "rsp_vld",
        rsp_busy.name: "rsp_busy",
    }
    missing = [port for port in ports.values() if port.name not in signal_for_port]
    if missing:
        found = ", ".join(f"{p.direction}:{canonical(p.name)}{p.width}" for p in missing)
        raise SystemExit(f"unconnected P2P top ports: {found}")

    connections = [
        port_connection(port, signal_for_port[port.name])
        for port in ports.values()
    ]

    response_scan_format = " ".join(["%d"] * (args.num_classes + 1) + ["%s", "%s"])
    response_scan_args = ", ".join(
        ["expected_valid"]
        + [f"expected_distance[{class_id}]" for class_id in range(args.num_classes)]
        + ["expected_predicted_token", "expected_actual_token"]
    )

    return f"""`timescale 1ns/1ps

module hdc_accelerator_rtl_tb;
    localparam integer NUM_FEATURES = {args.num_features};
    localparam integer NUM_CLASSES = {args.num_classes};
    localparam integer VECTOR_DIMENSION = {args.vector_dimension};
    localparam integer COMMAND_WIDTH = {command_width};
    localparam integer RESPONSE_WIDTH = {response_width};
    localparam integer COMMAND_KIND_BITS = {command_kind_bits};
    localparam integer CLASS_BITS = {class_bits};
    localparam integer LEVEL_BITS = {level_bits};
    localparam integer DISTANCE_BITS = {distance_bits};
    localparam integer TIMEOUT_CYCLES = {args.timeout_cycles};
    localparam integer PROGRESS_CYCLES = {args.progress_cycles};
    localparam integer RESET_CYCLES = {args.reset_cycles};
    localparam string COMMAND_PATH = {command_path};
    localparam string RESPONSE_PATH = {response_path};

    logic clk;
    logic rst;
    logic{width_or_scalar(cmd_data.width)} cmd_data;
    logic cmd_vld;
    wire cmd_busy;
    wire{width_or_scalar(rsp_data.width)} rsp_data;
    wire rsp_vld;
    logic rsp_busy;

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
    integer cmd_first_accept_cycle;
    integer cmd_last_accept_cycle;
    integer cmd_max_accept_gap;
    integer outstanding;
    integer next_kind;
    integer accepted_kind;
    integer next_class_id;
    integer next_levels [0:NUM_FEATURES-1];
    integer has_command;
    integer issue_cycles [0:1048575];
    integer issue_head;
    integer issue_tail;
    integer total_latency;
    integer max_latency;
    integer error_count;
    integer top_cmd_kind_fires [0:4];
    integer counter_index;
{p2p_counter_logic["decls"]}

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

    task pack_next_command;
        integer feature;
        integer low;
        begin
            cmd_data = '0;
            cmd_data[0 +: COMMAND_KIND_BITS] = next_kind;
            cmd_data[COMMAND_KIND_BITS +: CLASS_BITS] = next_class_id;
            for (feature = 0; feature < NUM_FEATURES; feature = feature + 1) begin
                low = COMMAND_KIND_BITS + CLASS_BITS + (feature * LEVEL_BITS);
                cmd_data[low +: LEVEL_BITS] = next_levels[feature];
            end
        end
    endtask

    task drive_next_command;
        begin
            pack_next_command();
            cmd_vld = 1'b1;
        end
    endtask

    function integer parse_expected_signed_token;
        input string token;
        integer value;
        integer rc;
        begin
            if (token == "-1") begin
                parse_expected_signed_token = -1;
            end else begin
                rc = $sscanf(token, "%d", value);
                if (rc != 1) begin
                    $fatal(1, "Malformed signed expected-response token: %s", token);
                end
                parse_expected_signed_token = value;
            end
        end
    endfunction

    task check_response;
        integer rc;
        integer expected_valid;
        integer expected_distance [0:NUM_CLASSES-1];
        integer expected_predicted;
        integer expected_actual;
        string expected_predicted_token;
        string expected_actual_token;
        integer class_id;
        integer predicted;
        integer best_distance;
        integer actual_valid;
        integer actual_distance [0:NUM_CLASSES-1];
        integer response_has_unknown;
        logic [RESPONSE_WIDTH-1:0] response_snapshot;
        logic [DISTANCE_BITS-1:0] actual_distance_bits [0:NUM_CLASSES-1];
        integer low;
        integer latency;
        begin
            rc = $fscanf(response_fd, "{response_scan_format}", {response_scan_args});
            if (rc != (NUM_CLASSES + 3)) begin
                $fatal(1, "Malformed expected response %0d: parsed %0d fields",
                       responses_received, rc);
            end
            expected_predicted = parse_expected_signed_token(expected_predicted_token);
            expected_actual = parse_expected_signed_token(expected_actual_token);

            response_snapshot = rsp_data;
            response_has_unknown = 0;
            if (^response_snapshot === 1'bx) begin
                $error("response %0d contains X/Z bits: rsp_data=0x%0h",
                       responses_received, response_snapshot);
                error_count = error_count + 1;
                response_has_unknown = 1;
            end

            actual_valid = response_snapshot[0];
            for (class_id = 0; class_id < NUM_CLASSES; class_id = class_id + 1) begin
                low = 1 + (class_id * DISTANCE_BITS);
                actual_distance_bits[class_id] = response_snapshot[low +: DISTANCE_BITS];
                actual_distance[class_id] = actual_distance_bits[class_id];
                if (^actual_distance_bits[class_id] === 1'bx) begin
                    $error("distance[%0d] contains X/Z at response %0d: raw=0x%0h expected=%0d",
                           class_id, responses_received, actual_distance_bits[class_id],
                           expected_distance[class_id]);
                    error_count = error_count + 1;
                    response_has_unknown = 1;
                end
            end

            if (responses_received < 8 || response_has_unknown) begin
                $display("debug response idx=%0d raw=0x%0h valid got=%0d expected=%0d distances got=%0d,%0d,%0d,%0d,%0d expected=%0d,%0d,%0d,%0d,%0d expected_pred=%0d",
                         responses_received, response_snapshot, actual_valid, expected_valid,
                         actual_distance[0], actual_distance[1], actual_distance[2],
                         actual_distance[3], actual_distance[4],
                         expected_distance[0], expected_distance[1], expected_distance[2],
                         expected_distance[3], expected_distance[4], expected_predicted);
            end

            if ((actual_valid != 0) != (expected_valid != 0)) begin
                $error("valid_prediction mismatch at response %0d: got %0d expected %0d",
                       responses_received, actual_valid, expected_valid);
                error_count = error_count + 1;
            end

            predicted = 0;
            best_distance = actual_distance[0];
            for (class_id = 0; class_id < NUM_CLASSES; class_id = class_id + 1) begin
                if (!response_has_unknown && actual_distance[class_id] != expected_distance[class_id]) begin
                    $error("distance[%0d] mismatch at response %0d: got %0d expected %0d",
                           class_id, responses_received,
                           actual_distance[class_id], expected_distance[class_id]);
                    error_count = error_count + 1;
                end
                if (!response_has_unknown && class_id > 0 && actual_distance[class_id] < best_distance) begin
                    best_distance = actual_distance[class_id];
                    predicted = class_id;
                end
            end

            if (!response_has_unknown && expected_valid != 0 && predicted != expected_predicted) begin
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

{generate_debug_task(module_body, "cmd_vld", "!cmd_busy", "rsp_vld", "!rsp_busy", " top_first_accept=%0d top_last_accept=%0d top_max_accept_gap=%0d", ", cmd_first_accept_cycle, cmd_last_accept_cycle, cmd_max_accept_gap")}

    task finish_if_complete;
        integer extra;
        integer rc;
        real average_latency;
        begin
            if (!has_command && !cmd_vld && outstanding == 0) begin
                rc = $fscanf(response_fd, "%d", extra);
                if (rc == 1) begin
                    $fatal(1, "Expected response file contains extra response data");
                end

                average_latency = (responses_received == 0)
                    ? 0.0
                    : (1.0 * total_latency) / responses_received;
                $display("RTL P2P simulation complete");
                $display("cycles=%0d", cycle_count);
                $display("commands_sent=%0d", commands_sent);
                $display("inference_sent=%0d", inference_sent);
                $display("responses_received=%0d", responses_received);
                $display("command_stall_cycles=%0d", command_stall_cycles);
                $display("response_stall_cycles=%0d", response_stall_cycles);
                $display("average_inference_latency=%0f", average_latency);
                $display("max_inference_latency=%0d", max_latency);
                $display("errors=%0d", error_count);
                print_dut_debug();

                $fclose(command_fd);
                $fclose(response_fd);
                if (error_count != 0) begin
                    $fatal(1, "RTL response comparison failed");
                end
                $finish;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        cmd_vld = 1'b0;
        cmd_data = '0;
        rsp_busy = 1'b0;
        cycle_count = 0;
        commands_sent = 0;
        inference_sent = 0;
        responses_received = 0;
        command_stall_cycles = 0;
        response_stall_cycles = 0;
        cmd_first_accept_cycle = -1;
        cmd_last_accept_cycle = -1;
        cmd_max_accept_gap = 0;
        outstanding = 0;
        accepted_kind = 0;
        has_command = 0;
        issue_head = 0;
        issue_tail = 0;
        total_latency = 0;
        max_latency = 0;
        error_count = 0;
        for (counter_index = 0; counter_index < 5; counter_index = counter_index + 1) begin
            top_cmd_kind_fires[counter_index] = 0;
        end
{p2p_counter_logic["inits"]}

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
                $fatal(1, "RTL P2P simulation timeout after %0d cycles", cycle_count);
            end

            if (!cmd_vld && has_command) begin
                drive_next_command();
            end

            @(posedge clk);
            #1;
            cycle_count = cycle_count + 1;
            if (PROGRESS_CYCLES > 0 && (cycle_count % PROGRESS_CYCLES) == 0) begin
                $display("progress cycles=%0d commands=%0d inference=%0d responses=%0d outstanding=%0d cmd_vld=%0b cmd_busy=%0b rsp_vld=%0b rsp_busy=%0b",
                         cycle_count, commands_sent, inference_sent, responses_received, outstanding,
                         cmd_vld, cmd_busy, rsp_vld, rsp_busy);
            end
{p2p_counter_logic["updates"]}

            if (cmd_vld && !cmd_busy) begin
                accepted_kind = cmd_data[0 +: COMMAND_KIND_BITS];
                if (cmd_first_accept_cycle < 0) begin
                    cmd_first_accept_cycle = cycle_count;
                end
                if (cmd_last_accept_cycle >= 0 &&
                    (cycle_count - cmd_last_accept_cycle) > cmd_max_accept_gap) begin
                    cmd_max_accept_gap = cycle_count - cmd_last_accept_cycle;
                end
                cmd_last_accept_cycle = cycle_count;
                commands_sent = commands_sent + 1;
                if (accepted_kind >= 0 && accepted_kind <= 4) begin
                    top_cmd_kind_fires[accepted_kind] = top_cmd_kind_fires[accepted_kind] + 1;
                end
                if (commands_sent < 64) begin
                    $display("debug accepted_cmd idx=%0d trace_kind=%0d packed_kind=%0d class=%0d cycle=%0d",
                             commands_sent, next_kind, accepted_kind,
                             cmd_data[COMMAND_KIND_BITS +: CLASS_BITS], cycle_count);
                end
                if (accepted_kind == 4) begin
                    inference_sent = inference_sent + 1;
                    outstanding = outstanding + 1;
                    issue_cycles[issue_tail] = cycle_count;
                    issue_tail = issue_tail + 1;
                end
                cmd_vld = 1'b0;
                read_next_command(has_command);
            end else if (cmd_vld && cmd_busy) begin
                command_stall_cycles = command_stall_cycles + 1;
            end

            if (rsp_vld && rsp_busy) begin
                response_stall_cycles = response_stall_cycles + 1;
            end

            if (rsp_vld && !rsp_busy) begin
                if (outstanding <= 0) begin
                    $fatal(1, "Unexpected response with no outstanding inference: cycle=%0d responses=%0d rsp_data=0x%0h",
                           cycle_count, responses_received, rsp_data);
                end
                check_response();
                responses_received = responses_received + 1;
                outstanding = outstanding - 1;
            end

            finish_if_complete();
        end
    end
endmodule
"""


def generate_tb(args: argparse.Namespace) -> str:
    module_body = read_module_body(args.top, "HDC_Accelerator")
    p2p_counter_logic = generate_p2p_counter_logic(module_body)
    ports = parse_ports(args.top, "HDC_Accelerator")
    clk = find_exact(ports, "clk")
    rst = find_exact(ports, "rst")
    if find_optional_exact(ports, "cmd_valid") is None:
        return generate_p2p_tb(args, module_body, ports)

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
    localparam integer VECTOR_DIMENSION = {args.vector_dimension};
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
    integer total_latency;
    integer max_latency;
    integer error_count;
    integer command_hold_cycles;
    integer top_cmd_kind_fires [0:4];
    integer counter_index;
{p2p_counter_logic["decls"]}

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
        begin
            rc = $fscanf(response_fd, "%d", expected_valid);
            if (rc != 1) begin
                $fatal(1, "Missing expected response for received response %0d", responses_received);
            end

            for (class_id = 0; class_id < NUM_CLASSES; class_id = class_id + 1) begin
                rc = $fscanf(response_fd, "%d", expected_distance[class_id]);
                if (rc != 1) begin
                    $fatal(1, "Malformed expected response distance %0d", class_id);
                end
            end
            rc = $fscanf(response_fd, "%d %d", expected_predicted, expected_actual);
            if (rc != 2) begin
                $fatal(1, "Malformed expected response predicted/actual fields");
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
                print_dut_debug();

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
        total_latency = 0;
        max_latency = 0;
        error_count = 0;
        command_hold_cycles = 0;
        for (counter_index = 0; counter_index < 5; counter_index = counter_index + 1) begin
            top_cmd_kind_fires[counter_index] = 0;
        end
{p2p_counter_logic["inits"]}

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
            if (PROGRESS_CYCLES > 0 && (cycle_count % PROGRESS_CYCLES) == 0) begin
                $display("progress cycles=%0d commands=%0d inference=%0d responses=%0d outstanding=%0d",
                         cycle_count, commands_sent, inference_sent, responses_received, outstanding);
                print_dut_debug();
            end

            if (!cmd_valid && command_hold_cycles > 0) begin
                command_hold_cycles = command_hold_cycles - 1;
            end
{p2p_counter_logic["updates"]}

            if (cmd_valid && cmd_ready) begin
                commands_sent = commands_sent + 1;
                if (cmd_kind >= 0 && cmd_kind <= 4) begin
                    top_cmd_kind_fires[cmd_kind] = top_cmd_kind_fires[cmd_kind] + 1;
                end
                if (cmd_kind == 4) begin
                    inference_sent = inference_sent + 1;
                    outstanding = outstanding + 1;
                    issue_cycles[issue_tail] = cycle_count;
                    issue_tail = issue_tail + 1;
                end
                cmd_valid = 1'b0;
                command_hold_cycles = POST_COMMAND_HOLD_CYCLES;
                read_next_command(has_command);
            end else if (cmd_valid && !cmd_ready) begin
                command_stall_cycles = command_stall_cycles + 1;
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
    parser.add_argument("--vector-dimension", type=int, default=1024)
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
