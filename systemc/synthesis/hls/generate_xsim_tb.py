#!/usr/bin/env python3
"""Generate an XSim SystemVerilog testbench for the Stratus RTL top module."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class Port:
    direction: str
    name: str
    width: str


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


def parse_ports(verilog_path: pathlib.Path, module_name: str) -> Dict[str, Port]:
    text = strip_comments(verilog_path.read_text())
    body = module_text(text, module_name)
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


def generate_tb(args: argparse.Namespace) -> str:
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

        command_fd = $fopen(COMMAND_PATH, "r");
        if (command_fd == 0) begin
            $fatal(1, "Failed to open command file: %s", COMMAND_PATH);
        end
        response_fd = $fopen(RESPONSE_PATH, "r");
        if (response_fd == 0) begin
            $fatal(1, "Failed to open expected response file: %s", RESPONSE_PATH);
        end
        read_next_command(has_command);

        repeat (4) @(posedge clk);
        #1;
        rst = 1'b0;

        forever begin
            if (cycle_count > TIMEOUT_CYCLES) begin
                $fatal(1, "RTL simulation timeout after %0d cycles", cycle_count);
            end

            if (!cmd_valid && can_drive_next_command()) begin
                drive_next_command();
            end
            rsp_ready = (outstanding > 0);

            @(posedge clk);
            #1;
            cycle_count = cycle_count + 1;

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
    parser.add_argument("--max-outstanding", type=int, default=32)
    parser.add_argument("--timeout-cycles", type=int, default=50000000)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generate_tb(args))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
