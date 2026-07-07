`timescale 1ns/1ps

module p2p_pipeline_rtl_tb;
    localparam integer NUM_TOKENS = 16;
    localparam integer TIMEOUT_CYCLES = 5000;

    logic clk;
    logic rst;

    logic in_valid;
    wire in_ready;
    logic [2:0] in_kind;
    logic [7:0] in_value;

    wire out_valid;
    logic out_ready;
    wire [2:0] out_kind;
    wire [7:0] out_value;
    wire [15:0] out_sample_checksum;
    wire [15:0] out_encoded_checksum;

    wire [15:0] source_count;
    wire [15:0] stage_count;
    wire [15:0] sink_count;

    integer cycle_count;
    integer sent;
    integer received;
    integer errors;
    integer expected_kind;
    integer expected_value;
    integer input_value;
    integer expected_sample_checksum_value;
    integer expected_encoded_checksum_value;

    function integer expected_sample_checksum;
        input integer kind;
        input integer value;
        integer index;
        integer checksum;
        begin
            checksum = 0;
`ifdef P2P_PAYLOAD_SAMPLE
            for (index = 0; index < 32; index = index + 1) begin
                checksum = (checksum + ((value + kind + 3 * index) & 8'hff)) & 16'hffff;
            end
`elsif P2P_PAYLOAD_FULL
            for (index = 0; index < 32; index = index + 1) begin
                checksum = (checksum + ((value + kind + 3 * index) & 8'hff)) & 16'hffff;
            end
`endif
            expected_sample_checksum = checksum;
        end
    endfunction

    function integer expected_encoded_checksum;
        input integer kind;
        input integer value;
        integer index;
        integer checksum;
        integer low;
        integer high;
        begin
            checksum = 0;
`ifdef P2P_PAYLOAD_ENCODED
            for (index = 0; index < 16; index = index + 1) begin
                low = value | (index << 8) | (kind << 16) | ((13'h123 + index) << 19);
                high = 32'habc00000 + (index << 4);
                checksum = checksum ^ (low & 16'hffff);
                checksum = checksum ^ ((low >> 16) & 16'hffff);
                checksum = checksum ^ (high & 16'hffff);
                checksum = checksum ^ ((high >> 16) & 16'hffff);
            end
`elsif P2P_PAYLOAD_FULL
            for (index = 0; index < 16; index = index + 1) begin
                low = value | (index << 8) | (kind << 16) | ((13'h123 + index) << 19);
                high = 32'habc00000 + (index << 4);
                checksum = checksum ^ (low & 16'hffff);
                checksum = checksum ^ ((low >> 16) & 16'hffff);
                checksum = checksum ^ (high & 16'hffff);
                checksum = checksum ^ ((high >> 16) & 16'hffff);
            end
`endif
            expected_encoded_checksum = checksum & 16'hffff;
        end
    endfunction

    P2PPipeline dut (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_kind(in_kind),
        .in_value(in_value),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_kind(out_kind),
        .out_value(out_value),
        .out_sample_checksum(out_sample_checksum),
        .out_encoded_checksum(out_encoded_checksum),
        .source_count(source_count),
        .stage_count(stage_count),
        .sink_count(sink_count)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        in_valid = 1'b0;
        in_kind = '0;
        in_value = '0;
        out_ready = 1'b0;
        cycle_count = 0;
        sent = 0;
        received = 0;
        errors = 0;

        repeat (5) @(posedge clk);
        #1;
        rst = 1'b0;
        out_ready = 1'b1;
        repeat (2) @(posedge clk);
        #1;

        forever begin
            if (cycle_count > TIMEOUT_CYCLES) begin
                $fatal(1,
                       "P2P RTL timeout cycles=%0d sent=%0d received=%0d source_count=%0d stage_count=%0d sink_count=%0d",
                       cycle_count, sent, received, source_count, stage_count, sink_count);
            end

            if (!in_valid && sent < NUM_TOKENS) begin
                in_kind = sent % 5;
                in_value = 10 + sent;
                in_valid = 1'b1;
            end

            @(posedge clk);
            #1;
            cycle_count = cycle_count + 1;

            if ((cycle_count % 100) == 0) begin
                $display("progress cycles=%0d sent=%0d received=%0d in_v=%0b in_r=%0b out_v=%0b out_r=%0b counts=%0d/%0d/%0d",
                         cycle_count, sent, received, in_valid, in_ready,
                         out_valid, out_ready, source_count, stage_count, sink_count);
            end

            if (in_valid && in_ready) begin
                sent = sent + 1;
                in_valid = 1'b0;
            end

            if (out_valid && out_ready) begin
                expected_kind = received % 5;
                input_value = 10 + received;
                expected_value = input_value + 1;
                expected_sample_checksum_value =
                    expected_sample_checksum(expected_kind, input_value);
                expected_encoded_checksum_value =
                    expected_encoded_checksum(expected_kind, input_value);

                if (out_kind !== expected_kind[2:0] ||
                    out_value !== expected_value[7:0] ||
                    out_sample_checksum !== expected_sample_checksum_value[15:0] ||
                    out_encoded_checksum !== expected_encoded_checksum_value[15:0]) begin
                    $error("Mismatch token=%0d got kind=%0d value=%0d sample=%0d encoded=%0d expected kind=%0d value=%0d sample=%0d encoded=%0d",
                           received, out_kind, out_value,
                           out_sample_checksum, out_encoded_checksum,
                           expected_kind, expected_value,
                           expected_sample_checksum_value,
                           expected_encoded_checksum_value);
                    errors = errors + 1;
                end

                received = received + 1;
            end

            if (sent == NUM_TOKENS && received == NUM_TOKENS && !in_valid) begin
                $display("P2P RTL simulation complete");
                $display("cycles=%0d", cycle_count);
                $display("sent=%0d", sent);
                $display("received=%0d", received);
                $display("source_count=%0d", source_count);
                $display("stage_count=%0d", stage_count);
                $display("sink_count=%0d", sink_count);
                $display("errors=%0d", errors);

                if (source_count !== NUM_TOKENS[15:0] ||
                    stage_count !== NUM_TOKENS[15:0] ||
                    sink_count !== NUM_TOKENS[15:0]) begin
                    $fatal(1, "Counter mismatch");
                end

                if (errors != 0) begin
                    $fatal(1, "P2P RTL comparison failed");
                end

                $finish;
            end
        end
    end
endmodule
