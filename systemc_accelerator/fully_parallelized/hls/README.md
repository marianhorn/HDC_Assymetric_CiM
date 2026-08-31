# Fully Parallelized Accelerator: HLS and FPGA Flow

This directory synthesizes `HDC_Accelerator` with Cadence Stratus, verifies the
generated RTL with Vivado XSim, and implements the AXI-stream wrapper with
Vivado. Run all commands in this directory on `cae02`.

The configured target is `xcvu57p-fsvk2892-3-e` with a 10 ns clock. The HLS
boundary contains only `../src/hdc_accelerator.cpp`; controller, dataset loading,
quantization, and result evaluation remain software/testbench code.

## Environment

```sh
cd systemc_accelerator/fully_parallelized/hls
source setup_cae_cosim.sh
```

The setup script loads Vivado 2023.2 and Stratus 22.01 and configures Xcelium.
Load Vivado before Stratus if configuring the modules manually.

## Stratus HLS

For a clean synthesis:

```sh
rm -rf bdw_work Makefile.prj
make Makefile.prj
/usr/bin/time -v make hls_HDC_Accelerator_HLS_BASIC 2>&1 | tee output_hls.txt
```

Generated RTL is written to:

```text
bdw_work/modules/HDC_Accelerator/HLS_BASIC/hdc_accelerator_rtl.v
```

## RTL Simulation

The script generates an AXI wrapper and XSim testbench around the Stratus RTL.
Pass either the smoke trace or full dataset-00 trace directory:

```sh
bash run_rtl_sim_hdc.sh ../rtl_trace_dataset00_smoke20 2>&1 | tee output_rtl_smoke20.txt
bash run_rtl_sim_hdc.sh ../rtl_trace_dataset00 2>&1 | tee output_rtl_full.txt
```

The trace directory must contain `commands.txt` and
`expected_responses.txt`. Temporary XSim files are written under
`vivado_rtl_sim_hdc/`.

## Vivado Synthesis and Implementation

Run this only after successful HLS generation:

```sh
rm -rf vivado_synth_hdc
vivado -mode batch -source vivado_synth_hdc.tcl 2>&1 | tee output_vivado.txt
```

The script synthesizes and routes `HDC_AcceleratorAxis` for
`xcvu57p-fsvk2892-3-e`. Important outputs are:

```text
vivado_synth_hdc/utilization_synth.rpt
vivado_synth_hdc/timing_synth.rpt
vivado_synth_hdc/high_fanout_synth.rpt
vivado_synth_hdc/utilization_impl.rpt
vivado_synth_hdc/timing_impl.rpt
vivado_synth_hdc/high_fanout_impl.rpt
vivado_synth_hdc/post_route.dcp
```

Timing is met when the implementation timing summary reports non-negative WNS
and WHS. Resource percentages are reported in `utilization_impl.rpt`.

## Alternative FPGA Configuration

`project_smallfpga.tcl` and `vivado_synth_hdc_smallfpga.tcl` retain the smaller
FPGA experiment. They are not the default fully parallelized configuration.
