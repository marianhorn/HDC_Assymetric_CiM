set_attr fpga_tool "vivado"
set_attr fpga_part "xcvu57p-fsvk2892-3-e"

# Clock period in ns for the synthesized accelerator.
set_attr clock_period 10
set_attr default_input_delay 1.0
set_attr flatten_arrays none
set_attr output_style_reset_all on

# Enable Stratus' partial-constant propagation pass. This was suggested by
# HINT 00333 in the HLS report.
set_attr inline_partial_constants on

# Spend more compile time searching for a better schedule and implementation.
# The explicit HLS unroll directives still define the intended parallelism.
set_attr sched_effort high
set_attr sharing_effort_parts low
set_attr sharing_effort_regs low
set_attr parts_effort high
set_attr relax_timing on
set_attr sched_asap on
set_attr lsb_trimming on

# Copied from the Stratus tutorial. Keep this available for later explicit
# memory mapping experiments.
use_hls_lib "mem_lib"

# Full SystemC testbench for temporary RTL cosimulation attempts.
define_system_module ../src/main.cpp
define_system_module ../src/controller.cpp
define_system_module ../src/hdc_memory.cpp
define_system_module ../src/foot_dataset_loader.cpp

# Only HDC_Accelerator is the HLS target. Controller, dataset loading, main,
# CSV parsing, quantization, and metrics remain software/testbench code.
define_hls_module HDC_Accelerator ../src/hdc_accelerator.cpp

define_hls_config HDC_Accelerator HLS_BASIC

define_sim_config RTL_BASIC {HDC_Accelerator RTL_V HLS_BASIC}
