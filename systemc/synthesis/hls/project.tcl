set_attr fpga_tool "vivado"
set_attr fpga_part "xc7v2000tflg1925-2"

# Clock period in ns for the synthesized accelerator.
set_attr clock_period 10

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

define_hls_config HDC_Accelerator HLS_BASIC {
    set_attr inline_partial_constants on
}

define_sim_config RTL_BASIC {HDC_Accelerator RTL_V HLS_BASIC}
