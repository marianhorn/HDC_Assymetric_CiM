set_attr fpga_tool "vivado"
set_attr fpga_part "xc7v2000tflg1925-2"

# Clock period in ns for the synthesized accelerator.
set_attr clock_period 10

# Copied from the Stratus tutorial. Keep this available for later explicit
# memory mapping experiments.
use_hls_lib "mem_lib"

# Only HDC_Accelerator is the HLS target. Controller, dataset loading, main,
# CSV parsing, quantization, and metrics remain software/testbench code.
define_hls_module HDC_Accelerator ../src/hdc_accelerator.cpp

define_hls_config HDC_Accelerator HLS_BASIC
