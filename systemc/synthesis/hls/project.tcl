set_attr fpga_tool "vivado"
set_attr fpga_part "xc7v2000tflg1925-2"

# Clock period in ns for the synthesized accelerator.
set_attr clock_period 10

# Copied from the Stratus tutorial. Keep this available for later explicit
# memory mapping experiments.
use_hls_lib "mem_lib"

# Only HDC_Accelerator is the HLS target. Controller, dataset loading, main,
# CSV parsing, quantization, and metrics remain software/testbench code.
set hdc_accelerator_src "../src/hdc_accelerator.cpp"
if {[info exists ::env(HDC_HLS_SOURCE)] && $::env(HDC_HLS_SOURCE) != ""} {
    set hdc_accelerator_src "../src/$::env(HDC_HLS_SOURCE)"
} elseif {[info exists ::env(HDC_HLS_IMPORT_STUB)] && $::env(HDC_HLS_IMPORT_STUB) == "1"} {
    set hdc_accelerator_src "../src/hdc_accelerator_import_stub.cpp"
}

define_hls_module HDC_Accelerator $hdc_accelerator_src

define_hls_config HDC_Accelerator HLS_BASIC
