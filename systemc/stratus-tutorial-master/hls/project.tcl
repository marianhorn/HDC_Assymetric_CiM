set_attr fpga_tool "vivado"
set_attr fpga_part "xc7v2000tflg1925-2"

set_attr clock_period 10

use_hls_lib "mem_lib"

define_system_module ringbuffer_tb ../src/ringbuffer_tb.cpp
define_system_module sc_main ../src/sc_main.cpp

define_hls_module ringbuffer ../src/ringbuffer.cpp

define_hls_config ringbuffer HLS_BASIC

define_sim_config "ringbuffer_SystemC" "ringbuffer BEH HLS_BASIC"
