set_attr fpga_tool "vivado"
set_attr fpga_part "xcvu57p-fsvk2892-3-e"
set_attr clock_period 10
set_attr default_input_delay 1.0
set_attr output_style_reset_all on

set_attr sched_effort low
set_attr sharing_effort_parts low
set_attr sharing_effort_regs low
set_attr parts_effort low
set_attr relax_timing on
set_attr sched_asap on

# Encoder-style P2P experiment: one input token, 16 local compute cycles,
# then one output token. This mimics HDC_Accelerator::encoder_thread.
set_attr D P2P_ENCODER_MIMIC

# Uncomment for explicit nonblocking P2P API experiments.
# set_attr D P2P_EXPERIMENT_NB

define_system_module ../src/main.cpp
define_hls_module P2PPipeline ../src/p2p_pipeline.cpp

define_hls_config P2PPipeline HLS_BASIC
define_sim_config RTL_BASIC {P2PPipeline RTL_V HLS_BASIC}
