set part_name xcvu57p-fsvk2892-3-e
set top_name HDC_AcceleratorAxis
set out_dir vivado_synth_hdc_bigfpga
set hls_dir bdw_work/modules/HDC_Accelerator/HLS_BASIC
set top_rtl $hls_dir/hdc_accelerator_rtl.v
set axis_wrapper $out_dir/hdc_accelerator_axis_wrapper.sv
set generated_rtl [glob -nocomplain $hls_dir/v_rtl/*.v]
set memlib_rtl [glob -nocomplain mem_lib/*.v]

file mkdir $out_dir

if {![file exists $top_rtl]} {
    error "Missing HLS top RTL: $top_rtl. Run HLS with project_bigfpga.tcl first."
}
if {[llength $generated_rtl] == 0} {
    error "Missing generated memory RTL under $hls_dir/v_rtl. Run HLS with project_bigfpga.tcl first."
}
if {[llength $memlib_rtl] == 0} {
    error "Missing Stratus memory library RTL under mem_lib."
}

exec python3 generate_axis_wrapper.py --top $top_rtl --out $axis_wrapper

read_verilog $memlib_rtl
read_verilog $generated_rtl
read_verilog $top_rtl
read_verilog -sv $axis_wrapper

synth_design -top $top_name -part $part_name
create_clock -name aclk -period 10.000 [get_ports aclk]

report_utilization -file $out_dir/utilization_synth.rpt
report_timing_summary -check_timing_verbose -max_paths 10 -file $out_dir/timing_synth.rpt
report_high_fanout_nets -file $out_dir/high_fanout_synth.rpt

write_checkpoint -force $out_dir/post_synth.dcp

opt_design -directive Explore
place_design -directive ExtraNetDelay_high
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
phys_opt_design -directive AggressiveExplore

report_utilization -file $out_dir/utilization_impl.rpt
report_timing_summary -check_timing_verbose -max_paths 10 -file $out_dir/timing_impl.rpt
report_high_fanout_nets -file $out_dir/high_fanout_impl.rpt

write_checkpoint -force $out_dir/post_route.dcp
