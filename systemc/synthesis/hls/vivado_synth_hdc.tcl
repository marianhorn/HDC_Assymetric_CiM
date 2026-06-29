set part_name xcvu57p-fsvk2892-3-e
set top_name HDC_Accelerator
set out_dir vivado_synth_hdc
set hls_dir bdw_work/modules/HDC_Accelerator/HLS_BASIC
set top_rtl $hls_dir/hdc_accelerator_rtl.v
set generated_rtl [glob -nocomplain $hls_dir/v_rtl/*.v]
set memlib_rtl [glob -nocomplain mem_lib/*.v]

file mkdir $out_dir

if {![file exists $top_rtl]} {
    error "Missing HLS top RTL: $top_rtl. Run make hls_HDC_Accelerator_HLS_BASIC first."
}
if {[llength $generated_rtl] == 0} {
    error "Missing generated memory RTL under $hls_dir/v_rtl. Run make hls_HDC_Accelerator_HLS_BASIC first."
}
if {[llength $memlib_rtl] == 0} {
    error "Missing Stratus memory library RTL under mem_lib."
}

read_verilog $memlib_rtl
read_verilog $generated_rtl
read_verilog $top_rtl

synth_design -top $top_name -part $part_name
create_clock -name clk -period 10.000 [get_ports clk]

report_utilization -file $out_dir/utilization_synth.rpt
report_timing_summary -check_timing_verbose -max_paths 10 -file $out_dir/timing_synth.rpt
report_high_fanout_nets -file $out_dir/high_fanout_synth.rpt

write_checkpoint -force $out_dir/post_synth.dcp
