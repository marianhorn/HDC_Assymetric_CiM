set part_name xc7v2000tflg1925-2
set top_name HDC_Accelerator
set out_dir vivado_synth_hdc

file mkdir $out_dir

read_verilog v_rtl/hl5_block_1w1r.v
read_verilog [glob bdw_work/modules/HDC_Accelerator/HLS_BASIC/v_rtl/*.v]
read_verilog bdw_work/modules/HDC_Accelerator/HLS_BASIC/hdc_accelerator_rtl.v

synth_design -top $top_name -part $part_name

report_utilization -file $out_dir/utilization_synth.rpt
report_timing_summary -file $out_dir/timing_synth.rpt
report_high_fanout_nets -file $out_dir/high_fanout_synth.rpt

write_checkpoint -force $out_dir/post_synth.dcp