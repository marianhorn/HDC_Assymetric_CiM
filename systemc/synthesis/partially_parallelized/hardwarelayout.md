# HDC Accelerator Hardware Layout

This file tracks the current HLS/RTL accelerator structure, measured implementation results, and planned hardware changes. It is intended to keep optimization decisions explicit and reproducible.

## Current Target

- Top module: `HDC_Accelerator`
- HLS tool: Cadence Stratus 22.01 on CAE
- RTL simulation: Vivado XSim 2023.2
- FPGA implementation: Vivado 2023.2
- Device targeted by Stratus and Vivado: `xczu3eg-sbva484-1-i`
- Clock target: 10 ns
- Vector dimension: 1024
- Hypervector storage: 16 words x 64 bits
- Dataset currently compiled into CiM ROM: dataset 00

## Top-Level Structure

The accelerator is a six-thread SystemC pipeline:

1. `command_thread`
2. `encoder_thread`
3. `ngram_thread`
4. `train_thread`
5. `distance_thread`
6. `response_thread`

The current HLS path uses Stratus P2P channels for stage-to-stage communication and for external command/response communication. The non-HLS SystemC reference path still uses manual valid/ready signals and is kept as a functional reference.

## External Protocol

Commands contain:

- command kind
- class id
- 32 quantized feature levels

Responses contain:

- valid-prediction flag
- one distance per class

The RTL smoke test uses generated `commands.txt` and `expected_responses.txt`. The accepted smoke scope is dataset 00 with 20 inference samples.

## Current Internal Data Layout

### Encoder

- Input: quantized sample levels.
- Output: packed 1024-bit encoded hypervector.
- CiM item memory is split into 32 explicit feature banks.
- Encoder bit loop inside each 64-bit word is fully unrolled.
- Feature-word loads are scalarized before the bit loop to avoid artificial local-array or ROM-port bottlenecks.
- Current HLS encoder experiment processes all hypervector words per compute step.
- This changes the encoder from 16 word steps per sample to 1 word-parallel compute step if Stratus can schedule the required CiM reads.
- Main risk: every feature bank must provide 16 word reads per encoder step; if ROM port limits block this, explicit word-lane CiM duplication/banking is the next fix.
- Measured 4-word encoder result:
  - HLS: 0 errors, 0 warnings.
  - RTL smoke20: `cycles=2002`, `average_inference_latency=59.8`, `max_inference_latency=62`, `errors=0`.
  - `enc_out max_fire_gap=168`, `distance_done max_fire_gap=57`.
  - Vivado impl LUTs: 30670, about 2.35 percent.
  - Vivado impl FFs: 37521, about 1.44 percent.
  - Vivado BRAM: 32, about 1.59 percent.
  - Vivado worst listed data path: about 8.40 ns at 10 ns.
- Conclusion: keep 4-word encoder. Runtime improves substantially; timing worsens versus class-parallel-distance baseline but still closes.
- Current full-word encoder experiment is not measured yet.

### Ngram

- HLS path stores the ngram ring buffer as packed `hv_bits_t` values.
- Encoder output is stored directly into the packed buffer; no unpack to `hv_t` in HLS.
- Ngram binding uses packed 1024-bit rotate-left-by-one and XOR.
- Current change: rotate and XOR are split into separate FSM states:
  - `NGRAM_ROTATE`: register `rotated_bits` and `rhs_bits`.
  - `NGRAM_XOR`: compute `next_bits = rotated_bits ^ rhs_bits`.
- This is register slicing, not full multi-token pipelining.
- Expected cost: roughly one extra cycle per bind round.
- Expected benefit: shorter critical route from packed rotate/XOR and lower timing pressure.

### Train / Bundler

- Trainer uses explicit 64 bit banks for bundling scores:
  - `m_bundling_score_0[word]` through `m_bundling_score_63[word]`.
- This avoids dynamic bit-bank indexing and lets Stratus implement bitwise train add/finalize in parallel.
- The explicit bank layout was required; multidimensional arrays did not produce the expected speedup.

### Distance

- Distance uses packed ngram input and class associative-memory words.
- Word-level Hamming distance uses SWAR popcount for each 64-bit diff word.
- Previous HLS distance baseline was class-parallel and word-serial:
  - one `word_index` was processed per compute step.
  - all five class distances were updated in parallel using explicit accumulators.
  - this replaced the previous serial `class_id` loop.
- Bit-level distance inside each 64-bit word is already parallelized by SWAR popcount.
- Current experiment is full word-parallel distance:
  - all five classes are still processed in parallel.
  - all 16 words per class are compared in parallel.
  - each word uses SWAR popcount.
  - each class uses an explicit balanced 16-word reduction tree.
  - valid ngram distance computation becomes one compute state plus P2P input/output overhead.
- Measured class-parallel result:
  - HLS: 0 errors, 0 warnings.
  - RTL smoke20: `cycles=3052`, `average_inference_latency=95.8`, `max_inference_latency=98`, `errors=0`.
  - `distance_in max_fire_gap=74`, `distance_done max_fire_gap=90`.
  - Vivado impl LUTs: 25970, about 1.99 percent.
  - Vivado impl FFs: 32233, about 1.24 percent.
  - Vivado BRAM: 32, about 1.59 percent.
  - Vivado worst listed data path: about 7.60 ns at 10 ns.
- Conclusion: keep class-parallel distance. It improves both runtime and timing compared with the ngram-slice baseline.

## Optimization History And Measurements

All cycle numbers below refer to the RTL smoke20 flow unless noted otherwise.

### Warning-Clean Stable Baseline

- HLS succeeded with 0 errors and 0 warnings.
- RTL smoke20 passed after fixing command hold and expected trace generation.
- Design was functionally correct but slow.

### 32-Bank CiM And Encoder Feature Unroll

Result:

- `cycles=146107`
- `average_inference_latency=13395.2`
- `max_inference_latency=15370`
- RTL errors: 0

Vivado summary:

- Impl LUTs: about 20058
- Impl FFs: about 30191
- BRAM: 33
- Worst listed data path: about 6.4 ns at 10 ns

Conclusion:

- Removed the main CiM memory-port bottleneck.
- Very large runtime speedup with small resource cost.

### Distance SWAR Popcount

Result:

- `cycles=89387`
- `average_inference_latency=2872.7`
- `max_inference_latency=2956`
- RTL errors: 0

Vivado summary:

- Impl LUTs: about 19871
- Impl FFs: about 29907
- BRAM: 33
- Worst listed data path: about 8.83 ns at 10 ns

Conclusion:

- Distance became much faster.
- Timing margin decreased but still closed.

### Encoder Bit Unroll

Result:

- `cycles=55008`
- `average_inference_latency=1470.4`
- `max_inference_latency=1662`
- RTL errors: 0

Vivado summary:

- Impl LUTs: about 18945
- Impl FFs: about 28037
- BRAM: 33
- Worst listed data path: about 8.61 ns at 10 ns

Conclusion:

- Encoder no longer dominates as strongly.
- Runtime bottleneck shifted toward trainer/ngram/distance cadence.

### Explicit Trainer Score Banks

Result:

- `cycles=28546`
- `average_inference_latency=1470.4`
- `max_inference_latency=1662`
- RTL errors: 0
- `bundler_in max_fire_gap=821`
- `distance_in max_fire_gap=821`
- `distance_done max_fire_gap=901`

Vivado summary:

- Impl LUTs: 24509, about 1.88 percent
- Impl FFs: 30018, about 1.15 percent
- BRAM: 32, about 1.59 percent
- DSP: 0
- Worst listed data path: about 8.58 ns at 10 ns

Conclusion:

- Explicit bit-bank storage was necessary for actual train parallelism.
- Multidimensional-array banking attempts did not produce the same runtime improvement.

### Packed Ngram Rotate/XOR

Result before register slicing:

- `cycles=3384`
- `average_inference_latency=217.6`
- `max_inference_latency=250`
- RTL errors: 0
- `ngram_latency train_avg=7 train_max=8`
- `ngram_latency infer_avg=63 infer_max=82`
- `enc_out max_fire_gap=207`
- `distance_done max_fire_gap=152`

Vivado summary before register slicing:

- Impl LUTs: 23490, about 1.80 percent
- Impl FFs: 27200, about 1.04 percent
- BRAM: 32, about 1.59 percent
- DSP: 0
- Worst listed data path: about 9.31 ns at 10 ns
- Route dominated timing: about 6.7 ns route vs about 2.6 ns logic
- High fanout included a packed shift/rotate-related net with fanout about 1357

Conclusion:

- Packed ngram removed the ngram cycle bottleneck.
- Timing became tighter and routing dominated.

### Ngram Register Slice

Current source change:

- Split packed ngram bind into `NGRAM_ROTATE` and `NGRAM_XOR` states.
- Register `rotated_bits` and `rhs_bits` across a `wait()`.
- Functional intent unchanged.

Measured result:

- `cycles=3388`
- `average_inference_latency=218.2`
- `max_inference_latency=250`
- RTL errors: 0
- `ngram_latency train_avg=9 train_max=10`
- `ngram_latency infer_avg=64 infer_max=82`
- Vivado worst listed data path improved from about 9.31 ns to about 8.34 ns.
- Runtime cost versus unsliced packed ngram: about 4 total cycles in smoke20.

## Current Bottleneck Assessment

After packed ngram and the ngram register slice, the old ngram bottleneck is gone. The remaining runtime bottleneck appears distributed across:

- encoder output cadence
- distance done cadence
- P2P backpressure around distance and response

The current timing bottleneck is route-dominated wide logic, especially wide packed operations and high-fanout control/data nets.

## Register Slicing Strategy

Register slicing means splitting one long combinational path across multiple `SC_CTHREAD` states with `wait()` between them. An intermediate C++ variable alone is not a register unless it lives across a `wait()`.

Good candidates:

1. Ngram packed rotate/XOR. Done.
2. Distance XOR -> popcount -> accumulation. Serial slicing was rejected because it worsened cycles and timing.
3. Encoder score accumulation if future unrolling makes timing worse.

## Future Parallelization Candidates

### Distance

Current target after verifying ngram register slice and class-parallel distance.

Options:

- Class-parallel distance calculation. Done and beneficial.
- Full word-parallel distance calculation. Current experiment:
  - 5 classes in parallel.
  - 16 words per class in parallel.
  - 80 SWAR popcount lanes total.
  - balanced reduction tree per class.
- If HLS cannot schedule the concurrent associative-memory reads, explicitly bank associative memory by class and word.
- If Vivado timing worsens, add one register slice between popcount outputs and final reductions or fall back to 4/8 words per cycle.

Risk:

- More distance parallelism can increase route delay and high fanout.
- If timing worsens badly, inspect whether associative memory is still implemented as a shared memory and consider explicit class banks.

### Encoder

Already has CiM feature banking, bitwise word unroll, and a measured-good 4-word-per-cycle encoder. Current experiment is full-word-per-cycle encoding.

Potential changes:

- Keep 4-word-per-cycle as fallback because HLS schedules and RTL improves.
- Test full-word-per-cycle encoding. If HLS fails, RTL does not improve, or timing gets too tight, fall back to 4 or try 8 words per cycle.
- If increasing beyond 4 words per cycle fails due to memory access, duplicate or bank CiM by word lane so each feature can supply more word reads per step.
- Register between CIM feature-word load and feature-score accumulation.
- Split feature accumulation into partial sums if timing worsens.

Current experiment:

- CiM ROM is explicitly banked by feature and hypervector word for HLS:
  `HDC_CIM_ROM_DATASET00_Fxx_Wyy[level]`.
- This targets the full-word encoder case where all 16 hypervector words are encoded in the same encoder step.
- Expected HLS improvement: the encoder should no longer report 16 variable reads from each feature ROM
  `HDC_CIM_ROM_DATASET00_Fxx[level][word]`; reads should be distributed across independent `Fxx_Wyy` banks.
- Non-HLS valid/ready reference path remains unchanged.

### Trainer

Trainer already uses explicit 64 score banks.

Current experiment:

- HLS trainer processes all `HV_WORDS` per step for add-ngram, finalize-class, score reset, and associative-memory clear.
- This means the trainer attempts 16-way word parallelism and 64-way bit-bank parallelism in the same cycle.
- Expected cycle benefit: valid training ngram add drops from about 16 word steps to 1 scheduled trainer step.
- Main risk: large fanout and routing pressure from 1024 score updates/finalize decisions in one state.

Potential changes:

- Register-slice finalize path if high fanout from score banks becomes critical.
- If timing degrades, fall back to 4 or 8 train words per cycle.

## Stratus HLS Effort

The current mixed-effort HLS experiment uses high effort for scheduling and part
selection while avoiding the expensive resource-sharing search:

- `sched_effort high`
- `sharing_effort_parts low`
- `sharing_effort_regs low`
- `parts_effort high`

The explicit loop-unroll directives continue to define the intended parallelism.
The previous all-high experiment took about 37 hours, preserved the same cycle
schedule, and reduced post-route WNS from 0.987 ns to 0.636 ns. The mixed setup
tests whether stronger scheduling and part selection can help without the LUT and
routing pressure introduced by high sharing effort.

## Vivado Implementation Effort

The Vivado flow uses higher-effort implementation directives for the current high-parallelism experiments:

- `opt_design -directive Explore`
- `place_design -directive ExtraNetDelay_high`
- `phys_opt_design -directive AggressiveExplore`
- `route_design -directive AggressiveExplore`
- post-route `phys_opt_design -directive AggressiveExplore`

This does not change RTL behavior. It gives Vivado more effort to handle route-dominated timing and high-fanout nets.

## Test Commands

### HLS

```sh
cd systemc/synthesis/hls
rm -rf bdw_work Makefile.prj vivado_rtl_sim_hdc xsim.dir
make Makefile.prj
make hls_HDC_Accelerator_HLS_BASIC | tee output_hls_<name>.txt
```

### RTL Smoke20

```sh
cd systemc/synthesis/hls
rm -rf vivado_rtl_sim_hdc xsim.dir
bash run_rtl_sim_hdc.sh ../rtl_trace_dataset00_smoke20 | tee output_rtl_smoke20_<name>.txt
```

### Vivado Utilization And Timing

```sh
cd systemc/synthesis/hls
rm -rf vivado_synth_hdc
vivado -mode batch -source vivado_synth_hdc.tcl | tee output_vivado_<name>.txt

{
  echo "== utilization_synth =="
  grep -E "CLB LUTs|LUT as Logic|LUT as Memory|CLB Registers|Register as Flip Flop|Block RAM Tile|URAM|DSPs" \
    vivado_synth_hdc/utilization_synth.rpt

  echo
  echo "== utilization_impl =="
  grep -E "CLB LUTs|LUT as Logic|LUT as Memory|CLB Registers|Register as Flip Flop|Block RAM Tile|URAM|DSPs" \
    vivado_synth_hdc/utilization_impl.rpt

  echo
  echo "== timing_impl =="
  grep -E "WNS|TNS|WHS|THS|Requirement|Data Path Delay" \
    vivado_synth_hdc/timing_impl.rpt | head -60

  echo
  echo "== high_fanout_impl =="
  head -90 vivado_synth_hdc/high_fanout_impl.rpt
} | tee output_vivado_<name>_summary.txt
```

## Acceptance Criteria For Each Hardware Change

- HLS succeeds with 0 errors.
- HLS warnings are reviewed; avoid reintroducing protocol warnings.
- RTL smoke20 finishes with `errors=0` and all 20 responses received.
- Vivado implementation completes.
- Timing closes at 10 ns with positive WNS.
- Resource growth is justified by cycle reduction or timing margin.
