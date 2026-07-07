# Stratus P2P Experiments

Minimal standalone experiments for `cynw_p2p_direct` communication between
`SC_CTHREAD`s. This is intentionally separate from the HDC accelerator so P2P
semantics can be tested without the full pipeline.

## Structure

- `src/p2p_pipeline.h/.cpp`: small three-thread pipeline.
- `src/main.cpp`: SystemC testbench that sends deterministic tokens and checks
  the returned values.
- `hls/project.tcl`: Stratus HLS project for the experiment module.

The default build uses blocking `input.put()` / `output.get()`.

Define `P2P_EXPERIMENT_NB` to test explicit `nb_can_put` / `nb_put` and
`nb_can_get` / `nb_get` state-machine handshakes.

Define `P2P_ENCODER_MIMIC` to test the encoder-like case: the middle stage gets
one token, performs 16 word-compute cycles using the 32-entry sample payload,
then puts one output token. This mirrors the important scheduling shape of
`HDC_Accelerator::encoder_thread`.

Define `P2P_ENCODER_MIMIC_NB` to run the same encoder-like test with split
nonblocking P2P handshakes. It implies `P2P_ENCODER_MIMIC` and
`P2P_EXPERIMENT_NB`.

In encoder-mimic mode the broad `*_cycle` protocol markers are disabled because
they force Stratus to schedule local array copies/checksum work as fixed-latency
protocols. The experiment is meant to validate P2P behavior around a multi-cycle
FSM, not local array-loop protocol timing.

## Local SystemC

```sh
cd systemc/synthesis/p2p_experiments
make clean && make && ./p2p_experiment
```

Nonblocking variant:

```sh
make clean && make EXTRA_CXXFLAGS=-DP2P_EXPERIMENT_NB && ./p2p_experiment
```

Encoder-mimic variant:

```sh
make clean && make EXTRA_CXXFLAGS=-DP2P_ENCODER_MIMIC_NB && ./p2p_experiment
```

## Stratus

```sh
cd systemc/synthesis/p2p_experiments/hls
rm -rf bdw_work Makefile.prj
make Makefile.prj
make hls_P2PPipeline_HLS_BASIC | tee output_p2p_experiment_hls.txt
```

The checked-in HLS project currently enables:

```tcl
set_attr D P2P_ENCODER_MIMIC_NB
```

so the default Stratus run tests the split-nonblocking encoder-mimic variant.

## RTL Simulation

After HLS succeeds:

```sh
cd systemc/synthesis/p2p_experiments/hls
bash run_p2p_rtl_sim.sh | tee output_p2p_experiment_rtl.txt
```

Expected success summary:

```text
P2P RTL simulation complete
sent=16
received=16
source_count=16
stage_count=16
sink_count=16
errors=0
```

For the nonblocking variant, add the define in `hls/project.tcl`:

```tcl
set_attr D P2P_EXPERIMENT_NB
```

For the simple pass-through transform experiment, remove or comment out
`set_attr D P2P_ENCODER_MIMIC` in `hls/project.tcl`.
