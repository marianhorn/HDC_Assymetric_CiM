# Partially Parallelized SystemC Accelerator

This project contains the resource-reduced SystemC/HLS implementation of the
foot-EMG HDC classifier. The native SystemC executable models the complete
controller/accelerator flow; Cadence Stratus synthesizes only
`src/hdc_accelerator.cpp`.

## Architecture

The controller loads and quantizes the datasets, sends commands to the
accelerator, and evaluates returned class distances. The accelerator is split
into command, encoder, n-gram, bundler/trainer, distance, and response threads.
The HLS path uses packed hypervectors and Stratus P2P channels internally and
at its external command/response boundary.

The partially parallelized configuration reduces the largest spatial stages:

- 16 x 64-bit words for each 1024-bit hypervector;
- feature-banked CiM ROM;
- four hypervector words per encoder step;
- packed rotate/XOR n-gram formation;
- 64 explicit bit banks and four words per trainer step;
- the same class-parallel, word-parallel SWAR distance implementation as the
  fully parallelized variant.

The encoder and trainer constants are defined in `src/hdc_accelerator.h` as
`ENCODER_WORDS_PER_CYCLE` and `TRAIN_WORDS_PER_CYCLE`.

## Imported Reference Data

The model expects `import/cim_datasetXX.txt` and
`import/quantizer_datasetXX.txt` for datasets 00 through 03. These files are
generated from the C reference model. From the repository root:

```sh
make export_systemc_imports
./c_model/build/export_systemc_foot_imports systemc_accelerator/partially_parallelized/import
```

The dataset-00 CiM used by HLS is compiled into a feature-banked ROM header:

```sh
cd systemc_accelerator/partially_parallelized
python3 tools/generate_cim_rom.py \
  --input import/cim_dataset00.txt \
  --output src/generated_cim_rom_dataset00_banked.h \
  --layout feature-banked
```

## Native SystemC Simulation

Requirements are SystemC, GNU Make, and a C++17 compiler. Run from this
directory because import paths are relative to it:

```sh
cd systemc_accelerator/partially_parallelized
make clean
make
./main
```

Override a nonstandard SystemC installation with `SYSTEMC_HOME` and, if
needed, `SYSTEMC_LIB_DIR`.

## Regression and RTL Traces

```sh
make golden
./systemc_hdc_golden_regression golden_regression_check.txt
diff -u golden_regression_current.txt golden_regression_check.txt

make rtl-trace
./systemc_hdc_rtl_trace_export
```

The checked-in `rtl_trace_dataset00_smoke20/` and `rtl_trace_dataset00/`
directories provide the command and expected-response files used by XSim.

## HLS, RTL Simulation, and Vivado

See [`hls/README.md`](hls/README.md) for the complete CAE flow, including
Stratus synthesis, smoke/full RTL simulation, Vivado implementation, timing,
and utilization reports. The default FPGA is `xczu3eg-sbva484-1-i`.
