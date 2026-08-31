# HDC-Based Foot-EMG Classification

This repository contains the software and hardware implementations developed for
the accompanying master's thesis. The project classifies foot movements from
EMG time series using binary hyperdimensional computing (HDC).

## Repository Structure

- [`c_model`](./c_model): C reference model, datasets, configuration, and export tools.
- [`systemc_accelerator`](./systemc_accelerator): Fully and partially parallelized SystemC/HLS accelerator implementations.
- [`analysis`](./analysis): Benchmarking, quantizer, genetic-algorithm, and hardware-analysis scripts.
- [`thesis`](./thesis): LaTeX source, figures, bibliography, and final thesis PDF.

## Build and Run the C Model

Requirements are a C11 compiler, GNU Make, and optionally OpenMP.

```bash
make foot
./c_model/build/hdc_model
```

The model configuration is defined in [`c_model/include/config.h`](./c_model/include/config.h).
Build-time overrides can be passed directly to Make, for example:

```bash
make foot VECTOR_DIMENSION=10000 NUM_LEVELS=40 ITEM_MEM_SEED=1
./c_model/build/hdc_model
```

Additional helper targets are available:

```bash
make export_naive_cim
make evaluate_final_cims
make clean
```

## Author

Marian Horn
