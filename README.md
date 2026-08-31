# HDC-Based Foot-EMG Classification

This repository contains the software and hardware implementations developed for
the accompanying master's thesis. The project classifies foot movements from
EMG time series using binary hyperdimensional computing (HDC).

## Repository Structure

- [`c_model`](./c_model): C reference model, datasets, configuration, and export tools.
- [`systemc_accelerator`](./systemc_accelerator): Fully and partially parallelized SystemC/HLS accelerator implementations. Each variant has a README for native SystemC simulation and a separate `hls/README.md` for Stratus, RTL simulation, and Vivado.
- [`analysis`](./analysis): Thesis experiment runners and plotting scripts. See [`analysis/README.md`](./analysis/README.md) for the commands used to reproduce the thesis figures.
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

## Quantization

Select the value-to-level quantizer at build time with `BINNING_MODE`:

| Value | Quantizer |
|---|---|
| `UNIFORM_BINNING` | Equally spaced baseline |
| `QUANTILE_BINNING` | Per-feature quantile boundaries |
| `KMEANS_1D_BINNING` | Per-feature one-dimensional k-means |
| `DECISION_TREE_1D_BINNING` | Per-feature supervised decision tree |
| `CHIMERGE_BINNING` | Per-feature supervised ChiMerge |

For example:

```bash
make foot BINNING_MODE=KMEANS_1D_BINNING NUM_LEVELS=40 VECTOR_DIMENSION=10000
./c_model/build/hdc_model
```

Uniform binning is the default. The learned quantizers fit their boundaries on
the training split before model training and evaluation.

## Genetic CCIM Optimization

Enable optimization of the continuous item memory (CCIM) with
`USE_GENETIC_ITEM_MEMORY=1`. The thesis configuration uses 128 individuals,
64 generations, and uniform binning:

```bash
make foot \
  USE_GENETIC_ITEM_MEMORY=1 \
  BINNING_MODE=UNIFORM_BINNING \
  GA_DEFAULT_POPULATION_SIZE=128 \
  GA_DEFAULT_GENERATIONS=64 \
  GA_DEFAULT_SEED=1 \
  ITEM_MEM_SEED=1 \
  NUM_LEVELS=40 \
  VECTOR_DIMENSION=10000
./c_model/build/hdc_model
```

Quantizer and CCIM optimization can be combined by selecting a learned
`BINNING_MODE` in the same command. Full experiment grids and thesis plotting
commands are documented in [`analysis/README.md`](./analysis/README.md).

## Author

Marian Horn
