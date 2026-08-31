# Stratus HLS Setup For HDC_Accelerator

This directory is copied from the Stratus tutorial and adapted so the first
HLS target is only `HDC_Accelerator`.

## CAE Setup

Run on `cae00` or `cae02`:

```sh
module load Core/vivado/vivado2023.2
module load Core/cadence/stratus22_23
```

Load Vivado before Stratus.

## Generate Make Targets

```sh
cd systemc_accelerator/partially_parallelized/hls
make Makefile.prj
make help
```

## Synthesize

```sh
make hls_HDC_Accelerator_HLS_BASIC
```

The expected generated RTL location is under:

```text
bdw_work/modules/HDC_Accelerator/HLS_BASIC/
```

## Boundary

Only `../src/hdc_accelerator.cpp` is registered as an HLS module.
Controller, dataset loading, CSV parsing, quantization, metrics, and `main.cpp`
remain software/testbench code and are intentionally not part of this Stratus
project.

## Notes

The copied `mem_lib` and `mem_virtex7` directories are kept from the tutorial
for later explicit memory mapping experiments. The first synthesis attempt
should show whether Stratus accepts the current internal memories directly or
whether the hypervector memories need to be split into narrower word arrays.
