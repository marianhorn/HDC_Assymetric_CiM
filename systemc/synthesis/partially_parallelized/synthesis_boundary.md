# SystemC Synthesis Boundary

This document defines the intended first HLS/SystemC synthesis boundary for the SystemC project.

## Target

Only `HDC_Accelerator` is intended to become synthesizable in the first hardware-oriented version.

Relevant synthesis-target files:

- `systemc/synthesis/src/hdc_accelerator.h`
- `systemc/synthesis/src/hdc_accelerator.cpp`

The accelerator is currently driven through command/response FIFOs in simulation. FIFO cleanup, explicit clock/reset handling, and hardware-style memory interfaces are later phases.

## Not Synthesized

The following parts are simulation or software models and are explicitly outside the first HLS synthesis target:

- `Controller`
- dataset loader
- `main.cpp`
- golden regression harness
- metrics/statistics printing
- file loading
- CSV parsing
- raw EMG preprocessing
- floating-point quantization

Relevant software/simulation files:

- `systemc/synthesis/src/controller.h`
- `systemc/synthesis/src/controller.cpp`
- `systemc/synthesis/src/foot_dataset_loader.h`
- `systemc/synthesis/src/foot_dataset_loader.cpp`
- `systemc/synthesis/src/main.cpp`
- `systemc/synthesis/src/golden_regression.cpp`

## Accelerator Interface

The accelerator input boundary is `AccelCommand`.

It currently carries only synthesis-appropriate accelerator input:

- command kind
- class id
- quantized sample levels

The accelerator must not receive raw EMG values in the first synthesis version.

The accelerator must not perform floating-point quantization in the first synthesis version.

Do not add `double` fields, raw EMG arrays, CSV data, or quantizer fitting data to `AccelCommand`.

## Memory Status

`HDC_Memory` is still part of the current simulation model.

It stores:

- continuous item memory (CiM)
- quantizer boundaries
- associative memory class vectors
- memory access statistics

For synthesis, `HDC_Memory` will later need cleanup or replacement with hardware-like memory arrays, banks, or explicit ports. This phase does not redesign memory access.

Relevant memory-model files:

- `systemc/synthesis/src/hdc_memory.h`
- `systemc/synthesis/src/hdc_memory.cpp`

## Shared Types

Shared type and transaction definitions are:

- `systemc/synthesis/src/systemc_types.h`
- `systemc/synthesis/src/hdc_transactions.h`
- `systemc/synthesis/src/config_systemc.h`

`hdc_transactions.h` is the important accelerator boundary file. `systemc_types.h` also contains simulation result/statistic structs, so not every type in that file is part of the final HLS boundary.
