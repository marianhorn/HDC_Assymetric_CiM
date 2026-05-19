# SystemC HDC Accelerator Model

This folder contains the SystemC model of the HDC EMG classifier. The goal is to model the current C implementation as a hardware/software architecture, not just as another functional C++ program.

The model uses the native C code as the reference source for generated data:

- precomputed CiM vectors are imported from text files
- learned quantizer boundaries are imported from text files
- raw EMG CSV datasets are loaded from the existing `foot/data/datasetXX` folders

The SystemC side then runs the full flow for all configured datasets: quantization, training, inference, statistics, and golden regression checking.

## Architecture

The model is split into three main blocks.

### Controller

Files:

- `src/controller.h`
- `src/controller.cpp`

The controller represents the software/RISC-V side of the system. It is an active SystemC module with its own `SC_THREAD(main_thread)`.

Responsibilities:

- configure datasets before simulation starts
- load CiM and quantizer text files
- keep quantization in software
- send accelerator commands through FIFOs
- compute final prediction from returned class distances
- compute accuracy, transition errors, and confusion matrix
- collect per-dataset memory and accelerator statistics

The controller owns the `HDC_Memory` and `HDC_Accelerator` objects for this model and binds them together during construction.

### HDC_Memory

Files:

- `src/hdc_memory.h`
- `src/hdc_memory.cpp`

The memory block stores:

- precomputed item memory / CiM: `level x feature -> hypervector`
- quantizer boundaries: `feature x threshold`
- associative memory: `class -> hypervector`

It also counts memory traffic:

- quantizer row reads and bytes
- CiM reads and bytes
- associative-memory reads and bytes
- associative-memory writes and bytes

The current memory model is functional: accesses return immediately and counters are updated. It does not yet model ports, banks, contention, arbitration, or per-access latency.

### HDC_Accelerator

Files:

- `src/hdc_accelerator.h`
- `src/hdc_accelerator.cpp`

The accelerator represents the HDC hardware datapath. It communicates externally only through command/response FIFOs.

External command flow:

```text
Controller
  -> AccelCommand FIFO
HDC_Accelerator::command_thread
  -> internal pipeline
  -> AccelResponse FIFO
Controller
```

Internal pipeline:

```text
command_thread
  -> encoder_thread
  -> ngram_thread
  -> bundler_thread   (training path)
  -> distance_thread  (inference path)
```

Command types are defined in `src/hdc_transactions.h`:

- `ResetTraining`
- `ResetInference`
- `TrainSample`
- `InvalidTrainingStep`
- `InferSample`
- `Shutdown`

## Parallelism Model

The accelerator models hardware parallelism with SystemC threads and events. This is still an architectural model, not RTL.

### Encoding Parallelism

Controlled by:

```cpp
ENCODER_PES
```

Encoding is parallelized across hypervector dimensions. Each encoder PE computes a slice of the output hypervector. For every output dimension, it reads the relevant CiM bits for all features and applies the majority rule.

### N-Gram Binding Parallelism

Controlled by:

```cpp
NGRAM_PES
```

N-gram binding is also parallelized across hypervector dimensions. Each n-gram PE computes a slice of:

```text
output = permute(input) XOR rhs
```

The implementation avoids in-place hazards by writing each parallel permute/XOR step into a temporary output vector.

### Distance Parallelism

Distance computation is parallelized across classes. The accelerator spawns one distance PE per class:

```text
NUM_CLASSES distance PEs
```

Each class PE computes the Hamming distance between the query hypervector and one associative-memory class vector.

### Bundling

Bundling is currently sequential. It is training-side only and updates one class bundling buffer before writing the final class vector to associative memory.

## Simulation Timing

Simulation time is SystemC modeled time, not wall-clock runtime.

The latency constants are defined in `src/config_systemc.h`:

```cpp
ACCEL_LATENCY_ENCODE_NS
ACCEL_LATENCY_NGRAM_NS
ACCEL_LATENCY_BUNDLE_NS
ACCEL_LATENCY_DISTANCE_NS
```

The model advances time with `wait(...)` calls in the accelerator pipeline stages. The printed per-dataset simulation time is measured around training plus test evaluation for that dataset.

## Captured Metrics

The normal executable prints one block per dataset.

### Functional Metrics

From `EvaluationResult`:

- test accuracy
- test accuracy excluding transition windows
- correct count
- wrong count
- transition-error count
- total evaluated n-grams

The golden regression also stores the confusion matrix for each dataset.

### Memory Metrics

From `MemoryStats`:

- `quantizer_row_reads`
- `quantizer_row_read_bytes`
- `cim_reads`
- `cim_read_bytes`
- `assoc_reads`
- `assoc_read_bytes`
- `assoc_writes`
- `assoc_write_bytes`
- total read accesses and bytes
- total write accesses and bytes

Byte accounting is based on the current memory API granularity. A CiM or associative-memory access counts one full hypervector.

### Accelerator Metrics

From `AcceleratorStats`:

- command count
- training samples sent to accelerator
- inference samples sent to accelerator
- encoded samples
- n-gram samples
- valid n-grams
- bundled n-grams
- bundle flushes
- distance requests
- valid distance requests

These counters help check whether architectural changes affect the expected dataflow.

## Project Structure

```text
systemc_hdc/
  Makefile
  README.md
  golden_regression_current.txt
  import/
    cim_dataset00.txt
    cim_dataset01.txt
    cim_dataset02.txt
    cim_dataset03.txt
    quantizer_dataset00.txt
    quantizer_dataset01.txt
    quantizer_dataset02.txt
    quantizer_dataset03.txt
  src/
    config_systemc.h
    systemc_types.h
    hdc_transactions.h
    hdc_memory.h
    hdc_memory.cpp
    hdc_accelerator.h
    hdc_accelerator.cpp
    controller.h
    controller.cpp
    foot_dataset_loader.h
    foot_dataset_loader.cpp
    golden_regression.cpp
    main.cpp
```

Generated files:

```text
systemc_hdc/build/
systemc_hdc/main
systemc_hdc/systemc_hdc_golden_regression
```

These generated outputs are ignored by git.

## Build And Run

Run from the `systemc_hdc` directory:

```sh
make clean
make
./main
```

Equivalent Makefile shortcut:

```sh
make run
```

The Makefile tries to find SystemC under:

```text
/home/zora/systemc-install
/usr/local/systemc
```

If needed, override paths explicitly:

```sh
make SYSTEMC_HOME=/path/to/systemc-install
```

or:

```sh
make SYSTEMC_HOME=/path/to/systemc-install SYSTEMC_LIB_DIR=/path/to/systemc-lib
```

## Golden Regression

The golden regression is the functional safety net for refactoring. It records deterministic prediction behavior and confusion matrices.

Build and run:

```sh
make golden
./systemc_hdc_golden_regression golden_regression_check.txt
```

Compare against the checked-in baseline:

```sh
diff -u golden_regression_current.txt golden_regression_check.txt
```

Expected result after behavior-preserving refactors:

```text
no diff
```

Use this after changing controller/accelerator/memory logic. Timing and counters may evolve intentionally, but prediction output should not change unless the algorithm changed on purpose.

## Configuration

Main compile-time parameters are in `src/config_systemc.h`:

```cpp
VECTOR_DIMENSION
NUM_FEATURES
NUM_LEVELS
NUM_CLASSES
N_GRAM_SIZE
NUM_DATASETS
ENCODER_PES
NGRAM_PES
ACCEL_LATENCY_ENCODE_NS
ACCEL_LATENCY_NGRAM_NS
ACCEL_LATENCY_BUNDLE_NS
ACCEL_LATENCY_DISTANCE_NS
```

The import file headers are checked against the active SystemC configuration. If `NUM_LEVELS`, `NUM_FEATURES`, or `VECTOR_DIMENSION` do not match the imported files, the simulation fails early.

## Import Files

The model expects one CiM and one quantizer file per dataset:

```text
import/cim_dataset00.txt
import/quantizer_dataset00.txt
...
import/cim_dataset03.txt
import/quantizer_dataset03.txt
```

These files are produced by the native C code export path. The SystemC model imports them; it does not learn the quantizer or generate the CiM internally.

## Current Scope And Limitations

Current scope:

- binary HDC hypervectors
- fusion-style sample and n-gram encoding
- software-side quantization in the controller
- hardware-side encoding, n-gram binding, bundling, and Hamming distance
- all configured foot datasets
- per-dataset functional, timing, memory, and accelerator metrics

Current limitations:

- not RTL
- no clocked register-transfer implementation
- controller uses a mostly blocking command/response style
- memory is functional and does not model ports, banks, contention, or bandwidth stalls
- import files must already exist and match the compiled configuration
- bundling is sequential

## Presentation Summary

A concise way to describe the model:

```text
The controller models the RISC-V/software side: loading, quantization, labels, and evaluation.
The accelerator models the HDC datapath: encoding, n-gram binding, class-vector bundling, and Hamming distance.
The memory module separates CiM, quantizer boundaries, and associative memory and collects traffic metrics.
Golden regression ensures architectural refactors preserve prediction behavior.
```