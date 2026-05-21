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

The controller owns the `HDC_Memory` and `HDC_Accelerator` objects for this model and binds them together during construction.

### HDC_Memory

Files:

- `src/hdc_memory.h`
- `src/hdc_memory.cpp`

The memory block stores:

- precomputed item memory / CiM: `level x feature -> hypervector`
- quantizer boundaries: `feature x threshold`
- associative memory: `class -> hypervector`

The current memory model is functional: accesses return immediately. It does not model ports, banks, contention, arbitration, bandwidth stalls, or traffic statistics.

### HDC_Accelerator

Files:

- `src/hdc_accelerator.h`
- `src/hdc_accelerator.cpp`

The accelerator represents the HDC hardware datapath. It communicates externally only through command/response FIFOs.

It uses internal fixed-size memories for CiM and associative memory. In the current simulation model, CiM and associative-memory initialization happen through the preload helpers `set_cim()` and `set_assoc_class()`. Those helpers are simulation setup only; a real hardware build still needs ROM initialization, generated constants, or a dedicated preload interface before deployment.

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

## Datapath Execution

The synthesis variant keeps the command/FIFO pipeline structure but executes each datapath stage serially inside its owning thread.

- `encoder_thread`: serial sample encoding across dimensions and features
- `ngram_thread`: serial n-gram binding using `permute(input) XOR rhs`
- `distance_thread`: serial Hamming distance computation over all classes
- `bundler_thread`: sequential training-side bundling and class finalization

This keeps behavior aligned with the simulation model while simplifying the code for downstream HLS/synthesis work.

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

The synthesis variant intentionally does not print memory or accelerator traffic counters. Functional regression is checked through accuracy, counts, and confusion matrices.

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

Use this after changing controller/accelerator/memory logic. Counters may evolve intentionally, but prediction output should not change unless the algorithm changed on purpose.

## Configuration

Main compile-time parameters are in `src/config_systemc.h`:

```cpp
VECTOR_DIMENSION
NUM_FEATURES
NUM_LEVELS
NUM_CLASSES
N_GRAM_SIZE
NUM_DATASETS
MAX_SAMPLES_IN_PIPELINE
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
- per-dataset functional metrics

Current limitations:

- not RTL
- no clocked register-transfer implementation
- controller streams data commands with a bounded inference window; reset, flush, and shutdown commands remain blocking stream boundaries
- memory is functional and does not model ports, banks, contention, or bandwidth stalls
- import files must already exist and match the compiled configuration
- bundling is sequential

## Presentation Summary

A concise way to describe the model:

```text
The controller models the RISC-V/software side: loading, quantization, labels, and evaluation.
The accelerator models the HDC datapath: encoding, n-gram binding, class-vector bundling, and Hamming distance.
The memory module keeps quantizer boundaries on the simulation side; the accelerator owns internal CiM and associative-memory arrays in the synthesis variant.
Golden regression ensures architectural refactors preserve prediction behavior.
```
