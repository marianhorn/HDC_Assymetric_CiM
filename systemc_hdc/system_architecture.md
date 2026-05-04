# SystemC HDC Architecture

## Goal
This document defines the target modular architecture for the SystemC implementation of the current HDC pipeline.

The design is intentionally simple and hardware-oriented:
- precomputed CiM only
- binary vectors only
- fusion n-gram only
- fixed `N_GRAM_SIZE = 3`
- no GA logic in SystemC
- no training-data preprocessing in SystemC except quantization lookup
- no heavy object-oriented structure

The design is split into 3 main components:
1. `Controller`
2. `HDC_Memory`
3. `HDC_Accelerator`

---

## Top-Level View

### `Controller`
Role:
- owns the program flow
- imports external data files
- performs training orchestration
- performs evaluation orchestration
- writes associative memory contents after training
- requests encoding or classification from the accelerator

The controller is the only block that:
- iterates over samples
- decides when to train
- decides when to classify
- loads CiM and quantizer data from files

It should not contain:
- CiM storage
- associative-memory storage
- bit-level encoding datapath logic
- Hamming-distance logic

---

### `HDC_Memory`
Role:
- stores the precomputed CiM
- stores the learned quantizer boundaries
- stores the associative memory

It is a data storage module, not a compute-heavy module.

Contained memories:
- `CiM[level][feature] -> hv`
- `QuantizerBoundaries[feature][level-1] -> threshold`
- `AssocMem[class] -> hv`

Operations:
- load CiM from file
- load quantizer boundaries from file
- read CiM entry
- quantize one raw feature value into one level
- write associative-memory class vector
- read associative-memory class vector

This module centralizes all persistent model state.

---

### `HDC_Accelerator`
Role:
- implements the fixed HDC datapath
- uses data stored in `HDC_Memory`

Operations:
- `encode(...)`
- `classify(...)`

The accelerator does not own long-term model state itself.
It reads CiM / quantizer / associative memory from `HDC_Memory`.

Internal datapath responsibilities:
- quantized sample handling
- timestamp encoding
- fusion n-gram encoding
- Hamming-distance computation

The intended style is fixed loops over:
- `NUM_FEATURES`
- `VECTOR_DIMENSION`
- `NUM_CLASSES`

No dynamic allocation inside the accelerator.

---

## Module Interfaces

## `Controller`

### Main responsibilities
- load CiM file into `HDC_Memory`
- load quantizer file into `HDC_Memory`
- load dataset samples
- training:
  - for each valid training window
  - call `HDC_Accelerator.encode(...)`
  - accumulate class counters locally or in a small training helper
  - finalize class vectors
  - write final class vectors into `HDC_Memory`
- evaluation:
  - for each evaluation window
  - call `HDC_Accelerator.classify(...)`
  - derive predicted class from returned distances
  - compute accuracy statistics

### Suggested public methods
- `load_cim_file(const char *path)`
- `load_quantizer_file(const char *path)`
- `load_dataset_file(...)`
- `train_dataset(...)`
- `evaluate_dataset(...)`

### Important design choice
The controller should be the only place that knows:
- training semantics
- stable-window rule
- dataset iteration order
- accuracy calculation

So the accelerator remains purely computational.

---

## `HDC_Memory`

### Stored data
- `hv_t cim[NUM_LEVELS][NUM_FEATURES]`
- `double quantizer_boundaries[NUM_FEATURES][NUM_LEVELS - 1]`
- `hv_t assoc_mem[NUM_CLASSES]`

### Suggested public methods
- `clear_all()`
- `load_cim_flat(const hv_t *flat_cim)`
- `load_cim_text(const char *path)`
- `load_quantizer_boundaries(const double *flat_boundaries)`
- `load_quantizer_text(const char *path)`
- `read_cim(level_t level, unsigned feature) const`
- `quantize_sample(const double *raw_sample, level_t *quantized_sample) const`
- `write_assoc_class(unsigned class_id, const hv_t &class_hv)`
- `read_assoc_class(unsigned class_id) const`

### Notes
- quantization should be a simple threshold lookup
- `quantize_sample(...)` should quantize one timestamp:
  - input: raw `NUM_FEATURES` values
  - output: `NUM_FEATURES` discrete levels
- boundaries are imported, not learned in SystemC

### File import expectation
The files produced by the C code should later be converted into:
- flat CiM dump
- flat boundary dump

The exact file format is still open, but the memory layout should match:
- CiM flat order:
  - `(level * NUM_FEATURES + feature)`
- quantizer flat order:
  - `(feature * (NUM_LEVELS - 1) + cut)`

---

## `HDC_Accelerator`

### External behavior

#### `encode(...)`
Input:
- 3 quantized samples
- each sample has `NUM_FEATURES` levels

Output:
- one encoded hypervector

Behavior:
1. encode timestamp 0
2. encode timestamp 1
3. encode timestamp 2
4. apply fusion recurrence:
   - `H0 = encode_timestamp(sample0)`
   - `H1 = permute_right(H0, 1) XOR encode_timestamp(sample1)`
   - `H2 = permute_right(H1, 1) XOR encode_timestamp(sample2)`
5. return `H2`

#### `classify(...)`
Input:
- same 3 quantized samples as `encode(...)`

Output:
- Hamming distance to every associative-memory class vector

Behavior:
1. internally call the same encoding logic as `encode(...)`
2. read all class vectors from `HDC_Memory`
3. compute XOR + popcount against each class vector
4. return `distance[class_id]` for all classes

### Internal structure
Suggested internal helper functions:
- `encode_timestamp(...)`
- `encode_ngram(...)`
- `compute_hamming_distances(...)`

### Timestamp encoding behavior
For one timestamp:
- for each feature:
  - read `CiM[level][feature]`
- for each bit position:
  - count ones over all features
  - output bit is `1` if `ones >= NUM_FEATURES / 2`

This matches the current binary bundling rule.

### Hamming-distance behavior
For one query and one class:
- XOR query with class vector
- popcount result
- return the count of differing bits

---

## File Split

Target source split:
- `src/controller.h`
- `src/controller.cpp`
- `src/hdc_memory.h`
- `src/hdc_memory.cpp`
- `src/hdc_accelerator.h`
- `src/hdc_accelerator.cpp`
- `src/tb_systemc.cpp`
- `src/config_systemc.h`

Optional shared utility file if needed:
- `src/systemc_types.h`

Build/output layout:
- `build/` for object files
- final executable at `systemc_hdc/systemc_hdc_tb`
- `import/` for CiM and quantizer files exported from the C implementation

---

## Data Flow

### Training
1. `Controller` loads CiM and quantizer into `HDC_Memory`
2. `Controller` reads 3 raw timestamps from dataset
3. `Controller` asks `HDC_Memory` to quantize each timestamp
4. `Controller` passes the 3 quantized timestamps to `HDC_Accelerator.encode(...)`
5. `Controller` accumulates class counters for the target class
6. after training, `Controller` finalizes one binary associative-memory vector per class
7. `Controller` stores those vectors into `HDC_Memory`

### Classification
1. `Controller` reads 3 raw timestamps
2. `Controller` asks `HDC_Memory` to quantize them
3. `Controller` calls `HDC_Accelerator.classify(...)`
4. `HDC_Accelerator` returns one distance per class
5. `Controller` chooses the minimum-distance class

---

## Hardware-Friendly Design Rules

The implementation should keep these rules:
- no inheritance-heavy design
- no STL-heavy containers in the datapath
- no dynamic allocation in hot compute paths
- fixed-size arrays based on compile-time constants
- simple loops instead of deeply nested helper abstractions
- memory and compute clearly separated

SystemC should model hardware structure, not software convenience.

---

## Current Gap to Target

Current SystemC code already contains most compute logic, but in monolithic form:
- CiM memory exists
- timestamp encoder exists
- fusion n-gram encoder exists
- associative-memory trainer exists
- classifier exists
- one top wrapper exists

Missing relative to the target architecture:
- separate `Controller`
- separate `HDC_Memory` with quantizer + assoc_mem together
- separate `HDC_Accelerator`
- file import path for CiM and quantizer boundaries
- explicit `encode(...)` and `classify(...)` interface centered on 3 samples

So the next implementation step is mainly a structural split, not a new algorithm.
