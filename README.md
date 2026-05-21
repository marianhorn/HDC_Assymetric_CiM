# HDC Foot EMG Classifier

This repository contains a simplified C implementation of a Hyperdimensional Computing (HDC) classifier for foot EMG time-series data.


## How To Use

Build the foot model:

```sh
make foot
```

Run all four datasets:

```sh
./modelFoot
```

The executable loads one preoptimized CiM per dataset from:

```text
CiMs/preoptimized/cim_dataset00.csv
CiMs/preoptimized/cim_dataset01.csv
CiMs/preoptimized/cim_dataset02.csv
CiMs/preoptimized/cim_dataset03.csv
```

It trains the associative memory on the training split and evaluates only the testing set.


Common parameters are defined in [`foot/configFoot.h`](foot/configFoot.h).

## Algorithmic Path

For each dataset `0..3`:

1. Load the preoptimized continuous item memory from CSV.
2. Load training, validation, and testing CSV files.
3. Fit the uniform quantizer.
4. Encode training samples as overlapping n-grams.
6. Bundle encoded training n-grams per class into associative memory class vectors.
7. Encode testing samples as overlapping n-grams.
8. Classify each testing n-gram by Hamming-distance similarity to the class vectors.
9. Report per-dataset test accuracy and the mean test accuracy.

The validation split is still created because it preserves the current training split behavior, but validation is not evaluated in the simplified main path.

## Project Structure

```text
.
|-- Makefile
|-- README.md
|-- CiMs/
|-- foot/
`-- hdc_infrastructure/
```

## Top-Level Files

- [`Makefile`](Makefile): Builds the active foot model target `modelFoot`.
- [`README.md`](README.md): Current project overview and usage notes.
- `CiMs/preoptimized/`: Input directory for the preoptimized item-memory CSV files used by `modelFoot`.

## `foot/`

- [`foot/configFoot.h`](foot/configFoot.h): Main configuration constants such as vector dimension, number of quantization levels, n-gram size, dataset dimensions, output mode, and CiM import path.
- [`foot/modelFoot.c`](foot/modelFoot.c): Main program. Runs datasets `0..3`, loads CiMs, fits the quantizer, trains, evaluates testing data, and prints accuracies.
- [`foot/dataReaderFootEMG.c`](foot/dataReaderFootEMG.c): Reads foot EMG CSV files, performs optional downsampling, and creates the training/validation/testing arrays.
- [`foot/dataReaderFootEMG.h`](foot/dataReaderFootEMG.h): Public interface for the foot dataset reader.
- `foot/data/datasetDescription.md`: Dataset description.
- `foot/data/dataset00/` to `foot/data/dataset03/`: EMG and label CSV files for training and testing.

## `hdc_infrastructure/`

- [`hdc_infrastructure/vector.c`](hdc_infrastructure/vector.c): Allocation and printing for packed binary hypervectors.
- [`hdc_infrastructure/vector.h`](hdc_infrastructure/vector.h): Binary vector type, storage layout, and inline bit operations.
- [`hdc_infrastructure/operations.c`](hdc_infrastructure/operations.c): HDC primitive operations: binding, bundling, permutation, and similarity.
- [`hdc_infrastructure/operations.h`](hdc_infrastructure/operations.h): Public interface for HDC primitive operations.
- [`hdc_infrastructure/item_mem.c`](hdc_infrastructure/item_mem.c): Item-memory creation, loading, storing, and CSV/SystemC export helpers.
- [`hdc_infrastructure/item_mem.h`](hdc_infrastructure/item_mem.h): Public item-memory interface.
- [`hdc_infrastructure/quantizer.c`](hdc_infrastructure/quantizer.c): Uniform quantizer fitting, level lookup, and export helpers.
- [`hdc_infrastructure/quantizer.h`](hdc_infrastructure/quantizer.h): Public quantizer interface.
- [`hdc_infrastructure/encoder.c`](hdc_infrastructure/encoder.c): Encodes one timestamp and streams encoded timestamps through an n-gram ring buffer.
- [`hdc_infrastructure/encoder.h`](hdc_infrastructure/encoder.h): Encoder structures and public encoding functions.
- [`hdc_infrastructure/trainer.c`](hdc_infrastructure/trainer.c): Time-series training. Bundles encoded n-grams per class and writes the final class vectors to associative memory.
- [`hdc_infrastructure/trainer.h`](hdc_infrastructure/trainer.h): Public trainer interface.
- [`hdc_infrastructure/evaluator.c`](hdc_infrastructure/evaluator.c): Direct time-series evaluation with overlapping n-grams, confusion matrix, accuracy, and transition-error accounting.
- [`hdc_infrastructure/evaluator.h`](hdc_infrastructure/evaluator.h): Public evaluator interface and result struct.
- [`hdc_infrastructure/assoc_mem.c`](hdc_infrastructure/assoc_mem.c): Associative memory initialization, class-vector storage, classification, and load/store helpers.
- [`hdc_infrastructure/assoc_mem.h`](hdc_infrastructure/assoc_mem.h): Public associative-memory interface.

## Output Modes

Set `OUTPUT_MODE` in [`foot/configFoot.h`](foot/configFoot.h) or override it at build time:

- `OUTPUT_NONE`: no output
- `OUTPUT_BASIC`: dataset and overall test accuracies
- `OUTPUT_DETAILED`: additional loading/training/evaluation messages
- `OUTPUT_DEBUG`: debug-level output such as class-vector prints

Example:

```sh
make foot OUTPUT_MODE=2
./modelFoot
```



## Notes

- `make foot` runs `clean` first, so object files are rebuilt for the selected configuration.
- The validation split remains in the data reader to keep the current train/test behavior stable.

