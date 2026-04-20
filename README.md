# HDC Framework Horn

## Build
Build the foot model:

```bash
make foot
```

Build with config overrides:

```bash
make foot VECTOR_DIMENSION=2048 NUM_LEVELS=20 OUTPUT_MODE=2
```

Clean:

```bash
make clean
```

## Run
Run the foot model:

```bash
./modelFoot
```

Main config file:

```text
foot/configFoot.h
```

## Config Defines

### Main Defines

| Define | Meaning |
|---|---|
| `VECTOR_DIMENSION` | Hypervector dimension. |
| `NUM_LEVELS` | Number of quantization levels. |
| `MIN_LEVEL` | Minimum signal value for uniform-style level mapping. |
| `MAX_LEVEL` | Maximum signal value for uniform-style level mapping. |
| `N_GRAM_SIZE` | Number of timestamps combined into one n-gram. |
| `DOWNSAMPLE` | Downsampling factor applied in preprocessing. |
| `NUM_CLASSES` | Number of target classes. |
| `NUM_FEATURES` | Number of input features per sample. |
| `PRECOMPUTED_ITEM_MEMORY` | Selects precomputed CiM storage. |
| `USE_GENETIC_ITEM_MEMORY` | Enables the CiM GA optimization step. |
| `OUTPUT_MODE` | Selects console output verbosity. |
| `RESULT_CSV_PATH` | CSV path for result logging. |
| `VALIDATION_RATIO` | Fraction of data used for validation. |
| `GA_DEFAULT_POPULATION_SIZE` | GA population size. |
| `GA_DEFAULT_GENERATIONS` | Number of GA generations. |
| `GA_DEFAULT_CROSSOVER_RATE` | GA crossover probability. |
| `GA_DEFAULT_MUTATION_RATE` | GA mutation probability. |
| `GA_DEFAULT_TOURNAMENT_SIZE` | Tournament size for parent selection. |
| `GA_DEFAULT_LOG_EVERY` | GA log interval. |
| `GA_DEFAULT_SEED` | Default GA random seed. |
| `GA_MAX_FLIPS_CIM` | Maximum total flip budget per CiM sequence. |
| `GA_INIT_UNIFORM` | Selects the GA population initialization style. |
| `GA_SELECTION_MODE` | Selects the GA selection objective. |
| `BINNING_MODE` | Selects the quantization method. |
| `GA_REFINED_EPSILON` | Smoothing constant for GA-refined binning. |
| `GA_REFINED_ALPHA` | Strength parameter for GA-refined interval sizing. |

### `OUTPUT_MODE` Values

| Define | Meaning |
|---|---|
| `OUTPUT_NONE` | No console output. |
| `OUTPUT_BASIC` | Final results only. |
| `OUTPUT_DETAILED` | Results plus intermediate information. |
| `OUTPUT_DEBUG` | Maximum debug output. |

### `BINNING_MODE` Values

| Define | Meaning |
|---|---|
| `UNIFORM_BINNING` | Uniform value-to-level mapping. |
| `QUANTILE_BINNING` | Per-feature quantile binning. |
| `KMEANS_1D_BINNING` | Per-feature 1D k-means binning. |
| `DECISION_TREE_1D_BINNING` | Per-feature supervised decision-tree binning. |
| `CHIMERGE_BINNING` | Per-feature supervised ChiMerge binning. |
| `GA_REFINED_BINNING` | Quantizer refined from a preprocessing GA run. |

### `GA_SELECTION_MODE` Values

| Define | Meaning |
|---|---|
| `GA_SELECTION_PARETO` | Pareto selection on accuracy and similarity. |
| `GA_SELECTION_MULTI` | Single score from accuracy minus similarity. |
| `GA_SELECTION_ACCURACY` | Accuracy-only selection. |
