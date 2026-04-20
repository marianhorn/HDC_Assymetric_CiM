
# INSTRUCTIONS: Genetic Asymmetric Item Memory (`asymItemMemory`)

This document describes, in **high detail but at a high level**, how to extend the existing C-based HDC implementation with a **genetic, asymmetric item memory generator** that is compatible with the current EMG data handling pipeline.

The intent is that an automated code generator (e.g., Codex) can implement the required code **inside this project** using only these instructions and the existing source files. Do **not** introduce a new dataset struct or abstraction: always work with the existing data loading functions and their pointer-based outputs.

---

## 1. Overall Goal

You will:

1. Introduce a new module, conceptually named `asymItemMemory.c` with a corresponding header `asymItemMemory.h`. This module will:
   - Run a **genetic algorithm (GA)** over a matrix of integers `B` that encode how many bits to flip between adjacent levels for each feature.
   - Evaluate each candidate `B` using the **existing HDC pipeline** (quantization, encoding, classification) on the existing EMG training data.
   - Return a final, optimized `B*` matrix.

2. Extend `item_mem.c` with **one additional public initialization function** that:
   - Calls the GA module to obtain `B*`.
   - Uses `B*` to construct the final item memory hypervectors for each feature–level combination.
   - Keeps the same memory layout and semantics as the current `init_precomp_item_memory` so that the rest of the project continues to work unchanged.

3. Integrate this new initialization into the control flow so that:
   - The GA-based generation is used in an **offline training mode** (e.g., on the host).
   - The resulting item memory can be stored to disk using the already existing binary store/load functions for deployment on embedded targets.

All design choices must be consistent with the current usage of `item_memory` in the project.

---

## 2. Use the Existing Data Pipeline (Do NOT Introduce New Dataset Types)

The project already includes a data reader in `dataReaderFootEMG.h` / `.c`, which provides at least the following functions (names based on the provided code):

- `getData(int dataset, double*** trainingData, double*** testingData, int** trainingLabels, int** testingLabels, int* trainingSamples, int* testingSamples)`
- `getTestingData(int dataset, double*** testingData, int** testingLabels, int* testingSamples)`

These functions:

- Load CSV files containing EMG signals and labels.
- Downsample the data via `down_sample(...)` from the preprocessor.
- Return:

  - `trainingData`: pointer to a 2D array `double**`, shape `[trainingSamples][NUM_FEATURES]`
  - `trainingLabels`: pointer to `int`, length `trainingSamples`
  - `testingData`: pointer to `double**`, shape `[testingSamples][NUM_FEATURES]`
  - `testingLabels`: pointer to `int`, length `testingSamples`

**Important constraints:**

- **Do not** create any new dataset struct such as `HDCDataset`.
- **Do not** duplicate or wrap these data structures in new types.
- All new functionality (GA, evaluation) must accept and use these pointer-based representations directly, plus the metadata (`trainingSamples`, `testingSamples`, `NUM_FEATURES`, etc.).
- Use `NUM_FEATURES` from the existing project (typically defined in a config header or preprocessor header).

Wherever this document refers to “dataset” or similar, interpret that as:

- `double** trainingData`
- `int* trainingLabels`
- `int numTrainingSamples`
- Potentially also `double** testingData` and `int* testingLabels` plus counts, if you decide to evaluate on a validation/testing subset.

---

## 3. Conceptual Design of the Genetic Asymmetric Item Memory

### 3.1. Current `item_mem.c` behavior (recap)

`item_mem.c` currently provides functions such as:

- `init_item_memory(...)` — random vectors per discrete item.
- `init_continuous_item_memory(...)` — continuous interpolation between two orthogonal vectors for levels.
- `init_precomp_item_memory(...)` — precomputed per-feature per-level vectors using two orthogonal endpoints per feature and interpolation.

In `init_precomp_item_memory`, the layout for feature/level hypervectors is:

- Total number of hypervectors = `num_levels * num_features`.
- Index mapping:
  - For a given `level` (0 to `num_levels - 1`) and `feature` (0 to `num_features - 1`), the vector is in
    `item_mem->base_vectors[level * num_features + feature]`.

Any new item memory initialization must **maintain exactly this layout**, so no downstream code needs to be changed.

### 3.2. What “asymmetric item memory” means here

We will introduce an alternative to the interpolation-based `init_precomp_item_memory`:

- For each feature `f`, we create a **sequence of hypervectors** across levels.
- For level `0` of feature `f`, we generate a single random hypervector.
- For each subsequent level `ℓ` (1 to `num_levels - 1`):
  - We obtain the hypervector at level `ℓ` by **flipping a certain number of bits** in the hypervector at level `ℓ - 1`.
  - The number of bits to flip is specified by an integer `B[f, ℓ]`.

The matrix `B` is thus:

- Conceptually shaped as `num_features` × `num_levels`.
- For each `f` and `ℓ`:
  - `B[f, ℓ]` is the number of bits to flip when going from level `ℓ - 1` to level `ℓ`.
  - For `ℓ = 0`, `B[f, 0]` can be set to 0 or ignored (level 0 is created purely randomly).

The GA will search over different choices of `B` to maximize classification performance when this asymmetric item memory is used in the existing HDC pipeline.

---

## 4. New Module: `asymItemMemory.h` and `asymItemMemory.c`

Create a new header file `asymItemMemory.h` and a new implementation file `asymItemMemory.c` in an appropriate location (for example, next to `item_mem.c` in the HDC infrastructure).

### 4.1. Public API in `asymItemMemory.h`

Define one main function that the rest of the project will call:

- Name the function something like `optimize_asym_item_memory`.
- Its purpose is:
  - To run a GA to optimize the `B` matrix based on classification accuracy.
  - To write the resulting best `B` into a caller-provided buffer.

The parameters should include:

1. A dataset identifier integer (e.g., `int dataset_id`) if you want the GA to load the data internally via `getData`. Alternately, pass the raw pointers and counts; both are acceptable, but avoid introducing a new dataset struct.
2. Pointers to:
   - Training data `double** trainingData`
   - Training labels `int* trainingLabels`
   - Number of training samples `int trainingSamples`
   - Number of features (use `NUM_FEATURES` or pass it explicitly)
   - Number of levels `int num_levels` (this must match the HDC quantization configuration).
3. Optionally:
   - Pointers and counts for validation/testing data if you want to evaluate on a held-out set during GA.
4. A pointer to a preallocated buffer for `B`:
   - Conceptually a 1D array of length `num_features * num_levels`.
   - The layout should be consistent and fixed: for feature `f` and level `ℓ`, access `B[f * num_levels + ℓ]`.
   - Each entry should be an integer type capable of storing values from 0 up to at least `VECTOR_DIMENSION` (e.g., `uint16_t`).
5. GA parameter struct or separate parameters, including at least:
   - Population size.
   - Number of generations.
   - Crossover probability.
   - Mutation probability.

Describe the function in the header in prose comments, for example:

- Explain that the GA will repeatedly:
  - Generate candidate `B` matrices.
  - For each candidate:
    - Build a **temporary item memory** from `B`.
    - Run the existing HDC training/encoding/classification code using that temporary item memory and the provided training data.
    - Compute the classification accuracy.
  - Use that accuracy as the **fitness**.
  - After several generations, return the `B` corresponding to the individual with the highest accuracy.

Do not define a new struct for the dataset; describe arguments only as raw pointers and integers.

### 4.2. Responsibilities of `asymItemMemory.c`

Inside `asymItemMemory.c`, you will implement:

1. **A representation of the GA population:**
   - Each individual corresponds to one candidate `B` matrix.
   - Store each individual as a contiguous block of integers in a 1D array of length `num_features * num_levels`.
   - Maintain an array of individuals (population), and optionally a separate array for offspring.

2. **Initialization function for individuals:**
   - A helper that fills a candidate `B` matrix with random yet reasonable values.
   - For each feature:
     - Initialize `B[f, 0]` to 0 (or ignore) since level 0 will be random.
     - For levels `ℓ` from 1 to `num_levels - 1`:
       - Choose a random number of flips within a reasonable range, e.g., from 0 up to some `max_flip`.
       - `max_flip` can be derived from `VECTOR_DIMENSION` and `num_levels`. For example, something like:
         - `VECTOR_DIMENSION / (2 * (num_levels - 1))` as an upper bound per level.
     - Optionally enforce that the **sum** over all levels for a given feature does not exceed some budget (for example, half the dimension), to keep early and late levels sufficiently different without overflipping.

3. **Mutation function:**
   - For each gene (each `B[f, ℓ]`), with a certain probability (mutation rate):
     - Randomly increase or decrease the number of flips by a small value (e.g., ±1).
     - Always clamp the value so that it stays within `0` and some `max_flip`.
   - Ensure you never mutate `B[f, 0]` (or treat it as always zero).

4. **Crossover function:**
   - Combine two parent individuals into a child individual.
   - Use a simple, robust scheme, such as:
     - One-point crossover (split at a random index and take the prefix from parent1 and the suffix from parent2).
     - Or uniform crossover (randomly select each gene from one of the two parents).
   - After crossover, enforce that entries corresponding to `ℓ = 0` are set to 0 for every feature.

5. **Evaluation function for a candidate `B`:**
   - This is the core function. It must:
     1. Build a **temporary item memory** based on `B`.
     2. Run the existing HDC pipeline using this temporary item memory.
     3. Compute and return the classification accuracy on the chosen dataset.

   The steps for this evaluation are:

   - Create a new local `struct item_memory tmp_im;`.
   - Allocate `tmp_im.base_vectors` as an array that can hold `num_levels * num_features` pointers to `Vector`.
   - For each entry, allocate a `Vector` using `create_uninitialized_vector()` (maintaining the existing conventions in `item_mem.c`).
   - For each feature `f` (from 0 to `num_features - 1`):
     - For level 0:
       - Generate a random hypervector using the existing helper `generate_random_hv`.
       - Store this hypervector at index `0 * num_features + f` in `tmp_im.base_vectors`.
     - Maintain an auxiliary array (e.g., a boolean or byte array of length `VECTOR_DIMENSION`) to track which positions (dimensions) have already been flipped for this feature, if you want to avoid flipping the same position too often.
     - For each subsequent level `ℓ` from 1 to `num_levels - 1`:
       - Copy the hypervector from level `ℓ - 1` into the current level for feature `f`.
       - Read the corresponding number of flips from `B[f * num_levels + ℓ]`.
       - Perform that many **bit flips** on the copied hypervector:
         - For bipolar mode (`BIPOLAR_MODE`), flipping a bit means negating the value (from -1 to 1 or vice versa).
         - For binary mode, flipping a bit means toggling between 0 and 1.
       - Decide whether to:
         - Prefer indices that have not been flipped yet (using the auxiliary tracking array), and/or
         - Just choose random indices without tracking.
       - Finally, store the resulting hypervector in the appropriate position in `tmp_im.base_vectors`.

   - After `tmp_im` is fully constructed, call the **existing HDC training and evaluation functions** to compute accuracy. To do this:
     - Locate the existing code paths in the project that:
       - Quantize raw EMG samples (the `double**` data) into discrete levels (indices between 0 and `num_levels - 1`).
       - Map each `(feature_index, level_index)` pair to the correct hypervector in `item_mem` (using the same indexing scheme as the rest of the project).
       - Bind/aggregate all feature hypervectors for a sample into a single hypervector.
       - Build class hypervectors (encoders) by accumulating sample hypervectors for each class label.
       - Classify samples by comparing their hypervectors to the class hypervectors and producing predicted labels.
       - Compute classification accuracy as the ratio of correctly classified samples to total samples.

     - Implement a helper function (within `asymItemMemory.c` or in another appropriate file, but without introducing a new dataset type) that:
       - Accepts:
         - `double** trainingData`, `int* trainingLabels`, `int trainingSamples`.
         - `int num_features`, `int num_levels`.
         - A pointer to `struct item_memory` (use `&tmp_im`).
       - Uses the existing pipeline to:
         - Train or build class hypervectors using `tmp_im`.
         - Evaluate one or more passes over the samples to compute accuracy.
       - Returns the accuracy as a floating-point value.

     - This helper must use the same functions and logic currently used for your HDC training/testing, only substituting `tmp_im` as the item memory. If the existing pipeline uses a global `item_mem`, redirect those references temporarily to `tmp_im` or refactor minimally to pass `item_memory` as a parameter.

   - After computing the accuracy:
     - Store the result in the fitness array for the GA.
     - Destroy `tmp_im` by calling `free_item_memory(&tmp_im)` to avoid leaks.

6. **GA main loop (`optimize_asym_item_memory`):**
   - Implement the GA steps:
     1. Allocate storage for the population (an array of individuals, each being a `B` matrix), an offspring population, and fitness values.
     2. Initialize all individuals using the random initialization helper.
     3. For each generation:
        - Evaluate each individual using the evaluation function described above.
        - Perform selection (e.g., tournament selection):
          - Repeatedly choose parent indices based on fitness (prefer higher accuracy).
        - Generate offspring population using crossover and mutation.
        - Replace the current population with the offspring.
        - Optionally, print progress (e.g., best accuracy in the current generation).
     4. After the final generation:
        - Evaluate all individuals one more time (or keep track of the best during the loop).
        - Select the individual with the highest fitness (highest accuracy).
        - Copy this individual’s `B` matrix into the caller-provided buffer (`B_out`).
     5. Free any allocated structures (population arrays, fitness arrays).

   - The GA should use **classification accuracy** as the primary and possibly only fitness measure. Multi-objective extensions (e.g., also using a robustness metric) can be added later if desired, but are not required for the first implementation.

---

## 5. Modifications to `item_mem.c`

Do not change existing functions unless necessary. Add a **new** function that:

- Initializes item memory for feature-level hypervectors using the GA-optimized `B*` matrix.
- Uses the same layout as `init_precomp_item_memory`.

### 5.1. New function: `init_genetic_precomp_item_memory`

Add a declaration in the appropriate header (e.g., `item_mem.h`):

- The function should take:
  - A pointer to `struct item_memory`.
  - The number of levels (`num_levels`).
  - The number of features (`num_features`).
  - Access to the training data and labels, or an identifier that can be used to load them via `getData`.

Implement the function in `item_mem.c` with the following conceptual steps:

1. **Allocate item memory vectors:**
   - Set `item_mem->num_vectors` to `num_levels * num_features`.
   - Allocate `item_mem->base_vectors` as an array of that many `Vector*`.
   - For each entry, allocate a `Vector` using `create_uninitialized_vector()`.

2. **Obtain or prepare training data for GA:**
   - Use the existing `getData` function from `dataReaderFootEMG` to load:
     - `trainingData` (double**),
     - `trainingLabels` (int*),
     - `trainingSamples` (int),
     - plus the testing data if needed.
   - The GA may run on the training set or on a subset; choose whichever is appropriate (training set is simplest at first).
   - Ensure that whatever quantization configuration (number of levels) used in the HDC pipeline matches the `num_levels` provided to this function.

3. **Call the GA optimizer (`optimize_asym_item_memory`):**
   - Prepare a buffer `B` of size `num_features * num_levels`.
   - Provide:
     - `trainingData`, `trainingLabels`, `trainingSamples`.
     - `num_features`, `num_levels`, `VECTOR_DIMENSION`.
     - GA parameters (population size, generations, etc.) chosen to balance runtime and quality. Start with moderate values and refine later.
   - After the call, `B` must contain the optimized flip counts `B*`.

4. **Generate the final hypervectors from `B*`:**
   - Reuse the same logic described for the evaluation function but now writing directly into `item_mem->base_vectors`:
     - For each feature:
       - Create a random hypervector for level 0 using `generate_random_hv`.
       - For each level `ℓ` > 0:
         - Copy the previous level’s hypervector.
         - Flip the number of bits indicated by `B[feature * num_levels + ℓ]`.
         - Store the result into `item_mem->base_vectors[level * num_features + feature]`.
       - Optionally use a local tracking array to avoid reusing the same flip positions too often, or accept random flips without tracking.

   - Ensure that all bit flipping respects `BIPOLAR_MODE` vs. binary mode, exactly as the rest of `item_mem.c` does.

5. **Optionally, free the training data:**
   - If the GA function or `init_genetic_precomp_item_memory` itself loaded the data via `getData`, release it using the existing `freeData` and `freeCSVLabels` functions once it is no longer needed.
   - Be careful not to double-free if the caller is responsible for freeing.

6. **Debug and logging:**
   - If `OUTPUT_MODE` is configured for detailed or debug output:
     - Print a message that GA-based item memory initialization is in use.
     - Optionally, print a subset of the generated vectors to verify structure.

Once this function is implemented, the rest of the project can call it instead of `init_precomp_item_memory` when GA-based initialization is desired.

---

## 6. Integration into the Training / Execution Flow

1. **Add `asymItemMemory.c` to the build:**
   - Update the project’s build system (Makefile, CMake, etc.) to compile and link `asymItemMemory.c` with the rest of the project.

2. **Introduce a configuration flag:**
   - In `config.h` or a similar configuration header, define a flag such as:
     - `USE_GENETIC_ITEM_MEMORY`.
   - Use this flag to select at compile time which item memory initialization is used.

3. **Replace calls to `init_precomp_item_memory`:**
   - Locate where `init_precomp_item_memory` is currently called (initialization of item memory for HDC).
   - Wrap this call in a conditional that chooses between:
     - The original interpolation-based precomputed item memory (when `USE_GENETIC_ITEM_MEMORY` is disabled).
     - The new `init_genetic_precomp_item_memory` (when `USE_GENETIC_ITEM_MEMORY` is enabled).

4. **Support offline GA training and binary export:**
   - For embedded targets, you typically do not want to run GA at runtime.
   - Add a mode (e.g., via command-line argument or separate executable) that:
     - Runs the GA-based initialization on the host.
     - After `init_genetic_precomp_item_memory` completes, calls `store_item_mem_to_bin(...)` to export the resulting hypervectors.
   - For the embedded or deployment build:
     - Skip GA entirely.
     - Load the precomputed item memory from the binary file using `load_item_mem_from_bin(...)`.

This pattern mirrors the existing `store_item_mem_to_bin` / `load_item_mem_from_bin` usage, but now with GA-optimized contents.

---

## 7. Testing and Validation

Before relying on the GA-based item memory, perform the following checks:

1. **Compilation tests:**
   - Ensure the project builds successfully:
     - With `USE_GENETIC_ITEM_MEMORY` disabled (old behavior).
     - With `USE_GENETIC_ITEM_MEMORY` enabled (new GA-based behavior).

2. **Functional tests:**
   - With GA disabled, verify that classification performance is identical to your current baseline.
   - With GA enabled:
     - Confirm that the GA runs for the configured number of generations without crashing.
     - Print progress (e.g., best accuracy per generation) to confirm that optimization is occurring.
     - Compare final classification accuracy to the baseline to ensure the new item memory is at least competitive.

3. **Binary file tests:**
   - After running GA and exporting the item memory via `store_item_mem_to_bin`, verify:
     - The file size equals `num_vectors * VECTOR_DIMENSION * sizeof(vector_element)`.
     - Loading the binary back via `load_item_mem_from_bin` yields equivalent classification performance.

4. **Performance tuning:**
   - If GA is too slow:
     - Reduce population size and number of generations.
     - Evaluate on a subset of training data.
   - Once correctness is validated, you can scale the GA parameters for better quality.

---

## 8. Summary of Required Steps for Codex

1. Create `asymItemMemory.h` with the declaration of `optimize_asym_item_memory`, using **only** existing data representations:
   - Arguments must be combinations of:
     - `double** trainingData`, `int* trainingLabels`, `int trainingSamples`,
     - `int num_features`, `int num_levels`, `int vector_dimension`,
     - Pointers and sizes for validation/testing data if desired,
     - GA configuration parameters,
     - A caller-allocated buffer for the result `B` of length `num_features * num_levels`.

2. Implement `asymItemMemory.c`:
   - Define GA data structures and arrays.
   - Implement:
     - Individual initialization for `B`.
     - Mutation operation.
     - Crossover operation.
     - Evaluation of individuals that:
       - Builds a temporary `item_memory` from `B`.
       - Uses the existing HDC pipeline to compute accuracy on `trainingData`/`trainingLabels`.
     - The full GA loop in `optimize_asym_item_memory` that returns the best `B`.

3. Extend `item_mem.c`:
   - Add `init_genetic_precomp_item_memory` that:
     - Allocates `item_mem->base_vectors` for `num_levels * num_features`.
     - Loads or receives training data via `getData` or as parameters.
     - Invokes `optimize_asym_item_memory` to fill `B*`.
     - Generates hypervectors from `B*` and stores them into `item_mem->base_vectors` in the same layout as `init_precomp_item_memory`.

4. Modify the main initialization flow to optionally call `init_genetic_precomp_item_memory` instead of `init_precomp_item_memory` based on a configuration flag.

5. Ensure all new code uses `double**` and `int*` data pointers returned by `getData` and does not introduce additional dataset abstractions.

6. Validate correctness, performance, and compatibility with the existing HDC codebase, and integrate binary export/import if required for deployment.

Following these instructions, Codex should be able to implement a complete GA-based asymmetric item memory generation pipeline that plugs into the existing C HDC system without changing the fundamental data structures.
