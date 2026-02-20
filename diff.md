# Difference Report: `modelFoot` vs `krischans_model`

## Scope
- I compared the current code paths used for foot EMG classification in:
  - `foot/modelFoot.c`
  - `hdc_infrastructure/*` (encoder/trainer/evaluator/item memory/data reader)
  - `krischans_model/src/*` + memory generation scripts
- This is a static, code-level investigation focused on **why accuracy numbers can differ**.

## Executive Summary (most likely causes of different results)
Assuming your current config (`VALIDATION_RATIO=0`, `DOWNSAMPLE=1`), the main drivers are:
1. **Temporal encoding and evaluation are not equivalent** (different task definition).
2. **Continuous item memory (CM/CiM) generation is fundamentally different** (monotonic permutation-based vs random-walk bitflips).
3. **Feature-bundling tie handling differs** (`> half` vs `>= half`), which can flip many bits.
4. **Label handling for sequences differs** (stable-window filtering + transition handling vs per-sample labels).
5. **Reproducibility differs** (deterministic C `rand` flow vs Python `random` scripts without fixed seed; fixed IM size script default).

---

## 1) Data handling and split protocol

### `modelFoot`
- Iterates all 4 datasets in one run: `foot/modelFoot.c:25`.
- Uses `getDataWithValSet(...)`: `foot/modelFoot.c:58`.
- Optional validation split via `VALIDATION_RATIO` (default 0): `foot/configFoot.h:53`.
- Downsampling is always part of the loading path (`DOWNSAMPLE`, default 1): `hdc_infrastructure/preprocessor.c:35-37`, `foot/configFoot.h:24`.
- Validation split is class-balanced but order-preserving first-occurrence assignment: `foot/dataReaderFootEMG.c:93-99`, `foot/dataReaderFootEMG.c:132-149`.

### `krischans_model`
- Loads train/test CSV directly per dataset path, no downsampling or validation API: `krischans_model/src/main.c:63-74`.

### Impact
- With your current config (`VALIDATION_RATIO=0`, `DOWNSAMPLE=1`), this is **not a primary cause** of the accuracy gap.
- It becomes relevant again if validation split or downsampling are changed.

---

## 2) Quantization of feature values to levels

### `modelFoot`
- Linear map with floor/truncation:
  - `hdc_infrastructure/encoder.c:72-82`
  - `level = (int)(normalized * (NUM_LEVELS - 1))`

### `krischans_model`
- Scales to `[0,20000]`, uses `ceil(...)`, then integer rounding style mapping:
  - `krischans_model/src/hdc_encode.c:12-20`

### Impact
- Borderline feature values map to different levels.
- This directly changes bound feature HVs and all downstream class vectors.

---

## 3) Spatial encoding tie rule (very important in binary mode)

### `modelFoot`
- Bundling over 32 features uses strict majority:
  - `result_bit = (count_true > num_vectors/2)` in `hdc_infrastructure/operations.c:126`

### `krischans_model`
- Bundling over 32 features uses `>=`:
  - `set_bit(..., cnt[bit] >= (N/2))` in `krischans_model/src/hdc_encode.c:66`

### Impact
- For `N=32`, tie at 16/16 is common for near-random bits.
- Your model writes tie bits as 0; colleague model writes tie bits as 1.
- This can alter a large fraction of encoded sample bits.

---

## 4) Temporal encoding is different (core mismatch)

### `modelFoot` (N-gram)
- Uses recursive permutation + bind for `N_GRAM_SIZE`:
  - `hdc_infrastructure/encoder.c:166-173` and `194-201`
- This is positional N-gram composition.

### `krischans_model` mode 1 (rolling block)
- Uses rolling XOR accumulator with slot-based rotation:
  - `krischans_model/src/block_accumulator.c:32-55`
- Rotation depends on ring-buffer position (`window_pos`), not equivalent to your recursive N-gram construction.

### Impact
- Even if both use window size 5, representations are mathematically different.
- Different inductive bias => different class prototypes and accuracy.

---

## 5) Sequence label policy and transition handling

### `modelFoot`
- Trains only when window is “stable” via `is_window_stable(...)`: `hdc_infrastructure/trainer.c:65`, `91`.
- Stability check compares first vs last only: `hdc_infrastructure/encoder.c:140-145`.
- On unstable windows in binary training, loop jumps by `N_GRAM_SIZE-1`: `hdc_infrastructure/trainer.c:99`.
- Evaluation is non-overlapping windows (`j += N_GRAM_SIZE`): `hdc_infrastructure/evaluator.c:222`.
- Uses mode label per N-gram and tracks transition errors separately: `hdc_infrastructure/evaluator.c:223`, `245-249`.

### `krischans_model`
- Mode 0: per-sample encoding/classification.
- Mode 1: rolling classification each sample after warm-up (`i >= BLOCK_WINDOW-1`): `krischans_model/src/main.c:153-164`.
- Uses current sample label `y[i]` for train/test decisions (no stable-window filter).

### Impact
- You are not scoring the same prediction units:
  - your direct eval: ~non-overlapping n-grams,
  - colleague: per-sample stream (or near per-sample after warm-up).
- Transition treatment is fundamentally different.

---

## 6) Continuous item memory generation differs strongly

### `modelFoot`
- Runtime CiM generation with random min vector + random permutation + cumulative flips:
  - `hdc_infrastructure/item_mem.c:170-219`
- Total flip budget controlled by `GA_MAX_FLIPS_CIM` (default = `VECTOR_DIMENSION`): `foot/configFoot.h:82`.

### `krischans_model`
- CM vectors are generated offline by script:
  - fixed per-level flips `bit_flips = D // 40`: `krischans_model/scripts/bitflipvector.py:11`
  - repeated random index flips with replacement: `krischans_model/scripts/bitflipvector.py:16-21,27-29`
- IM generation script default is hardcoded `vector_size = 2000`: `krischans_model/scripts/randomvector.py:4`.

### Impact
- Your CiM: structured monotonic path via one permutation.
- Colleague CiM: random-walk behavior over levels.
- These produce very different level geometry and separability.
- If colleague IM is not regenerated for current `D`, silent mismatch risk is high.

---

## 7) Class prototype training differences

### `modelFoot`
- Binary path stores encoded vectors per class and bundles once per class:
  - `hdc_infrastructure/trainer.c:78-108`
- Training windows are filtered by stability/transition logic.

### `krischans_model`
- Collects vectors per class, then majority train:
  - `krischans_model/src/main.c:135`
  - `krischans_model/src/hdc_train.c:5-16`
- No stability filter.

### Impact
- Different sample selection and class prototype composition.

---

## 8) Evaluation metric/reporting differences

### `modelFoot`
- Reports:
  - overall testing accuracy,
  - transition-adjusted accuracy,
  - class-average accuracy,
  - class-vector similarity, confusion matrix (depending on output mode):
  - `hdc_infrastructure/evaluator.c:256-281`

### `krischans_model`
- Reports plain accuracy only.
- In mode 1, first `BLOCK_WINDOW-1` samples are skipped (`continue`) but denominator still `test_count`:
  - skip: `krischans_model/src/main.c:162-163`
  - denominator: `krischans_model/src/main.c:170`

### Impact
- Reported numbers are not directly apples-to-apples.

---

## 9) Randomness and reproducibility

### `modelFoot`
- Uses C `rand()` in several initializers (e.g., item memory): `hdc_infrastructure/item_mem.c:87-90`, `170-183`.
- No explicit `srand(...)` in these paths => deterministic process-start sequence, but state advances with call order.

### `krischans_model`
- IM/CM scripts use Python `random` without fixed seed:
  - `krischans_model/scripts/randomvector.py`
  - `krischans_model/scripts/bitflipvector.py`

### Impact
- Colleague memory files can vary between runs unless seed and files are fixed.
- Your runtime memory can also change with changed call order/config.

---

## 10) Configuration/default mismatch summary

- Your defaults: `VECTOR_DIMENSION=10000`, `NUM_LEVELS=100`, `N_GRAM_SIZE=5`, binary mode:
  - `foot/configFoot.h:5,8,21,57`
- Colleague defaults: `D=10000`, `M=32`, `BLOCK_WINDOW=5`:
  - `krischans_model/src/hdc_types.c:3-4`, `krischans_model/include/block_accumulator.h:7`

### Impact
- If not explicitly aligned (`M` vs `NUM_LEVELS`, etc.), results are expected to differ.

---

## High-confidence conclusion
The two implementations are currently **not equivalent experiments**.  
The largest divergences are temporal encoding/evaluation protocol and CM/CiM construction. Those alone are sufficient to explain large accuracy gaps.

---

## Recommended apples-to-apples comparison protocol

1. **Match temporal protocol first**
   - Compare `krischans mode 0` against your model with `N_GRAM_SIZE=1`.
2. **Use identical IM/CM vectors**
   - Export from one pipeline and import into the other (same `D`, same level count).
3. **Align quantization and tie rule**
   - Use same level mapping formula and same majority tie policy.
4. **Align evaluation unit**
   - Either both per-sample or both non-overlapping windows.
5. **Fix randomness**
   - Lock seeds/scripts and avoid regenerating memory files mid-comparison.
6. **Only then compare advanced temporal versions**
   - Your recursive N-gram vs colleague rolling block can be compared as different methods.

---

## Addendum: Answers to your follow-up questions

### Q1) Data handling: with `VALIDATION_RATIO=0` and `DOWNSAMPLE=1`, are train/test effectively the same as in colleague model?

Short answer: **yes, almost**.

- In your loader, with `DOWNSAMPLE=1`, `down_sample(...)` keeps all rows (`new_length = original_size / 1`): `hdc_infrastructure/preprocessor.c:35-37`.
- With `VALIDATION_RATIO=0`, all downsampled training rows stay in training, validation stays empty:
  - target calculation gives 0 per class: `foot/dataReaderFootEMG.c:93-99`
  - assignment then sends all rows to training: `foot/dataReaderFootEMG.c:132-152`
- So semantically, train/test partitions match the CSV split.

Remaining minor differences:
- Your code still allocates/copies through preprocessing buffers.
- If preprocessing code changes later, this path can diverge again.

### Q2) Quantization: why does colleague model scale to `20000`?

In `krischans_model`, quantization does:
- `scaled = ceil(x * 10000 + 10000)` then clip to `[0,20000]`: `krischans_model/src/hdc_encode.c:12-15`
- `level = ((int)scaled * (M-1) + 10000) / 20000`: `krischans_model/src/hdc_encode.c:17`

Intent:
- map input assumed near `[-1,1]` to an integer range `[0,20000]`,
- then do mostly integer arithmetic for level mapping.

Equivalent intuition:
- approximately `level ~= round(((x + 1)/2) * (M-1))`, with clipping.

So it is mainly a fixed-point style mapping choice; it is not wrong, but it is **different** from your linear truncation mapping (`encoder.c:72-82`), which can shift boundary assignments.

### Q3) Spatial encoding difference: explain in detail. Would changing to `>=` make it equal?

How both pipelines do spatial encoding (binary):
1. For each of 32 features, build a feature HV by binding feature-ID and level HV (XOR).
2. Bundle those 32 feature HVs bitwise by majority.

The key mismatch is tie handling at each bit:
- your code: bit = 1 only if `count_true > 16` (`N=32`) -> ties become 0:
  - `hdc_infrastructure/operations.c:126`
- colleague code: bit = 1 if `count_true >= 16` -> ties become 1:
  - `krischans_model/src/hdc_encode.c:66`

Would changing your code to `>=` make spatial encoding equal?
- **Only for this one sub-step**, yes (assuming same 32 input feature-HVs).
- End-to-end equality still requires same quantization, same IM/CM vectors, same temporal method, same training/eval protocol.

### Q4) Temporal encoding difference: detailed rolling XOR in colleague model

Colleague mode-1 temporal logic:

State:
- `window[5]` stores last rotated sample HVs,
- `window_pos` is current slot (0..4),
- `out` is rolling HV.

Per new sample HV `s_t`:
1. `rotated = rotate_right(s_t, window_pos)` (`block_accumulator.c:37`).
2. If window not full:
   - `out ^= rotated`
   - `window[window_pos] = rotated`
3. Else (full window):
   - `out ^= window[window_pos]` (remove old vector in this slot),
   - `out ^= rotated` (add new one),
   - overwrite slot with `rotated`.
4. `window_pos = (window_pos + 1) % 5` (`block_accumulator.c:55`).

So after warm-up, `out` is XOR of exactly 5 stored rotated vectors (one per slot).

Important consequence:
- Rotation is tied to cyclic slot index, not a fixed N-gram composition rule like your recursive `permute+bind`.
- In testing, first 4 samples are skipped (`main.c:162-163`), but denominator still uses full `test_count` (`main.c:170`), which depresses reported accuracy in mode 1.

### Q6) Detailed IM/CM generation in colleague model

#### IM generation (`randomvector.py`)
- Generates `num_vectors=32` random binary vectors.
- Default vector length is hardcoded `vector_size=2000`: `krischans_model/scripts/randomvector.py:4-5`.
- Writes plain text bitstrings to `memoryfiles/position-vectors.txt`.

Implication:
- If you run model with `D != 2000` and did not regenerate IM accordingly, memory loading/geometry is inconsistent.

#### CM generation (`bitflipvector.py`)
- Inputs: `D`, `M`.
- Sets `bit_flips = D // 40`: `krischans_model/scripts/bitflipvector.py:11`.
- Level 0: one random vector.
- Each next level:
  - performs `bit_flips` random position toggles with replacement (`randint`): `bitflipvector.py:16-21,27-29`.

Implications:
- This is a random walk in Hamming space, not your monotonic permutation path.
- Same bit can be flipped multiple times in one step (cancellation possible).
- No fixed seed -> different CM each regeneration unless you add seeding.
