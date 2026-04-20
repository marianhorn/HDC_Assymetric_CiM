# SystemC HDC Pipeline v1

This folder contains a standalone SystemC implementation of the current `foot` precomputed-binary-fusion path.

## Included behavior
- precomputed CiM only
- binary vectors only
- `MODEL_VARIANT_FUSION` only
- no quantization
- no GA
- input = already quantized level indices per timestamp

## Structure
- `config_systemc.h`: minimal compile-time constants
- `hdc_systemc.h/.cpp`: CiM memory, timestamp encoder, fusion n-gram encoder, associative-memory trainer, classifier, top module
- `tb_systemc.cpp`: small untimed reference testbench
- `Makefile`: standalone build entry

## Notes
- The implementation mirrors the current C behavior for the binary non-Krischan `foot` path.
- The timestamp encoder uses the same majority rule as `bundle_multi`: output bit is `1` if ones `>= NUM_FEATURES / 2`.
- The final associative-memory vector uses the same current threshold rule as `trainer.c`.
- The SIMD datapath is modeled as fixed loops over `VECTOR_DIMENSION`, not as `VECTOR_DIMENSION` separate `SC_MODULE`s.

## Build
Set `SYSTEMC_HOME` to your SystemC installation root, then run:

```sh
export SYSTEMC_HOME="$HOME/systemc-install"
export CMAKE_PREFIX_PATH="$SYSTEMC_HOME:$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$SYSTEMC_HOME/lib:$LD_LIBRARY_PATH"
make -C systemc_hdc
make -C systemc_hdc run
```

If your SystemC library is not in `$(SYSTEMC_HOME)/lib-linux64`, set `SYSTEMC_LIB_DIR` explicitly.
