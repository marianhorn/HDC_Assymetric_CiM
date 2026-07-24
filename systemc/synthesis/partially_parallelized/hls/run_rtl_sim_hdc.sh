#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <trace-dir>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRACE_DIR="$(cd "$1" && pwd)"
HLS_DIR="bdw_work/modules/HDC_Accelerator/HLS_BASIC"
TOP_RTL="$SCRIPT_DIR/$HLS_DIR/hdc_accelerator_rtl.v"
OUT_DIR="$SCRIPT_DIR/vivado_rtl_sim_hdc"
TB_SV="$OUT_DIR/hdc_accelerator_rtl_tb.sv"
SIM_RTL_DIR="$OUT_DIR/rtl_with_timescale"
VIVADO_BIN="$(command -v vivado || true)"
if [[ -n "$VIVADO_BIN" ]]; then
    VIVADO_ROOT="$(cd "$(dirname "$VIVADO_BIN")/.." && pwd)"
else
    VIVADO_ROOT="${XILINX_VIVADO:-}"
fi
GLBL_V="${XILINX_VIVADO:-$VIVADO_ROOT}/data/verilog/src/glbl.v"

if [[ ! -f "$TRACE_DIR/commands.txt" ]]; then
    echo "ERROR: missing trace command file: $TRACE_DIR/commands.txt" >&2
    exit 1
fi
if [[ ! -f "$TRACE_DIR/expected_responses.txt" ]]; then
    echo "ERROR: missing trace response file: $TRACE_DIR/expected_responses.txt" >&2
    exit 1
fi
if [[ ! -f "$TOP_RTL" ]]; then
    echo "ERROR: missing HLS top RTL: $TOP_RTL" >&2
    echo "Run: make hls_HDC_Accelerator_HLS_BASIC" >&2
    exit 1
fi

shopt -s nullglob
MEM_RTL=("$SCRIPT_DIR"/mem_lib/*.v)
GENERATED_RTL=("$SCRIPT_DIR/$HLS_DIR"/v_rtl/*.v)
shopt -u nullglob

if [[ ${#MEM_RTL[@]} -eq 0 ]]; then
    echo "ERROR: no mem_lib Verilog found under $SCRIPT_DIR/mem_lib" >&2
    exit 1
fi
if [[ ${#GENERATED_RTL[@]} -eq 0 ]]; then
    echo "ERROR: no generated RTL found under $SCRIPT_DIR/$HLS_DIR/v_rtl" >&2
    exit 1
fi
if [[ ! -f "$GLBL_V" ]]; then
    echo "ERROR: missing Vivado glbl.v. Expected: $GLBL_V" >&2
    echo "Check XILINX_VIVADO or Vivado installation path." >&2
    exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR" "$SIM_RTL_DIR"

# Generated Stratus memories can contain $readmemh paths relative to this HLS
# directory, for example bdw_work/modules/.../*.memh. XSim runs from OUT_DIR,
# so expose the same relative path there without copying the full HLS tree.
if ln -s "$SCRIPT_DIR/bdw_work" "$OUT_DIR/bdw_work" 2>/dev/null; then
    :
else
    cp -R "$SCRIPT_DIR/bdw_work" "$OUT_DIR/bdw_work"
fi

python3 "$SCRIPT_DIR/generate_xsim_tb.py" \
    --top "$TOP_RTL" \
    --trace-dir "$TRACE_DIR" \
    --out "$TB_SV"

SIM_RTL=()
copy_with_timescale() {
    local src="$1"
    local dst="$SIM_RTL_DIR/$(basename "$src")"
    if grep -q '^[[:space:]]*`timescale' "$src"; then
        cp "$src" "$dst"
    else
        {
            printf '`timescale 1ns/1ps\n'
            cat "$src"
        } > "$dst"
    fi
    SIM_RTL+=("$dst")
}

for rtl in "${MEM_RTL[@]}" "${GENERATED_RTL[@]}" "$TOP_RTL"; do
    copy_with_timescale "$rtl"
done

pushd "$OUT_DIR" >/dev/null

xvlog -sv -work work -log xvlog.log \
    "${SIM_RTL[@]}" \
    "$TB_SV" \
    "$GLBL_V"

xelab -debug typical \
    -L unisims_ver \
    -L unimacro_ver \
    -L secureip \
    -top hdc_accelerator_rtl_tb \
    -top glbl \
    -snapshot hdc_accelerator_rtl_tb_snapshot \
    -log xelab.log

xsim hdc_accelerator_rtl_tb_snapshot -R -log xsim.log

popd >/dev/null

echo "RTL simulation logs: $OUT_DIR"
