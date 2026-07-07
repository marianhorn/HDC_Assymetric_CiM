#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HLS_DIR="bdw_work/modules/P2PPipeline/HLS_BASIC"
OUT_DIR="$SCRIPT_DIR/vivado_p2p_rtl_sim"
SIM_RTL_DIR="$OUT_DIR/rtl_with_timescale"
TB_SRC="$SCRIPT_DIR/p2p_pipeline_rtl_tb.sv"
TB_DST="$OUT_DIR/p2p_pipeline_rtl_tb.sv"
EXTERNAL_P2P=0

VIVADO_BIN="$(command -v vivado || true)"
if [[ -n "$VIVADO_BIN" ]]; then
    VIVADO_ROOT="$(cd "$(dirname "$VIVADO_BIN")/.." && pwd)"
else
    VIVADO_ROOT="${XILINX_VIVADO:-}"
fi
GLBL_V="${XILINX_VIVADO:-$VIVADO_ROOT}/data/verilog/src/glbl.v"

if [[ ! -d "$HLS_DIR" ]]; then
    echo "ERROR: missing HLS output directory: $SCRIPT_DIR/$HLS_DIR" >&2
    echo "Run: make hls_P2PPipeline_HLS_BASIC" >&2
    exit 1
fi

if [[ ! -f "$TB_SRC" ]]; then
    echo "ERROR: missing testbench: $TB_SRC" >&2
    exit 1
fi

shopt -s nullglob
TOP_CANDIDATES=("$SCRIPT_DIR/$HLS_DIR"/*_rtl.v)
GENERATED_RTL=("$SCRIPT_DIR/$HLS_DIR"/v_rtl/*.v)
shopt -u nullglob

if [[ ${#TOP_CANDIDATES[@]} -eq 0 ]]; then
    echo "ERROR: no top RTL matching $HLS_DIR/*_rtl.v" >&2
    exit 1
fi
if [[ ${#GENERATED_RTL[@]} -eq 0 ]]; then
    echo "ERROR: no generated RTL found under $HLS_DIR/v_rtl" >&2
    exit 1
fi
if [[ ! -f "$GLBL_V" ]]; then
    echo "ERROR: missing Vivado glbl.v. Expected: $GLBL_V" >&2
    exit 1
fi

TB_DEFINES=()
for define in P2P_PAYLOAD_SAMPLE P2P_PAYLOAD_ENCODED P2P_PAYLOAD_FULL P2P_ENCODER_MIMIC P2P_ENCODER_MIMIC_NB P2P_ENCODER_SCALAR_MIMIC P2P_INTERNAL_SOURCE P2P_ACCEL_CMD_MIMIC P2P_EXTERNAL_P2P; do
    if grep -Eq "^[[:space:]]*set_attr[[:space:]]+D[[:space:]]+${define}([[:space:]]|$)" project.tcl; then
        if [[ "$define" == "P2P_EXTERNAL_P2P" ]]; then
            EXTERNAL_P2P=1
        fi
        TB_DEFINES+=("-d" "${define}")
        if [[ "$define" == "P2P_ENCODER_MIMIC_NB" || "$define" == "P2P_ENCODER_SCALAR_MIMIC" || "$define" == "P2P_INTERNAL_SOURCE" || "$define" == "P2P_ACCEL_CMD_MIMIC" || "$define" == "P2P_EXTERNAL_P2P" ]]; then
            TB_DEFINES+=("-d" "P2P_ENCODER_MIMIC")
        fi
        if [[ "$define" == "P2P_INTERNAL_SOURCE" || "$define" == "P2P_ACCEL_CMD_MIMIC" || "$define" == "P2P_EXTERNAL_P2P" ]]; then
            TB_DEFINES+=("-d" "P2P_ENCODER_SCALAR_MIMIC")
        fi
    fi
done

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR" "$SIM_RTL_DIR"

if [[ "$EXTERNAL_P2P" -eq 1 ]]; then
    python "$SCRIPT_DIR/generate_external_p2p_tb.py" "${TOP_CANDIDATES[0]}" "$TB_DST"
else
    cp "$TB_SRC" "$TB_DST"
    if grep -q '\\in_sample_levels\[0\]' "${TOP_CANDIDATES[0]}"; then
        for index in $(seq 0 31); do
            sed -i "s/\\.in_sample_levels_${index}(/.\\\\in_sample_levels[${index}] (/g" "$TB_DST"
        done
    fi
fi

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

for rtl in "${GENERATED_RTL[@]}" "${TOP_CANDIDATES[@]}"; do
    copy_with_timescale "$rtl"
done

pushd "$OUT_DIR" >/dev/null

xvlog -sv -work work -log xvlog.log \
    "${TB_DEFINES[@]}" \
    "${SIM_RTL[@]}" \
    "$TB_DST" \
    "$GLBL_V"

xelab -debug typical \
    -L unisims_ver \
    -L unimacro_ver \
    -L secureip \
    -top p2p_pipeline_rtl_tb \
    -top glbl \
    -snapshot p2p_pipeline_rtl_tb_snapshot \
    -log xelab.log

xsim p2p_pipeline_rtl_tb_snapshot -R -log xsim.log

popd >/dev/null

echo "P2P RTL simulation logs: $OUT_DIR"
