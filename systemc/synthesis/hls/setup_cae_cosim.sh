#!/usr/bin/env bash
# Source this file on CAE before running Stratus RTL cosimulation:
#   source setup_cae_cosim.sh
#
# It keeps the Stratus module loaded and adds only the Xcelium paths needed by
# Stratus cosim. Executing this script directly cannot update the parent shell.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this script instead of executing it:" >&2
    echo "  source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_hdc_prepend_path() {
    case ":${PATH}:" in
        *":$1:"*) ;;
        *) PATH="$1:${PATH}" ;;
    esac
}

_hdc_find_xcelium() {
    local candidate

    # Prefer newer Xcelium installations. Xcelium 19.03 is visible on CAE, but
    # Stratus 22 rejects it because it cannot compile with
    # -D_GLIBCXX_USE_CXX11_ABI=1.
    for candidate in \
        /eda/cadence/2025-26/RHELx86/XCELIUM_* \
        /eda/cadence/2024-25/RHELx86/XCELIUM_* \
        /eda/cadence/2023-24/RHELx86/XCELIUM_* \
        /eda/cadence/2022-23/RHELx86/XCELIUM_* \
        /eda/cadence/2021-22/RHELx86/XCELIUM_* \
        /eda/cadence/2020-21/RHELx86/XCELIUM_*; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    return 1
}

if ! command -v module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    fi
fi

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: environment modules are not available in this shell." >&2
    return 1
fi

module load Core/vivado/vivado2023.2 || return 1
module load Core/cadence/stratus22_23 || return 1

# Xcelium is provided by generic Cadence modules, but those modules are in the
# same family as Stratus. Add simulator paths directly to avoid replacing the
# loaded Stratus module.
if [[ -z "${CDS_XCELIUM:-}" ]]; then
    CDS_XCELIUM="$(_hdc_find_xcelium)" || {
        echo "ERROR: no compatible Xcelium installation found under /eda/cadence." >&2
        echo "       Set CDS_XCELIUM explicitly before sourcing this script." >&2
        return 1
    }
fi

if [[ "${CDS_XCELIUM}" == *"/XCELIUM_19.03."* ]]; then
    echo "ERROR: ${CDS_XCELIUM} is too old for Stratus 22 cosim." >&2
    echo "       Use a newer Xcelium path, for example from cadence24-25 or cadence25-26." >&2
    return 1
fi

export CDS_XCELIUM
export BDW_INCISIVE_HOME="${BDW_INCISIVE_HOME:-${CDS_XCELIUM}}"
export CDS_LIC_FILE="${CDS_LIC_FILE:-5280@eplicense.e-technik.uni-erlangen.de}"

_hdc_prepend_path "${CDS_XCELIUM}/bin"
_hdc_prepend_path "${CDS_XCELIUM}/tools/bin"
_hdc_prepend_path "${CDS_XCELIUM}/tools/cdsgcc/gcc/bin"
export PATH

echo "Stratus: $(command -v stratus_hls || echo not found)"
echo "Xcelium: $(command -v xrun || echo not found)"
if command -v xrun >/dev/null 2>&1; then
    xrun -version | head -n 1
fi
echo "BDW_INCISIVE_HOME=${BDW_INCISIVE_HOME}"

if ! command -v stratus_hls >/dev/null 2>&1; then
    echo "ERROR: stratus_hls was not found after loading Stratus." >&2
    return 1
fi

if ! command -v xrun >/dev/null 2>&1; then
    echo "ERROR: xrun was not found. Check CDS_XCELIUM=${CDS_XCELIUM}" >&2
    return 1
fi
