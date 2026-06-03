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

# Xcelium is provided by the generic cadence19-20 module, but that module is in
# the same family as Stratus. Add the simulator paths directly to avoid
# replacing the loaded Stratus module.
export CDS_XCELIUM="${CDS_XCELIUM:-/eda/cadence/2019-20/RHELx86/XCELIUM_19.03.013}"
export BDW_INCISIVE_HOME="${BDW_INCISIVE_HOME:-${CDS_XCELIUM}}"
export CDS_LIC_FILE="${CDS_LIC_FILE:-5280@eplicense.e-technik.uni-erlangen.de}"

_hdc_prepend_path "${CDS_XCELIUM}/bin"
_hdc_prepend_path "${CDS_XCELIUM}/tools/bin"
_hdc_prepend_path "${CDS_XCELIUM}/tools/cdsgcc/gcc/bin"
export PATH

echo "Stratus: $(command -v stratus_hls || echo not found)"
echo "Xcelium: $(command -v xrun || echo not found)"
echo "BDW_INCISIVE_HOME=${BDW_INCISIVE_HOME}"

if ! command -v stratus_hls >/dev/null 2>&1; then
    echo "ERROR: stratus_hls was not found after loading Stratus." >&2
    return 1
fi

if ! command -v xrun >/dev/null 2>&1; then
    echo "ERROR: xrun was not found. Check CDS_XCELIUM=${CDS_XCELIUM}" >&2
    return 1
fi
