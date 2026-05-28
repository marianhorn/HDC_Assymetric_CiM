// SYNTHESIS IMPORT DIAGNOSTIC ONLY.
// This file keeps the real HDC_Accelerator module declaration/interface from
// hdc_accelerator.h but replaces the datapath body with a minimal clocked FSM.
#include "hdc_accelerator.h"

void HDC_Accelerator::set_cim(unsigned level, unsigned feature, const hv_t &value) {
    m_cim[level][feature] = value;
}

void HDC_Accelerator::set_assoc_class(unsigned class_id, const hv_t &value) {
    m_assoc_mem[class_id] = value;
}

void HDC_Accelerator::pipeline_fsm() {
    cmd_ready.write(false);
    rsp_valid.write(false);
    rsp_valid_prediction.write(false);
    rsp_distances.write(0);
    wait();

    while (true) {
        cmd_ready.write(true);
        rsp_valid.write(false);
        rsp_valid_prediction.write(false);
        rsp_distances.write(0);
        wait();
    }
}
