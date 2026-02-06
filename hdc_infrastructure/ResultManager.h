#ifndef RESULT_MANAGER_H
#define RESULT_MANAGER_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include "evaluator.h"

void result_manager_init(void);
void result_manager_close(void);
void addResult(const struct timeseries_eval_result *result, const char *info);

#endif // RESULT_MANAGER_H
