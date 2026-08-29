#ifndef RESULT_MANAGER_H
#define RESULT_MANAGER_H

#include "../foot/configFoot.h"

#include "evaluator.h"

void result_manager_init(void);
void result_manager_close(void);
void addResult(const struct timeseries_eval_result *result, const char *info);

#endif // RESULT_MANAGER_H
