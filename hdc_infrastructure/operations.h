#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include <stdbool.h>
#include "vector.h"

void bind(Vector* vector1,Vector* vector2,Vector* result);
void bundle(Vector* vector1, Vector* vector2, Vector* result);
void bundle_multi(Vector** vectors, int num_vectors, Vector* result);
void permute(Vector* vector, int offset, Vector* result);
double similarity_check(Vector *vec1, Vector *vec2);