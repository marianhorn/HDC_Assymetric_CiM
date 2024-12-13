#ifndef VECTOR_H
#define VECTOR_H

#ifdef HAND_EMG
#include "../hand/configHand.h"
#elif defined(FOOT_EMG)
#include "../foot/configFoot.h"
#elif defined(CUSTOM)
#include "../customModel/configCustom.h"
#else
#error "No EMG type defined. Please define HAND_EMG or FOOT_EMG."
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#if BIPOLAR_MODE
typedef int vector_element;
#else
typedef bool vector_element;
#endif
/**
 * @brief Represents the item memory used in Hyperdimensional Computing.
 *
 * The item memory stores base hypervectors for discrete or continuous input features.
 * It supports precomputed and dynamically generated item memories.
 * 
 * Members:
 * - **num_vectors**: The total number of base vectors in the item memory.
 * - **base_vectors**: Array of pointers to the base hypervectors.
 */
typedef struct {
    vector_element *data; /**< Array of elements representing the vector. */
} Vector;

// Function declarations
Vector* create_vector();
Vector* create_uninitialized_vector();
void free_vector(Vector* vec);
void print_vector(const Vector* vec);

#endif // VECTOR_H
