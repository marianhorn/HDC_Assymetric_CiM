/**
 * @file assoc_mem.c
 * @brief Implements functions for managing associative memory.
 *
 * @details
 * The associative memory module provides functionality to store hypervectors for various classes,
 * classify new input vectors, and update class vectors dynamically (in bipolar mode). It supports both bipolar
 * and binary modes of operation and includes utilities for loading and saving memory.
 *
 * @author Marian Horn
 */


#include "assoc_mem.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "operations.h"
#include "vector.h"
#include <stdio.h>

/**
 * @brief Initializes the associative memory structure.
 *
 * @details
 * Allocates memory for storing hypervectors for each class and initializes the vectors to zero.
 * The memory structure also tracks the number of stored vectors per class.
 *
 * @param assoc_mem A pointer to the `associative_memory` structure to initialize.
 */
void init_assoc_mem(struct associative_memory *assoc_mem) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Initializing associative memory for %d classes.\n",NUM_CLASSES);
    }
    assoc_mem->num_classes = NUM_CLASSES;
    assoc_mem->class_vectors = (Vector **)malloc(NUM_CLASSES *sizeof(Vector*));
    assoc_mem->counts = (int*)malloc(NUM_CLASSES*sizeof(int));
    if(assoc_mem->class_vectors==NULL||assoc_mem->counts ==NULL){
        printf("Failed to allocate memory for data");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        assoc_mem->class_vectors[i] = create_vector();
        memset(assoc_mem->class_vectors[i]->data, 0, VECTOR_DIMENSION * sizeof(vector_element));
        assoc_mem->counts[i] = 0;
    }
}

/**
 * @brief Adds or sets a hypervector to the associative memory for a specific class.
 *
 * @details
 * - In **bipolar mode**, this function incrementally updates the class vector by bundling the
 *   input hypervector with the existing class vector, but only if similarity of class and input hypervector is below CUTTING_ANGLE_THRESHOLD.
 * - In **binary mode**, since majority voting does not support incremental bundling, the class
 *   vector is directly set to the input hypervector.
 *
 * @param assoc_mem A pointer to the associative memory structure.
 * @param hv The input hypervector to add or set.
 * @param class_label The class label to which the hypervector belongs.
 *
 * @note The `class_label` must be within the range `[0, NUM_CLASSES - 1]`.
 * @warning Ensure the input hypervector (`hv`) is not `NULL`.
 */
int add_to_assoc_mem(struct associative_memory *assoc_mem, Vector *sample_hv, int class_id) {
    //Bipolar case: adds an encoded data sample to the associative memory
    //Binary case: sets a classvector in the associative memory
    if (class_id >= 0 && class_id < assoc_mem->num_classes) {

        #if BIPOLAR_MODE
        Vector *memory_hv = assoc_mem->class_vectors[class_id];
            if(assoc_mem->counts[class_id]==0){
                for (int i = 0; i < VECTOR_DIMENSION; i++) {
                    memory_hv->data[i] = sample_hv->data[i];
                }
                assoc_mem->counts[class_id]=1;
                return 1;
            }else{
                double angle = similarity_check(memory_hv, sample_hv);
                if(angle == -2) {
                    fprintf(stderr, "AddToAssocMemFailed");
                    exit(EXIT_FAILURE);
                }

                if (angle < CUTTING_ANGLE_THRESHOLD) {
                    Vector *temp_vector = create_vector();
                    if(assoc_mem->counts[class_id]>0){
                        bundle(memory_hv, sample_hv, temp_vector);
                    }
                    
                    for (int i = 0; i < VECTOR_DIMENSION; i++) {
                        memory_hv->data[i] = temp_vector->data[i];
                    }
                    free_vector(temp_vector);
                    assoc_mem->counts[class_id]++;
                    return 1;
                }
                else{return 0;}
            }
        #else
            for (int i = 0; i < VECTOR_DIMENSION; i++) {
                    assoc_mem->class_vectors[class_id]->data[i] = sample_hv->data[i];
                }
            assoc_mem->counts[class_id]=1;
            return 1;
        #endif
    } else {
        fprintf(stderr, "AddToAssocMem: Invalid class id\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Classifies an input hypervector based on its similarity to stored class vectors.
 *
 * @details
 * Compares the input vector with each class vector in the associative memory using
 * cosine similarity (for bipolar mode) or Hamming distance (for binary mode).
 *
 * @param assoc_mem A pointer to the associative memory structure.
 * @param hv The input hypervector to classify.
 * @return The predicted class label, or `-1` if no valid classification is possible.
 *
 */
int classify(struct associative_memory *assoc_mem, Vector *sample_hv) {
    int best_class = -1;
    double best_similarity = -1.0;

    for (int i = 0; i < assoc_mem->num_classes; i++) {
        double similarity = similarity_check(assoc_mem->class_vectors[i], sample_hv);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_class = i;
        }
    }

    return best_class;
}
/**
 * @brief Retrieves the class vector for a given class ID.
 *
 * This function fetches the class vector for the specified class ID.
 *
 * @param assoc_mem A pointer to the associative memory structure.
 * @param class_id The class ID for which the vector is to be fetched.
 *
 * @return A pointer to the class vector.
 *
 * @note If the class ID is invalid, the program will terminate with an error.
 */

Vector* get_class_vector(struct associative_memory *assoc_mem, int class_id) {
    if (class_id >= 0 && class_id < assoc_mem->num_classes) {
        return assoc_mem->class_vectors[class_id];
    }else{
        printf("Error fetching class vector for class ID: %i",class_id);
        exit(EXIT_FAILURE);
    }
}
/**
 * @brief Frees the memory allocated for associative memory.
 *
 * This function frees the memory allocated for the class vectors and count array in the associative memory.
 *
 * @param assoc_mem A pointer to the associative memory structure to be freed.
 */

// Free associative memory
void free_assoc_mem(struct associative_memory *assoc_mem) {
    for (int i = 0; i < NUM_CLASSES; i++) {
        free_vector(assoc_mem->class_vectors[i]);
    }
    free(assoc_mem->class_vectors);
    free(assoc_mem->counts);
}

/**
 * @brief Prints all the class vectors stored in the associative memory.
 *
 * This function prints the learned class vectors and the number of trained elements per class.
 *
 * @param assoc_mem A pointer to the associative memory structure to print.
 */
// Print all the learned class vectors
void print_class_vectors(struct associative_memory *assoc_mem) {
    printf("Number of trained elements per class:\n");
    for(int i = 0; i<NUM_CLASSES; i++){
        printf("%d ",assoc_mem->counts[i]);
    }
    printf("\nClass Vectors:\n");
    for (int i = 0; i < 10; i+=1) {
        for (int j = 0; j < assoc_mem->num_classes; j++) {
#if BIPOLAR_MODE
            printf("%d ", assoc_mem->class_vectors[j]->data[i]);
#else 
            printf("%d ", (bool)assoc_mem->class_vectors[j]->data[i]);
#endif
        }
        printf("\n");
    }
}
/**
 * @brief Normalizes the class vectors by dividing each element by the number of samples in the class.
 *
 * This function performs a simple normalization of the class vectors by dividing each element by the count of
 * data samples in that class. This helps in balancing the class vectors over time.
 *
 * @param assoc_mem A pointer to the associative memory structure.
 * @note Can be activated/deactivated by NORMALIZE in config.h
 */
void normalize(struct associative_memory *assoc_mem) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Normalizing associative memory\n");
    }
    for (int i = 0; i < assoc_mem->num_classes; i++) {
        int count = assoc_mem->counts[i];
        if (count > 0) {
            for (int j = 0; j < VECTOR_DIMENSION; j++) {
                assoc_mem->class_vectors[i]->data[j] /= count;
            }
        }
    }
}

/**
 * @brief Stores the associative memory to a binary file.
 *
 * This function writes the associative memory's class vectors to a binary file. Each class vector 
 * is stored sequentially in the file. The data layout is as follows:
 *
 * - For **bipolar mode**:
 *   Each element of the class vector is stored as an `int` with values `-1` or `1`.
 * 
 * - For **binary mode**:
 *   Each element of the class vector is stored as an `bool` with values `0` or `1`.
 * 
 * The total size of the binary file will be:
 * `NUM_CLASSES * VECTOR_DIMENSION * sizeof(vector_element)`
 *
 * @param assoc_mem A pointer to the associative memory structure.
 * @param file_path The path to the binary file where the memory should be stored.
 *
 * @note The caller must ensure the associative memory is properly initialized before calling this function.
 * @warning The file will be overwritten if it already exists.
 */
void store_assoc_mem_to_bin(struct associative_memory *assoc_mem, const char *file_path) {
    FILE *file = fopen(file_path, "wb");
    if (file == NULL) {
        perror("Failed to open associative memory file for writing");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < assoc_mem->num_classes; i++) {
        fwrite(assoc_mem->class_vectors[i]->data, sizeof(vector_element), VECTOR_DIMENSION, file);
    }

    fclose(file);
    printf("Associative memory successfully stored to %s\n", file_path);
}
/**
 * @brief Loads the associative memory from a binary file.
 *
 * This function reads the associative memory's class vectors from a binary file. The file is 
 * expected to follow the same data layout as written by `store_assoc_mem_to_bin`.
 *
 * - For **bipolar mode**:
 *   Each element of the class vector is read as an `int` and is expected to be `-1` or `1`.
 * 
 * - For **binary mode**:
 *   Each element of the class vector is read as an `bool` and is expected to be `0` or `1`.
 * 
 * The function allocates memory for the class vectors as needed.
 *
 * @param assoc_mem A pointer to the associative memory structure.
 * @param filepath The path to the binary file from which to load the memory.
 *
 * @warning The function will terminate the program if the file format or size is invalid.
 */
void load_assoc_mem_from_bin(struct associative_memory *assoc_mem, const char *filepath) {
    FILE *file = fopen(filepath, "rb");
    if (file == NULL) {
        perror("Failed to open file for reading associative memory");
        exit(EXIT_FAILURE);
    }

    init_assoc_mem(assoc_mem);

    for (int i = 0; i < NUM_CLASSES; i++) {
        if (fread(assoc_mem->class_vectors[i]->data, sizeof(vector_element), VECTOR_DIMENSION, file) != VECTOR_DIMENSION) {
            fprintf(stderr, "Error: Incomplete vector data for class %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
    printf("Associative memory successfully loaded from %s\n", filepath);
}
