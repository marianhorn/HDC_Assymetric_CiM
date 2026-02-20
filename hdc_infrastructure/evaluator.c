/**
 * @file evaluator.c
 * @brief Implements functions for evaluating the performance of hyperdimensional classifiers.
 * 
 * @details
 * This file contains evaluation functions for assessing the accuracy and performance 
 * of HDC models on EMG data. It supports both time-series-based and general data-based 
 * evaluation methods and generates detailed results, including confusion matrices.
 * 
 * The evaluation process involves encoding the input data, classifying it using 
 * the associative memory, and comparing the predictions with ground truth labels.
 * 
 * @date 2023-12-11
 * @author Marian Horn
 */
#include "evaluator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "assoc_mem.h"
#include "operations.h"

int mode(int *array, int size) {
    int max_value = 0, max_count = 0, i, j;

    for (i = 0; i < size; i++) {
        int count = 0;
        for (j = 0; j < size; j++) {
            if (array[j] == array[i])
                ++count;
        }
        if (count > max_count) {
            max_count = count;
            max_value = array[i];
        }else if(count == max_count){//handle edge case as Matlab implementation: choose smallest value
            if(array[i]<max_value){
                max_value = array[i];
            }
        }
    }
    return max_value;
}

static double compute_class_average_accuracy(const int confusion_matrix[NUM_CLASSES][NUM_CLASSES]) {
    double sum = 0.0;
    int classes_with_samples = 0;

    for (int i = 0; i < NUM_CLASSES; i++) {
        int row_total = 0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            row_total += confusion_matrix[i][j];
        }
        if (row_total > 0) {
            sum += (double)confusion_matrix[i][i] / (double)row_total;
            classes_with_samples++;
        }
    }

    if (classes_with_samples == 0) {
        return 0.0;
    }
    return sum / (double)classes_with_samples;
}

static double compute_class_vector_similarity(const struct associative_memory *assoc_mem) {
    if (!assoc_mem || assoc_mem->num_classes <= 1) {
        return 0.0;
    }

    double sum = 0.0;
    int pairs = 0;
    for (int i = 0; i < assoc_mem->num_classes; i++) {
        for (int j = i + 1; j < assoc_mem->num_classes; j++) {
            double sim = similarity_check(assoc_mem->class_vectors[i],
                                          assoc_mem->class_vectors[j]);
            if (sim == -2) {
                fprintf(stderr, "Got invalid cosine similarity\nTerminating...");
                exit(EXIT_FAILURE);
            }
            sum += sim;
            pairs++;
        }
    }

    if (pairs == 0) {
        return 0.0;
    }
    return sum / (double)pairs;
}
/**
 * @brief Evaluates the HDC model using a sliding window over time-series data.
 * 
 * @details
 * This function evaluates the performance of the HDC model on a time-series dataset 
 * by applying a sliding window approach. For each window, it determines the most 
 * frequent ground truth label, encodes the data for all N-grams in the window, and predicts the label of the N-gram with the 
 * highest similarity to the class vectors. Results are compared to the ground truth labels to calculate 
 * accuracy and generate a confusion matrix
 * 
 * @param enc A pointer to the encoder structure.
 * @param assoc_mem A pointer to the associative memory structure.
 * @param testing_data A 2D array of testing data.
 * @param testing_labels An array of ground truth labels for the testing data.
 * @param testing_samples The number of testing samples in the dataset.
 * 
 * @note The window's size is determined by WINDOW in config.h
 */
struct timeseries_eval_result evaluate_model_timeseries_with_window(struct encoder *enc,
                                                                    struct associative_memory *assoc_mem,
                                                                    double **testing_data,
                                                                    int *testing_labels,
                                                                    int testing_samples) {
    struct timeseries_eval_result result;
    result.correct = 0;
    result.not_correct = 0;
    result.transition_error = 0;
    result.total = 0;
    result.overall_accuracy = 0.0;
    result.class_average_accuracy = 0.0;
    result.class_vector_similarity = 0.0;
    memset(result.confusion_matrix, 0, sizeof(result.confusion_matrix));

    for (int j = 0; j < testing_samples-WINDOW; j+=WINDOW) {
        int actual_label = mode(testing_labels + j, WINDOW);
        double max_similarity = -1.0;
        int best_predicted_label = -1;

        for (int k = 0; k <= WINDOW - N_GRAM_SIZE; k++) {
            Vector* sample_hv = create_vector();
            int encoding_result = encode_timeseries(enc, &(testing_data[j+k]), sample_hv);
            int predicted_label = classify(assoc_mem, sample_hv);
            if(predicted_label==-1){
                printf("Encoding result: %i",encoding_result);
                printf("SampleHV number %i:\n",j+k);
                print_vector(sample_hv);
                fprintf(stderr, "Label not valid, terminating...");
                exit(EXIT_FAILURE);
            }
            double confidence = similarity_check(sample_hv,get_class_vector(assoc_mem,predicted_label));
            if(confidence==-2){
                fprintf(stderr,"Got invalid cosine similarity\nTerminating...");
                exit(EXIT_FAILURE);
            }
            if (confidence > max_similarity) {
            max_similarity = confidence;
            best_predicted_label = predicted_label;
            }
            free_vector(sample_hv);
        }
        result.confusion_matrix[actual_label][best_predicted_label]++;
        
        if (best_predicted_label == actual_label) {
            result.correct++;
        }else{result.not_correct++;}
    }
    result.total = result.correct + result.not_correct;
    result.overall_accuracy = result.total > 0 ? (double)result.correct / (double)result.total : 0.0;
    result.class_average_accuracy = compute_class_average_accuracy(result.confusion_matrix);
    result.class_vector_similarity = compute_class_vector_similarity(assoc_mem);
    if (output_mode >= OUTPUT_BASIC) {
        int number_total_tests = (int)result.total;
        float accuracy = number_total_tests > 0 ? (float)result.correct / (number_total_tests) : 0.0f;

        printf("Testing accuracy: %.3f%%\n", accuracy * 100);
        printf("Class-average accuracy: %.3f%%\n", result.class_average_accuracy * 100.0);
        printf("Class vector similarity: %.3f\n", result.class_vector_similarity);

        printf("Total: %ld of %d ngrams correctly classified\n",result.correct,number_total_tests);
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Confusion Matrix:\n");
            printf("True\\Predicted\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("\t%d", i);
            }
            printf("\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("%d", i);
                for (int j = 0; j < NUM_CLASSES; j++) {
                    printf("\t%d", result.confusion_matrix[i][j]);
                }
                printf("\n");
            }
        }
    }
    return result;
}
/**
 * @brief Directly evaluates the HDC model on a time-series dataset.
 * 
 * @details
 * This function evaluates the model without using a sliding window. Instead, it 
 * processes the data in fixed N-grams, encodes each N-gram, and predicts the label 
 * for that N-gram. Results are compared to the ground truth labels to calculate 
 * accuracy and generate a confusion matrix.
 * 
 * @param enc A pointer to the encoder structure.
 * @param assoc_mem A pointer to the associative memory structure.
 * @param testing_data A 2D array of testing data.
 * @param testing_labels An array of ground truth labels for the testing data.
 * @param testing_samples The number of testing samples in the dataset.
 */
struct timeseries_eval_result evaluate_model_timeseries_direct(struct encoder *enc,
                                                               struct associative_memory *assoc_mem,
                                                               double **testing_data,
                                                               int *testing_labels,
                                                               int testing_samples) {
    struct timeseries_eval_result result;
    result.correct = 0;
    result.not_correct = 0;
    result.transition_error = 0;
    result.total = 0;
    result.overall_accuracy = 0.0;
    result.class_average_accuracy = 0.0;
    result.class_vector_similarity = 0.0;
    memset(result.confusion_matrix, 0, sizeof(result.confusion_matrix));

    if (output_mode >= OUTPUT_DETAILED) {
#if ENCODER_ROLLING && !BIPOLAR_MODE
        printf("Evaluating HDC-Model (rolling XOR) for %d testing samples.\n",testing_samples);
#else
        printf("Evaluating HDC-Model for %d testing samples.\n",testing_samples);
#endif
    }

#if ENCODER_ROLLING && !BIPOLAR_MODE
    int window_size = N_GRAM_SIZE;
    Vector *rolling_acc = create_vector();
    Vector **window_vectors = (Vector **)malloc((size_t)window_size * sizeof(Vector *));
    if (!rolling_acc || !window_vectors) {
        fprintf(stderr, "Failed to allocate rolling evaluation buffers.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < window_size; i++) {
        window_vectors[i] = create_vector();
    }

    int window_filled = 0;
    int window_pos = 0;

    for (int i = 0; i < testing_samples; i++) {
        Vector *sample_hv = create_vector();
        Vector *rotated_hv = create_vector();
        encode_timestamp(enc, testing_data[i], sample_hv);
        permute(sample_hv, window_pos, rotated_hv);

        if (window_filled < window_size) {
            bind(rolling_acc, rotated_hv, rolling_acc);
            memcpy(window_vectors[window_pos]->data,
                   rotated_hv->data,
                   VECTOR_DIMENSION * sizeof(vector_element));
            window_filled++;
        } else {
            bind(rolling_acc, window_vectors[window_pos], rolling_acc);
            bind(rolling_acc, rotated_hv, rolling_acc);
            memcpy(window_vectors[window_pos]->data,
                   rotated_hv->data,
                   VECTOR_DIMENSION * sizeof(vector_element));
        }

        window_pos = (window_pos + 1) % window_size;

        if (i >= window_size - 1) {
            int actual_label = testing_labels[i];
            int predicted_label = classify(assoc_mem, rolling_acc);
            if (predicted_label < 0) {
                fprintf(stderr, "Label not valid, terminating...");
                exit(EXIT_FAILURE);
            }
            result.confusion_matrix[actual_label][predicted_label]++;
            if (predicted_label == actual_label) {
                result.correct++;
            } else {
                result.not_correct++;
            }
        }

        free_vector(rotated_hv);
        free_vector(sample_hv);
    }

    // Match colleague reporting: denominator is total test samples even with warm-up skipped.
    result.total = (size_t)testing_samples;
    result.overall_accuracy =
        testing_samples > 0 ? (double)result.correct / (double)testing_samples : 0.0;
    result.class_average_accuracy = compute_class_average_accuracy(result.confusion_matrix);
    result.class_vector_similarity = compute_class_vector_similarity(assoc_mem);

    if (output_mode >= OUTPUT_BASIC) {
        printf("Testing accuracy: %.3f%%\n", result.overall_accuracy * 100.0);
        printf("Class-average accuracy: %.3f%%\n", result.class_average_accuracy * 100.0);
        printf("Class vector similarity: %.3f\n", result.class_vector_similarity);
        printf("Total: %ld of %d samples correctly classified\n",
               result.correct,
               testing_samples);
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Confusion Matrix:\n");
            printf("True\\Predicted\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("\t%d", i);
            }
            printf("\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("%d", i);
                for (int j = 0; j < NUM_CLASSES; j++) {
                    printf("\t%d", result.confusion_matrix[i][j]);
                }
                printf("\n");
            }
        }
    }

    for (int i = 0; i < window_size; i++) {
        free_vector(window_vectors[i]);
    }
    free(window_vectors);
    free_vector(rolling_acc);
#else
    //Iterate over testing data 
    for (int j = 0; j < testing_samples-N_GRAM_SIZE+1; j+=N_GRAM_SIZE) {
        int actual_label = mode(testing_labels + j, N_GRAM_SIZE);
        Vector* sample_hv = create_vector();
        int encoding_result = encode_timeseries(enc, &(testing_data[j]), sample_hv);
        int predicted_label = classify(assoc_mem, sample_hv);
        if(predicted_label==-1){
            printf("Encoding result: %i",encoding_result);
            printf("SampleHV number %i:\n",j);
            print_vector(sample_hv);
            fprintf(stderr, "Label not valid, terminating...");
            exit(EXIT_FAILURE);
        }
        double confidence = similarity_check(sample_hv,get_class_vector(assoc_mem,predicted_label));
        if(confidence==-2){
            fprintf(stderr,"Got invalid cosine similarity\nTerminating...");
            exit(EXIT_FAILURE);
        }

        free_vector(sample_hv);
        result.confusion_matrix[actual_label][predicted_label]++;
        
        if (predicted_label == actual_label) {
            result.correct++;
        }else if(testing_labels[j]!=testing_labels[j+N_GRAM_SIZE-1])
        {
            result.transition_error++;

        } else{result.not_correct++;}
    }

    result.total = result.correct + result.not_correct + result.transition_error;
    result.overall_accuracy = result.total > 0 ? (double)result.correct / (double)result.total : 0.0;
    result.class_average_accuracy = compute_class_average_accuracy(result.confusion_matrix);
    result.class_vector_similarity = compute_class_vector_similarity(assoc_mem);
    if (output_mode >= OUTPUT_BASIC) {
        int number_total_tests = (int)result.total;
        float accuracy = number_total_tests > 0 ? (float)result.correct / (number_total_tests) : 0.0f;
        float accuracyTranz = number_total_tests > 0
            ? ((float)result.correct + (float)result.transition_error) / (number_total_tests)
            : 0.0f;
        printf("Testing accuracy: %.3f%%\n", accuracy * 100);

        printf("Accuracy excluding gesture transitions: %.3f%%\n",accuracyTranz*100);
        printf("Class-average accuracy: %.3f%%\n", result.class_average_accuracy * 100.0);
        printf("Class vector similarity: %.3f\n", result.class_vector_similarity);
        printf("Total: %ld of %d ngrams correctly classified\n",result.correct,number_total_tests);
        printf("Transition error: %ld\n",result.transition_error);
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Confusion Matrix:\n");
            printf("True\\Predicted\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("\t%d", i);
            }
            printf("\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("%d", i);
                for (int j = 0; j < NUM_CLASSES; j++) {
                    printf("\t%d", result.confusion_matrix[i][j]);
                }
                printf("\n");
            }
        }
    }
#endif
    return result;
}

/**
 * @brief Directly evaluates the HDC model on general (non-time-series) data.
 * 
 * @details
 * This function evaluates the model by encoding individual data points (rather 
 * than N-grams) and predicting their labels. It computes accuracy metrics and 
 * generates a confusion matrix for the predictions.
 * 
 * @param enc A pointer to the encoder structure.
 * @param assoc_mem A pointer to the associative memory structure.
 * @param testing_data A 2D array of testing data.
 * @param testing_labels An array of ground truth labels for the testing data.
 * @param testing_samples The number of testing samples in the dataset.
 */
struct timeseries_eval_result evaluate_model_general_direct(struct encoder *enc,
                                                            struct associative_memory *assoc_mem,
                                                            double **testing_data,
                                                            int *testing_labels,
                                                            int testing_samples) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Evaluating HDC-Model for %d testing samples.\n",testing_samples);
    }
    struct timeseries_eval_result result;
    result.correct = 0;
    result.not_correct = 0;
    result.transition_error = 0;
    result.total = 0;
    result.overall_accuracy = 0.0;
    result.class_average_accuracy = 0.0;
    result.class_vector_similarity = 0.0;
    memset(result.confusion_matrix, 0, sizeof(result.confusion_matrix));


    for (int j = 0; j < testing_samples; j++) {
        int actual_label = testing_labels[j];
        Vector* sample_hv = create_vector();
        int encoding_result = encode_general_data(enc, testing_data[j], sample_hv);
        int predicted_label = classify(assoc_mem, sample_hv);
        if(predicted_label==-1){
            printf("Encoding result: %i",encoding_result);
            printf("SampleHV number %i:\n",j);
            print_vector(sample_hv);
            fprintf(stderr, "Label not valid, terminating...");
            exit(EXIT_FAILURE);
        }
        double confidence = similarity_check(sample_hv,get_class_vector(assoc_mem,predicted_label));
        if(confidence==-2){
            fprintf(stderr,"Got invalid cosine similarity\nTerminating...");
            exit(EXIT_FAILURE);
        }

        free_vector(sample_hv);
        result.confusion_matrix[actual_label][predicted_label]++;
        
        if (predicted_label == actual_label) {
            result.correct++;
        }else{result.not_correct++;}
    }

    result.total = result.correct + result.not_correct;
    result.overall_accuracy = result.total > 0 ? (double)result.correct / (double)result.total : 0.0;
    result.class_average_accuracy = compute_class_average_accuracy(result.confusion_matrix);
    result.class_vector_similarity = compute_class_vector_similarity(assoc_mem);
    if (output_mode >= OUTPUT_BASIC) {
        int number_total_tests = (int)result.total;
        float accuracy = number_total_tests > 0 ? (float)result.correct / (number_total_tests) : 0.0f;
        printf("Testing accuracy: %.3f%%\n", accuracy * 100);
        printf("Class-average accuracy: %.3f%%\n", result.class_average_accuracy * 100.0);
        printf("Class vector similarity: %.3f\n", result.class_vector_similarity);

        printf("Total: %ld of %d ngrams correctly classified\n",result.correct,number_total_tests);
        if (output_mode >= OUTPUT_DETAILED) {
            printf("Confusion Matrix:\n");
            printf("True\\Predicted\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("\t%d", i);
            }
            printf("\n");
            for (int i = 0; i < NUM_CLASSES; i++) {
                printf("%d", i);
                for (int j = 0; j < NUM_CLASSES; j++) {
                    printf("\t%d", result.confusion_matrix[i][j]);
                }
                printf("\n");
            }
        }
    }
    return result;
}

