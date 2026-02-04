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
    if (output_mode >= OUTPUT_BASIC) {
        int number_total_tests = (int)(result.correct + result.not_correct);
        float accuracy = number_total_tests > 0 ? (float)result.correct / (number_total_tests) : 0.0f;

        printf("Testing accuracy: %.3f%%\n", accuracy * 100);

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
    result.total = result.correct + result.not_correct;
    result.overall_accuracy = result.total > 0 ? (double)result.correct / (double)result.total : 0.0;
    result.class_average_accuracy = compute_class_average_accuracy(result.confusion_matrix);
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
    memset(result.confusion_matrix, 0, sizeof(result.confusion_matrix));


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

    if (output_mode >= OUTPUT_BASIC) {
        int number_total_tests = (int)(result.correct + result.not_correct + result.transition_error);
        float accuracy = number_total_tests > 0 ? (float)result.correct / (number_total_tests) : 0.0f;
        float accuracyTranz = number_total_tests > 0
            ? ((float)result.correct + (float)result.transition_error) / (number_total_tests)
            : 0.0f;
        printf("Testing accuracy: %.3f%%\n", accuracy * 100);

        printf("Accuracy excluding gesture transitions: %.3f%%\n",accuracyTranz*100);
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
    result.total = result.correct + result.not_correct + result.transition_error;
    result.overall_accuracy = result.total > 0 ? (double)result.correct / (double)result.total : 0.0;
    result.class_average_accuracy = compute_class_average_accuracy(result.confusion_matrix);
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

    if (output_mode >= OUTPUT_BASIC) {
        int number_total_tests = (int)(result.correct + result.not_correct);
        float accuracy = number_total_tests > 0 ? (float)result.correct / (number_total_tests) : 0.0f;
        printf("Testing accuracy: %.3f%%\n", accuracy * 100);

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
    result.total = result.correct + result.not_correct;
    result.overall_accuracy = result.total > 0 ? (double)result.correct / (double)result.total : 0.0;
    result.class_average_accuracy = compute_class_average_accuracy(result.confusion_matrix);
    return result;
}

