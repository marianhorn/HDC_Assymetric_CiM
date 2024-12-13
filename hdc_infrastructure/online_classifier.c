/**
 * @file online_classifier.c
 * @brief Implements real-time classification for EMG signals using HDC models.
 *
 * This file contains functionality to initialize an online classifier, process
 * streaming EMG data in batches, and calculate predictions based on similarity measures.
 * 
 * @details
 * The online classifier uses a batch-based approach to process EMG signals and 
 * classify them using hyperdimensional computing principles. It supports both 
 * bipolar and binary vector modes and utilizes the associative memory for classification.
 * @author Marian Horn
 */
#include "online_classifier.h"
#include "vector.h"
/**
 * @brief Initializes an online classifier for real-time EMG signal evaluation.
 *
 * This function sets up the online classifier by associating it with an encoder
 * and an associative memory structure and specifying the batch size for processing.
 *
 * @param classifier A pointer to the `onlineClassifier` structure to initialize.
 * @param assMem A pointer to the associative memory used for classification.
 * @param enc A pointer to the encoder used for feature extraction.
 * @param batchSize The number of samples to process in one batch.
 */
void init_online_classifier(struct onlineClassifier* classifier, 
                            struct associative_memory* assMem, 
                            struct encoder* enc, 
                            int batchSize) {
    classifier->assoc_mem = assMem;
    classifier->enc = enc;
    classifier->batch_size = batchSize;
}
/**
 * @brief Calculates the predicted label for a batch of testing data.
 *
 * This function encodes each time-series data sample in the batch using the provided encoder,
 * classifies it using the associative memory, and determines the best-predicted label based
 * on similarity scores.
 *
 * @param classifier A pointer to the `onlineClassifier` structure.
 * @param testing_data A 2D array of testing data samples, where each row is a feature vector.
 * 
 * @return The best-predicted label for the batch.
 *
 * @note The function assumes that the batch size is larger than `N_GRAM_SIZE`. If this 
 *       condition is not met, the behavior is undefined.
 * 
 * @details
 * - **Encoding:** Each sample is encoded into a hypervector.
 * - **Classification:** The hypervector is compared against class vectors in the associative
 *   memory to determine the closest match.
 * - **Confidence:** The similarity score (e.g., cosine similarity) is used to evaluate the
 *   prediction confidence.
 * 
 * @warning If the classification result is invalid (`-1`), the function terminates the program.
 */

int calculateUpdate(struct onlineClassifier* classifier,double** testing_data){
    double max_similarity = -1.0;
    int best_predicted_label = -1;
    //TODO: FEHLER wenn numsamples kleiner als ngram
    for(int i = 0; i<classifier->batch_size-N_GRAM_SIZE; i++){
        Vector* sample_hv = create_vector();
        int encodingResult = encode_timeseries(classifier->enc, &(testing_data[i]), sample_hv);
        int predicted_label = classify(classifier->assoc_mem, sample_hv);
        if(predicted_label==-1){
            printf("Encoding result: %i",encodingResult);
            printf("SampleHV number %i:\n",i);
            print_vector(sample_hv);
            fprintf(stderr, "Label not valid, terminating...");
            exit(EXIT_FAILURE);
        }
        double confidence = similarity_check(sample_hv,get_class_vector(classifier->assoc_mem,predicted_label));
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
    return best_predicted_label;
}