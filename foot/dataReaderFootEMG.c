#include "dataReaderFootEMG.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hdc_infrastructure/preprocessor.h"

#define INITIAL_CAPACITY 1024

static void get_file_paths(int dataset_id, char *training_emg_file, char *training_labels_file, char *testing_emg_file, char *testing_labels_file) {
    sprintf(training_emg_file, "foot/data/dataset%02d/training_emg.csv", dataset_id);
    sprintf(training_labels_file, "foot/data/dataset%02d/training_labels.csv", dataset_id);
    sprintf(testing_emg_file, "foot/data/dataset%02d/testing_emg.csv", dataset_id);
    sprintf(testing_labels_file, "foot/data/dataset%02d/testing_labels.csv", dataset_id);
}
// Function to count rows in a CSV file
size_t countCSVRows(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    char line[1024];
    size_t rows = 0;

    // Skip the header row
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file)) {
        rows++;
    }

    fclose(file);
    return rows;
}

// Function to load EMG data from a CSV file
double** loadEMGData(char* filename, size_t rows, size_t cols) {
    double** data = (double**)malloc(rows * sizeof(double*));
    if (data == NULL) {
        perror("Malloc failed for data");
        exit(EXIT_FAILURE);
    }

    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    char line[1024];

    // Skip the header row
    fgets(line, sizeof(line), file);

    size_t row = 0;
    while (fgets(line, sizeof(line), file)) {
        data[row] = (double*)malloc(cols * sizeof(double));
        if (data[row] == NULL) {
            perror("Malloc failed for data[row]");
            exit(EXIT_FAILURE);
        }

        char* token = strtok(line, ",");
        size_t col = 0;
        while (token != NULL && col < cols) {
            data[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(file);
    return data;
}

// Function to load labels from a CSV file
int* loadLabels(char* filename, size_t rows) {
    int* labels = (int*)malloc(rows * sizeof(int));
    if (labels == NULL) {
        perror("Malloc failed for labels");
        exit(EXIT_FAILURE);
    }

    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        exit(EXIT_FAILURE);
    }

    char line[1024];

    // Skip the header row
    fgets(line, sizeof(line), file);

    size_t row = 0;
    while (fgets(line, sizeof(line), file)) {
        labels[row] = atoi(line);
        row++;
    }

    fclose(file);
    return labels;
}


// Function to ONLY testing data
void getTestingData(int dataset, double*** testingData, int** testingLabels, int* testingSamples) {
    char training_emg_file_name[128], training_labels_file_name[128], testing_emg_file_name[128], testing_labels_file_name[128];
    get_file_paths(dataset,training_emg_file_name,training_labels_file_name,testing_emg_file_name,testing_labels_file_name);
    size_t testingRows = countCSVRows(testing_emg_file_name);

    double** rawTestingData = loadEMGData(testing_emg_file_name, testingRows, NUM_FEATURES);
    int* rawTestingLabels = loadLabels(testing_labels_file_name, testingRows);

    if (rawTestingData == NULL) {
        fprintf(stderr, "Data or Labels loading failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Downsample training and testing data
    double** downSampledDataTest = NULL;
    int* downSampledLabelsTest = NULL;
    size_t downSampledSizeTest = 0;
    down_sample(rawTestingData, rawTestingLabels, testingRows, &downSampledDataTest, &downSampledLabelsTest, &downSampledSizeTest);
    *testingData = downSampledDataTest;
    *testingLabels = downSampledLabelsTest;
    *testingSamples = (int)downSampledSizeTest;
    if (downSampledDataTest == NULL) {
        fprintf(stderr, "Error: downSampleDataTest.\n");
        exit(EXIT_FAILURE);
    }
    freeData(rawTestingData, testingRows);
    freeCSVLabels(rawTestingLabels);
}
// Function to get training and testing data
void getData(int dataset,double*** trainingData, double*** testingData, int** trainingLabels, int** testingLabels, int* trainingSamples, int* testingSamples) {
    #if OUTPUT_MODE>=OUTPUT_DETAILED
        printf("Reading data.\n");
    #endif
    // Load raw training data
    char training_emg_file_name[128], training_labels_file_name[128], testing_emg_file_name[128], testing_labels_file_name[128];
    get_file_paths(dataset,training_emg_file_name,training_labels_file_name,testing_emg_file_name,testing_labels_file_name);
    size_t trainingRows = countCSVRows(training_emg_file_name);
    size_t testingRows = countCSVRows(testing_emg_file_name);

    double** rawTrainingData = loadEMGData(training_emg_file_name, trainingRows, NUM_FEATURES);
    int* rawTrainingLabels = loadLabels(training_labels_file_name, trainingRows);

    double** rawTestingData = loadEMGData(testing_emg_file_name, testingRows, NUM_FEATURES);
    int* rawTestingLabels = loadLabels(testing_labels_file_name, testingRows);

    if (rawTestingData == NULL || rawTrainingLabels == NULL) {
        fprintf(stderr, "Data or Labels loading failed\n");
        exit(EXIT_FAILURE);
    }
    
    // Downsample training and testing data
    double** downSampledDataTrain = NULL;
    int* downSampledLabelsTrain = NULL;
    size_t downSampledSizeTrain = 0;
    double** downSampledDataTest = NULL;
    int* downSampledLabelsTest = NULL;
    size_t downSampledSizeTest = 0;
    down_sample(rawTrainingData, rawTrainingLabels, trainingRows, &downSampledDataTrain, &downSampledLabelsTrain, &downSampledSizeTrain);
    down_sample(rawTestingData, rawTestingLabels, testingRows, &downSampledDataTest, &downSampledLabelsTest, &downSampledSizeTest);
    *trainingData = downSampledDataTrain;
    *trainingLabels = downSampledLabelsTrain;
    *trainingSamples = (int)downSampledSizeTrain;

    *testingData = downSampledDataTest;
    *testingLabels = downSampledLabelsTest;
    *testingSamples = (int)downSampledSizeTest;
    if (downSampledDataTest == NULL) {
        fprintf(stderr, "Error: downSampleDataTest.\n");
        exit(EXIT_FAILURE);
    }
    if (*trainingData == NULL) {
        fprintf(stderr, "Error: Downsampling training data failed. Data is NULL.\n");
        exit(EXIT_FAILURE);
    }

    // Free raw data
    freeData(rawTrainingData, trainingRows);
    freeCSVLabels(rawTrainingLabels);
    freeData(rawTestingData, testingRows);
    freeCSVLabels(rawTestingLabels);
}

// Function to free CSV data
void freeData(double** data, size_t rows) {
    for (size_t i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
}

// Function to free CSV labels
void freeCSVLabels(int* labels) {
    free(labels);
}
