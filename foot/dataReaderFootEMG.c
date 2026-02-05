#include "dataReaderFootEMG.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../hdc_infrastructure/preprocessor.h"

#define INITIAL_CAPACITY 1024

size_t countCSVRows(const char* filename);
double** loadEMGData(char* filename, size_t rows, size_t cols);
int* loadLabels(char* filename, size_t rows);

static void get_file_paths(int dataset_id, char *training_emg_file, char *training_labels_file, char *testing_emg_file, char *testing_labels_file) {
    sprintf(training_emg_file, "foot/data/dataset%02d/training_emg.csv", dataset_id);
    sprintf(training_labels_file, "foot/data/dataset%02d/training_labels.csv", dataset_id);
    sprintf(testing_emg_file, "foot/data/dataset%02d/testing_emg.csv", dataset_id);
    sprintf(testing_labels_file, "foot/data/dataset%02d/testing_labels.csv", dataset_id);
}

void getDataWithValSet(int dataset,
                       double*** trainingData,
                       double*** validationData,
                       double*** testingData,
                       int** trainingLabels,
                       int** validationLabels,
                       int** testingLabels,
                       int* trainingSamples,
                       int* validationSamples,
                       int* testingSamples,
                       double validationRatio) {
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Reading data.\n");
    }

    if (validationRatio < 0.0) {
        validationRatio = 0.0;
    } else if (validationRatio > 1.0) {
        validationRatio = 1.0;
    }

    char training_emg_file_name[128], training_labels_file_name[128], testing_emg_file_name[128], testing_labels_file_name[128];
    get_file_paths(dataset, training_emg_file_name, training_labels_file_name, testing_emg_file_name, testing_labels_file_name);
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

    double** downSampledDataTrain = NULL;
    int* downSampledLabelsTrain = NULL;
    size_t downSampledSizeTrain = 0;
    double** downSampledDataTest = NULL;
    int* downSampledLabelsTest = NULL;
    size_t downSampledSizeTest = 0;
    down_sample(rawTrainingData, rawTrainingLabels, trainingRows, &downSampledDataTrain, &downSampledLabelsTrain, &downSampledSizeTrain);
    down_sample(rawTestingData, rawTestingLabels, testingRows, &downSampledDataTest, &downSampledLabelsTest, &downSampledSizeTest);

    if (downSampledDataTrain == NULL || downSampledLabelsTrain == NULL) {
        fprintf(stderr, "Error: downSampleDataTrain.\n");
        exit(EXIT_FAILURE);
    }
    if (downSampledDataTest == NULL || downSampledLabelsTest == NULL) {
        fprintf(stderr, "Error: downSampleDataTest.\n");
        exit(EXIT_FAILURE);
    }

    int class_counts[NUM_CLASSES];
    int class_targets[NUM_CLASSES];
    int class_assigned[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        class_counts[i] = 0;
        class_targets[i] = 0;
        class_assigned[i] = 0;
    }

    for (size_t i = 0; i < downSampledSizeTrain; i++) {
        int label = downSampledLabelsTrain[i];
        if (label >= 0 && label < NUM_CLASSES) {
            class_counts[label] += 1;
        }
    }

    size_t validation_total = 0;
    for (int cls = 0; cls < NUM_CLASSES; cls++) {
        int target = (int)(class_counts[cls] * validationRatio + 0.5);
        if (target > class_counts[cls]) {
            target = class_counts[cls];
        }
        class_targets[cls] = target;
        validation_total += (size_t)target;
    }

    size_t training_total = downSampledSizeTrain - validation_total;
    *trainingSamples = (int)training_total;
    *validationSamples = (int)validation_total;
    *testingSamples = (int)downSampledSizeTest;

    if (training_total > 0) {
        *trainingData = (double**)malloc(training_total * sizeof(double*));
        *trainingLabels = (int*)malloc(training_total * sizeof(int));
        if (*trainingData == NULL || *trainingLabels == NULL) {
            fprintf(stderr, "Malloc failed for training data.\n");
            exit(EXIT_FAILURE);
        }
    } else {
        *trainingData = NULL;
        *trainingLabels = NULL;
    }

    if (validation_total > 0) {
        *validationData = (double**)malloc(validation_total * sizeof(double*));
        *validationLabels = (int*)malloc(validation_total * sizeof(int));
        if (*validationData == NULL || *validationLabels == NULL) {
            fprintf(stderr, "Malloc failed for validation data.\n");
            exit(EXIT_FAILURE);
        }
    } else {
        *validationData = NULL;
        *validationLabels = NULL;
    }

    size_t train_idx = 0;
    size_t val_idx = 0;
    for (size_t i = 0; i < downSampledSizeTrain; i++) {
        int label = downSampledLabelsTrain[i];
        int to_validation = 0;
        if (label >= 0 && label < NUM_CLASSES && class_assigned[label] < class_targets[label]) {
            to_validation = 1;
        }

        if (to_validation) {
            if (val_idx < validation_total) {
                (*validationData)[val_idx] = (double*)malloc(NUM_FEATURES * sizeof(double));
                if ((*validationData)[val_idx] == NULL) {
                    fprintf(stderr, "Malloc failed for validation data row.\n");
                    exit(EXIT_FAILURE);
                }
                memcpy((*validationData)[val_idx], downSampledDataTrain[i], NUM_FEATURES * sizeof(double));
                (*validationLabels)[val_idx] = label;
                class_assigned[label] += 1;
                val_idx++;
            }
        } else {
            if (train_idx < training_total) {
                (*trainingData)[train_idx] = (double*)malloc(NUM_FEATURES * sizeof(double));
                if ((*trainingData)[train_idx] == NULL) {
                    fprintf(stderr, "Malloc failed for training data row.\n");
                    exit(EXIT_FAILURE);
                }
                memcpy((*trainingData)[train_idx], downSampledDataTrain[i], NUM_FEATURES * sizeof(double));
                (*trainingLabels)[train_idx] = label;
                train_idx++;
            }
        }
    }

    *testingData = downSampledDataTest;
    *testingLabels = downSampledLabelsTest;

    if (output_mode >= OUTPUT_DETAILED) {
        printf("Loaded data: training %d x %d, validation %d x %d, testing %d x %d\n",
               *trainingSamples,
               NUM_FEATURES,
               *validationSamples,
               NUM_FEATURES,
               *testingSamples,
               NUM_FEATURES);
    }

    freeData(downSampledDataTrain, downSampledSizeTrain);
    freeCSVLabels(downSampledLabelsTrain);
    freeData(rawTrainingData, trainingRows);
    freeCSVLabels(rawTrainingLabels);
    freeData(rawTestingData, testingRows);
    freeCSVLabels(rawTestingLabels);
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
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Reading data.\n");
    }
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
    if (output_mode >= OUTPUT_DETAILED) {
        printf("Loaded data: training %d x %d, testing %d x %d\n",
               *trainingSamples,
               NUM_FEATURES,
               *testingSamples,
               NUM_FEATURES);
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
