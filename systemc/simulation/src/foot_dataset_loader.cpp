#include "foot_dataset_loader.h"
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <systemc>

namespace hdc_systemc {

namespace {

void build_dataset_paths(int dataset_id,
                         char *training_emg_path,
                         char *training_labels_path,
                         char *testing_emg_path,
                         char *testing_labels_path) {
    std::snprintf(training_emg_path, 128, "../foot/data/dataset%02d/training_emg.csv", dataset_id);
    std::snprintf(training_labels_path, 128, "../foot/data/dataset%02d/training_labels.csv", dataset_id);
    std::snprintf(testing_emg_path, 128, "../foot/data/dataset%02d/testing_emg.csv", dataset_id);
    std::snprintf(testing_labels_path, 128, "../foot/data/dataset%02d/testing_labels.csv", dataset_id);
}

DatasetSplit load_emg_split(const char *emg_path, const char *labels_path) {
    if (emg_path == 0 || labels_path == 0) {
        SC_REPORT_FATAL("foot_dataset_loader", "dataset paths must not be null");
    }

    std::ifstream emg_file(emg_path);
    if (!emg_file.is_open()) {
        SC_REPORT_FATAL("foot_dataset_loader", "failed to open EMG csv");
    }

    std::ifstream label_file(labels_path);
    if (!label_file.is_open()) {
        SC_REPORT_FATAL("foot_dataset_loader", "failed to open label csv");
    }

    DatasetSplit split;
    std::string line;

    if (!std::getline(emg_file, line)) {
        SC_REPORT_FATAL("foot_dataset_loader", "EMG csv is empty");
    }
    if (!std::getline(label_file, line)) {
        SC_REPORT_FATAL("foot_dataset_loader", "label csv is empty");
    }

    while (std::getline(emg_file, line)) {
        if (line.empty()) {
            continue;
        }

        std::istringstream iss(line);
        std::string token;
        int feature_count = 0;
        while (std::getline(iss, token, ',')) {
            if (feature_count >= NUM_FEATURES) {
                SC_REPORT_FATAL("foot_dataset_loader", "too many EMG columns in csv row");
            }
            split.data.push_back(std::atof(token.c_str()));
            ++feature_count;
        }

        if (feature_count != NUM_FEATURES) {
            SC_REPORT_FATAL("foot_dataset_loader", "wrong number of EMG columns in csv row");
        }
        ++split.samples;
    }

    split.labels.reserve(static_cast<std::vector<int>::size_type>(split.samples));
    while (std::getline(label_file, line)) {
        if (line.empty()) {
            continue;
        }
        split.labels.push_back(std::atoi(line.c_str()));
    }

    if (static_cast<int>(split.labels.size()) != split.samples) {
        SC_REPORT_FATAL("foot_dataset_loader", "EMG/label row count mismatch");
    }

    return split;
}

} // namespace

FootDataset load_foot_dataset_by_id(int dataset_id) {
    char training_emg_path[128];
    char training_labels_path[128];
    char testing_emg_path[128];
    char testing_labels_path[128];

    build_dataset_paths(dataset_id,
                        training_emg_path,
                        training_labels_path,
                        testing_emg_path,
                        testing_labels_path);

    FootDataset dataset;
    dataset.training = load_emg_split(training_emg_path, training_labels_path);
    dataset.testing = load_emg_split(testing_emg_path, testing_labels_path);
    return dataset;
}

} // namespace hdc_systemc
