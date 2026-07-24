// SIMULATION / SOFTWARE ONLY: This file is not part of the HLS synthesis target.
// It defines host-side containers for CSV-loaded foot EMG datasets.
#ifndef SYSTEMC_HDC_FOOT_DATASET_LOADER_H
#define SYSTEMC_HDC_FOOT_DATASET_LOADER_H

#include <vector>
#include "config_systemc.h"

namespace hdc_systemc {

struct DatasetSplit {
    std::vector<double> data;
    std::vector<int> labels;
    int samples;

    DatasetSplit() : samples(0) {}

    const double *raw_data() const { return data.empty() ? 0 : data.data(); }
    const int *raw_labels() const { return labels.empty() ? 0 : labels.data(); }
};

struct FootDataset {
    DatasetSplit training;
    DatasetSplit testing;
};

FootDataset load_foot_dataset_by_id(int dataset_id);

} // namespace hdc_systemc

#endif
