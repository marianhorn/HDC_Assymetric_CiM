# Dataset Overview

Datasets in the subfolders `/dataset<number>` contain the following measurements:

## Datasets
- **00:** Healthy reference data 1
- **01:** Healthy reference data 2
- **02:** Patient data 1
- **03:** Patient data 2

## Data Processing Details
- **Moving Window:**  
  A moving window of **252 ms** (504 samples) was applied to the raw EMG data to calculate the RMS (Root Mean Square).

- **Task Duration:**  
  All tasks were recorded for **10 seconds**, where the subject had to hold the foot in the end position.

- **Data Split:**  
  The dataset is split into:
  - **80% Training Data**
  - **20% Testing Data**

## Scaling Methodology
The data is scaled in two steps:
1. **Standard Scaling:**  
   Data is standardized to have a mean of 0 and a standard deviation of 1.
2. **Min-Max Scaling:**  
   Data is scaled to the value range \([-1, 1]\).
