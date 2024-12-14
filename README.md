# HDC-Framework Horn

**Computer Science Project for Master CE**  
**Author: Marian Horn**

---

## Description
This project implements a well-structured framework for Hyperdimensional Computing (HDC). 

**Evaluation Example:** EMG-Classification of foot movements for the NOE-EMY project.

**Documentation:** 
- System and function details are available in the [`/doc`](./doc) folder and [`Abschlussbericht.pdf`](./Abschlussbericht.pdf).
- The development repository with all intermediate artifacts is available at: [Development Repository](https://gitlab.cs.fau.de/ew14ozom/emghandgestureshdc).

---

## Getting Started

### Clone Repository
```bash
git clone https://gitlab.cs.fau.de/ew14ozom/hdcframeworkhorn
```

### Build and Run the Model for EMG Classification
To build the model:
```bash
make foot
```

To train and evaluate the model for 4 different subjects:
```bash
./modelFoot
```

To explore the effects of different system parameters, modify the constants in [`configFoot.h`](./foot/configFoot.h), rebuild, and rerun the model.

---

## Building Your Own HDC Model

A template with dummy data is available in [`/customModel`](./customModel). This model can be run immediately with the dummy data:

```bash
make custom
```
```bash
./modelCustom
```

### Customization Options

#### Set System Parameters
Set system parameters in [`configCustom.h`](./customModel/configCustom.h).

#### Implement a Data Reader
Implement a data-reading function specific to your problem and data in [`dataReaderCustom.c`](./customModel/dataReaderCustom.c). 
- The training and testing data should be a 2D array of doubles, where each row represents a sample and each column represents a feature.
- The training and testing labels should be a 1D array of integers, where each element corresponds to the label of a sample.

#### Modify the Model
The main file of the custom model is [`modelCustom.c`](./customModel/modelCustom.c). 
- If working with categorical or discrete data, use the predefined functions in [`modelCustom.c`](./customModel/modelCustom.c).
- If working on time-series classification, replace `train_model_general_data()` and `evaluate_model_general_direct()` with `train_model_timeseries()` and `evaluate_model_timeseries_direct()`.
- For other types of data, extend the encoder to include additional encoding algorithms (e.g., for images, music, etc.). Use the following resources:
    - [`vector.h`](./hdc_infrastructure/vector.h): Provides vector structures (binary or bipolar, depending on [`configCustom.h`](./customModel/configCustom.h)) and related functions for initialization and modification.
    - [`operations.h`](./hdc_infrastructure/operations.h): For binding, bundling, permutation, and similarity checks.

#### Other Features
- Include [`preprocessor.h`](./hdc_infrastructure/preprocessor.h) to downsample the data.
- Include [`trainer.h`](./hdc_infrastructure/trainer.h) to process the training dataset, call the encoder on it, and set the associative memory.
- Include [`evaluator.h`](./hdc_infrastructure/evaluator.h) to test the model using the testing dataset.
- Include [`online_classifier.h`](./hdc_infrastructure/online_classifier.h) to evaluate a trained model on live data.

---

## Author
**Marian Horn**  
Email: [marian.horn@fau.de](mailto:marian.horn@fau.de)
