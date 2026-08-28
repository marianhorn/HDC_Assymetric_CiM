# HDC-Framework Horn

**Computer Science Project for Master CE**  
**Author: Marian Horn**

---

## Description
This project implements a well-structured framework for Hyperdimensional Computing (HDC). 

**Evaluation Example:** EMG-Classification of foot movements for the NOE-EMY project.

**Documentation:** 
- System and function details are available in the [`/doc`](./doc/html/mainpage_8h_source.html) folder.
- The development repository with all intermediate artifacts is available at: [Development Repository](https://gitlab.cs.fau.de/ew14ozom/emghandgestureshdc).

---

## Getting Started

### Clone Repository
```bash
git clone https://gitlab.cs.fau.de/ew14ozom/hdcframeworkhorn
cd hdcframeworkhorn/
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

## Core Components

- [`hdc_infrastructure`](./hdc_infrastructure): Core HDC implementation, including item memory, encoder, trainer, evaluator, quantizer, and vector operations.
- [`foot`](./foot): Foot EMG data reader, configuration, and executable model entry point.
- [`systemc/synthesis`](./systemc/synthesis): SystemC/HLS accelerator implementations and synthesis support files.

### Other Features
- Include [`preprocessor.h`](./hdc_infrastructure/preprocessor.h) to downsample the data.
- Include [`trainer.h`](./hdc_infrastructure/trainer.h) to process the training dataset, call the encoder on it, and set the associative memory.
- Include [`evaluator.h`](./hdc_infrastructure/evaluator.h) to test the model using the testing dataset.
- Include [`online_classifier.h`](./hdc_infrastructure/online_classifier.h) to evaluate a trained model on live data.

---

## Author
**Marian Horn**  
Email: [marian.horn@fau.de](mailto:marian.horn@fau.de)
