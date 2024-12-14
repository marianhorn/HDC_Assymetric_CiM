# emgHandGesturesHDC

Computer science project for master CE (Marian Horn)

## Description
Implementation of a well-structured framework for hyperdimensional computing (HDC).
Evaluation Example: EMG-Classification of foot movement for NOE-EMY project.
For documentation about system and functions please refer to /doc and Abschlussbericht.pdf.
The development repository of this project with all its intermediate artifacts is available at: https://gitlab.cs.fau.de/ew14ozom/emghandgestureshdc


## Getting started

Clone repository

```git clone https://gitlab.cs.fau.de/ew14ozom/hdcframeworkhorn```


To build the model for EMG classification run: 

```make foot```

To train and evaluate the model for 4 different subjects run:

```./modelFoot```

To explore effects of different system parameters change Constants in /foot/configFoot.h, rebuild and rerun.

## Building your own HDC model

A template with dummy data is available in /customModel. The model can be run immediately with only dummy data by:

```make custom```

```./modelCustom```

System parameters can be set in /customModel/configCustom.h

A data reading function specific to your problem and data has to be implemented by the user in /customModel/dataReaderCustom.c

The models main file is /customModel/modelCustom.c

If working on timeseries classification change train_model_general_data() and evaluate_model_general_direct() to train_model_timeseries() and evaluate_model_timeseries_direct().

If necessary extend the encoder by additional encoding algorithms (eg. for pictures, music...).

    Include /hdc_infrastructure/operations.h to use binding, bundling, permutation and similarity check.
    Include /hdc_infrastructure/vector.h to utilize vector struct, that contains binary or bipolar vectors, depending on configCustom.h, and functions for initialization and modification.

Include /hdc_infrastructure/online_classifier.h to evaluate a trained model on live data
## Author
Marian Horn

marian.horn@fau.de

    