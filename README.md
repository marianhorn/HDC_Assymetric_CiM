# emgHandGesturesHDC

Computer science project for master CE (Marian Horn)

## Goal
Implement a well-structured version of EMG gesture recognition with HDC in C.
Should have the same functionality as torchHDExampleCode.py.

This serves for evaluation of HDC for EMG classification (hand and foot).

## Usage

To build the executable please execute:

```make hand```

```make foot```

To run the models execute: 

```./modelHand```

```./modelFoot```

## Files
- #### torchHDExampleCode.py:

    Contains python code for HDC classification of the handEMG dataset. Serves as reference program. Downloaded from [here](https://github.com/hyperdimensional-computing/torchhd/blob/main/examples/emg_hand_gestures.py).
- #### model.c:
    Main program, clones the functionality of torchHDExampleCode.py
- #### hdc_infrastructure:
    directory for general hdc components;
    contains:
    - #### assoc_mem.c: 
        Associative memory stores the trained class vectors
    - #### item_mem.c: 
        Generates and stores the hdc base vectors for encoding
    - #### encoder.c:
        Provides methods for encoding the input signals into hypervectors
    - #### operations.c:
        Provides methods for binding, bundling and permutation
    - #### trainer.c
        Provides method train_model, fills the assoc mem
    - #### evaluator.c
        Inference for testing data and evaluation of predictions

## Author
Marian Horn

marian.horn@fau.de

    