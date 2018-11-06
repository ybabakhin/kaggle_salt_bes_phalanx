Hello!

Below you can find the instructions to reproduce the first place solution for the 'TGS Salt Identification Challenge' by team 'b.e.s. &amp; phalanx'.

If you run into any trouble with the setup/code or have any questions please contact us at y.babakhin@gmail.com (b.e.s.) or ritskitamura@gmail.com (phalanx).

To read the detailed solution, please, refer to *First_Place_TGS_Salt.pdf*.

## ARCHIVE CONTENTS

* bes                      : contains train and prediction code for b.e.s. models
* data                     : contains all the initial data + generated folds, mosaics, pseudolabels
* docker                   : docker utilities
* phalanx                  : contains train and prediction code for phalanx models
* predictions              : model predictions

## HARDWARE (The following specs were used to create the original solution)

* Ubuntu 16.04 LTS (256 GB boot disk)
* 12 CPUs, 30 GB memory
* 1 x GeForce GTX 1080 Ti

## SOFTWARE

* Python 3.5.2
* CUDA 9.0
* cuddn 7
* nvidia drivers v.384
* Docker (https://www.docker.com/)
* nvidia-docker (https://github.com/NVIDIA/nvidia-docker)

## DATA SETUP

Inital Train and Test image data together with .csv files are already available in data/ directory.
One could specify local path to the new test images in SETTINGS.json file (NEW_TEST_IMAGES_DATA field). The initial test data is used by default.

## DOCKER SETUP

To build and start a docker container run:
```bash
cd docker 
./build.sh
./run.sh
```

## MODEL BUILD (There are three options to produce the solution)

1. very fast prediction

    a) takes about 20 minutes
    
    b) uses precomputed neural network predictions
    
2. ordinary prediction

    a) expect this to run for 3.5 hours for 18,000 test images
    
    b) uses saved model weights
    
    c) could be used to predict new test images
    
3. retrain models

    a) expect this to run for about 16 days
    
    b) trains all models from scratch
    
    c) follow this with (2) to produce entire solution from scratch

Commands to run each build are presented below:

### 1. very fast prediction (overwrites predictions/test_prediction.csv)
```bash
./make_submission.sh
```

### 2. ordinary prediction (overwrites predictions/test_prediction.csv)
```bash
./predict.sh
```

### 3. retrain models (overwrites model weights in bes/weights and phalanx/weights)
```bash
/.train.sh
```

## ADDITIONAL NOTES

1. Model weights are located in bes/weights and phalanx/weights for b.e.s. and phalanx models respectively

2. Individual model predictions before ensembling are stored in bes/predictions (lots of .png images) and phalanx/predictions (.npy files)

3. Scripts to generate initial folds and jigsaw mosaics are located in bes/datasets: generate_folds.py and Generate_Mosaic.R