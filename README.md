Hello!

Below you can find an outline of how to reproduce first place solution for the TGS Salt Identification competition by team 'b.e.s. &amp; phalanx'.
If you run into any trouble with the setup/code or have any questions please contact us at y.babakhin@gmail.com (b.e.s.) or ritskitamura@gmail.com (phalanx).

### ARCHIVE CONTENTS

* bes                      : contains train and prediction code for b.e.s. models
* data                     : contains all the initial data + generated folds, mosaics, pseudolabels
* docker                   : docker utilities
* phalanx                  : contains train and prediction code for phalanx models
* predictions              : model predictions

#HARDWARE: (The following specs were used to create the original solution)

* Ubuntu 16.04 LTS (256 GB boot disk)
* 12 CPUs, 30 GB memory
* 1 x GeForce GTX 1080 Ti

#SOFTWARE:

* Python 3.5.2
* CUDA 9.0
* cuddn 7
* nvidia drivers v.384
* Docker (https://www.docker.com/)
* nvidia-docker (https://github.com/NVIDIA/nvidia-docker)

#DATA SETUP

Inital Train and Test data together with .csv files are already available in data/ directory.
One could specify local path to the new test images in SETTINGS.json file (NEW_TEST_IMAGES_DATA field). The initial test data is used by default.

#DOCKER SETUP

To build and start a docker container run:
```bash
cd docker 
./build.sh
./run.sh
```

#MODEL BUILD: There are three options to produce the solution.

1. very fast prediction

    a) runs for about 20 minutes
    
    b) uses precomputed neural network predictions
    
2. ordinary prediction

    a) expect this to run for 3.5 hours for 18,000 test images
    
    b) uses saved model weights
    
    c) could be used to predict new test images
    
3. retrain models

    a) expect this to run for about 16 days
    
    b) trains all models from scratch
    
    c) follow this with (2) to produce entire solution from scratch

shell commands to run each build is below

#1) very fast prediction (overwrites predictions/test_prediction.csv)
```bash
./make_submission.sh
```

#2) ordinary prediction (overwrites predictions/test_prediction.csv)
```bash
./predict.sh
```

#3) retrain models (overwrites model weights in bes/weights and phalanx/weights)
```bash
/.train.sh
```