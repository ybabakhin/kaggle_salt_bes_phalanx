## Kaggle TGS Salt Identification Challenge. b.e.s. & phalanx 1st Place Solution

To read the detailed solution, please, refer to the [Kaggle post] (https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69291)

## ENVIRONMENT

The solution is available as a Docker container. The following dependecies should be installed:

* Python 3.5.2
* CUDA 9.0
* cuddn 7
* nvidia drivers v.384
* Docker (https://www.docker.com/)
* nvidia-docker (https://github.com/NVIDIA/nvidia-docker)

## DATA SETUP

Download and unzip [competition data] (https://www.kaggle.com/c/tgs-salt-identification-challenge/data) into data/ directory.
One could specify local path to the new test images in SETTINGS.json file (NEW_TEST_IMAGES_DATA field). The competition test data is used by default.

## DOCKER SETUP

To build and start a docker container run:
```bash
cd docker 
./build.sh
./run.sh
```

## MODEL BUILD

1. train models

    a) expect this to run for about 16 days on a single GTX1080Ti
    
    b) trains all models from scratch
    
2. make prediction

    a) expect this to run for 3.5 hours for 18,000 test images on a single GTX1080Ti
    
    b) uses saved model weights
    

Commands to run each build are presented below:

### 1. train models (creates model weights in bes/weights and phalanx/weights)
```bash
/.train.sh
```

### 2. ordinary prediction (creates predictions/test_prediction.csv)
```bash
./predict.sh
```


## ADDITIONAL NOTES

1. Model weights are saved in bes/weights and phalanx/weights for b.e.s. and phalanx models respectively

2. Individual model predictions before ensembling are stored in bes/predictions (lots of .png images) and phalanx/predictions (.npy files)

3. Scripts to generate initial folds and jigsaw mosaics are located in bes/datasets: generate_folds.py and Generate_Mosaic.R
