# Kaggle TGS Salt Identification Challenge. b.e.s. & phalanx 1st Place Solution

## Paper describing the solution: 

**Semi-Supervised Segmentation of Salt Bodies in Seismic Images using an Ensemble of Convolutional Neural Networks**  
 ***German Conference on Pattern Recognition (GCPR), 2019***  
*Yauhen Babakhin, Artsiom Sanakoyeu, Hirotoshi Kitamura*   
https://arxiv.org/abs/1904.04445 

Kaggle post about the solution: [link](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69291).

## ENVIRONMENT

The solution is available as a Docker container. The following dependecies should be installed:

* Python 3.5.2
* CUDA 9.0
* cuddn 7
* nvidia drivers v.384
* [Docker](https://www.docker.com/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## DATA SETUP

Download and unzip [competition data](https://www.kaggle.com/c/tgs-salt-identification-challenge/data) into `data/` directory.
One could specify local path to the new test images in `SETTINGS.json` file (`NEW_TEST_IMAGES_DATA` field). The competition test data is used by default.

## WEIGHTS SETUP

To get the weights from the final stage models download them from [google drive](https://drive.google.com/file/d/12iXDUhBTC6596MLAC2aiN-GDVqBbGBWh/view?usp=sharing) and unzip into corresponding `bes/weights/` and `phalanx/weights` directories.

## DOCKER SETUP

To build and start a docker container run:
```bash
cd docker 
./build.sh
./run.sh
```

## MODEL BUILD

1. train models from scratch

    a) trains all models from scratch

    b) expect this to run for about 16 days on a single GTX1080Ti
    
2. make prediction

    a) uses weights from the final stage models to make predictions

    b) expect this to run for 3.5 hours for 18,000 test images on a single GTX1080Ti

Commands to run each build are presented below:

### 1. train models (creates model weights in bes/weights and phalanx/weights)
```bash
./train.sh
```

### 2. make prediction (creates predictions/test_prediction.csv)
```bash
./predict.sh
```

## ADDITIONAL NOTES

1. Model weights are saved in bes/weights and phalanx/weights for b.e.s. and phalanx models respectively

2. Individual model predictions before ensembling are stored in bes/predictions (lots of .png images) and phalanx/predictions (.npy files)

3. Scripts to generate initial folds and jigsaw mosaics are located in bes/datasets: generate_folds.py and Generate_Mosaic.R

## CITATION
If you find this code useful, please cite the preprint:

```
@journal{tgsSaltBodiesSegmentation2019,
  title={Semi-Supervised Segmentation of Salt Bodies in Seismic Images using an Ensemble of Convolutional Neural Networks},
  author={Babakhin, Yauhen, and Sanakoyeu, Artsiom, and Kitamura, Hirotoshi},
  journal={German Conference on Pattern Recognition (GCPR)},
  year={2019}
}
```
