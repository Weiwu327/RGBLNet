# RGBLNet-ear-counting
## Prerequisite

Python 3.7

Pytorch 1.7.0

## Code structure

`density_map.py` To generate the density map. 

`dataset.py` and `transforms.py` For data preprocess and augmentation. 

`model.py` The structure of the network. 

`train.py` To train the model. 

`eval.py` To test the model. 

`predict_density.py` To predict the density maps. 

## Train & Test

For training, run

`python train.py --data_path="path to dataset" --save_path="path to save checkpoint"`

For testing, run

`python eval.py --data_path="path to dataset" --save_path="path to checkpoint"`

## Result

ShanghaiTech part A: epoch367 MAE 60.43 MSE 98.24

![](./logs/A.png)

ShanghaiTech part B: epoch432 MAE 6.38 MSE 10.99

![](./logs/B.png)
