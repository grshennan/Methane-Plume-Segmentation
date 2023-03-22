# Methane-Plume-Segmentation

This repository contains the code base for our work entitled *Methane plume detection with U-Net segmentation on Sentinel-2 image data*

It is based on the work of Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D., "Characterization of Industrial Smoke Plumes from Remote Sensing Data" Tackling Climate Change with Machine Learning Workshop, NeurIPS 2020.

 
## About this Project

Methane ($CH_4$) emissions have a significant impact on increasing global warming. To help track methane leaks and uncontrolled emissions, we looked into an open source way to detect methane plumes on Sentinel-2 satellite images.
Many satellites have on-board instruments measuring radiation at methane-absorbing wavelengths allowing for physical-based methods to infer methane plumes detection and quantification. However these solutions can be error-prone and rely heavily on image processing and manual fine tuning. Therefore we decided to use deep-learning and particularly a U-Net Convolutional Neural Network for effective methane plume detection.

We built a train dataset of approximatly 5000 pictures and a test dataset of 200 images composed of polygons of geographical coordinates defining the outline of methane plumes from Ehret \textit{et al.}'s dataset and the corresponding Sentinel-2 satellite images. The U-Net we trained for methane detection with this dataset shows mixed results. Even though we can visually validate that the model is able to correctly generate segmentation masks for the training data, the IoU (Intersection over Union) is very unstable on the test dataset, varying from 0 to 0.6 with a standard deviation of 0.15 meaning the model is not generalizing properly. 
Despite numerous attempts to prevent overfitting using various regularization and data augmentation methods, the performances on the test dataset did not improve. This leads us to think that the poor generalization capacity of our model comes from the size and variety of our dataset.

The full code and data are available in this respository.

## Content

`data/`: Train Dataset + Test Dataset

Each dataset containing : 

`input_tiles/` : tiles used for the input

`output_matrix/` : matrix containing the polygons used as a label

`tci_tiles/` : true color tiles

 
## How to Use

Download this repository and decompress the latter. For
both model training and evaluation, you will not have to modify the directory
paths appropriately so that they point to the image and segmentation label
data.

The model can be trained by invoking:

    python train.py
