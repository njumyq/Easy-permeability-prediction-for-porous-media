# Easy-permeability-prediction-for-porous-media
Feature extraction of porous media; permeability prediction; machine learning; LSTM
## Description

## Installation
* numpy
* pandas
* matplotlib
* sklearn
* scipy
* scikit-image
* PyTorch
## Data preparation
You should pre-express the 3D image of the porous medium as a folder composed of multiple slices (csv file format or image format). The name of each csv file is the slice number, and the folder name is the sample number.
## Feature extraction

From feature_extraction directory, run:`python feature_extraction.py`  
Then you could achieve **porosity_2d.csv**, **pecific_perimeter.csv**, **euler_number.csv** and **euler_number_std.csv** of the test samples.
